# OCRDINO：DINO 检测与共享特征 CTC 识别

基于 **MMDetection 3.3.0** 的单模型 OCR 第一版，采用 **DINO 文本区域检测 + 共享 C2 区域采样 + 4 层卷积 + 单层 BiLSTM + CTC**。输入一张完整图片，由同一个模型先定位文本区域，再读取区域特征识别字符串。训练数据只需要文本区域的 `bbox + text`，不需要字符框、字符归属或阅读序号标注。

检测部分继承官方 **R50 DINO-4scale improved**，使用用户准备的 COCO 预训练权重初始化。训练默认使用全部训练图片，不划分验证集，预先指定训练轮数并保留最后一轮权重。本版先面向英文水平单词或单行文字，同时提供明确的字符字典接口，便于之后训练中文。

新增逻辑集中在 `projects/OCRDINO`：项目自己的 `OCRDINO` 继承 MMDetection 的 DINO，独立实现共享 C2、区域识别、有限 DN 和数据接口。原 `mmdet/models/detectors/dino.py` 及 MMDetection 核心实现不作修改。

**本次四个英文数据包的推荐流程**：先按[数据审计与转换](#四个英文数据包审计与转换)准备数据，再按[24 轮预训练与 12 轮微调](#英文实验24-轮预训练--12-轮-total-text-微调)训练。MLT2017、SynText1、SynText2 合并用于预训练，Total-Text train 用于微调；Total-Text test 单独保留。后面的通用训练示例供其他自备数据使用。

## 模型结构

### 1. “两阶段，但只有一个模型”是什么意思

两个计算阶段分别是**文本区域检测**和**区域内字符串识别**，它们共用一个 ResNet-50，一次前向完成。识别阶段从共享特征图取样，不将原图裁成小图片后再运行第二个完整 backbone，也不加载另一个独立识别模型。

训练同时计算 DINO 检测损失与 CTC 识别损失。识别分支通过区域特征向共享 R50 反向传播；检测分支通过四尺度 Transformer 向相同 R50 反向传播。因此两个任务共同学习视觉表示。

DINO 配置中的 `as_two_stage=True` 是 DINO 内部“Encoder 候选框初始化 Decoder”的术语，与这里“检测后识别”的两个计算阶段不是同一个划分。

### 2. 整体数据流

```text
输入整图 [B,3,Hp,Wp]
                │
          共享 ResNet-50
                │
     ┌──────────┴─────────────────────────────────┐
     │                                            │
     ▼                                            ▼
 C3/C4/C5                              C2 [B,256,Hp/4,Wp/4]
     │                                            │
 ChannelMapper                             1×1 Conv 256→128
     │                                            │
 P3/P4/P5/P6（各256通道）                   共享高分辨率识别特征
     │                                            │
 6层 Deformable Encoder                           │
     │                                            │
 单类文本候选 + Top-900                           │
     │                                            │
 900个内容query + 图像相关参考框                    │
     │                                            │
 6层 DINO Decoder                                 │
     │                                            │
 文本前景分数 + bbox ─── 选择文本区域 ──────────────┤
                                                  ▼
                                      RoIAlign [M,128,8,W]
                                                  │
                                4层卷积：高度8→8→4→2→1
                                                  │
                                       序列 [M,W,128]
                                                  │
                                单层BiLSTM，每方向hidden=128
                                                  │
                                       序列 [M,W,256]
                                                  │
                                   Linear 256→V+1（含blank）
                                                  │
                                       CTC贪心解码为字符串
                                                  ▼
                                     原图bbox + 分数 + text
```

图示是推理路径。训练默认用 GT 区域代替筛选后的预测框来采样；可选地混入定位质量足够好的匹配预测框。`B` 是每卡图片数，`Hp/Wp` 是补齐后的图片尺寸，`M` 是实际进入识别头的区域数，`W` 是该区域宽度桶，`V` 是字典中的字符种类数。只有一个检测 query 分支，没有字符 query、字符归属头或第二次检测 Decoder 调用。

### 3. Backbone、四尺度特征与 C2 分支

| R50 输出 | stride | 通道数 | 用途 |
| --- | ---: | ---: | --- |
| C2 | 4 | 256 | 用于区域识别，保留较多笔画细节 |
| C3 | 8 | 512 | 进入四尺度检测 neck |
| C4 | 16 | 1024 | 进入四尺度检测 neck |
| C5 | 32 | 2048 | 进入检测 neck，并生成第四个更粗尺度 |

检测 neck 保持官方 ChannelMapper：C3/C4/C5 经通道投影得到 256 通道特征，另从 C5 生成名义 stride 64 的特征。四尺度输入的名义 stride 是 8、16、32、64。这里没有额外引入 FPN 或 BiFPN；尺度间交互由 Deformable Transformer 处理。

C2 经一个无 bias 的 `1×1 Conv(256,128)` 投影后用于区域采样。**整张 C2 不进入 Encoder，也不先放大回整图分辨率。** 保留 C2 输出不改变 R50-4scale 的 Transformer 尺度数，其余检测主体的参数形状仍与官方 checkpoint 对齐。

原图先等比例缩放，再在右下侧补齐到 32 的倍数。采样框使用缩放后的输入像素坐标，RoIAlign 的 `spatial_scale=1/4` 将框映射到 C2；最终输出再恢复为原图坐标。批次 padding 不改变原始标注语义。

### 4. DINO 如何检测文字

检测部分保持六层 Encoder、六层 Decoder、256 维 Transformer 特征和 900 个匹配 query。Encoder 在四尺度特征上使用可变形注意力，输出整图 memory，并给出文本候选分数和参考框；按分数选择 Top-900 参考框，配合可学习的 900 行内容 embedding 进入 Decoder。

Decoder 的 self-attention 让文本候选交互，deformable cross-attention 从四尺度特征读取信息，各层回归头迭代细化参考框。每个 query 只负责一个完整文本区域，分类头输出一个“文本前景”类别；完整文字内容由后面的区域识别头输出。

检测采用 Hungarian 一对一匹配，代价和损失沿用 DINO 的前景 focal、框 L1、GIoU。Encoder 候选、六层 Decoder 和 DN 分支接受各自的检测监督。识别仅运行一次，不在六层 Decoder 上分别生成字符序列。

### 5. 900 个 query、文字长度与字典大小的关系

本版的 900 个 query 对应**900 个文本区域槽位**，不对应 900 个字符。一个区域 `OPEN` 只需要一个检测 query，识别头在该区域的时间步上输出四个字符；一个含 30 个字的文本行也只占一个检测 query。

每个识别时间步使用同一个 `Linear(256,V+1)` 字符分类器。英文的 95 个可打印 ASCII 字符、将来的数千汉字，都是这个分类器可选择的类别；不需要为每种字单独增加一个 query。字典大小决定输出类别轴，序列宽度决定可用时间步，query 数量决定文本实例容量，三者相互独立。

推理默认在检测输出中按分数筛选，最多识别 100 个文本区域；这是为显存和延迟设置的运行限制，并不是 900 个 query 的理论容量。如果一张图有更多区域，需要提高 `model.rec_test_cfg.max_per_img`，并同时检查 `model.test_cfg.max_per_img`。有效训练区域数超过检测 query 容量时应调整 query 容量，而不是将多余 GT 当作背景。

### 6. 区域采样与宽度分桶

区域采样使用 MMCV RoIAlign，输出高度固定为 8。宽度由区域宽高比决定：

```text
requested_width = ceil(bbox_width / bbox_height × width_scale)
width_scale = 16
width_buckets = (32, 64, 128)
```

将请求宽度向上取到最近的桶，超过最大桶时使用最大桶。相同桶中的区域一起进入识别头，各桶分开计算，不将全部短单词补齐到最长行的宽度。没有跨桶的序列 padding，因此双向 LSTM 不会把补齐区域当作真实右侧上下文。

训练和推理使用相同的几何选桶规则。训练不会根据真实字符串偷偷增加采样宽度，避免训练时才有的额外信息改变输入分布。CTC 能否容纳标注会显式检查，详见下文；桶或宽度比例不够时需要调整配置。

第一版只支持水平轴对齐 bbox，不预测旋转角、四边形或曲线控制点。倾斜文字的外接矩形可以读取部分信息，但不能等价于几何校正。弯曲 Total-Text、竖排、一个框内多行文字，需要后续实现相应的区域表示和展开方式。

### 7. 四层卷积与单层 BiLSTM

| 操作 | 输出形状 | 作用 |
| --- | --- | --- |
| RoIAlign，采样率 2 | `[M,128,8,W]` | 从同一 C2 特征提取区域 |
| 3×3 Conv，stride 1，GN、ReLU | `[M,128,8,W]` | 提取局部笔画结构 |
| 3×3 Conv，stride `(2,1)`，GN、ReLU | `[M,128,4,W]` | 压缩高度，保留横向位置 |
| 3×3 Conv，stride `(2,1)`，GN、ReLU | `[M,128,2,W]` | 继续聚合竖直方向信息 |
| 2×1 Conv，ReLU | `[M,128,1,W]` | 高度压到 1 |
| 去掉高度维并转置 | `[M,W,128]` | 形成横向序列 |
| 单层 BiLSTM，每方向 hidden 128 | `[M,W,256]` | 融合左右视觉上下文 |
| Linear，256→`V+1` | `[M,W,V+1]` | 每个时间步预测字符或 blank |

四个区域卷积均不带 bias。前三层 GroupNorm 使用 32 组，最后一层不添加归一化。BiLSTM 的前向和后向输出拼接，不是相加，也没有堆叠第二层 LSTM。没有词典搜索、额外语言模型或自回归文字 Transformer。

卷积负责视觉特征，LSTM 负责序列上下文，CTC 负责可变长度对齐。字符分类器在所有区域和时间步间共享，因此字符 `A` 不绑定某个固定位置，也不将 `ABCD` 当作一个完整类别。有限训练内容仍会产生数据偏置，不能仅凭这种结构保证只训练 `ABCD` 就一定识别所有新排列。

### 8. CTC 如何使用整串标注训练

给定区域框和文字 `HELLO`，不需要提供每个字母的边界。模型可产生如下时间步路径，其中 `∅` 为 blank：

```text
H H ∅ E E L ∅ L O O
```

CTC 先合并相邻重复字符，再删除 blank，得到 `HELLO`。中间 blank 允许两个相邻的 `L` 保留下来。训练时 CTC 对能得到目标字符串的有效路径计算总概率，并不是先人为划出五个字符框。

字典只保存真实字符；blank 固定为 ID 0，字典中第 `i` 个字符对应 ID `i+1`。默认大小写、标点和空格均区分，不自动 lower、不去首尾空格、不把未知字映射成 UNK。空格是可学习字符，blank 是对齐符号，两者不同。

设文字长度为 `L`，相邻相同字符对数为 `R`，必须满足：

```text
实际时间步 T >= L + R
```

例如 `BOOK` 长度 4，含一对相邻 `OO`，至少需要 5 个时间步。代码在训练前检查这一条件，不依赖 `zero_infinity=True` 将不可能的样本静默置零。文字有未知字符、非法控制符或序列长度不足时，必须修正字典/标注/宽度配置，或明确设置 `rec_ignore=true`，不能在未说明的情况下丢弃识别监督。

为了稳定训练，RoIAlign、BiLSTM 和 CTC 的数值敏感路径使用 FP32；卷积在 AMP 上下文中运行。CTC 的 FP32 logits/log-softmax 仍然占用显存，中文大字典尤其需要控制区域数和时间步。

### 9. 训练时使用什么区域

默认 `rec_train_cfg.pred_box_prob=0.0`，识别分支使用有效 GT bbox。这让新初始化的识别头首先看到完整、位置可靠的文字。每批最多随机选择 128 个可识别区域，限制识别激活；这个限制只作用于识别任务，全部有效检测 GT 仍参与正常 Hungarian 匹配。

可选地将 `pred_box_prob` 调大。在达到 `pred_box_start_epoch` 后，对与 GT 匹配、IoU 至少为 `min_pred_iou`、覆盖 GT 面积至少 95%，且其宽度桶容得下该转录的预测框，以给定概率替代对应 GT 框；不合格预测仍使用 GT。epoch 按 MMEngine 从 0 开始计数，默认阈值 5 表示第 6 轮起允许混入。它用于减少训练时精确 GT 框与推理时预测框的差异，不是额外的模型。

第一版的采样框坐标不向框回归器传递识别梯度；采样特征仍保留梯度，识别损失更新共享 R50 和识别头。联合训练不等于所有离散选择、Hungarian 匹配与输出文字都可微。

### 10. 损失、有限 DN 与异常监督

| 损失 | 默认权重 | 监督 |
| --- | ---: | --- |
| 文本前景 focal | 2.0 | 文本实例与背景 |
| bbox L1 | 5.0 | 轴对齐区域坐标 |
| GIoU | 2.0 | 区域重叠与几何关系 |
| CTC | 1.0 | 完整字符串 |

检测权重沿用 improved DINO，CTC 接在最后的区域特征上。检测辅助层和 DN 损失按原 DINO 方式分别计算，不是只对最后一层进行定位监督。识别有效区域数在多卡下进行同步归一化，避免不同 rank 文字数量不同导致损失权重含义变化。

默认 DN query 总数有硬上限 128。每图只为 DN 随机采样最多 64 个 GT，若本批最多采样 `K` 个，使用 `max(1, floor(128/(2K)))` 组正负去噪副本，实际总槽数不超过 128。DN target 使用同一子集和次序，普通检测匹配始终保留全部 GT。训练 Decoder 长度最多为 `900+128=1028`，推理去掉 DN 后为 900，避免去噪长度随密集文字数量无界增加。全空 GT 批次不生成有效 DN 正例。

异常标注区分处理：

- 框有效、文字未知或损坏：`rec_ignore=true`，保留检测，关闭该区域 CTC。
- 可定位但整体忽略：`ignore=true`，不作为正例；正常 Hungarian 匹配与 Encoder 候选损失中，未匹配预测覆盖 ignored 区域的比例（IoF）达到 0.5 时，屏蔽其背景分类监督。DN 仍针对其采样到的有效 GT 构造正负监督。
- 框非有限、非正面积或超界：转换器记录异常，默认排除该框，不删除图片。若因此漏掉可检测文本，该图标记标注不完整，关闭未匹配 query 的背景监督，避免把缺失 GT 当背景。
- `--clip-boxes` 是显式的数据转换选择，不在训练时静默裁框。坐标问题和原始记录进入审计报告，自动规则不保证语义转录本身正确。

### 11. 推理输出

推理在缩放后的图像坐标中得到检测框，按文本检测分数和区域数量上限筛选后执行识别，再将 bbox 恢复为原图坐标。识别使用 CTC 贪心解码，不做 beam search 或词典纠错。

检测分数与识别分数分别输出。识别分数由贪心解码中实际保留字符的概率几何平均得到，空串分数为 0；它不等于整串完全正确的校准概率。另提供 `ocr_score=检测分数×识别分数` 供排序使用，不覆盖两个原始分数，也不声称乘积经过概率校准。输出可能包含空串，查看结果时应同时检查检测位置和识别内容。

| `pred_instances` 字段 | 含义 |
| --- | --- |
| `bboxes` | 原图像素 xyxy：`[x1,y1,x2,y2]` |
| `scores` / `labels` | 检测前景分数 / 文本类别 0 |
| `texts` | CTC 解码字符串 |
| `text_scores` | 识别分数 |
| `ocr_scores` | 检测与识别分数的乘积 |

推理脚本 JSON 将它们导出为每条预测的 `bbox/text/detection_score/text_score/ocr_score`，顶层标明 `bbox_format="xyxy"`。注意输出 xyxy 与训练 JSON 的 xywh 格式不同。

### 12. 官方预训练权重迁移

复用官方 R50、四尺度 ChannelMapper、Encoder、Decoder、900 行 query embedding 和兼容的框回归参数。R50 原本就包含 C2，因此保留 C2 输出不需要另找 R50-5scale 权重。

COCO 的 80 类分类分支与 DN 类别 embedding 不迁移到单类文本任务。新文本前景分类器、C2 的 1×1 投影、四层区域卷积、BiLSTM、字符分类层按本项目初始化。预训练迁移提供定位和视觉表示起点，不赋予模型现成的英文或中文识别能力，识别仍需相应数据训练。

使用转换工具生成新的初始化 checkpoint，不覆盖官方源权重。初始化用于 `load_from`；训练中断恢复则使用训练 checkpoint 的 `--resume`，两种用途不能混用。

## 显存控制

本实现将下列约束落实到结构和配置中：

1. 主干只运行一次；C2 仅区域采样，不进入全图 Transformer。
2. 不将 C2 上采样到输入分辨率；同通道下 stride 4 特征的空间面积约为全分辨率的 1/16。
3. 识别只在最后运行一次；不会对每层 Decoder 或全部背景 query 执行字符序列识别。
4. 识别区域训练默认最多 128 个、推理默认最多 100 个；短文本按宽度分桶。
5. 小型识别头使用 128 通道和一层每方向 128 维 BiLSTM。
6. DN 总数不超过 128；所有有效 GT 的主检测监督保留。
7. 默认 AMP、R50 activation checkpointing、每卡 batch 1、梯度累积 4。这里的 checkpointing 作用于 R50；没有声称 Encoder/Decoder 已启用重算。
8. 默认等比例 Resize 的短边目标 800、长边上限 1280，不使用会破坏完整转录的随机裁剪或文字镜像。

识别新增参数约为 `772864 + 257×(V+1)`，包括 C2 投影和整个识别头，不包括 DINO。默认 95 个字符时约 0.80M，6000 个字符时约 2.32M。参数小不代表中间激活没有成本：100 个 `128×8×128` FP32 区域张量本身约 50 MiB；100 个区域、128 步、6001 类的一份 FP32 logits 约 293 MiB，训练还会保存其他激活和梯度。

梯度累积改变有效 batch，不要求同时保留所有微批次的计算图，也不能降低单个微批次本身的峰值。有效 batch 为 `每卡 batch × GPU 数 × 累积次数`；单卡默认 4，两卡默认 8，学习率不自动缩放。

**尚无这套模型正式尺寸的服务器峰值显存或 OCR 精度数据。** 小头、AMP 与上限是减少开销的设计依据，不是“保证只占多少 GB”的测试结论。应使用真实样本执行下文 smoke 并观察正式训练中的密集文字和长文本批次。

## 目录与源码入口

```text
mmdetection/
├── README.md
├── configs/dino/                         # 官方配置，保持原样
├── mmdet/                               # MMDetection 核心，保持原样
├── tools/train.py                       # 原训练入口
├── tests/test_projects/test_ocrdino/     # 数据、权重、识别与检测测试
├── data/ocr/
│   ├── train.json                       # 自行准备的全量聚合标注
│   └── images/                          # 自行准备的图片
└── projects/OCRDINO/
    ├── configs/
    │   ├── ocrdino_r50_4scale_ctc.py                 # 通用基础配置
    │   ├── ocrdino_r50_4scale_ctc_pretrain_24e.py    # MLT + Syn1 + Syn2
    │   └── ocrdino_r50_4scale_ctc_totaltext_12e.py   # Total-Text 微调
    ├── requirements.txt
    ├── reports/english_datasets/        # 本地四包审计及转换结果
    ├── dictionaries/english.json        # 有序 95 字符，blank 不在文件中
    ├── ocrdino/
    │   ├── detector.py                  # 共享特征、检测与识别连接
    │   ├── heads.py                     # 文本检测、ignored 区域和 DN target
    │   ├── recognition.py               # 区域采样、4 Conv、BiLSTM、CTC
    │   ├── dataset.py                   # 聚合 JSON 读取
    │   ├── schema.py / vocabulary.py    # 格式约束与有序字符表
    │   ├── transforms.py                # 同步加载和打包 bbox/text
    │   └── hooks.py                     # 训练与 checkpoint 字典一致性
    ├── tools/
    │   ├── prepare_dataset.py           # COCO/ReCTS 聚合与审计
    │   ├── prepare_english_datasets.py   # 四个英文包转换、图片准备与预训练合并
    │   ├── english_archives.py           # ZIP/TAR/嵌套 ZIP 读取及 EXIF 修复
    │   ├── build_dictionary.py          # 从转录建立有序字典
    │   ├── audit_dataset.py             # 开训前全量 CTC 容量/字典/框检查
    │   ├── convert_checkpoint.py        # 官方 COCO 权重转换
    │   ├── smoke_test.py                # 真实数据训练检查与显存报告
    │   └── infer.py                     # 单图推理输出 JSON
```

建议按数据集、检测器、识别头、权重迁移工具的顺序阅读；DINO 主体实现仍在 MMDetection 原类中复用。

## 环境安装：只使用 pip

以下是 Linux 服务器、Python 3.10、CUDA 12.1 版 PyTorch 的一套明确组合。优先使用已经能运行 MMDetection CUDA ops 的兼容环境；新建环境时可执行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'numpy>=1.23,<2' 'opencv-python<4.12'
python -m pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install mmcv==2.1.0 --only-binary=mmcv \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
python -m pip install -r requirements/runtime.txt
python -m pip install -r projects/OCRDINO/requirements.txt
python -m pip install -e .

python -c "import torch, mmcv, mmengine, mmdet; from mmcv.ops import MultiScaleDeformableAttention, RoIAlign; print('torch',torch.__version__,'CUDA',torch.version.cuda,'available',torch.cuda.is_available()); print('mmcv',mmcv.__version__,'mmengine',mmengine.__version__,'mmdet',mmdet.__version__)"
```

本仓库要求 MMCV `>=2.0.0rc4,<2.2.0`、MMEngine `>=0.7.1,<1.0.0`。上面的项目 requirements 固定 MMEngine 0.10.7，并避免旧 PyTorch 与 NumPy 2 的组合。需要完整 `mmcv` 的 CUDA ops，不能以 `mmcv-lite` 代替。

MMCV wheel 必须匹配 PyTorch 编译的 CUDA 与 PyTorch 版本，以 `torch.version.cuda` 为准，不以 `nvidia-smi` 显示的最高 CUDA 版本代替。这里的 [cu121/torch2.1 官方索引](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html) 有 MMCV 2.1.0 的 Python 3.10 Linux wheel；换组合时应查阅 [MMCV 官方 pip 安装说明](https://github.com/open-mmlab/mmcv/blob/v2.1.0/docs/en/get_started/installation.md#install-with-pip)。导入成功仍需真实 GPU 前向、反向确认算子运行。

## 聚合数据集格式

一个 UTF-8 JSON 集中描述全部训练图片与所有文本区域：

```json
{
  "schema_version": "ocrdino.text.v1",
  "categories": [{"id": 1, "name": "text"}],
  "images": [
    {"id": 1, "file_name": "example.jpg", "width": 640, "height": 480, "annotation_complete": true}
  ],
  "annotations": [
    {
      "id": 1, "image_id": 1, "category_id": 1,
      "bbox": [50, 80, 220, 48],
      "text": "Hello 2026", "ignore": false, "rec_ignore": false
    }
  ]
}
```

字段规则：

- `bbox=[x,y,width,height]` 为**原图绝对像素**坐标，`x/y` 是左上角；不是 xyxy，也不是 0～1 归一化值。宽高必须正、坐标有限，框必须在图片范围内。
- `text` 是整个框的完整字符串，保留大小写、标点、简繁体与空格；一个框应对应一个横排单词或一行文字。
- `category_id` 固定为 1，表示文本前景；Dataset 转换成内部类别 0。字符类别在独立字典中，不是 COCO 的 categories。
- `ignore=true` 表示该区域整体忽略；`rec_ignore=true` 表示保留框定位、关闭文字监督。不要用这两个字段互相替代。
- 图片及标注 ID 分别唯一，每条标注必须指向已有图片。没有有效框的图片仍可以保留。
- `file_name` 相对于配置的图片路径，默认是 `$OCRDINO_DATA_ROOT/images/`；可用 `OCRDINO_IMAGES` 指定其他相对目录或绝对路径。完整图片文件不嵌入 JSON。
- 图片的 `annotation_complete` 默认为 true。转换器因无效几何排除文本区域时将它设为 false，表示未匹配 query 不宜作为可靠背景监督；原始异常保存在 `invalid_annotations` 和审计报告中。

默认 `projects/OCRDINO/dictionaries/english.json` 格式为：

```json
{
  "schema_version": "ocrdino.dictionary.v1",
  "characters": [" ", "!", "\"", "#", "..."]
}
```

此处 `...` 仅为文档省略，不是可复制的真实字符字典。实际文件包含从 U+0020 空格到 U+007E 波浪线的 95 个不同单字符。blank ID 0 由识别头添加。字典顺序决定分类器权重的含义，不能只保持字符数量相同就更换顺序。

### 转换已有 COCO 风格 bbox + text 数据

在仓库根目录执行，下面服务器路径可以按实际位置替换：

```bash
cd /root/autodl-tmp/OCRDINO
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OCRDINO_DATA_ROOT=/root/autodl-tmp/data/ocr
mkdir -p "$OCRDINO_DATA_ROOT"

python projects/OCRDINO/tools/prepare_dataset.py coco \
  --input /root/autodl-tmp/data/source/train.json \
  --output "$OCRDINO_DATA_ROOT/train.json" \
  --image-root "$OCRDINO_DATA_ROOT/images" \
  --text-key text \
  --dictionary projects/OCRDINO/dictionaries/english.json
```

源文件的图片路径需能在 `--image-root` 下找到。不同公开数据集的 transcription 字段、ignore 规则、四点坐标或 mask 格式并不统一；工具的 `coco` 模式接收 COCO xywh 框及指定文字字段，不能把任意 Total-Text 原始标注直接改文件名就当作本格式。

转换会同时生成 `train.audit.json`，记录无效框、文字问题与被排除记录。有效转录含字典外字符时会明确报错，不把整套中文数据悄悄设为 `rec_ignore`。需要自定义字符表时，先不传 `--dictionary` 完成转换，再运行字典生成和下文的全量审计。工具不从图像自动生成或猜测文字转录，不保证数据发布方原始文字一定正确。

### 由 ReCTS 区域标注聚合

```bash
python projects/OCRDINO/tools/prepare_dataset.py rects \
  --image-root /root/autodl-tmp/data/rects/img \
  --annotation-root /root/autodl-tmp/data/rects/gt_unicode \
  --output /root/autodl-tmp/data/rects/train.json

python projects/OCRDINO/tools/build_dictionary.py \
  --input /root/autodl-tmp/data/rects/train.json \
  --output /root/autodl-tmp/data/rects/dictionary.json
```

ReCTS 模式使用 `lines` 文本区域与 transcription，不使用 `chars`，不再推导字符归属。四边形转换成轴对齐外接框，因此对明显倾斜区域会包含额外背景；这是第一版 bbox 模型的数据适配，不是四边形识别能力。英文第一版不建议直接用默认 ASCII 字典训练中文。

训练这份 ReCTS 转换结果时，设置 `OCRDINO_DATA_ROOT=/root/autodl-tmp/data/rects`、`OCRDINO_IMAGES=img` 和 `OCRDINO_DICTIONARY=/root/autodl-tmp/data/rects/dictionary.json`。转换器输出的 `file_name` 相对于 `--image-root`，不会自动增加 `img/` 前缀。

这次没有重新审计全部 ReCTS 数据。旧项目的全量审计统计和字符关系覆盖率不适用于本版本，不能作为本次转换结果照搬。

## 四个英文数据包：审计与转换

本节针对本地 `dataset/eng` 中实际提供的 `mlt2017.zip`、`syntext1.tar`、`syntext2.zip`、`totaltext.zip`，不是假设它们符合官方原始发布格式。四包都已经被上游处理为 COCO 风格 JSON：图片在 `images`，区域在 `annotations`；文字存在固定 25 位整数数组 `rec`，而不是直接可读的 `text`。标注还包含 `bbox=[x,y,w,h]`、16 个数的 `bezier_pts` 等。Total-Text train 另有 polygon `segmentation`，test 没有这个字段。

### 全量标注审计结果

以下数量来自对实际压缩包全部标注的扫描，并经过实际转换核验，默认 `--full-length-policy rec-ignore`。识别可用数表示标签可供识别损失使用，不代表已经人工确认转录内容正确；每次训练仍按模型的 `max_rois=128` 上限抽取识别区域。

| 数据包 / split | JSON 图片数 | 文本框数 | 可用识别标签 | 仅定位、屏蔽识别 | 裁剪到图内的框 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLT2017 train | 9,999 | 89,418 | 58,740 | 30,678 | 139 |
| SynText1 train | 94,723 | 1,318,214 | 1,317,925 | 289 | 37,180 |
| SynText2 train | 54,327 | 479,834 | 479,833 | 1 | 27,738 |
| Total-Text train | 1,255 | 9,722 | 9,708 | 14 | 100 |
| Total-Text test | 300 | 1,975 | 1,974 | 1 | 12 |
| **合并预训练：前三项** | **159,049** | **1,887,466** | **1,856,498** | **30,968** | **65,057** |

这些包中的区域都为 `category_id=1, iscrowd=0`。修复下面两张 Total-Text 图片的坐标系后，全部框都有正面积的图内交集；没有因为坏框删除图片或文本区域。原图 ID、原标注 ID、原文件名被保留，合并时重新分配唯一 ID，图片路径包含数据集和 split，避免各包同名图片冲突。

MLT 图片 ZIP 实际有 10,000 张图，但 JSON 只引用 9,999 张；`0060000.jpg` 无标注记录，转换器不会把它当作有完整负样本标注的图片补入训练。MLT JSON 中本来没有区域的 11 张图仍保留；Total-Text test 有 4 张空标注图片。所提供 MLT `train.json` 与官方原始划分的映射信息缺失，因此这里的 train 是**此包的文件名**，不能据此证明没有包含官方其他 split。

### `rec` 怎样还原成文字

按这批数据实际使用的字符表：`0..94 → chr(code+32)`，对应 95 个可打印 ASCII 字符；`96` 是尾部 padding；`95` 是不能从当前 JSON 还原原字符的编码。**源标签 ID 不是本模型的 CTC 类别 ID**：先转为真实字符串，再由模型的有序字符字典编码。

例如 `[67,79,84,84,79,78,96,...]` 转成 `cotton`，不能把重复的两个 `t` 合并。CTC 去重只发生在模型预测解码时，不能用于 GT 标签转换。

具体处理规则如下：

- MLT 有 30,607 条转录含 `95`，包括 25,102 条全部字符未知和 5,505 条部分未知。保留未知位置为 `�`、保存 `raw_rec`，整个区域设置 `rec_ignore=true`，继续训练检测。不会删掉未知字符拼成一个错误的短单词，也不会猜测原汉字。
- MLT 另有 14 条空转录。空标签、异常数组等都保留有效框，只屏蔽识别。
- 五个 split 分别有 130、289、1、14、1 条占满 25 个槽位且没有 padding 的标签。它们可能刚好 25 字，也可能已经被上游截断，现有数据无法区分。默认保留字符串和框、屏蔽识别；MLT 的 130 条中 73 条同时含 `95`，表中计数已去重。确认希望使用这些现有 25 字转录时，可指定 `--full-length-policy keep`，但脚本无法恢复截断内容。
- 不因为字符串恰好是 `***` 或 `###` 就擅自忽略。依据源 `ignore/iscrowd` 字段处理；本包 MLT 有 1 条、SynText1 有 11 条字面量 `***`，都按原标签保留。
- 没有语言字段，ASCII 可解码不等于一定是英语。这份 MLT 转换可保留多语言区域的检测监督，但不能从 `95` 恢复多语言识别标签。未来中文训练应换用保留完整 Unicode 转录的数据源。

### 框边界与两张 Total-Text 图片的修复

SynText1、SynText2 的越界框全部只超出右边或下边 1 像素。转换器取框与图片的正面积交集，保存原框到 `source_bbox`；不缩放文字，不改变其转录。个别框极薄，1 像素也会造成较大的面积比例变化，SynText1/SynText2 最大分别约 25.60%/33.33%。若其他数据出现完全无交集或无效框，会保留图片和 `invalid_annotations`，同时设 `annotation_complete=false`，排除无效位置监督。

Total-Text 有两张图片不能只按源 JSON 宽高裁剪，否则会误删真实区域：

| 图片 | 源 JSON 宽×高 | JPEG 原始像素宽×高 | 源 EXIF |
| --- | --- | --- | --- |
| train / `0000894.jpg` | 470×400 | 400×470 | Orientation=6 |
| test / `0000228.jpg` | 294×392 | 392×294 | Orientation=6 |

它们的框对应 JPEG 的**未旋转像素坐标**，源 JSON 尺寸却采用旋转后的显示方向。脚本将输出 JSON 改为原始像素宽高，并在输出图片副本中移除 EXIF，不旋转、不重新编码 JPEG。两张真实图片均已验证：处理前后解码像素字节完全相同。原始压缩包不修改。其他未审计过的 EXIF 旋转/镜像会在图片准备时明确报错，避免仅凭宽高猜坐标系。

当前模型使用水平 `bbox`，转换不使用 Bezier 或 polygon 来做弯曲文字矫正。Total-Text 中倾斜和弯曲文本仍会进入其水平包围框，框内可能带较多背景；这限制了此版的识别能力，不能将它视为已经复现曲线文本模型或官方 polygon 端到端评测。

完整机器可读结果见 [审计索引](projects/OCRDINO/reports/english_datasets/README.md)。本次覆盖全量标注和图片成员存在性；Total-Text 全部图片检查了头部尺寸/方向，其他三包未逐张解码或人工核对文字。默认宽度桶 `(32,64,128)`、`width_scale=16` 对全部未屏蔽转录均满足 CTC 的 `L+相邻重复数` 容量；单图最多 333 个检测 GT，低于 900 query 容量。

### 在服务器转换和准备图片

把四个压缩包放到一个目录，保留以下名称，无需手工解开嵌套 ZIP：

```text
/root/autodl-tmp/data/english_sources/
├── mlt2017.zip
├── syntext1.tar
├── syntext2.zip
└── totaltext.zip
```

在仓库根目录执行：

```bash
python -m pip install Pillow
python projects/OCRDINO/tools/prepare_english_datasets.py \
  --source-root /root/autodl-tmp/data/eng \
  --output-root /root/autodl-tmp/data/ocrdino_english \
  --extract-images \
  --temp-dir /root/autodl-tmp/ocrdino_convert_tmp
```

源目录和输出目录必须彼此独立，不能相同或相互嵌套。省略 `--extract-images` 只转换 JSON，**尚不能开始训练**；训练应使用该选项生成的图片副本，才能同步完成 Total-Text 方向修复。脚本需要标准库和 Pillow，不需要 PyTorch/CUDA。

MLT 和 SynText2 含嵌套图片 ZIP：第一次取图时流式写入磁盘临时文件，不把完整内层 ZIP 装入 RAM；一次只处理一个，最多需要约 10.4 GB 临时磁盘空间，正常结束会清理。输出图片还需要另外的磁盘空间。最大的源 JSON 约 486 MB，解码为 Python 对象需数 GB CPU 内存。已有输出图片仅在内容一致时复用，内容不同会报错；重新运行会更新生成的 JSON/报告，原始档案始终只读。

输出布局：

```text
ocrdino_english/
├── annotations/
│   ├── pretrain.json                 # 仅 MLT2017 + SynText1 + SynText2
│   ├── mlt2017_train.json
│   ├── syntext1_train.json
│   ├── syntext2_train.json
│   ├── totaltext_train.json          # 微调输入
│   ├── totaltext_test.json           # 独立保留，不进入训练/验证
│   ├── *.audit.json                 # 每个 split 的转换统计与异常原因
│   └── conversion_summary.json
└── images/
    ├── mlt2017/train/
    ├── syntext1/train/
    ├── syntext2/train/
    └── totaltext/{train,test}/
```

生成格式就是前述 `ocrdino.text.v1`。例如 `file_name="images/syntext1/train/00000001.jpg"` 已包含 `images/`，所以本流程的 `OCRDINO_IMAGES` 必须为空，图片根目录是整个 `ocrdino_english`，不能再追加一次 `images/`。新建的两个训练配置已默认采用这个约定。

只想转换某个包时可加 `--datasets totaltext` 等；仅当本次同时选择 `mlt2017 syntext1 syntext2` 时才生成 `pretrain.json`。SynText2 使用外层 `syntext2/train.json`，不再合并内层的另一份 `annotations/ecms_v1_maxlen25.json`，避免重复。

## 从英文扩展到中文

用训练集转录构造自己的字典：

```bash
python projects/OCRDINO/tools/build_dictionary.py \
  --input "$OCRDINO_DATA_ROOT/train.json" \
  --output "$OCRDINO_DATA_ROOT/dictionary.json"
export OCRDINO_DICTIONARY="$OCRDINO_DATA_ROOT/dictionary.json"
```

字典必须覆盖所有需要识别监督的字符，真实 Unicode 字符保留，不自动繁简转换或大小写合并。第一次用自定义字典训练时应重新建立相应字符输出层。之后恢复训练和推理必须用与 checkpoint 一致的有序字典；字典钩子检查并将字符表写入 checkpoint 元信息。

从英文训练权重扩展中文不等于把环境变量换成更长字典就能严格加载。若要保留已训练的英文字符权重，需要另行按字符映射迁移输出层；当前提供的是从官方 COCO DINO 初始化任意字典的流程。第一版没有声称未见汉字也能被识别。

长行可增加宽度桶，例如：

```bash
# 在训练命令后增加；先检查真实数据与显存，再采用。
--cfg-options 'model.recognition_head.width_buckets=[32,64,128,256]'
```

也可调整 `width_scale`。增加最大桶只使足够宽的区域进入更大桶；若较窄框内有很长转录，需要检查标注及实际选桶宽度，而不只修改全局最大值。更大的字典和时间步都会增加 logits/CTC 的显存。

## 官方预训练权重准备

本地源权重是：

```text
C:\Users\14222\Desktop\ocrdino\cn\dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth
```

上传到服务器后，先转换为单类检测初始化权重：

```bash
mkdir -p checkpoints
python projects/OCRDINO/tools/convert_checkpoint.py \
  --source /root/autodl-tmp/checkpoint/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth \
  --output checkpoints/ocrdino_init.pth \
  --report checkpoints/ocrdino_init.report.json

export OCRDINO_PRETRAINED="$PWD/checkpoints/ocrdino_init.pth"
```

转换报告列出保留和排除的参数。它不是训练恢复文件；第一次训练通过 `OCRDINO_PRETRAINED` 读取，不要对转换文件使用 `--resume`。配置默认读取 `checkpoints/ocrdino_init.pth`，没有准备该文件时应先转换，避免不知情地从零开始训练。

## 英文实验：24 轮预训练 → 12 轮 Total-Text 微调

此流程使用上一节生成的 `ocrdino_english`，运行目录始终为本仓库根目录。先完成环境安装、带 `--extract-images` 的数据转换，以及上一节官方 COCO 权重转换。两个阶段使用**同一个 95 字符有序字典和相同模型结构**。

### 1. 设置路径并做训练前检查

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OCRDINO_DATA_ROOT=/root/autodl-tmp/data/ocrdino_english
export OCRDINO_IMAGES=''
export OCRDINO_DICTIONARY="$PWD/projects/OCRDINO/dictionaries/english.json"
export OCRDINO_PRETRAINED="$PWD/checkpoints/ocrdino_init.pth"
export OCRDINO_BATCH_SIZE=1
export OCRDINO_ACCUMULATION=4
export OCRDINO_WORKERS=4

python projects/OCRDINO/tools/audit_dataset.py \
  "$OCRDINO_DATA_ROOT/annotations/pretrain.json" \
  --dictionary "$OCRDINO_DICTIONARY" --image-root "$OCRDINO_DATA_ROOT" \
  --width-buckets 32 64 128 --width-scale 16 --max-queries 900 \
  --output work_dirs/pretrain_data_audit.json

python projects/OCRDINO/tools/audit_dataset.py \
  "$OCRDINO_DATA_ROOT/annotations/totaltext_train.json" \
  --dictionary "$OCRDINO_DICTIONARY" --image-root "$OCRDINO_DATA_ROOT" \
  --width-buckets 32 64 128 --width-scale 16 --max-queries 900 \
  --output work_dirs/totaltext_data_audit.json

CUDA_VISIBLE_DEVICES=0 python projects/OCRDINO/tools/smoke_test.py \
  projects/OCRDINO/configs/ocrdino_r50_4scale_ctc_pretrain_24e.py \
  --checkpoint "$OCRDINO_PRETRAINED" --device cuda:0 --steps 2 \
  --output work_dirs/english_pretrain_smoke.json
```

这里 `--image-root` 必须指向整个数据根目录。全量审计确认路径、框、字典和序列容量；smoke 检查真实 GPU 前向、反向和峰值显存。每卡 batch 1、累积 4 是起始配置，不能据此预先承诺特定显存峰值。调整 batch 后需在自己的服务器重新测量。

### 2. MLT2017 + SynText1 + SynText2 预训练 24 轮

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  projects/OCRDINO/configs/ocrdino_r50_4scale_ctc_pretrain_24e.py \
  --work-dir work_dirs/ocrdino_pretrain24
```

每轮遍历合并后的 159,049 张图，不人为给三个来源各分配同样的采样配额。SynText1+2 占约 93.71% 的图片，因此是合成样本占主导的预训练；这里的 24 轮是**整个合并集的 24 次遍历**，并非每轮只取一个小子集。DDP 为对齐各 rank 的长度可能补齐极少量样本。

预训练 AdamW 基础学习率 `1e-4`，主干乘数 0.1；前 500 个 dataloader iteration warmup，第 19、22 轮的里程碑各乘 0.1。AMP、有限 DN、共享 C2、识别区域上限、梯度累积及 R50 checkpointing 均继承基础配置。单卡默认有效 batch 为 4；两卡相同配置为 8，未自动缩放学习率。

完成后取 `work_dirs/ocrdino_pretrain24/epoch_24.pth`。不划分验证集，不按测试集挑选 checkpoint。

### 3. 用完整 OCR 权重微调 Total-Text 12 轮

```bash
export OCRDINO_FINETUNE_FROM="$PWD/work_dirs/ocrdino_pretrain24/epoch_24.pth"

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  projects/OCRDINO/configs/ocrdino_r50_4scale_ctc_totaltext_12e.py \
  --work-dir work_dirs/ocrdino_totaltext12
```

微调只读 `annotations/totaltext_train.json` 的 1,255 张图，加载上一阶段的 backbone、检测分支、C2 投影和完整识别头。**不要再对这个 OCR checkpoint 执行 `convert_checkpoint.py`，也不要在切换阶段时加 `--resume`。** 配置用 `load_from` 加载模型参数、`resume=False`，重新建立优化器和训练计数，从第 1 轮开始完成 12 轮。

微调基础学习率 `5e-5`，主干仍乘 0.1，warmup 500 iteration，第 8、11 轮的里程碑各乘 0.1。这是首版明确给定的实验计划，还没有验证集调参结论。最后使用 `work_dirs/ocrdino_totaltext12/epoch_12.pth`。

### 4. 两卡、同阶段续训与最终推理

若改用两卡，将对应阶段的单卡命令替换为：

```bash
# 预训练；环境变量与上面保持一致
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
  tools/train.py projects/OCRDINO/configs/ocrdino_r50_4scale_ctc_pretrain_24e.py \
  --launcher pytorch --work-dir work_dirs/ocrdino_pretrain24

# 预训练完成后再微调
export OCRDINO_FINETUNE_FROM="$PWD/work_dirs/ocrdino_pretrain24/epoch_24.pth"
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
  tools/train.py projects/OCRDINO/configs/ocrdino_r50_4scale_ctc_totaltext_12e.py \
  --launcher pytorch --work-dir work_dirs/ocrdino_totaltext12
```

仅在**同一阶段中断后恢复**时，保持该阶段配置、工作目录和字典不变，在原启动命令末尾加 `--resume`。此时恢复的是该阶段优化器和进度。新阶段使用不同工作目录，避免把预训练进度当成微调进度。两个专用配置固定各自的 24/12 轮、数据文件、学习率和日程，不使用通用配置的 `OCRDINO_EPOCHS/OCRDINO_LR/OCRDINO_TRAIN_JSON` 来决定这些值；需要变更时明确编辑专用配置或使用 `--cfg-options`。

最终单图推理：

```bash
CUDA_VISIBLE_DEVICES=0 python projects/OCRDINO/tools/infer.py \
  projects/OCRDINO/configs/ocrdino_r50_4scale_ctc_totaltext_12e.py \
  work_dirs/ocrdino_totaltext12/epoch_12.pth \
  "$OCRDINO_DATA_ROOT/images/totaltext/test/0000001.jpg" \
  --device cuda:0 --out work_dirs/totaltext_example_pred.json
```

`totaltext_test.json` 已单独准备，但当前训练配置不启用验证和测试 loop，也没有实现 Total-Text 官方 polygon/lexicon 评测器。不要将单图推理或训练损失当作官方端到端 OCR 指标；以后评测时需使用独立的评测流程，并说明当前水平框输出及异常转录处理方式。

## 自备数据：先执行真实 GPU smoke

数据和字典准备好后，先完整检查字典覆盖、单图 query 容量、实际宽度桶与 `L+R` 的 CTC 可行性：

```bash
export OCRDINO_DATA_ROOT=/root/autodl-tmp/data/ocr
export OCRDINO_TRAIN_JSON=train.json
export OCRDINO_DICTIONARY="$PWD/projects/OCRDINO/dictionaries/english.json"

python projects/OCRDINO/tools/audit_dataset.py \
  "$OCRDINO_DATA_ROOT/train.json" \
  --dictionary "$OCRDINO_DICTIONARY" \
  --image-root "$OCRDINO_DATA_ROOT/images" \
  --width-buckets 32 64 128 --width-scale 16 --max-queries 900 \
  --output work_dirs/dataset_audit.json
```

调整过识别宽度、query 数量或图片目录时，审计命令也要使用相同设置。全量预检查可以发现随机采样 smoke 尚未碰到的长转录。接着运行真实 GPU smoke：

```bash
export OCRDINO_DATA_ROOT=/root/autodl-tmp/data/ocr
export OCRDINO_TRAIN_JSON=train.json
export OCRDINO_DICTIONARY="$PWD/projects/OCRDINO/dictionaries/english.json"
export OCRDINO_PRETRAINED="$PWD/checkpoints/ocrdino_init.pth"

CUDA_VISIBLE_DEVICES=0 python projects/OCRDINO/tools/smoke_test.py \
  projects/OCRDINO/configs/ocrdino_r50_4scale_ctc.py \
  --checkpoint "$OCRDINO_PRETRAINED" \
  --device cuda:0 --steps 2 \
  --output work_dirs/ocrdino_smoke.json
```

smoke 使用训练配置和真实 Dataset，检查模型实际前向、反向与优化步骤，并记录运行报告及 CUDA 峰值显存。有限的 smoke 样本不能代表整套数据最高密度、最长文本或正式训练精度；正式训练仍需观察峰值和异常样本。

## 自备数据：通用全量训练，不划分验证集

以下是基础配置的自备数据示例；四个英文数据包采用上面的专用 24→12 配置。基础配置默认 30 epochs、每卡 batch 1、累积 4、AdamW 学习率 `1e-4`、主干学习率乘数 0.1、AMP、R50 activation checkpointing。前 500 次 iteration 线性 warmup，在总轮数的 80% 和 90% 降低学习率。没有启用验证或测试 loop。

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OCRDINO_DATA_ROOT=/root/autodl-tmp/data/ocr
export OCRDINO_TRAIN_JSON=train.json
export OCRDINO_DICTIONARY="$PWD/projects/OCRDINO/dictionaries/english.json"
export OCRDINO_PRETRAINED="$PWD/checkpoints/ocrdino_init.pth"
export OCRDINO_BATCH_SIZE=1
export OCRDINO_ACCUMULATION=4
export OCRDINO_WORKERS=4
export OCRDINO_EPOCHS=30

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  projects/OCRDINO/configs/ocrdino_r50_4scale_ctc.py \
  --work-dir work_dirs/ocrdino_ctc_v1
```

配置已经使用 `AmpOptimWrapper`，不需要再加 `--amp`。每轮保存 checkpoint，保留最近三轮及最后权重。完成 30 轮后使用 `work_dirs/ocrdino_ctc_v1/epoch_30.pth`，它是预定最后一轮，不能称为验证集选出的 best。

两卡 DDP：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
  tools/train.py projects/OCRDINO/configs/ocrdino_r50_4scale_ctc.py \
  --launcher pytorch --work-dir work_dirs/ocrdino_ctc_v1_2gpu
```

从最近训练 checkpoint 恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  projects/OCRDINO/configs/ocrdino_r50_4scale_ctc.py \
  --work-dir work_dirs/ocrdino_ctc_v1 --resume
```

恢复时保留原字典、数据与模型结构；训练恢复会读取优化器和训练进度。修改字符顺序后继续恢复会改变分类语义，因此字典一致性检查不能绕开。

| 环境变量 | 默认值/用途 |
| --- | --- |
| `OCRDINO_DATA_ROOT` | `data/ocr`，图片和聚合 JSON 的根目录 |
| `OCRDINO_TRAIN_JSON` | `train.json`，相对数据根目录或绝对路径 |
| `OCRDINO_IMAGES` | `images/`，图片目录，相对数据根目录或绝对路径 |
| `OCRDINO_DICTIONARY` | `projects/OCRDINO/dictionaries/english.json` |
| `OCRDINO_PRETRAINED` | 转换后的初始化 checkpoint，默认 `checkpoints/ocrdino_init.pth` |
| `OCRDINO_BATCH_SIZE` | 每卡 1 |
| `OCRDINO_ACCUMULATION` | 4 |
| `OCRDINO_WORKERS` | 4，本机调试可设为 0 |
| `OCRDINO_EPOCHS` | 30 |
| `OCRDINO_LR` | 0.0001 |
| `OCRDINO_WORK_DIR` | `work_dirs/ocrdino_ctc_v1` |

可选预测框识别训练示例：

```bash
# 加在训练命令后；默认仍是纯 GT 框识别训练。
--cfg-options model.rec_train_cfg.pred_box_prob=0.2 \
  model.rec_train_cfg.pred_box_start_epoch=5 \
  model.rec_train_cfg.min_pred_iou=0.7
```

训练图片上的损失和可视结果只能用于检查拟合与程序运行。没有独立评测集时，不能给出可信的泛化 OCR 指标；未来可用独立 bbox+text 数据评测检测、识别和端到端结果。

## 单图推理

保持训练时相同的字典：

```bash
export OCRDINO_DICTIONARY="$PWD/projects/OCRDINO/dictionaries/english.json"
CUDA_VISIBLE_DEVICES=0 python projects/OCRDINO/tools/infer.py \
  projects/OCRDINO/configs/ocrdino_r50_4scale_ctc.py \
  work_dirs/ocrdino_ctc_v1/epoch_30.pth \
  /root/autodl-tmp/data/ocr/images/example.jpg \
  --device cuda:0 --out work_dirs/ocrdino_pred.json
```

推理输出 JSON 包含原图 bbox、文本检测分数、识别内容和识别分数。`--score-thr` 可调整检测阈值；调整它只改变哪些检测区域进入识别，不是重新训练。单张训练图预测不能替代独立数据上的评测。

## 当前验证范围

本次本地验证：45 项测试通过，两个依赖 Torch 的测试模块被跳过；OCRDINO 的 28 个 Python 源码文件通过 AST 语法检查。测试覆盖字符字典、异常标注保留、CTC 时间步容量、输入文件保护、嵌套 ZIP/TAR、EXIF 去除且像素不变、源 `rec` 解码、跨数据集 ID 合并和两阶段配置中的路径/日程，以及预训练权重筛选规则；权重筛选测试使用张量形状替身，不等于已经加载真实权重。

四个真实压缩包的全部 JSON 已实际转换，合并后的 159,049 图 / 1,887,466 框再次通过 schema、字典覆盖和 CTC 容量检查。Total-Text 的 1,555 张图片已实际提取，并通过尺寸/EXIF 校验；两张修复图额外验证了解码像素前后一致。其余三包检查了全部图片成员存在性，未在本地提取完整图片集。实际结果保存在 `projects/OCRDINO/reports/english_datasets/`，这是数据准备验证，不是 OCR 精度验证。

本机测试解释器没有 PyTorch、MMCV。另借助现有 MMEngine 0.10.7 缓存，以 `Config.fromfile(..., import_custom_modules=False)` 实际验证了两个阶段配置的继承合并和 `pretty_text` 序列化，但没有导入自定义模型或构建 Runner。识别头与检测器的运行时测试模块共两项被跳过，未执行真实权重转换、模型构建、CTC 张量数值计算或 GPU 前向反向。因此不能把本地检查表述成“训练已经跑通”。请在安装依赖后的服务器执行：

```bash
python -m pip install pytest
python -m pytest -o addopts='' tests/test_projects/test_ocrdino -q -rs
```

默认 `find_unused_parameters=False`，配合当前 ResNet 内置的重入 activation checkpointing；识别无有效区域、去噪无 GT 的分支都显式保留参数的零梯度路径。没有开启 `find_unused_parameters=True` 来搜索 checkpoint 内部的参数。DDP 行为仍需在目标多卡环境验证。

服务器完整 CUDA 训练、CUDA ops、AMP、DDP、多轮收敛、OCR 精度和正式尺寸峰值显存，仍需执行上述单元测试、真实 GPU smoke 及训练命令验证。小样本 smoke 通过也不代表已验证多轮收敛。

标准库数据/转换测试可独立执行；识别和检测测试需要兼容的 PyTorch、MMCV、MMEngine、MMDetection 环境：

```bash
python -m unittest discover -s tests/test_projects/test_ocrdino -p 'test_data.py'
python -m unittest discover -s tests/test_projects/test_ocrdino -p 'test_checkpoint.py'
python -m pip install pytest
python -m pytest tests/test_projects/test_ocrdino -q -o addopts=''
```

`-o addopts=''` 清除原仓库 pytest.ini 的额外插件参数，使本项目测试不依赖原仓库完整开发测试插件集。纯标准库测试通过仅证明相应逻辑，不替代 CUDA smoke。

## 设计依据

- [DINO：DETR with Improved DeNoising Anchor Boxes](https://arxiv.org/abs/2203.03605)：query 集合检测、参考框细化与去噪训练。
- [MMDetection R50 DINO-4scale improved 配置](https://github.com/open-mmlab/mmdetection/blob/main/configs/dino/dino-4scale_r50_improved_8xb2-12e_coco.py)：检测主体和官方预训练起点。
- [FOTS](https://arxiv.org/abs/1801.01671)：共享整图特征、区域采样与 CNN/BiLSTM/CTC 的单模型检测识别思路。
- [CRNN](https://arxiv.org/abs/1507.05717)：从二维文字视觉特征到一维序列与 CTC 转录。
- [TransDETR 当前公开识别头](https://github.com/weijiawu/TransDETR/blob/main/models/head/rec_head_ctc.py)：Transformer 检测框架连接区域卷积、单层 BiLSTM 与 CTC 的代码先例。当前代码与论文识别头叙述不能不加区分地视为同一实验配置。
- [CTC 原论文](https://www.cs.toronto.edu/~graves/icml_2006.pdf) 与 [PyTorch CTCLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)：无字符位置标注的序列对齐训练。

本项目实现保留这些方法的核心关系，具体宽度分桶、头部规模和 DINO 接入由本项目实现；没有声称复现其论文指标。新增代码沿用仓库 Apache 2.0 许可证，数据与预训练权重的使用按各自发布方说明执行。

## 导出最后一轮模型：bestepoch.pth

`projects/OCRDINO/tools/export_model.py` 将最后一轮完整训练 checkpoint 导出为较小的模型文件。这里按约定把第24轮称为 `bestepoch.pth`，不进行验证集评分或自动选轮。

- 保留全部模型参数和持久缓冲区，包括检测头、识别头、BN统计、字典hash，以及无CTC版的识别版本；数值、形状、dtype和参数名不变，统一保存到CPU。
- 只保留必要的模型元信息 `meta.ocrdino` 和 `meta.dataset_meta`。字符顺序、CTC blank或Query EOS语义用于正确加载模型；这些不是训练进度。
- 丢弃优化器、动量、AMP scaler、学习率调度、训练轮数/迭代数、完整配置、message_hub和其他训练状态。
- 默认核对 `meta.epoch == 24`，避免误导出尚未训练完成的文件。模型张量含NaN/Inf时明确拒绝；已丢弃的历史日志或优化器中出现NaN不影响模型张量检查。
- 源checkpoint保持原样；输出文件已存在时拒绝覆盖。工具不会修改 `last_checkpoint` 或训练配置。

在各自项目根目录、原来的PyTorch/MMEngine训练环境中执行，无需GPU，不构建模型：

~~~bash
python projects/OCRDINO/tools/export_model.py \
  --source work_dirs/ocrdino_pretrain24/epoch_24.pth
~~~

默认输出 `work_dirs/ocrdino_pretrain24/bestepoch.pth`。如果从FP32续训目录完成训练，`--source` 应改为该目录的 `epoch_24.pth`；以实际保存路径为准。也可以指定独立输出：

~~~bash
python projects/OCRDINO/tools/export_model.py \
  --source work_dirs/ocrdino_pretrain24/epoch_24.pth \
  --output checkpoints/bestepoch.pth
~~~

终端会打印导出前后文件大小、张量数量以及删除的字段。导出文件顶层只含 `state_dict` 和必要的 `meta`。

用于Total-Text新阶段微调时，可以设置：

~~~bash
export OCRDINO_FINETUNE_FROM="$PWD/work_dirs/ocrdino_pretrain24/bestepoch.pth"
~~~

继续使用原12轮微调配置的 `load_from`，或将该文件作为单图推理的checkpoint参数。**导出文件不再支持恢复旧优化器和训练进度，不能用于 `--resume`。** 断点续训仍用保留的完整 `epoch_24.pth`。

同一个工具也能导出微调最后一轮：将source换为 `epoch_12.pth`，并显式加 `--expected-epoch 12`。这与 `convert_checkpoint.py` 的COCO初始化转换不同，不会丢弃任何检测或识别权重。
