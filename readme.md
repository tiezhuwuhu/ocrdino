# OCRDINO：共享特征与 Character Query 的两阶段 OCR

本副本基于 MMDetection 3.3.0，在一个模型中联合训练 **DINO 文本框检测 + 区域字符识别**。输入为整图和 `bbox + text`，不需要字符框或字符位置。识别路径为 **共享 C2 → 1×1 Conv → RoIAlign → 4Conv → Visual Tokens → 三层 Character Transformer Decoder → 字符/EOS 分类**，使用有序交叉熵，完全移除实际识别路径对 BiLSTM 和 CTC 的依赖。

本次仅修改 `mmdetection_twostage_wo_ctc`。原项目、`mmdet/` 核心、官方 DINO 配置和 `heads.py` 的检测损失保持不变。旧 CTC 模块和配置在本副本中明确报迁移错误。英文数据继续采用 MLT2017 + SynText1 + SynText2 预训练24轮、Total-Text train微调12轮；不划分验证集，取预定最后一轮，test独立保留。

## 模型结构与张量形状

~~~text
整图 [B,3,Hp,Wp]
        │
        └─ 共享 ResNet-50，只运行一次
             ├─ C3/C4/C5 → ChannelMapper → 四尺度特征，各256通道
             │              → 6层 Deformable Encoder
             │              → Top-900参考框 + 900 Detection Query
             │              → 6层 DINO Decoder → 文本bbox + detection_score
             │
             └─ C2 [B,256,Hp/4,Wp/4]
                  → 1×1 Conv 256→128
                  → 按文本bbox RoIAlign [M,128,8,W]
                  → 4Conv，高度 8→8→4→2→1
                  → [M,128,1,W] → [M,W,128]
                  → Linear 128→256 → Visual Tokens [M,W,256]
                                               │
共享content [1,1,256] + 字符序号PE [1,N,256]    │
                  → Character Queries [M,N,256]
                  → 3层 Transformer Decoder ──┘
                     Self-Attention → Cross-Attention → FFN
                  → [M,N,256] → Linear 256→V+1
                  → [M,N,V+1] 字符/EOS logits
                  → 首个EOS之前的有序字符串
~~~

`B`为每卡图片数，`M`为一个识别块中的区域数，`W`为视觉采样宽度，`N`为字符槽数，`V`为真实字符类别数。不同区域可有不同`W/N`，同一块中二者一致。训练默认以GT框采样，推理使用DINO预测框。

### 共享骨干、四尺度检测与两类 Query

| R50输出 | stride | 通道 | 用途 |
| --- | ---: | ---: | --- |
| C2 | 4 | 256 | 区域识别，保留笔画细节 |
| C3 | 8 | 512 | DINO neck |
| C4 | 16 | 1024 | DINO neck |
| C5 | 32 | 2048 | DINO neck，并生成额外粗尺度 |

DINO输入的名义stride为8、16、32、64，各256通道。C2不进入全图Transformer，因此仍是官方R50-4scale，可复用improved COCO权重。六层Encoder、六层Decoder、900 query、Hungarian Matching、focal/L1/GIoU、辅助层和DN监督保持原版。

Detection Query对应完整文本区域；Character Query对应区域字符串的第1、第2……个字符。900个检测槽限制文本区域数，字符字典大小决定共享分类器类别数，二者独立。上千种汉字不需要上千个检测Query；一个HELLO区域仅占一个检测槽。DINO内部`as_two_stage=True`指Encoder proposals初始化Decoder，与这里“检测后识别”含义不同。识别不运行第二套backbone或检测器。

### RoIAlign与四层卷积

| 操作 | 输出形状 |
| --- | --- |
| C2的1×1 Conv，256→128，无bias | `[B,128,Hp/4,Wp/4]` |
| RoIAlign，height=8，sampling_ratio=2，aligned=True | `[M,128,8,W]` |
| 3×3 Conv，stride=1，GN(32组)，ReLU | `[M,128,8,W]` |
| 3×3 Conv，stride=(2,1)，GN，ReLU | `[M,128,4,W]` |
| 3×3 Conv，stride=(2,1)，GN，ReLU | `[M,128,2,W]` |
| 2×1 Conv，ReLU | `[M,128,1,W]` |
| 压缩高度、转置、Linear(128,256) | `[M,W,256]` |

四个区域卷积均无bias，不压缩横向顺序，也不全局平均池化。采样框使用缩放后的输入像素坐标，`spatial_scale=1/4`。框坐标不传识别梯度给DINO回归器；识别特征向共享C2、ResNet-50及新头反向传播。

### 两套位置编码与 Cross-Attention

视觉位置编码标记区域左到右第x个token；字符位置编码标记输出序号i。二者采用标准正弦/余弦位置编码，D=256，位置从0开始：

~~~text
PE(p,2k)   = sin(p / 10000^(2k/D))
PE(p,2k+1) = cos(p / 10000^(2k/D))
~~~

角度用FP32计算后转为特征dtype。初始字符Query为`shared_content + PE(i)`：只有一个共享可学习256维向量，不是500个类别embedding。每层Self-Attention的Q/K/V是当前字符表示；Cross-Attention的**Q为当前字符表示，K=visual+PE(x)，V=visual**。

每层包含Self-Attention、Cross-Attention、两层FFN及对应残差和LayerNorm。默认固定为3层、256维、8 heads、FFN 1024、dropout 0.1。没有causal mask、GT字符输入、teacher forcing、自回归、字符级Hungarian或单调位置约束。全部位置并行预测。

第3个Query表示第3个字符，不表示图片固定横坐标；视觉位置由Cross-Attention学习。有序监督和位置编码提供学习顺序的机制，**不保证学习到正确对齐，也没有证据表明本版比CTC更准确或稳定**。共享字符分类器也不保证只见过ABCD就能识别所有新排列或未见字符。

### 动态Query桶与视觉分辨率

~~~text
requested_slots = ceil(bbox_width / bbox_height * query_width_scale)
query_width_scale = 16.0
query_buckets = (32,64,128,256,500)
~~~

选择不小于需求的最小桶，超出最大桶则上限为500：20→32、50→64、110→128、200→256、350→500。训练和推理使用同一规则，选桶函数不接收文字长度。

视觉宽度单独使用`width_buckets/width_scale`，默认同为`(32,64,128,256,500)/16.0`。增大字符槽时需同时考虑视觉宽度。RoIAlign加宽只是更密集采样C2，不能恢复源图或stride=4已丢失的笔画；水平bbox也不等于弯曲、竖排或多行校正。

长度L必须满足`N>=L+1`。默认500槽含EOS，最多监督499个真实Unicode码点；500个真实字符需要显式增加容量。超容量会报出原文字、Query桶和所需槽数，不截断、不利用GT临时升桶、不静默屏蔽。

`max_rois`抽样前检查全部有效GT，预测框混入后及head内再次检查。离线审计复用几何规则，并报告视觉桶及可能的视觉分辨率不足。图片resize舍入和预测框变化可能改变桶，运行时检查不可省略。

### 字符、EOS、Ignore与损失

真实字符ID为0..V-1，EOS为V，共V+1类。没有blank或PAD预测类别。`-100`仅为CE的ignore_index，不占类别。95个可打印ASCII字符包含大小写、空格和标点；字典顺序及Unicode语义保持，未知有效字符明确报错。

~~~text
Query:   Q1 Q2 Q3 Q4 Q5 Q6  Q7 ... QN
Target:   H  E  L  L  O EOS -100...-100
~~~

每个区域的损失定义：

~~~text
loss_char = mean(CE(logits[i], text[i]), i=0...L-1)
loss_eos  = CE(logits[L], EOS)
region_loss = loss_char + lambda_eos * loss_eos   # 默认1.0
loss_rec = loss_weight * mean(region_loss)       # 默认1.0
loss_total = loss_dino + loss_rec
~~~

先平均区域内真实字符CE，再加该区域EOS CE，最后平均有效区域。EOS后所有槽不计算损失，不因大量空槽稀释监督。真实字符位置预测EOS会被惩罚，不能仅学习结束符逃避字符损失。无需字符框的原因是第i槽直接监督字符串第i字，空间对齐交给注意力学习。

DDP每rank的区域损失之和除以`全局有效区域数/world_size`，经梯度平均得到全局区域平均。正分母可以小于1，不能夹成1；仅用极小正数防零。空本地批次也参与collective，为C2和全部识别参数保留零计算图。

### 训练区域和输出

默认以GT bbox识别，`pred_box_prob=0`，每批随机选择最多128个识别区域，正常检测GT全部保留。`rec_ignore`仅屏蔽识别，`ignore`延续原ignored监督，无效几何仍记录并设`annotation_complete=false`。

可选后期混入预测框，保留原IoU、GT覆盖率、概率条件；容量条件改为预测框几何Query桶容得下`len(text)+1`，不合格则回退GT。默认`pred_box_start_epoch=5`按零起始epoch表示第6轮。

推理每图最多识别100个区域，所有槽并行argmax，读到首个EOS停止，重复字符不合并：`H E L L O EOS A B → HELLO`。首槽EOS得到空串、分数0。没有EOS时输出全部字符，记录`text_terminated=false`；占满500槽的未结束结果不等于合法499字加EOS序列。

`text_score`为EOS前输出字符概率的几何平均，不计EOS，不是校准后的整串正确概率。`ocr_score=detection_score*text_score`仅供排序。分桶和分块后按原区域索引回填，框在采样后恢复原图坐标。

| 输出字段 | 含义 |
| --- | --- |
| `bbox` | 原图绝对像素xyxy，与训练xywh不同 |
| `detection_score` | 文本检测分数 |
| `text` | 首个EOS之前的字符串 |
| `text_score` | 输出字符概率几何平均，空串0 |
| `ocr_score` | 检测与识别分数乘积 |
| `text_terminated` | 是否出现EOS |

## 显存与计算约束

保留DN总槽上限128与900匹配Query，训练检测Decoder总长最多1028；DN子采样不减少正常检测GT。C2不进入整图Transformer，识别只执行一次，不处理全部背景Query，也不在DINO六层输出上重复识别。

按相同`(W,N)`分组，默认`chunk_size=8`。训练`with_cp=True`对每块RoIAlign、4Conv、Decoder和区域CE做非重入重计算，保存标量损失，反向重算并恢复dropout RNG。单纯循环分块仍会保存所有训练激活，不能声称同样节省显存。分块不改变区域损失和归一化；不同块大小的dropout随机样本可能不同。

Attention设`need_weights=False`，允许支持的后端使用SDPA，不保证任何硬件都走Flash Attention。RoIAlign用FP32 feature/rois；卷积、Linear和Attention跟随AMP；CE及解码log-softmax用FP32。

500槽Self-Attention为平方开销。显式存一份8区域、8头、500×500的FP32 attention矩阵约61MiB，尚未计其他层、反向和检测；这不是实测峰值。可先减小chunk_size；调整Query或视觉桶必须重新审计，不可截断标签。

默认每卡batch1、累积4、AMP、AdamW、R50 activation checkpointing。单卡有效batch4，两卡8，LR不自动缩放；累积不减少单个微批次峰值。CPU验证不能替代CUDA AMP/DDP、正式尺寸显存、收敛或OCR准确率验证。

## 代码入口

以下路径相对于`projects/OCRDINO/`：

| 文件 | 内容 |
| --- | --- |
| `ocrdino/recognition_query.py` | 4Conv、PE、Decoder、CE和分块重计算 |
| `ocrdino/recognition_capacity.py` | 几何桶与len+1规则 |
| `ocrdino/detector.py` | 共享C2、区域选择、容量检查和输出 |
| `ocrdino/vocabulary.py`、`hooks.py` | 字符/EOS、hash和checkpoint保护 |
| `ocrdino/dataset.py`、`transforms.py` | bbox+text读取和字段对齐，延续原版 |
| `ocrdino/heads.py` | 原检测损失与DN对应关系，未修改 |
| `configs/ocrdino_r50_4scale_query*.py` | 通用、24轮预训练、12轮微调 |
| `tools/audit_dataset.py`、`prepare_english_datasets.py` | Query容量审计与四包转换 |
| `tools/convert_checkpoint.py`、`smoke_test.py`、`infer.py` | 权重初始化、GPU检查和推理 |
| `tools/cpu_smoke_test.py`、`cpu_ddp_check.py` | 完整模型CPU检查、识别头两进程Gloo检查 |

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
python -m pip install -e . --no-build-isolation

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

此处 `...` 仅为文档省略，不是可复制的真实字符字典。实际文件包含从 U+0020 空格到 U+007E 波浪线的 95 个不同单字符。真实字符 ID 为0..94，EOS为95；没有blank/PAD预测类别。字典顺序决定分类器权重的含义，不能只保持字符数量相同就更换顺序。

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

按这批数据实际使用的字符表：`0..94 → chr(code+32)`，对应 95 个可打印 ASCII 字符；`96` 是尾部 padding；`95` 是不能从当前 JSON 还原原字符的编码。**源数据特殊编码与模型字符/EOS编码须分别处理**：先转为真实字符串，再由模型的有序字符字典编码。

例如 `[67,79,84,84,79,78,96,...]` 转成 `cotton`，不能把重复的两个 `t` 合并。本模型训练和预测都保留重复字符，不做CTC去重。

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

此前 CTC 版本完成的源数据扫描、转换和 Total-Text 图片检查保留在[历史数据审计索引](projects/OCRDINO/reports/english_datasets/README.md)；其中旧 CTC 容量与配置测试不是本版验收结果。本版对合并预训练 JSON 和 Total-Text train/test 重新执行了完整 Query 容量审计，使用 (32,64,128,256,500) 桶和 L+1 规则，错误和警告均为 0；单图最多 333 个检测 GT。新报告见 [pretrain](projects/OCRDINO/reports/query/pretrain.query_audit.json)、[Total-Text train](projects/OCRDINO/reports/query/totaltext_train.query_audit.json)、[Total-Text test](projects/OCRDINO/reports/query/totaltext_test.query_audit.json)。本轮没有重复提取或逐张解码原始图片。

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
  --source-root /root/autodl-tmp/data/english_sources \
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


## 新旧checkpoint与中文扩展

官方COCO checkpoint继续转换为detection-only初始化：去掉80类分类分支及DN类别embedding，保留兼容的R50、neck、Encoder、Decoder、900行query embedding和框回归参数。C2投影、四层CNN、视觉Linear、共享content、字符Decoder和分类器随机初始化。

旧CTC与新Query都可能输出96类，但旧blank=0/字符1..V，新字符0..V-1/EOS=V，**同形状不代表可加载**。新head保存`recognition_version`与有序字典`dictionary_hash`；即使strict=False，旧CTC/LSTM参数、缺版本或字典顺序不同也会抛错。Runner的`after_load_checkpoint`还检查新架构、EOS、ignore_index和字典元信息。

不直接恢复旧CTC训练checkpoint，不自动迁移其识别权重。新Query预训练到微调用完整新OCR权重，不再转换；同阶段恢复才加`--resume`。原`recognition.py`及三个旧CTC配置只保留明确错误提示。

中文通过`build_dictionary.py`生成有序Unicode字典，格式不变。改变字典须新建字符分类器，不能把不同字典权重直接加载。扩长时同时考虑视觉和Query桶，例如：

~~~bash
--cfg-options 'model.recognition_head.width_buckets=[32,64,128,256,512]' \
  'model.recognition_head.query_buckets=[32,64,128,256,512]'
~~~

## Linux GPU服务器使用

在项目根目录执行，先完成前面纯pip安装。原数据源只读，转换输出可放项目内部：

~~~bash
export PYTHONPATH="$PWD:$PYTHONPATH"
python projects/OCRDINO/tools/prepare_english_datasets.py \
  --source-root /root/autodl-tmp/data/english_sources \
  --output-root "$PWD/data/english" --extract-images \
  --temp-dir "$PWD/work_dirs/convert_tmp"

export OCRDINO_DATA_ROOT="$PWD/data/english"
export OCRDINO_IMAGES=''
export OCRDINO_DICTIONARY="$PWD/projects/OCRDINO/dictionaries/english.json"
export OCRDINO_BATCH_SIZE=1
export OCRDINO_ACCUMULATION=4
export OCRDINO_WORKERS=4
~~~

已有`ocrdino.text.v1`数据可复用，不必重新提图，但要重新审计Query容量。四包转换的file_name包含images/，所以OCRDINO_IMAGES为空；自备file_name=example.jpg时按实际目录设为images/等。

### 配置检查与数据审计

~~~bash
python -c "from mmengine import Config; c=Config.fromfile('projects/OCRDINO/configs/ocrdino_r50_4scale_query_pretrain_24e.py'); print(c.model.recognition_head); print(c.train_dataloader.dataset); print(c.train_cfg)"

python projects/OCRDINO/tools/audit_dataset.py \
  "$OCRDINO_DATA_ROOT/annotations/pretrain.json" \
  --dictionary "$OCRDINO_DICTIONARY" --image-root "$OCRDINO_DATA_ROOT" \
  --query-buckets 32 64 128 256 500 --query-width-scale 16 \
  --visual-width-buckets 32 64 128 256 500 --visual-width-scale 16 \
  --max-queries 900 --output work_dirs/pretrain_query_audit.json

python projects/OCRDINO/tools/audit_dataset.py \
  "$OCRDINO_DATA_ROOT/annotations/totaltext_train.json" \
  --dictionary "$OCRDINO_DICTIONARY" --image-root "$OCRDINO_DATA_ROOT" \
  --query-buckets 32 64 128 256 500 --query-width-scale 16 \
  --visual-width-buckets 32 64 128 256 500 --visual-width-scale 16 \
  --max-queries 900 --output work_dirs/totaltext_query_audit.json
~~~

审计参数对应head的query_buckets/query_width_scale与width_buckets/width_scale，修改模型时同步审计。max-queries是DINO检测槽数。图片根目录指向整个data/english，不再追加images/。

### COCO初始化与GPU smoke

~~~bash
python projects/OCRDINO/tools/convert_checkpoint.py \
  --source /root/autodl-tmp/checkpoint/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth \
  --output checkpoints/ocrdino_init.pth \
  --report checkpoints/ocrdino_init.report.json
export OCRDINO_PRETRAINED="$PWD/checkpoints/ocrdino_init.pth"

CUDA_VISIBLE_DEVICES=0 python projects/OCRDINO/tools/smoke_test.py \
  projects/OCRDINO/configs/ocrdino_r50_4scale_query_pretrain_24e.py \
  --checkpoint "$OCRDINO_PRETRAINED" --device cuda:0 --steps 2 \
  --output work_dirs/query_gpu_smoke.json
~~~

转换不会覆盖源权重，也拒绝覆盖已有输出；已有本项目转换结果可直接复用。GPU smoke用真实Dataset、模型和CUDA算子检查前向/反向/预测及峰值显存，不代表整套数据极端批次或收敛。该工具明确要求CUDA。

### 单卡：24轮预训练→12轮微调

~~~bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  projects/OCRDINO/configs/ocrdino_r50_4scale_query_pretrain_24e.py \
  --work-dir work_dirs/ocrdino_query_pretrain24

export OCRDINO_FINETUNE_FROM="$PWD/work_dirs/ocrdino_query_pretrain24/epoch_24.pth"
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  projects/OCRDINO/configs/ocrdino_r50_4scale_query_totaltext_12e.py \
  --work-dir work_dirs/ocrdino_query_totaltext12
~~~

预训练每轮遍历合并159,049张图，基础LR=1e-4，里程碑19/22；微调每轮遍历1,255张Total-Text train图片，LR=5e-5，里程碑8/11。均warmup 500 dataloader iterations、backbone LR乘0.1。合成图约占预训练图片93.71%，没有等额重采样。

微调用load_from加载第24轮完整Query模型，resume=False重建优化器及计数，从第1轮跑12轮。**不再转换OCR权重，不在跨阶段时加--resume。** 最终使用`work_dirs/ocrdino_query_totaltext12/epoch_12.pth`，它不是验证集挑选的best。

### 双卡DDP、恢复与推理

~~~bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
  tools/train.py projects/OCRDINO/configs/ocrdino_r50_4scale_query_pretrain_24e.py \
  --launcher pytorch --work-dir work_dirs/ocrdino_query_pretrain24

# 预训练完成后再执行
export OCRDINO_FINETUNE_FROM="$PWD/work_dirs/ocrdino_query_pretrain24/epoch_24.pth"
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
  tools/train.py projects/OCRDINO/configs/ocrdino_r50_4scale_query_totaltext_12e.py \
  --launcher pytorch --work-dir work_dirs/ocrdino_query_totaltext12

CUDA_VISIBLE_DEVICES=0 python projects/OCRDINO/tools/infer.py \
  projects/OCRDINO/configs/ocrdino_r50_4scale_query_totaltext_12e.py \
  work_dirs/ocrdino_query_totaltext12/epoch_12.pth \
  "$OCRDINO_DATA_ROOT/images/totaltext/test/0000001.jpg" \
  --device cuda:0 --out work_dirs/query_prediction.json
~~~

同阶段中断时在原配置/工作目录命令末尾加--resume，恢复该阶段优化器与进度。字典和结构保持一致。训练关闭val/test loop，test不参与训练；没有实现Total-Text官方polygon/lexicon评测器，单图预测不能当作官方OCR指标。

通用`ocrdino_r50_4scale_query.py`默认30轮；专用24/12配置固定各自数据和日程。配置已启用AmpOptimWrapper，无需再加--amp。

| 环境变量 | 默认值/用途 |
| --- | --- |
| OCRDINO_DATA_ROOT | 通用data/ocr，专用data/english |
| OCRDINO_IMAGES | 通用images/，专用空字符串 |
| OCRDINO_TRAIN_JSON | 仅通用配置：train.json |
| OCRDINO_DICTIONARY | projects/OCRDINO/dictionaries/english.json |
| OCRDINO_PRETRAINED | 转换后的COCO权重，默认checkpoints/ocrdino_init.pth |
| OCRDINO_FINETUNE_FROM | 新Query预训练第24轮完整权重 |
| OCRDINO_BATCH_SIZE / OCRDINO_ACCUMULATION / OCRDINO_WORKERS | 1 / 4 / 4 |
| OCRDINO_EPOCHS / OCRDINO_LR | 仅通用：30 / 1e-4 |
| OCRDINO_WORK_DIR | 通用输出目录 |
| OCRDINO_PRETRAIN_WORK_DIR / OCRDINO_FINETUNE_WORK_DIR | 专用输出目录，可由--work-dir覆盖 |

## Windows本地开发

在PowerShell进入副本，使用兼容PyTorch/MMCV/MMEngine解释器：

~~~powershell
Set-Location 'C:\Users\14222\Desktop\ocrdino\twostage\mmdetection_twostage_wo_ctc'
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
$env:OCRDINO_DICTIONARY = "$PWD\projects\OCRDINO\dictionaries\english.json"
$env:OCRDINO_WORKERS = '0'

# 仅解析配置，不导入自定义模型
python -c "from mmengine import Config; c=Config.fromfile('projects/OCRDINO/configs/ocrdino_r50_4scale_query.py', import_custom_modules=False); print(c.model.recognition_head); print(c.train_cfg)"

# 完整兼容环境：注册、构建及数值测试
python -m pip install pytest
python -m pytest -o addopts='' tests/test_projects/test_ocrdino -q -rs

# 换成实际JSON路径；省略image-root仅检查标注
python projects/OCRDINO/tools/audit_dataset.py `
  'data\english\annotations\pretrain.json' `
  --dictionary $env:OCRDINO_DICTIONARY `
  --query-buckets 32 64 128 256 500 --query-width-scale 16 `
  --visual-width-buckets 32 64 128 256 500 --visual-width-scale 16 `
  --output work_dirs\windows_query_audit.json

# 源权重只读，输出仅在副本内部；同名输出存在时不要重复转换
python projects/OCRDINO/tools/convert_checkpoint.py `
  --source 'C:\Users\14222\Desktop\ocrdino\cn\dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth' `
  --output checkpoints\ocrdino_init.pth `
  --report checkpoints\ocrdino_init.report.json
~~~

仅装MMEngine时可用import_custom_modules=False检查配置，但不等于构建模型。CPU版Torch/MMCV可做CPU数值验证，不能运行CUDA smoke；正式训练推荐Linux兼容CUDA环境。所有安装使用pip，不使用mim。

## 本次实际验收：2026-10-16

**92 项测试、38 个子测试通过，0 失败、0 跳过。** 使用 Python 3.10.20、PyTorch 2.1.2+cpu、torchvision 0.16.2+cpu、MMCV 2.1.0 的真实 CPU 算子和 MMEngine 0.10.7。依赖由 pip 隔离安装到本副本的 `.test_deps/`，未修改全局 Python 环境。

[完整验收报告](projects/OCRDINO/reports/query_validation/README.md)保存测试名称、18项对应关系、原始日志和复跑方式；[pytest XML](projects/OCRDINO/reports/query_validation/pytest.xml)保存逐项执行结果。

| 编号 | 要求 | 状态 | 实际证据 |
| --- | --- | --- | --- |
| 1 | MMEngine 正确解析配置 | 通过 | `test_training_configs.py::test_mmengine_resolves_official_dino_inheritance` 解析三个新配置并检查官方 DINO 继承；完整 smoke 还执行 custom_imports 与构建 |
| 2 | MODELS 注册、构建识别头 | 通过 | `test_registered_head_has_required_production_defaults` 用真实注册器构建默认 256 维、3 层、8 头识别头 |
| 3 | RoIAlign + 4Conv 形状 | 通过 | `test_real_roi_convolution_shapes_and_loss_gradients_reach_shared_c2` 使用真实 MMCV RoIAlign，检查 H=8 到 H=1 |
| 4 | Character Query 的位置编码 | 通过 | `test_positions_and_shared_content_define_distinct_ordered_queries` 检查同一内容向量加不同正弦位置 |
| 5 | Decoder 输出形状 | 通过 | `test_decoder_parallel_shape_and_visual_gradient`；默认真实完整模型也完成前后向 |
| 6 | 字典与 EOS ID | 通过 | 默认英文 V=95，EOS=95，输出 96 类；HELLO 目标逐项检查 |
| 7 | 无 PAD / blank 类别 | 通过 | 检查无 `pad_idx`、`blank_idx`，无 LSTM / CTCLoss 模块 |
| 8 | HELLO + EOS 训练目标 | 通过 | `test_hello_targets_have_one_eos_no_pad_class_and_keep_ll` 得到 H E L L O EOS -100 -100 |
| 9 | LL 不合并 | 通过 | 字典与解码测试保留 HELLO 中两个 L；无 EOS 的 LLLL 也保留 |
| 10 | EOS 后 Query 不计损失 | 通过 | `test_slots_after_eos_have_exactly_zero_loss_gradient` 修改后续 logits 不改变 loss，后续 logits 梯度严格为零 |
| 11 | 全 EOS 不能优于正确输出 | 通过 | `test_all_eos_is_penalized_more_than_correct_characters_and_eos` |
| 12 | bbox 动态选桶 | 通过 | `test_geometry_buckets_are_independent_and_identical_in_train_eval` 覆盖边界与上限 |
| 13 | 容量不足显式报错 | 通过 | 容量、未知字符、空标签错误；`max_rois` 采样前校验全部有效 GT；499 字符加 EOS 接受，500 字符加 EOS 拒绝 |
| 14 | 训练、推理选桶规则相同 | 通过 | train/eval 对同一几何得到完全相同视觉宽度与 Query 数 |
| 15 | 空 GT 零 loss 与参数图参与 | 通过 | head 测试、完整模型空 GT backward、真实 Gloo 单 rank/双 rank 全空三种覆盖 |
| 16 | 识别梯度传到 C2 与 head | 通过 | 真实 RoIAlign 的 C2 梯度非零；额外完整模型只反传 `loss_rec`，共享 ResNet stem 梯度非零 |
| 17 | 框与文字正确对应 | 通过 | 解码桶顺序回填测试；多图、无效框过滤、空图、按检测分数排序、坐标恢复与 text_scores / EOS 标志对应 |
| 18 | 字典顺序不一致拒绝 checkpoint | 通过 | 包括 MMEngine `strict=False`，缺 hash、旧 CTC/LSTM、错误识别版本均拒绝 |

除上述单元测试外，执行了：

- **完整模型 CPU 前向、反向和推理**：保留 R50、四尺度、900 个检测 Query、6+6 层 DINO 和默认三层字符 Decoder，加载实际转换的官方 checkpoint；输入为 `[1,3,256,256]` 合成图，分别检查有 GT 和空 GT。额外仅反传 `loss_rec` 时，共享 `backbone.conv1.weight` 梯度范数为 **0.07932898**。该脚本没有执行优化器更新或真实数据训练。
- **真实两进程 CPU/Gloo DDP**：对识别头检查两 rank 区域数 `[1,2]`、`[0,1]`、`[0,0]`，与全局区域平均参考的最大参数梯度误差为 **2.38e-7**。这里使用较小识别头、dropout=0、chunk_size=1 和激活重计算，未运行完整 DINO 的 GPU DDP。
- **全量新容量审计**：对已有合并预训练、Total-Text train/test JSON 使用新 Query 桶和 `len(text)+1` 规则复核，均为 **0 错误、0 警告**。本轮复核对象是标注 JSON，不是重新解码全部数据图片。
- **源码边界检查**：与修改前 SHA-256 基线比较，`mmdet/`、官方 `configs/dino/`、OCRDINO 的 `heads.py`、`dataset.py`、`transforms.py` 均未改动。详细文件清单见[完整性报告](projects/OCRDINO/reports/query_validation/source_integrity.json)。

CPU 测试对不同 chunk 的 loss/梯度等价、dropout 重计算一致性、保存激活存储减少作了检查。保存激活的 CPU 检查不能替代 GPU 峰值显存测量。

### 复跑本机 CPU 检查

以下命令复用本次在副本内安装的依赖；该解释器自身环境没有被安装或升级包。不要把 Windows 的 `.test_deps` 搬到 Linux 使用。

~~~powershell
Set-Location 'C:\Users\14222\Desktop\ocrdino\twostage\mmdetection_twostage_wo_ctc'
$env:PYTHONPATH = "$PWD\.test_deps;$PWD"
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = "$PWD\.test_tmp"
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = "$PWD\.test_tmp\matplotlib"
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$python = 'C:\Users\14222\anaconda3\envs\data_treate\python.exe'

& $python -B -m pytest -o 'addopts=' -p no:cacheprovider tests/test_projects/test_ocrdino -q
& $python -B projects/OCRDINO/tools/cpu_smoke_test.py
& $python -B projects/OCRDINO/tools/cpu_ddp_check.py
~~~

完整 CPU 模型脚本默认读取副本内 `checkpoints/ocrdino_init.pth`，输出位于 `projects/OCRDINO/reports/query_validation/`。本次已完成该 checkpoint 的转换与加载：保留 612 个 tensor，去掉 15 个 COCO 类别相关 tensor；转换报告见[checkpoint_conversion.json](projects/OCRDINO/reports/query_validation/checkpoint_conversion.json)。

### 尚未验证与实际限制

- **未执行 CUDA 算子、FP16 AMP、真实 GPU 峰值显存、完整模型双 GPU/NCCL DDP**；服务器应先跑前面的 GPU smoke。没有依据承诺这一版在指定显卡上占用多少 GB。
- 未执行真实数据上的优化器训练、24轮预训练和12轮微调；尚无收敛、字符准确率、整串准确率或 Total-Text 官方端到端指标。
- 字符位置编码和有序 CE 只是对齐的学习机制，不能保证对齐成功、未见排列泛化或优于 CTC。区域水平框仍不具备曲线矫正、多行顺序或竖排识别能力。
- 本批源数据原有 25 位转录上限、未知字符与缺少原始 split 映射的问题仍存在；模型支持更长序列不能恢复源标注中已经丢失的文字。
- 中文大词表、500槽长文本的真实视觉信息与学习效果未训练验证。扩大字符容量时需一起检查图像清晰度、C2分辨率、视觉桶和显存。

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
  --source work_dirs/ocrdino_query_pretrain24/epoch_24.pth
~~~

默认输出 `work_dirs/ocrdino_query_pretrain24/bestepoch.pth`。如果从FP32续训目录完成训练，`--source` 应改为该目录的 `epoch_24.pth`；以实际保存路径为准。也可以指定独立输出：

~~~bash
python projects/OCRDINO/tools/export_model.py \
  --source work_dirs/ocrdino_query_pretrain24/epoch_24.pth \
  --output checkpoints/bestepoch.pth
~~~

终端会打印导出前后文件大小、张量数量以及删除的字段。导出文件顶层只含 `state_dict` 和必要的 `meta`。

用于Total-Text新阶段微调时，可以设置：

~~~bash
export OCRDINO_FINETUNE_FROM="$PWD/work_dirs/ocrdino_query_pretrain24/bestepoch.pth"
~~~

继续使用原12轮微调配置的 `load_from`，或将该文件作为单图推理的checkpoint参数。**导出文件不再支持恢复旧优化器和训练进度，不能用于 `--resume`。** 断点续训仍用保留的完整 `epoch_24.pth`。

同一个工具也能导出微调最后一轮：将source换为 `epoch_12.pth`，并显式加 `--expected-epoch 12`。这与 `convert_checkpoint.py` 的COCO初始化转换不同，不会丢弃任何检测或识别权重。
