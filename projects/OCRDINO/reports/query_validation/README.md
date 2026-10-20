# Character Query 版 OCRDINO 实际验证记录

验证日期：2026-10-16。项目：`mmdetection_twostage_wo_ctc`。

## 结果与环境

- 最终回归：**92 passed、38 subtests passed、0 failed、0 skipped**，7.35 秒。
- 测试 Python：3.10.20；PyTorch：2.1.2+cpu；torchvision：0.16.2+cpu；MMCV：2.1.0（真实编译 CPU ops）；MMEngine：0.10.7；NumPy：1.26.4。
- 依赖通过 pip 安装在当前副本 `.test_deps/`，临时文件在 `.test_tmp/`。没有修改旧项目或全局 Python 包。
- [pytest 原始输出](pytest.log)、[JUnit 测试结果](pytest.xml)。
- 真实 MMCV 测试发现并修复了 RoIAlign 参数调用问题：MMCV 2.1 的 functional API 是 `autograd.Function.apply`，必须传位置参数。修复后卷积、识别 loss、激活重计算及反向传播测试均通过。
- 剩余三个 pytest 警告来自 MMCV 旧 import 路径与 MMDetection 原生 docstring，不是测试失败；没有为消除警告修改 `mmdet/` 核心源码。

## 用户要求的 18 项测试

下表的通过状态均来自此次实际 CPU 测试。它们验证实现契约，不代表识别准确率或收敛性。

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

补充测试验证了逐区域“字符 CE 均值 + EOS CE”的 loss 定义、chunk 大小变化时 loss/梯度等价、checkpoint 重计算的 dropout RNG 保持，以及 checkpoint 减少需保存的区域激活存储。后者是 CPU autograd 保存张量存储检查，**不是 GPU 峰值显存测量**。

## 完整模型 CPU smoke

脚本：[cpu_smoke_test.py](../../tools/cpu_smoke_test.py)。报告：[cpu_model_smoke.json](cpu_model_smoke.json)。日志：[cpu_model_smoke.log](cpu_model_smoke.log)。

实际构建并加载初始化权重：

`ResNet-50 → 4-scale DINO (900 Query, 6 encoder + 6 decoder) + C2 → RoIAlign → 4Conv → 3-layer Character Query Decoder`。

- 输入为 `[1,3,256,256]` 合成 RGB 张量，GT 为两个 bbox 与 HELLO / BOOK。尺寸 256 使检测 encoder 的四尺度 token 数足够选择 900 个 proposal。
- 有 GT：联合 loss **91.14211**，识别 loss **9.07122**；backbone、encoder、decoder、识别头梯度均有限且非零。
- 空 GT：识别 loss **0**；识别头所有可训练参数均产生零梯度；检测分支正常反向。
- **单独反传识别 loss**：`loss_rec=9.25605`，共享 `backbone.conv1.weight` 梯度范数 **0.07932898**。这直接证明识别监督回传到共享 ResNet，不仅是检测损失造成 backbone 梯度。
- 推理返回 3 个框及对应字符串、text_scores、text_terminated；原图坐标恢复与 `ocr_scores=scores*text_scores` 检查通过。
- 为保证随机初始化的识别头实际执行，smoke **仅覆盖推理筛选参数**为 score_thr=0、max_per_img=3；正式配置仍是 0.3 / 100。网络与训练容量没有缩小。
- 随机识别头输出无意义文字是预期行为；本 smoke 没有声称识别准确。
- 实测有 GT 前后向约 2.31 秒、空 GT 约 1.99 秒、额外识别反向约 0.84 秒；仅反映本机此次小输入 CPU 运行。

## 两进程真实 CPU/Gloo 验证

脚本：[cpu_ddp_check.py](../../tools/cpu_ddp_check.py)。报告：[cpu_ddp.json](cpu_ddp.json)。

两个真实 `DistributedDataParallel` 进程、Gloo backend、真实 MMCV RoIAlign；开启 `with_cp=True`、`chunk_size=1`，小识别头维度用于加速，dropout=0 用于确定性梯度比较。

| 两 rank 有效区域数 | DDP 平均 loss | 全局参考 loss | 最大参数梯度绝对误差 |
| --- | --- | --- | --- |
| 1 / 2 | 2.48922038 | 2.48922038 | 1.79e-7 |
| 0 / 1 | 2.40290046 | 2.40290046 | 2.38e-7 |
| 0 / 0 | 0 | 0 | 0 |

所有参数参与梯度；同时检查了 C2 梯度与全局参考对应关系。这验证识别 head 的真实 DDP 归一化与空 rank 行为，**没有执行完整 DINO 的双 GPU DDP**。

## 官方 checkpoint 转换

读取用户给定的官方 R50 improved 4-scale checkpoint，输出当前副本 `checkpoints/ocrdino_init.pth`；源文件未写入。

- 保留 612 个 tensor。
- 删除 7 个 COCO 80 类分类分支的 weight/bias，以及 DN 的 80 类 embedding，共 15 个 tensor。
- 新文字检测分类分支、DN 文字 embedding 和 Character Query 识别头随机初始化。
- 转换报告：[checkpoint_conversion.json](checkpoint_conversion.json)。
- 完整 CPU 模型已经实际加载此输出；日志中的缺失类别头与识别头参数符合上述随机初始化范围。

## 复跑命令

在项目根目录，使用已有兼容环境：

```bash
python -m pytest -o addopts= -p no:cacheprovider tests/test_projects/test_ocrdino -q
python projects/OCRDINO/tools/cpu_smoke_test.py
python projects/OCRDINO/tools/cpu_ddp_check.py
```

此次 Windows 本机隔离依赖的设置：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) '.test_deps'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = Join-Path (Get-Location) '.test_tmp'
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.test_tmp/matplotlib'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$python = 'C:/Users/14222/anaconda3/envs/data_treate/python.exe'
& $python -B -m pytest -o 'addopts=' -p no:cacheprovider tests/test_projects/test_ocrdino -q
& $python -B projects/OCRDINO/tools/cpu_smoke_test.py
& $python -B projects/OCRDINO/tools/cpu_ddp_check.py
```

## 未执行

- CUDA 算子、CUDA FP16 AMP、真实 GPU 峰值显存、GPU throughput。
- 完整检测器双 GPU/NCCL DDP。
- 实际训练数据上的优化器训练、24 轮预训练 / 12 轮微调、准确率与收敛对比。
- 中文大词表训练与长文本真实视觉对齐质量。

本机虽然有 NVIDIA 显卡，但此次安装的是兼容 MMDetection/MMCV 的 CPU 测试环境。GPU 命令保留在项目 README，须在服务器的兼容 CUDA 环境继续验证。

