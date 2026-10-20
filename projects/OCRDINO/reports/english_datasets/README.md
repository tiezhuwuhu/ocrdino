# 四个英文数据包审计记录

审计对象是用户提供的 `dataset/eng/{mlt2017.zip,syntext1.tar,syntext2.zip,totaltext.zip}`。2026-10-14 至 2026-10-15 全量扫描源标注，2026-10-15 实际完成全部 JSON 转换及预训练合并。原始档案始终只读。本目录只保存报告，不将大型数据集或图片提交到源码中。

## 报告入口

| 文件 | 含义 |
| --- | --- |
| [conversion_summary.json](conversion_summary.json) | 四包、五个 split 的实际转换统计；本次完整转换为 JSON-only |
| [pretrain_schema_audit.json](pretrain_schema_audit.json) | 对真实合并后的 339,880,597 字节 `pretrain.json` 再次检查 schema、字典和 CTC 容量 |
| [mlt2017_train.audit.json](mlt2017_train.audit.json) | MLT 转换统计与标签屏蔽原因 |
| [syntext1_train.audit.json](syntext1_train.audit.json)、[syntext2_train.audit.json](syntext2_train.audit.json) | 两份 SynText 转换统计 |
| [totaltext_train.audit.json](totaltext_train.audit.json)、[totaltext_test.audit.json](totaltext_test.audit.json) | Total-Text 转换统计及明确的图像坐标系修复记录 |
| [totaltext_train.images_checked.audit.json](totaltext_train.images_checked.audit.json)、[totaltext_test.images_checked.audit.json](totaltext_test.images_checked.audit.json) | 追加执行 `--datasets totaltext --extract-images`，1,555 张图实际提取并检查尺寸/EXIF 的结果 |
| [source_scan/](source_scan/) | 修复前源标注的全量扫描、归档成员核对、SynText CTC 容量与越界程度统计 |
| [config_validation.json](config_validation.json) | MMEngine 0.10.7 真实配置继承/序列化验证，未构建模型或 Runner |

`source_scan` 是原始输入审计，所以 Total-Text 越界框数会大于转换报告；两张图的源 JSON 宽高与其框坐标系不一致，修复后再裁剪得到最终数量。不要把修复前的几何问题数误当成转换后的错误数。

## 核实的结果

| split | 图片 | 检测框 | 可用识别标签 | 屏蔽识别 |
| --- | ---: | ---: | ---: | ---: |
| MLT2017 train | 9,999 | 89,418 | 58,740 | 30,678 |
| SynText1 train | 94,723 | 1,318,214 | 1,317,925 | 289 |
| SynText2 train | 54,327 | 479,834 | 479,833 | 1 |
| Total-Text train | 1,255 | 9,722 | 9,708 | 14 |
| Total-Text test | 300 | 1,975 | 1,974 | 1 |
| **预训练合并（前三项）** | **159,049** | **1,887,466** | **1,856,498** | **30,968** |

合并文件 image ID、annotation ID、file_name 唯一，引用和 bbox 合法；未屏蔽标签全部由默认英文字符表覆盖，并满足 `(32,64,128)`、`width_scale=16` 的 CTC 时间步要求。最多 333 个检测 GT/图，未超过 900 query。Total-Text test 没有加入预训练或微调。

默认对占满 25 位的 `rec` 屏蔽识别，原因是无法从现有标注判断是否被上游截断；不是认定每条都错误。MLT 含 `95` 的 30,607 条转录无法从当前包恢复原文字，另有 14 条空转录。屏蔽原因有交集，总数已去重。所有有效检测框保留。

源 `rec` 的 `0..94` 对应 ASCII 32..126，`96` 为 padding。GT 重复字符不合并。MLT 唯一字面量特殊字符串为 `***`（源标注 19332，图片 0062899.jpg），SynText1 另有 11 条 `***`；它们的源 `iscrowd=0`，转换也按普通文字保留，没有猜测忽略标记。

## 验证范围

- 全部标注和所引用的图片归档成员已扫描；MLT 另有一张未被 JSON 引用的 `0060000.jpg`，不自行补入负样本。
- Total-Text 全部图片经过提取及头部尺寸/方向检查；两张 EXIF 修复图额外验证了 Pillow 解码像素前后完全一致。
- MLT/SynText 未本地完整提取或逐张解码。服务器运行 `--extract-images` 时会检查每张输出图的头部尺寸/方向；这仍不等于人工检查文字、逐像素质量或正确标注率。
- 数据准备工具的测试使用小型归档 fixture 验证嵌套 ZIP、TAR、路径和编码规则；真实合并 JSON 则通过独立全量审计。
- 无 GPU 训练或 OCR 准确率结论。MLT 包缺少与官方原始划分的映射、语言字段和未知字符原文，不据此宣称纯英语或官方 split 无泄漏。

重跑方式与训练命令见[仓库 README](../../../../README.md)的“四个英文数据包：审计与转换”和“英文实验：24 轮预训练 → 12 轮 Total-Text 微调”。
