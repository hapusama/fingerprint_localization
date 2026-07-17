# Old-370 全 split 1:10 与保守 TTA

更新日期：2026-07-16

## 决策

论文实验路线回到旧版严格对齐的 370 个真实包。固定 source split 保持为
train/validation/test = 223/73/74，seed 为 `20260626`，不重新划分。

本实验给 train、validation、test 的每个 source 分别使用 10 个已有高斯扰动视图，
得到 2230/730/740 行。三组 source overlap 为 0，但增强视图不是独立样本，因此
validation/test 的统计单位仍是 73/74 个原始 source packet。

## ACO 配置

- ACO v4，Top-5，RSSI class k=3，4 segments。
- validation 的 `T_seg=0.022065795990103106`，由 train-only 数据自动标定。
- 正式 train+validation refit 使用 296 个 source、2960 个增强行。
- 正式 test 的 `T_seg=0.02403918092026683`，测试为 74 source、740 个相关视图。

## 无条件 TTA

- Validation 行级诊断：547/730 = 74.93%。
- Validation 10 视图多数投票：59/73 = 80.82%。
- Test 行级诊断：613/740 = 82.84%。
- Test 10 视图多数投票：66/74 = 89.19%。
- 原始未增强主线：67/74 = 90.54%。
- TTA 相对原始主线：1 W2R、2 R2W、另有 1 个错误改为另一错误，净 -1，
  McNemar exact 双侧 p=1.0。

所以，无条件使用 TTA 多数票不能保持原结果，也不能将 `613/740` 作为 740 个独立
测试包的论文准确率。

## 保守门控

默认保留原始 ACO 预测。仅当 TTA 与原预测不一致、且 10 视图多数票置信度达到阈值时
才允许替换。阈值网格为 0.50--1.00 及 never-trigger；只用 validation 选取，准确率
并列时依次选择更少 R2W、更少触发和更高阈值。

- Validation 选择阈值：0.8。
- Validation：60/73 -> 61/73，1 W2R、0 R2W、1 次触发。
- Test：67/74 -> 67/74，0 W2R、0 R2W、0 次触发。
- 最终保持 90.54%，但没有产生新的测试增益。

由于保守门控是在无条件 TTA test 已经查看之后提出的，本结果属于探索性分析。论文可以
继续使用预先存在的未增强原包结果 67/74；不能把门控结果包装成新的独立确认性测试。

## 文件

- 构建脚本：`experiments/aco_source_safe_1to10/build_old370_all_split_1to10.py`
- 聚合脚本：`experiments/aco_source_safe_1to10/summarize_old370_all_split_1to10.py`
- 门控脚本：`experiments/aco_source_safe_1to10/run_old370_guarded_tta.py`
- 无条件 TTA 报告：
  `experiments/aco_source_safe_1to10/old370_all_split_1to10/results/report/`
- 门控报告：
  `experiments/aco_source_safe_1to10/old370_all_split_1to10/results/guarded_tta/`

## 已知限制

当前复用了旧版 Gaussian-noise pool。该 pool 在 source split 之前按完整旧数据的列标准差
确定噪声尺度，因此存在轻度的无标签 transductive preprocessing。若结果需要作为新论文
协议重新确认，应按同一 223/73/74 source split，仅用 223 个 train source 估计噪声尺度，
重新生成三组视图，并使用新的独立测试或预注册规则复核。
