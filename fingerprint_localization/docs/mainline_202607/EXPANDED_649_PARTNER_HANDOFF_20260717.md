# Expanded-649 无 alpha 主线交接说明

更新日期：2026-07-17

## 1. 交付范围

本次交付包含四部分：

1. ExpandedReal-649 预处理后的算法就绪数据包和固定 source-safe 划分数据包；
2. `expanded_LDA_ACO_no_alpha` 的冻结配置、核心代码、模型和逐包结果；
3. 继续做外部基线、solver、弱包和可扩展性实验时必须遵守的协议；
4. Expanded-649 主线与旧 370 主线的差异说明。

原始 22.61 GiB IQ 不在 GitHub 交付范围内。算法包保留了 packet start、QC、恢复记录、schema、数据字典和 SHA-256，可追溯到原始提取过程。

## 2. 数据包

### 2.1 未划分算法就绪包

路径：

```text
data/expanded_real_32points_20260716/algorithm_ready/
  LoRaMorph_ExpandedReal649_v1_20260716.tar.gz
  LoRaMorph_ExpandedReal649_v1_20260716.tar.gz.sha256
```

内容：649 个真实 source packet、32 个位置、RSSI+、17-bin S17、q=1/q=4 长频谱、27 维 ML 特征、symbol 特征、NPZ、QC/恢复记录和数据字典。该包没有 train/validation/test 划分，不能在全包上预先拟合 scaler 或增强统计。

### 2.2 固定 source-safe 训练包

路径：

```text
data/expanded_real_32points_20260716/source_safe_1to10/deliverables/
  ExpandedReal649_source_safe_1to10_seed20260626_partner_20260716.tar.gz
  ExpandedReal649_source_safe_1to10_seed20260626_partner_20260716.tar.gz.sha256
```

协议：393/128/128 个 source，三组 overlap 为 0；只增强 393 个 train source，每个生成 10 个视图；validation/test 保留 128+128 个原始包。

验证报告状态为 `PASS`，覆盖 source overlap、原始 evaluation 包逐字段一致性、train-only 统计、每包 352 行频谱、q1/q4 物理一致性和 ACO loader 数量。

### 2.3 解压与校验

在仓库根目录执行：

```bash
python3 fingerprint_localization/scripts/verify_expanded_handoff.py

cd fingerprint_localization/data/expanded_real_32points_20260716/algorithm_ready
shasum -a 256 -c LoRaMorph_ExpandedReal649_v1_20260716.tar.gz.sha256
tar -xzf LoRaMorph_ExpandedReal649_v1_20260716.tar.gz

cd ../source_safe_1to10/deliverables
shasum -a 256 -c ExpandedReal649_source_safe_1to10_seed20260626_partner_20260716.tar.gz.sha256
tar -xzf ExpandedReal649_source_safe_1to10_seed20260626_partner_20260716.tar.gz -C ..
```

### 2.4 生成 formal refit 数据

返回仓库根目录后执行：

```bash
python3 fingerprint_localization/experiments/aco_source_safe_1to10/build_expanded_trainval_refit.py \
  --parent-dir fingerprint_localization/data/expanded_real_32points_20260716/algorithm_ready/LoRaMorph_ExpandedReal649_v1_20260716 \
  --split-dir fingerprint_localization/data/expanded_real_32points_20260716/source_safe_1to10/ExpandedReal649_source_safe_1to10_seed20260626 \
  --output-dir fingerprint_localization/data/expanded_real_32points_20260716/trainval_refit/ExpandedReal649_trainval_refit_seed20260626
```

预期输出：5210 个 train rows、128 个未增强 test rows、1,878,976 个 spectrum rows、train/test source overlap 为 0。增强统计仍只来自最初 393 个 train source，不含原 validation 和 test。

## 3. 当前 no-alpha 主线

### 3.1 模型与候选

LDA 输入共 27 维：

- RSSI+：`snr`、average/median/mode RSSI、RSSI variance、residual；
- S17/raw：17 个 peak-aligned FFT magnitude bins、peak-to-residual、detect score、`s17_c_s`、`s17_j_s`。

模型为 `StandardScaler + LinearDiscriminantAnalysis(solver="svd")`。候选直接取 LDA posterior Top-5，不再使用 alpha 将 RSSI 和 LDA 排名提前融合。

### 3.2 ACO 与最终输出

- 4 segments；16 ants；12 iterations；4 elite ants；seed `20260626`；
- RSSI 只作为 segment observation cost 和弱先验；
- ACO 产生 candidate-internal Score4；
- 最终分数为 `(1-beta) * normalized Score4 + beta * normalized LDA posterior`；
- validation 选择 `beta=0.6`，同分时取更小 beta；formal 在冻结后才加载和评估。

### 3.3 冻结结果

| Split | LDA Top-1 | LDA Top-5 recall | ACO search | 最终 | MAE/P95 | 严重错误率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 117/128 | 128/128 | 106/128 | 119/128 | 0.472/3.390 m | 3.12% |
| formal | 120/128 | 128/128 | 105/128 | 120/128 | 0.315/3.390 m | 0.78% |

formal 中最终结果与纯 LDA 同为 120/128。ACO 的价值目前是候选约束、segment 物理证据和逐包审计，不是已证明的净 accuracy 增益。

## 4. 代码地图

按调用关系阅读：

1. `run_no_alpha_validation_refreeze.py`：当前入口，validation 冻结 beta 后运行 formal。
2. `run_candidate_recall_and_controlled_weakness.py`：加载 validation/formal context、诊断分组和扰动工具。
3. `run_search_mechanism_ablation.py`：平均代价、贪心、无信息素、完整 ACO 定义。
4. `run_expanded_lda_aco_mainline.py`：训练并保存 validation/formal LDA 模型的上一阶段入口。
5. `run_expanded_supervised_ensemble.py`：27 维 feature table、LDA 模型和 refit table。
6. `run_expanded_aco_ml_prior.py`：ACO stage args、模型 posterior 与 source-level ranker。
7. `run_aco_v4_source_level_on_split.py`：source-level templates/prototypes 和 packet loader。
8. `run_aco_v4_on_split.py`、`model/v3/aco_packet_path_v4.py`、`model/v3/aco_packet_path.py`：ACO v4 和底层路径搜索。

当前结果目录还包含 `validation_lda_model.joblib` 和 `formal_lda_model.joblib`；no-alpha 脚本复用这两个模型，因为删除 alpha 不改变 LDA 的训练目标或参数。

## 5. 复现顺序

推荐 Python 3.8/3.9，NumPy 1.24.x、scikit-learn 1.3.2、joblib。先生成/校验数据，再运行：

```bash
python3 fingerprint_localization/experiments/aco_source_safe_1to10/run_expanded_lda_aco_mainline.py
python3 fingerprint_localization/experiments/aco_source_safe_1to10/run_no_alpha_validation_refreeze.py
```

如果只想复核已冻结结果，可直接使用已提交的模型和 no-alpha 入口。逐包预测、配置、manifest 和 checksum 位于：

```text
results/expanded_source_safe_1to10/aco_lda_only_mainline/
results/expanded_source_safe_1to10/aco_lda_only_no_alpha_refrozen_20260717/
```

## 6. 下一轮对比实验规则

- 新算法必须读取同一 `split_assignments.csv`；不得自行重新划分。
- validation 只能用于选择超参数；选定后记录完整配置和输入 SHA-256，再运行 formal。
- formal 已被查看，只能作 exploratory 横向比较。强结论需要新盲测包或预注册多 split。
- 每个方法同时报告 accuracy、MAE/P95、>10 m severe error、逐包预测、时延和峰值内存。
- 任何 correction/gate 必须报告 coverage、correction precision、W2R/R2W 和 paired test。
- solver 消融必须共享同一候选、观测代价、先验和最终融合，只更换搜索机制。
- 当前 feature-space 弱包扰动只作为开发诊断；论文鲁棒性结论需要 raw-IQ 扰动。

## 7. 当前未解决问题

完整 ACO 尚未证明不可替代：当前 no-alpha formal 中，无信息素最终 121/128，完整 ACO 120/128；路径的 segment switch/garbage 使用率也过低。下一步不是继续调 formal，而是在 validation/新盲测上让路径机制真正产生可观测的 segment 选择，再做多 seed 配对比较。
