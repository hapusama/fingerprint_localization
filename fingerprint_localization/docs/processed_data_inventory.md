# Processed Data Inventory

生成时间：2026-06-22

本文档记录当前本地已经处理出来的数据文件，以及每一类文件的用途。路径均相对于项目根目录：

```text
D:\Desktop\fingerprint_localization\fingerprint_localization
```

注意：`data/processedData/`、`model/v1/input/`、`model/v1/output/` 已经被 `.gitignore` 忽略，通常作为本地实验产物保存，不建议直接推送大文件。

## 1. 当前最重要的主线产物

### USRP LoRa 真实频谱幅相特征

这条线对应当前主问题：从 USRP `.bin` 里提 LoRa 前导码频谱幅相特征，再用真实特征做分类和可分性分析。

| 文件 | 说明 |
|---|---|
| `data/processedData/usrp_preamble_fft_magphase16_54loc_20pkt.csv` | USRP 原始 `.bin` 提取后的包级特征 CSV；1 个 LoRa 包 = 1 个样本；每个样本包含 16-bin 幅度和 16-bin 相位。当前 370 行。 |
| `data/processedData/usrp_magphase16_real_54loc_strict.csv` | 严格过滤后的 USRP 幅相数据表，保留原始列和标准化特征列。当前 370 行。 |
| `data/processedData/usrp_magphase16_real_54loc_strict.metadata.json` | 严格数据集的元信息。当前 `features_2x16 = [370, 2, 16]`，有效位置 32/54。 |
| `model/v1/input/usrp_magphase16_real_54loc_strict.pth` | 严格 USRP 分类数据集，包含 `features_2x16`、`features_flat`、`label`，以及 raw 特征。 |
| `data/processedData/usrp_magphase16_real_min10.csv` | 从 strict 数据里筛出样本数不少于 10 的位置后的 CSV。 |
| `data/processedData/usrp_magphase16_real_min10_dataset.csv` | min10 版本的分类输入 CSV。当前 318 行。 |
| `data/processedData/usrp_magphase16_real_min10_dataset.metadata.json` | min10 数据集元信息。当前 `features_2x16 = [318, 2, 16]`，有效位置 22/54。 |
| `model/v1/input/usrp_magphase16_real_min10.pth` | min10 版本的 `.pth` 分类数据集。 |

当前严格数据集的有效位置：

```text
15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 48, 50
```

当前缺失位置：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
41, 46, 47, 49, 51, 52, 53
```

### USRP 可分性诊断

这条线用于回答：同一位置的 packet 频谱是否相似，不同位置是否不同。

| 文件 | 说明 |
|---|---|
| `data/processedData/usrp_similarity_analysis/loc_mag_mean_std.png` | 每个位置 16-bin 幅度均值曲线和标准差。 |
| `data/processedData/usrp_similarity_analysis/loc_phase_mean_std.png` | 每个位置 16-bin 相位圆周均值和圆周标准差。 |
| `data/processedData/usrp_similarity_analysis/pairwise_distance_distributions.png` | 类内/类间 pairwise distance 分布对比图。 |
| `data/processedData/usrp_similarity_analysis/loc_shape_stats.csv` | 每个位置的样本数、平均检测分数、类内距离统计。 |
| `data/processedData/usrp_similarity_analysis/similarity_summary.json` | 可分性分析的汇总 JSON。 |

当前诊断结论：

| 距离 | 类内 median | 类间 median | 分布重叠系数 | `P(类间 > 类内)` |
|---|---:|---:|---:|---:|
| magnitude z-score | 4.6815 | 4.8781 | 0.920 | 0.517 |
| phase circular | 6.5448 | 6.7706 | 0.874 | 0.528 |
| mag + phase sin/cos z-score | 9.5505 | 9.7852 | 0.911 | 0.530 |

这说明当前 2x16 幅相特征的类内/类间分布高度重叠，直接分类器效果弱主要是特征可分性不足，不是单纯模型没调好。

### USRP 原始文件和检测探针

| 文件 | 说明 |
|---|---|
| `data/processedData/usrp_raw_file_inventory.csv` | USRP raw `.bin` 文件清单，主要记录 position、文件名、大小。 |
| `data/processedData/usrp_raw_energy_inventory_auto.csv` | 每个 `.bin` 的块能量统计，用于判断缺失位置是低能量还是检测参数问题。 |
| `data/processedData/usrp_diagnostics/*.png` | 单包 STFT、dechirp FFT 等诊断图。当前主要是 `2_0_33_11_2_16` 的示例图。 |
| `data/processedData/usrp_preamble_fft_diagnostic_sample.csv` | 早期诊断样例提取结果。 |
| `data/processedData/usrp_preamble_fft_magphase16_benchmark.csv` | 提取参数 benchmark 结果。 |
| `data/processedData/usrp_preamble_fft_magphase16_periodic_smoke.csv` | periodic extraction 小规模冒烟测试。 |
| `data/processedData/usrp_preamble_fft_magphase16_periodic_fasttest.csv` | periodic extraction 快速测试。 |
| `data/processedData/usrp_magphase16_flow_smoke_extract.csv` | USRP 流程冒烟测试的特征提取 CSV。 |
| `data/processedData/usrp_magphase16_flow_smoke_dataset.csv` | 冒烟测试构建出的分类 CSV。 |
| `data/processedData/usrp_magphase16_flow_smoke_dataset.metadata.json` | 冒烟测试数据集 metadata。 |
| `model/v1/input/usrp_magphase16_flow_smoke.pth` | 冒烟测试 `.pth`。 |
| `data/processedData/usrp_magphase16_flow_readme.md` | 记录 USRP mag/phase 处理链路和复现命令的说明。 |

下面这些是针对单个位置或参数组合的探针文件，主要用于排查漏检，不建议作为最终训练数据：

```text
data/processedData/usrp_preamble_fft_magphase16_corridor2_sf10_probe.csv
data/processedData/usrp_preamble_fft_magphase16_probe_pos0.csv
data/processedData/usrp_preamble_fft_magphase16_probe_pos0_loose.csv
data/processedData/usrp_preamble_fft_magphase16_probe_pos0_pick.csv
data/processedData/usrp_preamble_fft_magphase16_probe_pos0_sf10.csv
data/processedData/usrp_preamble_fft_magphase16_probe_pos0_sf12.csv
data/processedData/usrp_preamble_fft_magphase16_probe_pos15_strict.csv
data/processedData/usrp_probe_pos0_bw250_sf11.csv
data/processedData/usrp_probe_pos0_fs250_sf11.csv
data/processedData/usrp_probe_pos0_sf8.csv
data/processedData/usrp_probe_pos0_sf9.csv
data/processedData/usrp_probe_pos0_sf9_recheck.csv
data/processedData/usrp_probe_pos15_seed1200.csv
data/processedData/usrp_probe_pos3_noncoherent.csv
data/processedData/usrp_probe_pos3_seed4000.csv
data/processedData/usrp_probe_pos6_noncoherent.csv
data/processedData/usrp_probe_pos6_seed1200.csv
data/processedData/usrp_probe_zero_pos0_seed1200.csv
```

## 2. Lab1 spectrum peak16 baseline

这条线是较早的 RSSI/频谱峰值 baseline，目前 RGM 配置还主要指向它。

| 文件 | 说明 |
|---|---|
| `data/processedData/lab1_spectrum_peak16_all.csv` | lab1 数据整理后的 16-bin 频谱峰值特征总表。 |
| `data/processedData/lab1_spectrum_peak16_all.metadata.json` | lab1 peak16 数据 metadata。 |
| `model/v1/input/lab1_spectrum_peak16_pretrain_dataset.pth` | RGM pretrain 数据，`rssi = [572, 16]`，`label = [572]`。 |
| `model/v1/input/lab1_spectrum_peak16_finetune_dataset.pth` | RGM finetune 数据，`rssi = [124, 16]`，`label = [124]`。 |
| `model/v1/input/lab1_spectrum_peak16_test_dataset.pth` | 测试数据，`rssi = [124, 16]`，`label = [124]`。 |
| `model/v1/input/lab1_spectrum_peak16_finger_dataset.pth` | 指纹分类数据，`features = [572, 18]`，`label = [572]`。 |
| `model/v1/output/lab1_spectrum_peak16_area_fake_rgm.pth` | RGM 生成出的 lab1 peak16 fake database。 |
| `model/v1/output/classifier_lab1_spectrum_peak16_metrics.json` | lab1 peak16 分类器指标。 |
| `model/v1/output/classifier_lab1_spectrum_peak16_per_label.png` | lab1 peak16 按 label 的分类表现图。 |

## 3. Multipath condition 产物

这条线用于给扩散模型准备多径 condition 向量，目前 `dataset.py` 已经支持读取 `multipath_*_norm` 列。

| 文件 | 说明 |
|---|---|
| `data/processedData/multipath_conditions.csv` | 每个位置的多径 condition 汇总表，包含归一化列。 |
| `data/processedData/multipath_segments.csv` | 多径分析过程中的分段/候选 tap 信息。 |
| `data/processedData/multipath_summaries.json` | 每个位置的多径 summary。 |
| `data/processedData/multipath_condition_metadata.json` | 多径 condition 归一化和字段 metadata。 |
| `model/v1/output/location_vector_multipath.csv` | 加入 multipath condition 后的 location vector。 |
| `data/processedData/multipath_probe_conditions.csv` | 小样本 probe 的多径 condition 表。 |
| `data/processedData/multipath_probe_segments.csv` | probe 的多径 segments。 |
| `data/processedData/multipath_probe_summaries.json` | probe 的多径 summary。 |
| `data/processedData/multipath_probe_condition_metadata.json` | probe condition 的 metadata。 |
| `model/v1/output/location_vector_multipath_probe.csv` | probe 版本 location vector。 |
| `data/processedData/multipath_probe_3_0_34.csv` | 单文件/单位置多径探针 CSV。 |
| `data/processedData/multipath_probe_3_0_34.json` | 单文件/单位置多径探针 JSON。 |
| `data/processedData/multipath_probe_sf11.csv` | SF11 多径探针 CSV。 |

## 4. 历史楼层/RSSI 处理数据

这些是原项目或早期实验留下的按楼层、SF 划分的数据。它们体量较大，主要用于已有 baseline、RGM 历史实验或对照实验。

| 目录 | 文件数 | 大小 | 说明 |
|---|---:|---:|---|
| `data/processedData/FLOOR2/` | 21 | 48.542 MB | FLOOR2 的处理后 RSSI/特征 CSV。 |
| `data/processedData/FLOOR3/` | 136 | 249.241 MB | FLOOR3 的处理后 RSSI/特征 CSV，是历史数据中最大的一组。 |
| `data/processedData/FLOOR4/` | 22 | 50.607 MB | FLOOR4 的处理后 RSSI/特征 CSV。 |
| `data/processedData/FLOOR5/` | 22 | 50.834 MB | FLOOR5 的处理后 RSSI/特征 CSV。 |

对应的 `.pth` 数据集集中在 `model/v1/input/`，例如：

```text
floor2_sf_11_pretrain_dataset.pth
floor2_sf_11_finetune_dataset.pth
floor2_sf_11_test_dataset.pth
floor3_sf_7_pretrain_dataset.pth
floor3_sf_7_finetune_dataset.pth
floor3_sf_7_test_dataset.pth
floor3_sf_8_pretrain_dataset.pth
floor3_sf_8_finetune_dataset.pth
floor3_sf_8_test_dataset.pth
floor3_sf_9_pretrain_dataset.pth
floor3_sf_9_finetune_dataset.pth
floor3_sf_9_test_dataset.pth
floor3_sf_10_pretrain_dataset.pth
floor3_sf_10_finetune_dataset.pth
floor3_sf_10_test_dataset.pth
floor3_sf_11_pretrain_dataset.pth
floor3_sf_11_finetune_dataset.pth
floor3_sf_11_test_dataset.pth
floor3_sf_12_pretrain_dataset.pth
floor3_sf_12_finetune_dataset.pth
floor3_sf_12_test_dataset.pth
floor4_sf_11_pretrain_dataset.pth
floor4_sf_11_finetune_dataset.pth
floor4_sf_11_test_dataset.pth
floor5_sf_11_pretrain_dataset.pth
floor5_sf_11_finetune_dataset.pth
floor5_sf_11_test_dataset.pth
```

还有一些历史对照或消融数据：

```text
ab_residual_pretrain_sf11_floor3.pth
ab_residual_finetune_sf10_floor3.pth
ab_residual_test_sf10_floor3.pth
ab_residual_finetune_sf11_floor4.pth
ab_residual_test_sf11_floor4.pth
ab_spatial_pretrain_sf11_floor3.pth
ab_spatial_finetune_sf10_floor3.pth
ab_spatial_test_sf10_floor3.pth
ab_spatial_finetune_sf11_floor4.pth
ab_spatial_test_sf11_floor4.pth
knn_sf11_floor3_dataset.pth
20m_sf11_floor3_pretrain.pth
20m_sf11_floor3_finetune.pth
20m_sf9_floor3_test.pth
5m_sf11_floor3_pretrain.pth
5m_sf11_floor3_finetune.pth
5m_sf9_floor3_test.pth
```

## 5. `model/v1/output/` 中的数据相关产物

`model/v1/output/` 里混有模型权重、生成数据、分类器指标和 location vector。严格来说这不是原始“处理后数据”，但它们是数据处理和训练链路的下游产物。

### Location vector

```text
model/v1/output/location_vector.csv
model/v1/output/location_vector_v2.csv
model/v1/output/location_vector_5m.csv
model/v1/output/location_vector_20m.csv
model/v1/output/location_vector_multipath.csv
model/v1/output/location_vector_multipath_probe.csv
```

### 生成数据库 / fake 数据

```text
model/v1/output/lab1_spectrum_peak16_area_fake_rgm.pth
model/v1/output/floor2_sf_11_area_fake_rgm.pth
model/v1/output/floor3_sf_7_area_fake_rgm.pth
model/v1/output/floor3_sf_8_area_fake_rgm.pth
model/v1/output/floor3_sf_9_area_fake_rgm.pth
model/v1/output/floor3_sf_10_area_fake_rgm.pth
model/v1/output/floor3_sf_11_area_fake_rgm.pth
model/v1/output/floor3_sf_12_area_fake_rgm.pth
model/v1/output/floor4_sf_11_area_fake_rgm.pth
model/v1/output/floor5_sf_11_area_fake_rgm.pth
model/v1/output/ab_residual_fake_sf10_floor3.pth
model/v1/output/ab_residual_fake_sf11_floor4.pth
model/v1/output/ab_spatial_fake_sf10_floor3.pth
model/v1/output/ab_spatial_fake_sf11_floor4.pth
```

### USRP 分类器输出

```text
model/v1/output/classifier_usrp_magphase16_flow_check.ckpt
model/v1/output/classifier_usrp_magphase16_flow_check_metrics.json
model/v1/output/classifier_usrp_magphase16_real_54loc_strict.ckpt
model/v1/output/classifier_usrp_magphase16_real_54loc_strict_metrics.json
model/v1/output/classifier_usrp_magphase16_real_min10.ckpt
model/v1/output/classifier_usrp_magphase16_real_min10_metrics.json
model/v1/output/classifier_usrp_magphase16_retest.ckpt
model/v1/output/classifier_usrp_magphase16_retest_metrics.json
```

当前 retest 指标：

```text
MLP accuracy: 0.1081
KNN accuracy: 0.0946
```

### RGM 权重和日志

```text
model/v1/output/1_pretrained_lab1_spectrum_peak16_rgm.ckpt
model/v1/output/2_finetuned_lab1_spectrum_peak16_rgm.ckpt
model/v1/output/rgm_log/pretrain_run_metrics.json
model/v1/output/rgm_log/finetune_run_metrics.json
model/v1/output/rgm_log/generate_run_metrics.json
```

历史权重还包括：

```text
model/v1/output/1_pretrained_rgm.ckpt
model/v1/output/2_finetuned_rgm.ckpt
model/v1/output/floor2_finetuned_rgm.ckpt
model/v1/output/floor4_finetuned_rgm.ckpt
model/v1/output/floor5_finetuned_rgm.ckpt
model/v1/output/sf7_finetuned_rgm.ckpt
model/v1/output/sf8_finetuned_rgm.ckpt
model/v1/output/sf9_finetuned_rgm.ckpt
model/v1/output/sf10_finetuned_rgm.ckpt
model/v1/output/sf12_finetuned_rgm.ckpt
```

## 6. 推荐后续使用顺序

如果继续当前 USRP 真实特征路线，优先看这几个文件：

1. `data/processedData/usrp_preamble_fft_magphase16_54loc_20pkt.csv`
2. `data/processedData/usrp_magphase16_real_54loc_strict.csv`
3. `model/v1/input/usrp_magphase16_real_54loc_strict.pth`
4. `data/processedData/usrp_similarity_analysis/pairwise_distance_distributions.png`
5. `data/processedData/usrp_raw_energy_inventory_auto.csv`

如果回到 RGM baseline，优先看：

1. `data/processedData/lab1_spectrum_peak16_all.csv`
2. `model/v1/input/lab1_spectrum_peak16_pretrain_dataset.pth`
3. `model/v1/input/lab1_spectrum_peak16_finetune_dataset.pth`
4. `model/v1/input/lab1_spectrum_peak16_test_dataset.pth`
5. `model/v1/output/location_vector_multipath.csv`

## 7. 一句话总结

当前数据产物可以分成三层：

1. 历史 RSSI/楼层 baseline：`FLOOR2/3/4/5` 和大量 `floor*_sf*_*.pth`。
2. RGM 当前 baseline：`lab1_spectrum_peak16_*` 和 multipath condition。
3. 最新 USRP 验证线：`usrp_preamble_fft_magphase16_54loc_20pkt.csv`、`usrp_magphase16_real_54loc_strict.pth`、`usrp_similarity_analysis/*`。

其中最新 USRP 线已经证明当前 2x16 包级幅相特征类内/类间高度重叠，下一步应优先改特征或增加有效样本，而不是直接加深分类器。
