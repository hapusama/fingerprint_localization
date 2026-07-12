# LoRa preamble zero-padding FFT experiment

这个目录用于尝试附件里的流程：对 LoRa packet 的 preamble upchirp 做 dechirp，然后在时域符号后做零填充 FFT，观察原始 LoRa FFT 单位 bin 内的亚 bin 频谱形状是否出现可用于判断多径的肩峰、偏移或不对称。

## 核心脚本

```bash
python fingerprint_localization/experiments/zero_padding_fft/zero_padding_lora_fft.py
```

默认输入：

```text
INFOCOM_origin_data/LoRaRSSI+IQ_data/*.bin
```

默认只处理 1 个文件、每个文件 3 个 packet，并输出到：

```text
v2_output/20260624_zero_padding_fft/
```

这是为了避免第一次试跑直接扫完整 23GB 原始数据。确认参数可用后，可以增加 `--max-files` 和 `--max-packets-per-file`，或用 `--max-files 0` 处理全部文件。

## 快速试跑

```bash
python fingerprint_localization/experiments/zero_padding_fft/zero_padding_lora_fft.py \
  --input INFOCOM_origin_data/LoRaRSSI+IQ_data \
  --max-files 1 \
  --max-packets-per-file 3 \
  --q-values 1,4,8,16,32 \
  --feature-symbols 8 \
  --skip-preamble-symbols 1
```

如果你使用 Codex 桌面自带 Python 或系统 Python 遇到 `numpy` 缺失，请改用包含 `numpy`/`matplotlib` 的环境，例如 Radioconda、Conda 或项目实验环境。

## 和附件流程的对应关系

Step A: preamble 对齐与 dechirp

- 脚本先在 `.bin` 中检测 preamble 起点。
- 每个 upchirp symbol 下采样到 LoRa 原始 FFT 长度 `N = 2^SF`。
- 用本地生成的 downchirp 做 `y[n] = r[n] * s_up^*[n]`。

Step B: 零填充 FFT

- 对每个 dechirped symbol 分别计算 `FFT(y[n], qN)`。
- 默认 `q in {1,4,8,16,32}`，可用 `--q-values` 修改。

Step C: 截取 bin0 附近亚 bin 频谱片段

- 默认截取 `[-2,+2]` original bins，对应 `[-2q,+2q]` zero-padding samples。
- 默认 `--profile-alignment per-symbol` 会把每个 symbol 的原始 FFT 主峰作为中心 bin，再看主峰附近的亚 bin 形状。
- 如果想严格看理论 bin0，使用 `--profile-alignment bin0`。
- 如果想多个 preamble symbol 共用一个主峰中心，使用 `--profile-alignment noncoherent`。

Step D: 三点抛物线插值

- 脚本在 zero-padding 局部峰值附近取三个对数幅度点，输出：
  - `peak_offset_bins`
  - `parabolic_delta_zp`
  - `interpolated_peak_offset_bins`

其中 `interpolated_peak_offset_bins` 是相对中心 LoRa bin 的最终亚 bin 峰值偏移。

## 输出文件

`subbin_spectrum_long.csv`

- 长表。每一行是一个 packet / preamble symbol / q / 亚 bin 采样点。
- 主要字段：
  - `subbin_offset`: 相对中心 bin 的亚 bin 偏移，单位是 original bin。
  - `mag_db_rel_peak`: 相对局部峰值的幅度 dB。
  - `phase_rad_rel_center`: 相对中心点旋转后的相位。
  - `real_norm`, `imag_norm`: 归一化后的复数局部频谱。

`symbol_peak_summary.csv`

- 每个 packet / symbol / q 一行。
- 主要字段：
  - `interpolated_peak_offset_bins`: 三点插值后的峰值亚 bin 偏移。
  - `local_peak_width_3db_bins`: 零填充后估计的 3 dB 宽度，单位 original bin。
  - `side_power_fraction`: 排除中心 `+-0.5` bin 后的侧向功率占比。
  - `asymmetry`: 右侧功率与左侧功率的不对称度。
  - `secondary_peak_rel_db`: 排除主峰附近后的次峰相对 dB。

`packet_q_summary.csv`

- 每个 packet / q 一行，对多个 preamble symbols 做均值和标准差汇总。
- 适合后续和位置、RSSI、已有多径指标做相关性分析。

`point_q_summary.csv`

- 每个点位 / q 一行，由 `packet_q_summary.csv` 聚合得到。
- 分组键为 `experiment_id, corridor_id, position_id, SF, tx_power, preamble_len, q`。
- `packet_count` 是该点位实际聚合到的 packet 数。
- 后缀 `_point_mean/_point_std/_point_median/_point_min/_point_max` 表示跨 packet 的点位级统计。

`plots/*.png`

- 默认每个文件前 2 个 packet 会画一张 q 值对比曲线。

`run_config.json`

- 记录本次运行的参数、输入文件和说明。

## 处理完整数据集

如果要和之前已经提取过的 LoRa 频域 packet 口径保持一致，推荐直接复用旧结果里的 `sample_start`：

```bash
python fingerprint_localization/experiments/zero_padding_fft/zero_padding_lora_fft.py \
  --input INFOCOM_origin_data/LoRaRSSI+IQ_data \
  --packet-starts-csv v2_output/20260623_from_raw/model_results/v2_usrp_packet_features.csv \
  --max-files 0 \
  --q-values 4 \
  --skip-preamble-symbols 0 \
  --feature-symbols 16 \
  --plot-packets-per-file 0 \
  --output-dir v2_output/20260624_zero_padding_fft_q4_from_trusted_starts
```

这会跳过重新检测，直接对 CSV 中已有的有效 packet 起点做零填充 FFT，并输出点位级 `point_q_summary.csv`。

如果要从原始 IQ 重新检测全部候选，可以运行：

```bash
python fingerprint_localization/experiments/zero_padding_fft/zero_padding_lora_fft.py \
  --input INFOCOM_origin_data/LoRaRSSI+IQ_data \
  --max-files 0 \
  --max-packets-per-file 0 \
  --q-values 4 \
  --skip-preamble-symbols 0 \
  --feature-symbols 16 \
  --output-dir v2_output/20260624_zero_padding_fft_full
```

其中 `--max-packets-per-file 0` 表示不限制每个文件的 packet 数。

默认启用 `--periodic-extract`，先检测一个 seed packet，再按 `--packet-period-seconds 5.0` 附近搜索后续 packet。若数据不是固定 5 秒一包，可以关闭：

```bash
python fingerprint_localization/experiments/zero_padding_fft/zero_padding_lora_fft.py --no-periodic-extract
```

## 解释注意

零填充 FFT 会把有限长 dechirped symbol 的 DTFT 采样得更密，因此能更清楚地看主峰附近的偏移、不对称和肩部变化；但它不会突破 LoRa 带宽和符号时长决定的真实物理分辨率。若看到稳定的亚 bin 形状差异，它更适合作为多径相关的形状指纹或弱证据；要严格分离路径时延，仍需要更宽带的信道探测或匹配滤波链路。
