# Fingerprint Localization 数据说明

INFOCOM 2027 当前主线、数据下载和接续实验说明见 `HANDOFF.md`。

本目录用于保存 LoRa 指纹定位实验的离线特征数据。当前数据位于
`data/packet_features_analysis/`，每个 CSV 文件是一组已经整理过的
`packet_features.csv` 结果，每一行对应一个被成功检测并完成特征提取的 LoRa 数据包。

## 数据来源

这些 CSV 的原始特征由
`../gr-lora_sdr/examples/lora_file_preamble_fft.py` 生成。该脚本读取 USRP 采集得到的
fc32/cfile 复数 IQ 文件，使用 `gr-lora_sdr` 接收链检测 LoRa 包，并输出每包级别的
前导码 FFT 特征。

脚本的主要处理流程如下：

1. 从 IQ 文件名解析实验元数据。脚本默认支持的文件名格式为
   `experiment_corridor_position_sf_tx_power_preamble.bin`，例如
   `1_0_10_12_10_16.bin`。
2. 通过 GNU Radio 接收链中的 `frame_sync` 检测包同步位置，得到对齐后的
   preamble/sync word/SFD 样本范围；通过 header 解码结果估计 payload 长度、编码率、
   CRC 标志等信息。
3. 使用 LoRa airtime 公式估计整包的样本范围，再从原始 IQ 中计算包平均功率。
4. 对每个前导码 upchirp 符号切片，下采样到 `2**SF` 点，乘以理想 downchirp 完成
   dechirp，然后做 FFT。
5. 将每个前导码符号的 FFT 主峰对齐，提取主峰集中度、3 dB 主瓣宽度，以及主峰左右
   若干 bin 的局部幅度谱。默认保留主峰左右各 8 个 bin。

脚本原始输出通常包括 `packet_features.csv` 和 `preamble_features.npz`。本项目的
`data/packet_features_analysis/*.csv` 是从这些 `packet_features.csv` 结果整合而来，
便于后续用表格工具或 Python 直接做指纹定位分析。

## CSV 字段含义

元数据字段：

- `file_name`：原始 IQ 文件名。
- `lab_name`：实验/数据批次名称，通常来自上一级 lab 文件夹或整理后的 CSV 文件名。
- `experiment_id`、`corridor_id`、`position_id`：从文件名中解析出的实验编号、走廊编号、
  位置编号，可作为定位标签或分组索引。
- `tx_power_dbm`：从文件名解析出的发射功率设置值，单位 dBm。
- `filename_sf`、`filename_tx_power_dbm`、`filename_preamble_len`：直接从文件名解析出的
  SF、发射功率和前导码长度。
- `header_packet_counter`：如果 payload/header 中带有包序号，脚本会写入该字段；否则可为空。

核心特征字段：

- `packet_avg_power_db`：整包 IQ 平均功率，计算口径为
  `10 * log10(mean(abs(iq) ** 2))`。它反映 USRP 复数基带样本尺度下的包强度，
  不是 dBm，也不是 SX1276 的 RSSI。由于 USRP 缺少与 SX1276 接收机一致的前端放大器、
  AGC 和寄存器 RSSI 映射，这里不尝试伪造 SX1276 RSSI；真正的 SX1276 RSSI 只能从
  SX1276 寄存器读取。为了避免和前一篇基于 SX1276 RSSI 的工作混淆，本文档将该字段
  记录为“USRP IQ 平均功率/包强度特征”。另外，它是平均功率，不是对整包样本求和的
  总能量。
- `preamble_peak_to_residual_db`：前导码 dechirp FFT 主峰集中度。对每个前导码符号，
  脚本先计算 FFT 幅度谱，再用主峰 bin 的能量除以除主峰外所有 residual bin 的总能量，
  最后取 `10 * log10(...)`，并在所有前导码符号上取平均。它可以理解为“主峰能量与其余
  频谱残余能量之比”，但不等价于传统接收机 SNR。
- `preamble_peak_width_3db_bins_avg`：前导码主峰的平均 3 dB 宽度，单位是 FFT bin。
  脚本以每个前导码符号的主峰幅度为基准，寻找幅度下降 3 dB 后的左右交点，并通过线性
  插值得到主瓣宽度；然后对所有前导码符号取平均。脚本默认阈值为 `--peak-width-db -3`，
  因此当前列名为 `3db`。
- `preamble_peak_mag_bin_-8` 到 `preamble_peak_mag_bin_+8`：主峰对齐后的局部 FFT
  幅度谱。对每个前导码符号，脚本把该符号的 FFT 主峰对齐到 offset `+0`，取主峰左右
  8 个 bin 的幅度，再在所有前导码符号上取平均。脚本默认使用 `--normalize max`，因此
  这些值通常是按每个符号主峰归一化后的相对幅度；`preamble_peak_mag_bin_+0` 通常为 1。

## 理解校准

当前对这些字段的理解基本正确，需要注意三点：

- `packet_avg_power_db` 更准确地说是 USRP IQ 平均功率或包强度特征，不是 SX1276 RSSI，
  也不是整包总能量。
- `preamble_peak_to_residual_db` 中的“其他峰能量”在脚本里实际是除主峰 bin 外所有 FFT
  bin 的 residual 总能量。
- `preamble_peak_mag_bin_-8` 到 `+8` 默认是归一化后的主峰邻域幅度谱，不是原始绝对幅度。
