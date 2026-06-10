# fingerprint_localization

本仓库当前重点是 `dong` 目录里的 CHIME-style LFM chirp sounding 实验链路：TX 用 USRP B210 发重复 chirp，RX 同步采集 IQ 并保存到本地 `.bin` 文件，后续用频谱图或匹配滤波做多径/路径分析。

## 当前实验配置

当前 `dong/TX` 和 `dong/RX` 已切到 20 MS/s 宽带实验配置：

| 项目 | 当前值 |
| --- | --- |
| 中心频率 | `487.7 MHz` |
| TX sample rate | `20 MS/s` |
| RX sample rate | `20 MS/s` |
| TX/RX RF bandwidth | `20 MHz` |
| TX chirp bandwidth | 约 `18 MHz`，不是严格 20 MHz |
| Chirp duration | `1 ms` |
| Chirp period | `20 ms` |
| TX waveform amplitude | `0.2` |
| TX 时长 | `50 s` |
| TX gain | `100 dB` |
| RX 采集时长 | `10 s` |
| RX 输出格式 | GNU Radio `complex64/fc32` |

注意：`20 MHz RF bandwidth` 和 `18 MHz chirp bandwidth` 是两个概念。现在的 TX `.bin` 是 20 MS/s 采样率下的 18 MHz LFM chirp，扫频大约从 `-8.91 MHz` 到 `+8.91 MHz`。这样是为了避开 `fs=20 MS/s` 时的 Nyquist 边界。

## 关键文件

- `dong/TX/chime_wideband_make_tx_waveform.py`  
  生成一个 20 ms 周期的 fc32 TX 波形文件。

- `dong/TX/chime_wideband_tx.grc`  
  TX 的 GNU Radio Companion 源文件。

- `dong/TX/chime_wideband_tx.py`  
  由 GRC 生成的 TX Python。当前是 no-GUI，链路为 `Vector Source -> Head -> USRP Sink`。TX 波形在启动时从 `.bin` 一次性读入内存，然后从内存循环输出，避免运行时 File Source 卡顿。优先改 `.grc`，再重新生成 `.py`。

- `dong/RX/chime_wideband_rx.grc`  
  RX 的 GNU Radio Companion 源文件。

- `dong/RX/chime_wideband_rx.py`  
  由 GRC 生成的 RX Python。当前是 no-GUI，链路为 `USRP Source -> Head -> File Sink`。

- `dong/RX/chime_wideband_spectrogram.py`  
  读取 RX `.bin` 并渲染频谱图。

- `dong/RX/chime_wideband_analyze.py`  
  匹配滤波/相关分析脚本。

## 目录和大文件

当前使用仓库内路径：

```text
dong/inputs/chime_test_tx_period_fc32.bin
dong/outputs/captures/chime_test_rx_fc32.bin
dong/outputs/analysis/
```

`.gitignore` 已忽略：

```text
dong/inputs/*.bin
dong/outputs/
```

这些文件通常很大，不要提交到 Git。20 MS/s、10 秒、fc32 的 RX 采集文件大小约：

```text
20e6 samples/s * 10 s * 8 bytes = 1.6 GB
```

## 生成 TX 波形

生成当前 20 MS/s / 18 MHz chirp 的 TX `.bin`：

```powershell
cd d:\Desktop\fingerprint_localization
& 'D:\mysoft2\radioconda\python.exe' dong/TX/chime_wideband_make_tx_waveform.py --fs 20e6 --chirp-bw 18e6 --chirp-duration 1e-3 --period 20e-3 --amplitude 0.2
```

预期输出文件：

```text
dong/inputs/chime_test_tx_period_fc32.bin
```

当前文件应接近：

```text
大小: 3,200,000 bytes
样本数: 400,000 complex64
按 20 MS/s 算: 20 ms
非零 chirp: 20,000 samples = 1 ms
幅度: 0.2
扫频跨度: 约 18 MHz
```

生成脚本自己的默认值仍是旧的低速测试参数 `fs=2e6`、`chirp_bw=1e6`。做当前实验时必须显式传 `--fs 20e6 --chirp-bw 18e6`。

## 运行 TX/RX

当前 TX 和 RX 都是 no-GUI，正式采集时建议用终端运行。

先启动 RX：

```powershell
cd d:\Desktop\fingerprint_localization
& 'D:\mysoft2\radioconda\python.exe' dong/RX/chime_wideband_rx.py
```

再启动 TX：

```powershell
cd d:\Desktop\fingerprint_localization
& 'D:\mysoft2\radioconda\python.exe' dong/TX/chime_wideband_tx.py
```

当前设备分配：

```text
TX: serial=2512552,num_send_frames=1024
RX: serial=2603160,num_recv_frames=512
```

如果两台 USRP 对调，修改 `.grc` 里的 `tx_device_addr` / `rx_device_addr`，然后重新生成 Python。

## 重新生成 GRC Python

`.grc` 是源文件，`.py` 是生成产物。改结构或参数时，优先改 `.grc`，然后运行：

```powershell
cd d:\Desktop\fingerprint_localization
& 'D:\mysoft2\radioconda\Scripts\grcc.exe' -o dong/TX dong/TX/chime_wideband_tx.grc
& 'D:\mysoft2\radioconda\Scripts\grcc.exe' -o dong/RX dong/RX/chime_wideband_rx.grc
```

语法检查：

```powershell
& 'D:\mysoft2\radioconda\python.exe' -m py_compile dong/TX/chime_wideband_tx.py dong/RX/chime_wideband_rx.py
```

## 分析和画图

频谱图脚本默认参数仍是旧的 `2 MS/s / 1 MHz`，当前 20M 实验必须显式传参：

```powershell
cd d:\Desktop\fingerprint_localization
& 'D:\mysoft2\radioconda\python.exe' dong/RX/chime_wideband_spectrogram.py --fs 20e6 --chirp-bw 18e6 --chirp-duration 1e-3 --period 20e-3 --duration 0.12
```

匹配滤波/多径分析同样要显式传当前参数：

```powershell
cd d:\Desktop\fingerprint_localization
& 'D:\mysoft2\radioconda\python.exe' dong/RX/chime_wideband_analyze.py --fs 20e6 --chirp-bw 18e6 --chirp-duration 1e-3 --period 20e-3
```

如果分析脚本用 `chirp_bw=20e6`，但实际 TX 是 `18e6`，相关峰会变宽、峰值会降低，时延/多径判断可能偏。最稳的做法是后续分析直接读取实际 TX `.bin` 作为 matched filter 模板，而不是重新假设一个 chirp。

## O 和 U 是什么

UHD/GNU Radio 里常见的两个实时流告警：

```text
O = RX overflow
U = TX underflow
```

### 本机实测结论

2026-06-10 在本机 20 MS/s 配置下，关闭 TX/RX GUI 后仍有偶发 `O` / `U`。加入 UHD 主机侧帧缓冲参数后，`O/U` 明显改善，一度测试到 0：

```text
TX: serial=2512552,num_send_frames=1024
RX: serial=2603160,num_recv_frames=512
```

因此当前这台机器上的首要结论是：主要瓶颈不是 USB3 平均带宽、CPU 频率或 USRP 硬件本身，而是 20 MS/s 实时流对主机调度/USB/写盘瞬时抖动比较敏感。`num_send_frames` 和 `num_recv_frames` 增大了 UHD 可用的主机侧缓冲，能吸收短时间卡顿，是当前最有效的修复项之一。

若 TX 侧后续仍偶发 `U`，当前已进一步把 TX 从运行时 `File Source` 改成启动时加载 `.bin` 的内存 `Vector Source`，减少磁盘/文件源参与实时发射路径。

如果在另一台电脑复现实验时出现少量 `O/U`，优先先检查这两个参数是否已经加上。若 `512` 启动异常，再降到 `128` 测试。

### RX overflow: `O`

含义：USRP 已经收到样本，但电脑没有及时从 USRP 取走，接收缓冲区满了。结果是 RX IQ 时间轴中间丢了一小段样本。

常见原因：

- RX 采样率高，当前是 `20 MS/s`
- RX 写盘瞬时卡顿
- USB 控制器或 Hub 调度抖动
- Windows 线程调度抖动
- GUI 全速画图拖慢处理

当前 RX 已关 GUI，并加入：

```text
num_recv_frames=512
```

本机实测加上该参数后 RX 侧偶发 `O` 消失。

处理建议：

1. 先只跑 RX。偶发 1 个 `O` 对流程测试问题不大，但正式数据最好重采到 0 个 `O`。
2. 确认 USRP 日志显示 `Operating over USB 3.`
3. 保存路径放本机 SSD，不要放网盘、移动盘或同步目录。
4. 关闭 RX GUI，当前已经关闭。
5. 如果大量连续 `O`，降低采样率到 `16 MS/s` 或 `10 MS/s` 做对照。
6. 如果两台 USRP 同机运行，尽量插到不同 USB 控制器；正式实验建议 TX/RX 分两台电脑。

一次 `O` 不等于只丢 1 个采样点。20 MS/s 下：

```text
0.1 ms 约 2,000 samples
1 ms   约 20,000 samples
10 ms  约 200,000 samples
```

具体丢多少取决于当时主机卡顿多久，UHD 通常不会直接报告精确丢点数。

### TX underflow: `U`

含义：USRP 正在发射，但电脑没有及时把下一批 IQ 样本送过去，TX 缓冲区空了。结果是发射波形中间断了一小段。

常见原因：

- TX 20 MS/s 连续发，主机调度不稳
- 旧版 File Source 循环读文件出现瞬时卡顿
- USB 控制器/Hub 抖动
- TX 和 RX 同机运行时总压力变大

当前 TX 已关 GUI，并加入 UHD 发送缓冲：

```text
num_send_frames=1024
```

同时 TX 源已从 File Source 改为 Vector Source：

```text
Vector Source -> Head -> USRP Sink
```

启动时会一次性读取 `dong/inputs/chime_test_tx_period_fc32.bin` 到内存，再从内存循环输出。这样可以排除运行中循环读文件造成的瞬时卡顿。

本机实测：先加大发送缓冲后 `U` 明显改善；若后续仍偶发 `U`，继续把 TX 改成内存 Vector Source，这是当前链路已采用的方案。当前 TX 使用 `num_send_frames=1024`，该值比 `2048` 更合适，当前测试可用。

处理建议：

1. 先只跑 TX。若 TX-only 仍持续 `U`，优先怀疑 TX 侧供样本不稳。
2. 若 TX-only 没 `U`，TX+RX 同机才出现 `U`，优先怀疑 USB/系统总负载。
3. 正式实验 TX/RX 分两台电脑运行。
4. 如果 `num_send_frames=1024` 启动异常、停止明显变慢或没有收益，可降回 `512` 或 `128`。
5. 如果内存源后仍有持续 `U`，再考虑降低采样率、提高进程优先级、换 USB 口/控制器或分机运行。

## USB 和系统注意事项

实验前建议确认：

```powershell
& 'D:\mysoft2\radioconda\Library\bin\uhd_find_devices.exe'
& 'D:\mysoft2\radioconda\Library\bin\uhd_usrp_probe.exe' --args "serial=2512552"
& 'D:\mysoft2\radioconda\Library\bin\uhd_usrp_probe.exe' --args "serial=2603160"
```

重点看：

```text
Operating over USB 3.
```

如果同一台电脑同时连两台 B210，尽量让它们分到不同 USB 控制器/Root Hub。单个 USB3 的理论带宽不一定是平均吞吐瓶颈，但 SDR 看的是实时连续性，短时间调度抖动也会导致 `O/U`。

Windows 上还可以考虑：

- 使用高性能/野兽模式电源计划
- 关闭 USB 选择性暂停
- 终端里用较高进程优先级运行 TX/RX

高优先级运行 TX 的 PowerShell 命令：

```powershell
cd d:\Desktop\fingerprint_localization
$p = Start-Process -FilePath 'D:\mysoft2\radioconda\python.exe' -ArgumentList 'dong\TX\chime_wideband_tx.py' -WorkingDirectory 'd:\Desktop\fingerprint_localization' -PassThru
$p.PriorityClass = 'High'
Get-Process -Id $p.Id | Select-Object Id,ProcessName,PriorityClass
```

一行版：

```powershell
cd d:\Desktop\fingerprint_localization; $p = Start-Process -FilePath 'D:\mysoft2\radioconda\python.exe' -ArgumentList 'dong\TX\chime_wideband_tx.py' -WorkingDirectory 'd:\Desktop\fingerprint_localization' -PassThru; $p.PriorityClass = 'High'; Get-Process -Id $p.Id | Select-Object Id,ProcessName,PriorityClass
```

RX 也可以同样方式启动，把 `dong\TX\chime_wideband_tx.py` 换成 `dong\RX\chime_wideband_rx.py`。不要使用 `Realtime` 优先级；它可能影响系统、USB 和桌面调度，反而更不稳定。

这些是优化项，不是每次测试都必须做。流程调通阶段，偶发 1 个 `O` 或 `U` 可以先记录；正式数据建议尽量重采到无连续 `O/U`。

## 开发日记

### 2026-06-09

- 把 `dong` 里的 CHIME-style TX/RX 链路整理到仓库内路径：
  - TX 输入：`dong/inputs/chime_test_tx_period_fc32.bin`
  - RX 输出：`dong/outputs/captures/chime_test_rx_fc32.bin`
- 加了 `.gitignore`，忽略 `dong/inputs/*.bin` 和 `dong/outputs/`，避免大体积 IQ 文件进入 Git。
- 生成并验证了当前 20 MS/s TX 波形：
  - `400000` 个 `complex64` 样本
  - 文件大小 `3.2 MB`
  - 一个周期 `20 ms`
  - chirp 时长 `1 ms`
  - 幅度 `0.2`
  - 瞬时频率约从 `-8.91 MHz` 到 `+8.91 MHz`
- 明确了一个重要参数区别：USRP RF bandwidth 是 `20 MHz`，但 TX chirp bandwidth 是约 `18 MHz`。后处理必须按 `chirp_bw=18e6` 分析，不能写死 `20e6`。
- TX/RX GRC 切到 `20 MS/s`、`20 MHz RF bandwidth`，RX 使用 `Head + File Sink` 保存 10 秒 IQ。

### 2026-06-10

- 发现 20 MS/s 下 TX/RX GUI 全速画 `|IQ|` 会增加实时压力，因此 TX/RX 都切成 no-GUI。
- RX 当前正式采集链路：

```text
USRP Source -> Head -> File Sink
```

- TX 当前正式发射链路：

```text
Vector Source -> Head -> USRP Sink
```

- TX 从运行时 `File Source` 改成启动时加载 `.bin` 的内存 `Vector Source`，减少发射过程中磁盘/文件源卡顿导致 `U` 的可能性。
- 排查了 `O` / `U`：
  - `O` 是 RX overflow，说明接收端来不及取走样本。
  - `U` 是 TX underflow，说明发送端来不及喂给 USRP 样本。
  - CPU 频率和 USB3 平均带宽不是主要矛盾，真正敏感的是 Windows/USB/UHD/写盘的瞬时调度抖动。
- 确认两台 B210 都能被 UHD 识别，并显示 `Operating over USB 3.`。
- 加入 UHD host-side frame 缓冲后，`O/U` 明显改善：
  - RX 当前使用 `num_recv_frames=512`
  - TX 当前使用 `num_send_frames=1024`
- 试过更大的 `num_send_frames=2048`，但当前回到 `1024` 后表现更合适；结论是该参数不是越大越好，过大可能带来更高延迟、停止变慢或收益不明显。
- 记录了 Windows 下高优先级运行 TX/RX 的 PowerShell 方法。后续如果 TX 仍偶发 `U`，优先尝试高优先级运行，而不是盲目继续增大 `num_send_frames`。

## 安全注意事项

当前 TX gain：

```text
tx_gain_db = 100
```

不要在较高增益下无衰减直连 TX/RX。正式实验前确认连接方式、天线、衰减器和 RX gain，避免过载 RX 或损坏设备。

## 推荐测试顺序

1. 生成 TX `.bin`。
2. 只跑 RX，确认能保存 `1.6 GB` 左右的文件，观察是否有 `O`。
3. 只跑 TX，观察是否有 `U`。
4. 先开 RX，再开 TX，跑完整 TX/RX。
5. 用显式 `--fs 20e6 --chirp-bw 18e6` 参数画频谱图和做相关分析。
6. 若正式数据里出现连续大量 `O/U`，先不要用于多径结论，按上面的排查项处理后重采。
