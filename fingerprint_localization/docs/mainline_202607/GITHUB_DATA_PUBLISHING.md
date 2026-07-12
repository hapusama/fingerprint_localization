# GitHub 数据与代码发布清单

本仓库实际 Git 根目录是当前 `fingerprint_localization/` 目录，不是其上层同名工作区。
远端 `main` 停在 2026-06-22；6 月 23 日之后的主线目前尚未提交。不要在上层仓库
执行 `git add -A`，也不要一次性提交当前所有未跟踪文件。

## 发布分层

### 1. 直接进入 Git

- 主线 Python/MJS 代码、README 和算法文档。
- 6/23 两个小型预计算输入：RSSI/packet-level 与 LoRa raw-bin。
- 位置表、chirp point-level structure、source split、metadata。
- 最终 summary、metrics 和 test predictions。
- 7/10-7/11 负对照的 compact summary/CSV；图只保留论文实际引用版本。

### 2. GitHub Release、Zenodo 或其他外部数据包

- 6/24 `subbin_spectrum_long.csv`，约 41 MB，是重建 noisy pool 的关键输入。
- 为让搭档立即续跑，可额外发布完整 `group_safe_trainval_refit/data`，约 348 MB。
- 其他 noisy spectrum 长表为可重建产物，分别约 277-430 MB；优先不重复发布，
  或放到同一个版本化数据包，禁止普通 Git 提交。

科研长期公开优先 Zenodo/机构存储并记录 DOI；GitHub Release 适合团队短期交接。
`docs/DATA_MANIFEST.csv` 给出当前文件大小和 SHA-256。

建议数据包结构：

```text
fingerloc-mainline-data-20260712/
  v2_output/20260624_zero_padding_fft_q1_q4_point_compare/subbin_spectrum_long.csv
  fingerprint_localization/experiments/aco_source_safe_1to10/group_safe_trainval_refit/data/
  DATA_MANIFEST.csv
```

### 3. 仅本地/实验室存储

- `INFOCOM_origin_data/`：约 58 GB 原始 LoRa/chirp IQ，GitHub 不适合承载。
- `v2_output_wrong/`：错误分支，仅留本地审计，不公开混入可信数据。
- `*_segment_costs.csv`、`*_candidate_scores.csv`、train LOOCV predictions、缓存和
  inspect ndjson：可重建诊断，不进入源码仓库。
- 原始 SDR capture、checkpoint、`*.pth/*.pt/*.ckpt`。

原始数据若需要共享，单独生成逐文件 SHA-256 manifest，连同采集参数、失败文件标记、
许可和下载地址放入数据仓库。当前 `multipath_data` 中带 `_fail` 的文件必须保留失败标签。

## 提交顺序

建议新建 `codex/mainline-handoff-20260712` 或同类分支，按以下顺序定向提交：

1. `docs`、README、`.gitignore` 和环境说明。
2. `fingerprint_localization/experiments/zero_padding_fft/`、`fingerprint_localization/model/v3-v4/`、
   `fingerprint_localization/experiments/aco_source_safe_1to10/*.py`。
3. 小型可信输入、source split/metadata。
4. 冻结结果 summary、metrics、test predictions 和必要论文图。
5. 发布外部数据包后，把 URL/DOI、版本和 SHA-256 回填到 README。

提交前检查：

```bash
git status --short
git diff --check
find . -type f -size +95M -not -path './.git/*' -print
```

必须定向 `git add <path...>`。不要使用 `git add -A`，因为工作区包含历史修改和大量
本地产物。`.gitignore` 是安全网，不是上传白名单。

## 发布前人工确认

- [ ] 确认采集数据和场地图是否允许公开，补 LICENSE/DATA LICENSE。
- [ ] 移除姓名、设备序列号、绝对本机路径等不必要元数据。
- [ ] 给数据包 URL/DOI 和版本号，不以“最新版”替代固定版本。
- [ ] 用干净 clone 下载数据包并跑一次最小复现。
- [ ] 确认论文表只混合相同协议下的结果，旧泄漏协议明确标为 historical。
- [ ] 确认 final refit 的 diagnostic val 没有被写成独立 validation 成绩。
