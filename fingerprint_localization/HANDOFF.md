# INFOCOM 2027 实验交接

## 当前主线

- 方法：ACO v4，`top_k=5`，`rssi_class_k=3`。
- 协议：先按 source packet 划分；train 增强 1:10；validation/test 不增强。
- validation 只用于选参；选参后合并 train+validation refit。
- 最终 test：`67/74 = 90.54%`，test 与训练 source overlap 为 0。


## 获取数据

数据包使用 Git LFS。安装 Git LFS 后，在仓库根目录执行：

```bash
git lfs pull --include="fingerprint_localization/data/mainline_202607/fingerloc-mainline-data-20260712.tar.gz"
tar -xzf fingerprint_localization/data/mainline_202607/fingerloc-mainline-data-20260712.tar.gz \
  --strip-components=1
python3 -B fingerprint_localization/scripts/verify_handoff.py --hash
```

数据包 SHA-256：

```text
24cafc87f9cc99c24230b37a13d403aa880c8e099cc38bb0b3f4933b9375ac70
```

## 继续实验

主线代码：`experiments/aco_source_safe_1to10/`。

实验：

1. 外部基线
-  KNN/probabilistic fingerprint； 
-  Random Forest / SVM / MLP； 
-  D-Trace RSSI+； 
-  OrchLoc； 
-  MC-LoRa。
2. Solver 对比
这是为了证明 ACO 不是随便选的：
-  exhaustive search small-scale； 
-  greedy segment selection； 
-  weighted voting； 
-  random search； 
-  ACO。
3. 参数敏感性与可扩展性。段数（4/8/16 段实验）、候选池大小 Top-k、一致性阈值、蚂蚁数/迭代数对精度和时延的影响曲线
4. 消融实验。
5. 性能试验、包端到端时延与峰值内存、计算复杂度。

调参只运行 `train_loocv,val`；配置冻结后才运行 `test`。refit 数据中的
73 条 `val` 已进入训练，只能作诊断，不能再作为独立 validation 成绩。

## 目录

- `data/mainline_202607/`：可信输入、特征和 split 清单。
- `model/v3/`：PGAR、ACO v2/v4 等模型实现。
- `experiments/aco_source_safe_1to10/results/frozen/`：最终结果与逐包预测。
- `experiments/zero_padding_fft/results/negative_controls/`：chirp-LoRa 直接匹配负结果。
- `docs/mainline_202607/`：完整时间线、算法和数据清单。

不要使用 `v2_output_wrong`、旧的增强后切分结果或 `_fail` chirp 文件。
