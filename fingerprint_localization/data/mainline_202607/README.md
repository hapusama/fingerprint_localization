# Mainline data

Git 中保留小型可信输入、特征、split 和 SHA-256 清单。大体积派生 CSV 打包为
`fingerloc-mainline-data-20260712.tar.gz`，通过 Git LFS 存储。

从仓库根目录解压：

```bash
git lfs pull --include="fingerprint_localization/data/mainline_202607/fingerloc-mainline-data-20260712.tar.gz"
tar -xzf fingerprint_localization/data/mainline_202607/fingerloc-mainline-data-20260712.tar.gz \
  --strip-components=1
```

SHA-256: `24cafc87f9cc99c24230b37a13d403aa880c8e099cc38bb0b3f4933b9375ac70`.

原始 58 GB IQ 不在 GitHub 中。完整清单见
`../../docs/mainline_202607/DATA_MANIFEST.csv`。
