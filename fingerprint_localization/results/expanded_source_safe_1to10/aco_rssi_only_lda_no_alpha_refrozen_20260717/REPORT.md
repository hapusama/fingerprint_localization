# RSSI-only LDA feature ablation (no alpha)

Date: 2026-07-17

Controlled change: LDA uses only the six RSSI+ features instead of RSSI+ plus 21 S17/raw features. ACO segment observations and all search parameters are unchanged.

## Validation beta refreeze

| beta | correct | accuracy | changes vs RSSI-only LDA |
| ---: | ---: | ---: | ---: |
| 0.0 | 101/128 | 78.91% | 30 |
| 0.1 | 101/128 | 78.91% | 30 |
| 0.2 | 101/128 | 78.91% | 30 |
| 0.3 | 101/128 | 78.91% | 30 |
| 0.4 | 103/128 | 80.47% | 28 |
| 0.5 **(selected)** | 105/128 | 82.03% | 22 |
| 0.6 | 104/128 | 81.25% | 9 |
| 0.7 | 104/128 | 81.25% | 5 |
| 0.8 | 103/128 | 80.47% | 2 |
| 0.9 | 102/128 | 79.69% | 1 |
| 1.0 | 101/128 | 78.91% | 0 |

Selection rule: maximum validation accuracy, then smaller beta on ties.

## LDA candidate recall

| Split | LDA inputs | Top-1 | Top-3 | Top-5 | cutoff errors |
| --- | --- | ---: | ---: | ---: | ---: |
| validation | RSSI+S17 | 117/128 | 126/128 | 128/128 | 0 |
| validation | RSSI+ only | 101/128 | 127/128 | 127/128 | 1 |
| formal_test | RSSI+S17 | 120/128 | 128/128 | 128/128 | 0 |
| formal_test | RSSI+ only | 112/128 | 128/128 | 128/128 | 0 |

## Frozen-beta full ACO

| Split | LDA inputs | LDA/search/final correct | final accuracy | MAE/P95 m | severe >10 m | correction precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | RSSI+ only | 101/101/105 | 82.03% | 0.918/5.560 | 3.91% | 45.45% |
| formal_test | RSSI+ only | 112/106/115 | 89.84% | 0.420/3.390 | 0.78% | 55.00% |

## Paired change versus RSSI+S17 no-alpha refreeze

| Split | Stage | old/new correct | delta pp | W2R/R2W | changed | McNemar p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | lda | 117/101 | -12.50 | 2/18 | 22 | 0.000402 |
| validation | search | 106/101 | -3.91 | 1/6 | 8 | 0.125000 |
| validation | final | 119/105 | -10.94 | 1/15 | 18 | 0.000519 |
| formal_test | lda | 120/112 | -6.25 | 7/15 | 23 | 0.133801 |
| formal_test | search | 105/106 | +0.78 | 2/1 | 4 | 1.000000 |
| formal_test | final | 120/115 | -3.91 | 5/10 | 15 | 0.301758 |

The formal split is exploratory because it was inspected by earlier experiments.
