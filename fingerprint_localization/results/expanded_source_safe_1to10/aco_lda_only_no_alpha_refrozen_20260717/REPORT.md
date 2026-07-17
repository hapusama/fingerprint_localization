# LDA/ACO without alpha candidate fusion

Date: 2026-07-17

Candidate policy: direct LDA Top-5. RSSI remains only inside ACO observation costs and weak priors.

## Validation beta freeze

| beta | Full ACO correct | accuracy | changes vs LDA |
| ---: | ---: | ---: | ---: |
| 0.0 | 106/128 | 82.81% | 21 |
| 0.1 | 106/128 | 82.81% | 21 |
| 0.2 | 106/128 | 82.81% | 21 |
| 0.3 | 106/128 | 82.81% | 21 |
| 0.4 | 108/128 | 84.38% | 19 |
| 0.5 | 114/128 | 89.06% | 10 |
| 0.6 **(selected)** | 119/128 | 92.97% | 2 |
| 0.7 | 118/128 | 92.19% | 1 |
| 0.8 | 118/128 | 92.19% | 1 |
| 0.9 | 117/128 | 91.41% | 0 |
| 1.0 | 117/128 | 91.41% | 0 |

Selection rule: maximize validation accuracy; choose the smaller beta on ties.

## Frozen-beta results

| Split | Method | search/final correct | final accuracy | MAE/P95 m | severe >10m | correction precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | Average cost | 105/119 | 92.97% | 0.472/3.390 | 3.12% | 100.00% |
| validation | Greedy path | 105/119 | 92.97% | 0.472/3.390 | 3.12% | 100.00% |
| validation | No pheromone | 105/119 | 92.97% | 0.472/3.390 | 3.12% | 100.00% |
| validation | Full ACO | 106/119 | 92.97% | 0.472/3.390 | 3.12% | 100.00% |
| formal_test | Average cost | 104/120 | 93.75% | 0.315/3.390 | 0.78% | n/a |
| formal_test | Greedy path | 104/120 | 93.75% | 0.315/3.390 | 0.78% | 50.00% |
| formal_test | No pheromone | 103/121 | 94.53% | 0.289/2.203 | 0.78% | 100.00% |
| formal_test | Full ACO | 105/120 | 93.75% | 0.315/3.390 | 0.78% | 50.00% |

## Full ACO comparison with old alpha=0.3 mainline

| Split | old/new correct | changed | W2R/R2W | net | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 117/119 | 5 | 3/1 | +2 | 0.625000 |
| formal_test | 120/120 | 4 | 2/2 | +0 | 1.000000 |

The formal split remains exploratory because it was inspected by prior experiments.
