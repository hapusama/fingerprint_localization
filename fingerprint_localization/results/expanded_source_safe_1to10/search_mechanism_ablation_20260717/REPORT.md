# Search-mechanism ablation

Date: 2026-07-17

All methods use identical LDA posterior, fused Top-5 candidates, segment costs, priors, and beta=0.5 final fusion.

## Clean

| Method | validation search/final | formal search/final | final MAE/P95 m | mean search ms |
| --- | --- | --- | --- | --- |
| Average cost | 117/118 | 114/121 | 0.29/2.20 | 0.001 |
| Greedy path | 117/116 | 117/121 | 0.31/2.20 | 0.080 |
| No pheromone | 117/117 | 117/121 | 0.29/2.20 | 4.678 |
| Full ACO | 117/117 | 118/120 | 0.37/3.39 | 4.900 |

## Strongest controlled degradation

| Condition | Method | search accuracy | final accuracy | MAE/P95 m | severe >10m | correction precision |
| --- | --- | --- | --- | --- | --- | --- |
| preamble_missing (4.0) | Average cost | 82.03% | 79.69% | 0.89/3.39 | 1.56% | 100.00% |
| preamble_missing (4.0) | Greedy path | 74.22% | 79.69% | 0.89/3.39 | 1.56% | 54.55% |
| preamble_missing (4.0) | No pheromone | 82.81% | 81.25% | 0.76/3.39 | 0.78% | 85.71% |
| preamble_missing (4.0) | Full ACO | 81.25% | 82.81% | 0.76/3.39 | 1.56% | 69.23% |
| amplitude_noise (1.0) | Average cost | 21.09% | 17.19% | 10.87/27.08 | 52.34% | 7.69% |
| amplitude_noise (1.0) | Greedy path | 21.09% | 17.97% | 10.87/27.01 | 52.34% | 10.53% |
| amplitude_noise (1.0) | No pheromone | 21.09% | 17.19% | 11.03/27.01 | 53.91% | 7.69% |
| amplitude_noise (1.0) | Full ACO | 21.09% | 17.97% | 10.71/27.01 | 52.34% | 9.52% |
| cfo_shift (1.0) | Average cost | 81.25% | 79.69% | 1.18/6.74 | 3.12% | 50.00% |
| cfo_shift (1.0) | Greedy path | 79.69% | 78.91% | 1.07/6.67 | 1.56% | 50.00% |
| cfo_shift (1.0) | No pheromone | 81.25% | 79.69% | 1.15/6.74 | 3.12% | 44.44% |
| cfo_shift (1.0) | Full ACO | 81.25% | 79.69% | 1.02/6.67 | 0.78% | 50.00% |
| segment_anomaly (1.0) | Average cost | 76.56% | 84.38% | 0.81/6.67 | 0.78% | 66.67% |
| segment_anomaly (1.0) | Greedy path | 81.25% | 83.59% | 0.81/6.67 | 1.56% | 50.00% |
| segment_anomaly (1.0) | No pheromone | 81.25% | 84.38% | 0.76/6.67 | 0.78% | 55.56% |
| segment_anomaly (1.0) | Full ACO | 79.69% | 82.03% | 0.89/6.67 | 1.56% | 38.46% |

## Full ACO versus ablations across 15 formal scenarios

| Ablation | Full wins/ties/losses | net correct packets |
| --- | --- | --- |
| Average cost | 7/2/6 | +9 |
| Greedy path | 7/2/6 | +4 |
| No pheromone | 4/3/8 | -8 |

## Mechanism check

- Full ACO and no-pheromone best paths had zero segment switches and zero garbage selections in every formal scenario.
- Full ACO versus no pheromone has no final-label McNemar result below 0.05 (minimum p=0.0625).
- Full ACO is strongest at amplitude noise 0.25/0.5 and four missing preamble symbols, but it is weaker under clean, CFO, and segment-anomaly cases.
- Therefore this experiment does not support a global claim that pheromone search is irreplaceable in the current configuration.

## Definitions

- Average cost: rank by negative mean C_obs; no path or transition term.
- Greedy path: one-step local minimum with the frozen dynamic switch penalty; no ants or pheromone.
- No pheromone: 16 ants x 12 iterations and the same heuristic/path cost, but no tau factor, evaporation, reinforcement, or pheromone score.
- Full ACO: frozen ACO v4 transition, evaporation, elite reinforcement, and Score4.
- Search accuracy is the candidate-internal score Top-1 before LDA final fusion.
- Final accuracy uses the same beta=0.5 LDA blend for every method.
- Perturbations are feature-space proxies because raw IQ is unavailable.
