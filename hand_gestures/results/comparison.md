| metric | raw EMG | filtered EMG |
|---|---|---|
| validation accuracy (ensemble) | 0.649 | 0.651 |
| test accuracy, mean of the single networks | 0.656 | 0.600 |
| test accuracy (ensemble) | 0.674 | 0.624 |
| test macro precision (ensemble) | 0.700 | 0.668 |
| test macro recall (ensemble) | 0.674 | 0.624 |
| test macro F1 (ensemble) | 0.674 | 0.632 |
| test accuracy, probabilities averaged over the last 3 windows | 0.687 | 0.627 |
| test accuracy, majority vote over a whole 6 s segment | 0.693 | 0.643 |
| test accuracy, merged classes (finger pair, rotation pair) | 0.727 | 0.683 |
| CALIBRATION: test accuracy with per-participant AdaBN | 0.851 | 0.845 |
| CALIBRATION: test macro F1 with per-participant AdaBN | 0.849 | 0.846 |
| test accuracy (logistic regression baseline) | 0.638 | 0.598 |
| test macro F1 (logistic regression baseline) | 0.651 | 0.619 |
| test accuracy (random forest baseline) | 0.598 | 0.598 |
| test macro F1 (random forest baseline) | 0.606 | 0.606 |

ensemble of 5; checkpoint epochs: raw [6, 3, 9, 3, 14], filtered [15, 5, 9, 8, 10]
test participants: [2, 3, 9, 14, 34, 39]

most confused pairs (raw): adduction -> neutral (123); pronation -> neutral (86); extension -> abduction (53); ulnar_dev -> abduction (49)
most confused pairs (filtered): pronation -> neutral (101); adduction -> neutral (92); neutral -> pronation (60); grip -> neutral (59)