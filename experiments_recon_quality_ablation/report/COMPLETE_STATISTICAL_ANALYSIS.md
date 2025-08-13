# Complete Statistical Analysis - MVBA Ablation Study

## How to Interpret This Report

### Reading the Comparisons (Model1 → Model2)
- **Mean Diff**: Model 1 - Model 2 value
- **Cohen's d**: Standardized effect size (see interpretation guide at bottom)
- **Improvement %**: 
  - **Positive** = Model 2 is better
  - **Negative** = Model 2 is worse 

### Metric-Specific Interpretation

| Metric | Nature | Good Direction | Mean Diff Interpretation | Cohen's d Interpretation |
|--------|--------|----------------|-------------------------|-------------------------|
| **MSE** | Error metric | Lower is better | Positive = Model 2 better<br>Negative = Model 1 better | Positive = Model 2 improves<br>Negative = Model 2 degrades |
| **PSNR** | Quality metric | Higher is better | Positive = Model 1 better<br>Negative = Model 2 better | Negative = Model 2 improves<br>Positive = Model 2 degrades |
| **SSIM** | Structural similarity<br>to original image | Higher is better (0-1) | Positive = Model 1 better<br>Negative = Model 2 better | Negative = Model 2 improves<br>Positive = Model 2 degrades |
| **L1** | Error metric | Lower is better | Positive = Model 2 better<br>Negative = Model 1 better | Positive = Model 2 improves<br>Negative = Model 2 degrades |

### Examples
- **MSE**: `baseline → full` has Mean Diff = +0.84, Cohen's d = +2.37 → Full model reduces error by 71% 
- **PSNR**: `baseline → full` has Mean Diff = -5.68, Cohen's d = -3.82 → Full model increases quality by 1781% 
- **Degradation**: `full_fixed → spatial` has MSE Mean Diff = -0.19, Cohen's d = -1.04 → Spatial is 46% worse

## Summary Statistics

| Variant | MSE (±std) | PSNR (±std) | SSIM (±std) | L1 (±std) |
|---------|------------|-------------|-------------|-----------|
| baseline | 1.1736 ± 0.4620 | -0.32 ± 1.89 | 0.0864 ± 0.0799 | 0.5664 ± 0.1996 |
| spatial_fixed | 1.1758 ± 0.4624 | -0.33 ± 1.89 | 0.0685 ± 0.0791 | 0.5357 ± 0.1912 |
| feature_fixed | 0.5438 ± 0.2994 | 3.31 ± 2.49 | 0.5883 ± 0.3019 | 0.2662 ± 0.1096 |
| full_fixed | 0.4108 ± 0.2421 | 4.61 ± 2.63 | 0.6210 ± 0.3489 | 0.2131 ± 0.0913 |
| spatial | 0.6003 ± 0.3052 | 2.81 ± 2.36 | 0.4844 ± 0.3600 | 0.3118 ± 0.1147 |
| feature | 0.4424 ± 0.2472 | 4.22 ± 2.51 | 0.6582 ± 0.3012 | 0.2219 ± 0.0905 |
| full | 0.3353 ± 0.1791 | 5.36 ± 2.36 | 0.7141 ± 0.2923 | 0.1902 ± 0.0703 |

- **Positive % = Improvement** (second model better than first)
- **Negative % = Degradation** (second model worse than first)
---

## MSE (Mean Squared Error) Comparisons
*MSE: Lower is better* 

*positive % means reduction in error*

| Comparison | Mean Diff | t-statistic | p-value | Cohen's d | Effect Size | Improvement % | 95% CI | Significant |
|------------|-----------|-------------|---------|-----------|-------------|---------------|--------|-------------|
| **Architecture-Only Comparisons (Fixed α=1)** | | | | | | | | |
| spatial_fixed → full_fixed | 0.7650 | 104.10 | < 1e-300 | **2.328** | Huge | 65.06% | [64.5, 65.7] | Yes |
| baseline → full_fixed | 0.7628 | 103.87 | < 1e-300 | **2.323** | Huge | 65.00% | [64.4, 65.6] | Yes |
| spatial_fixed → feature_fixed | 0.6320 | 93.19 | < 1e-300 | **2.084** | Huge | 53.75% | [53.0, 54.4] | Yes |
| baseline → feature_fixed | 0.6298 | 93.13 | < 1e-300 | **2.083** | Huge | 53.66% | [53.0, 54.4] | Yes |
| feature_fixed → full_fixed | 0.1330 | 34.00 | < 1e-199 | **0.760** | Medium | 24.46% | [23.3, 25.5] | Yes |
| baseline → spatial_fixed | -0.0022 | -6.27 | 4.49e-10 | **-0.140** | Negligible | -0.19% | [-0.2, -0.1] | Yes |
| **BBRE Enhancement Effects** | | | | | | | | |
| spatial_fixed → spatial | 0.5755 | 89.24 | < 1e-300 | **1.996** | Very Large | 48.95% | [48.2, 49.6] | Yes |
| feature_fixed → feature | 0.1014 | 30.94 | < 1e-171 | **0.692** | Medium | 18.65% | [17.7, 19.6] | Yes |
| full_fixed → full | 0.0755 | 22.07 | 8.15e-97 | **0.494** | Small | 18.38% | [17.0, 19.8] | Yes |
| **Enhanced Model Comparisons (α∈[1,3])** | | | | | | | | |
| baseline → full | 0.8383 | 105.80 | < 1e-300 | **2.366** | Huge | 71.43% | [71.0, 71.9] | Yes |
| baseline → feature | 0.7312 | 99.58 | < 1e-300 | **2.227** | Huge | 62.30% | [61.7, 62.9] | Yes |
| baseline → spatial | 0.5733 | 89.19 | < 1e-300 | **1.995** | Very Large | 48.85% | [48.1, 49.5] | Yes |
| spatial → full | 0.2650 | 60.39 | < 1e-300 | **1.351** | Very Large | 44.15% | [43.3, 44.9] | Yes |
| spatial → feature | 0.1579 | 43.66 | < 1e-292 | **0.976** | Large | 26.30% | [25.3, 27.3] | Yes |
| feature → full | 0.1071 | 36.27 | < 1e-221 | **0.811** | Large | 24.21% | [23.2, 25.2] | Yes |
| **Cross-Type Comparisons** | | | | | | | | |
| spatial_fixed → full | 0.8405 | 105.88 | < 1e-300 | **2.368** | Huge | 71.48% | [71.0, 72.0] | Yes |
| spatial_fixed → feature | 0.7334 | 99.69 | < 1e-300 | **2.230** | Huge | 62.37% | [61.7, 63.0] | Yes |
| feature_fixed → full | 0.2085 | 50.84 | < 1e-300 | **1.137** | Large | 38.34% | [37.5, 39.2] | Yes |
| full_fixed → feature | -0.0316 | -9.37 | 1.93e-20 | **-0.210** | Small | -7.69% | [-9.5, -6.0] | Yes |
| feature_fixed → spatial | -0.0565 | -15.83 | 3.01e-53 | **-0.354** | Small | -10.39% | [-11.7, -9.0] | Yes |
| full_fixed → spatial | -0.1895 | -46.28 | < 1e-300 | **-1.035** | Large | -46.13% | [-48.3, -43.9] | Yes |

---

## PSNR (Peak Signal-to-Noise Ratio) Comparisons
*Higher PSNR is better* 

*positive % means increase in quality*

| Comparison | Mean Diff | t-statistic | p-value | Cohen's d | Effect Size | Improvement % | 95% CI | Significant |
|------------|-----------|-------------|---------|-----------|-------------|---------------|--------|-------------|
| **Architecture-Only Comparisons (Fixed α=1)** | | | | | | | | |
| baseline → full_fixed | -4.9327 | -131.51 | < 1e-300 | **-2.941** | Huge | 1546.23% | [1212.8, 2088.9] | Yes |
| spatial_fixed → full_fixed | -4.9408 | -131.79 | < 1e-300 | **-2.948** | Huge | 1510.62% | [1202.4, 2021.1] | Yes |
| baseline → feature_fixed | -3.6340 | -106.08 | < 1e-300 | **-2.373** | Huge | 1139.13% | [904.5, 1518.0] | Yes |
| spatial_fixed → feature_fixed | -3.6421 | -106.06 | < 1e-300 | **-2.372** | Huge | 1113.54% | [892.5, 1484.8] | Yes |
| feature_fixed → full_fixed | -1.2987 | -42.48 | < 1e-280 | **-0.950** | Large | 39.18% | [36.8, 41.6] | Yes |
| baseline → spatial_fixed | 0.0081 | 7.08 | 1.94e-12 | **0.158** | Negligible | -2.52% | [-3.8, -1.7] | Yes |
| **BBRE Enhancement Effects** | | | | | | | | |
| spatial_fixed → spatial | -3.1375 | -101.19 | < 1e-300 | **-2.263** | Huge | 959.29% | [763.1, 1302.2] | Yes |
| feature_fixed → feature | -0.9093 | -39.09 | < 1e-248 | **-0.874** | Large | 27.43% | [25.6, 29.3] | Yes |
| full_fixed → full | -0.7483 | -23.93 | < 1e-110 | **-0.535** | Medium | 16.22% | [14.7, 17.7] | Yes |
| **Enhanced Model Comparisons (α∈[1,3])** | | | | | | | | |
| baseline → full | -5.6811 | -170.64 | < 1e-300 | **-3.817** | Huge | 1780.81% | [1399.2, 2462.3] | Yes |
| baseline → feature | -4.5433 | -127.29 | < 1e-300 | **-2.847** | Huge | 1424.17% | [1128.6, 1924.8] | Yes |
| baseline → spatial | -3.1295 | -101.19 | < 1e-300 | **-2.263** | Huge | 980.98% | [780.8, 1358.8] | Yes |
| spatial → full | -2.5516 | -87.91 | < 1e-300 | **-1.966** | Very Large | 90.79% | [86.8, 95.5] | Yes |
| spatial → feature | -1.4138 | -55.00 | < 1e-300 | **-1.230** | Very Large | 50.31% | [47.7, 53.1] | Yes |
| feature → full | -1.1377 | -47.66 | < 1e-300 | **-1.066** | Large | 26.93% | [25.5, 28.6] | Yes |
| **Cross-Type Comparisons** | | | | | | | | |
| spatial_fixed → full | -5.6891 | -170.63 | < 1e-300 | **-3.816** | Huge | 1739.42% | [1385.1, 2297.4] | Yes |
| spatial_fixed → feature | -4.5514 | -127.35 | < 1e-300 | **-2.848** | Huge | 1391.56% | [1120.5, 1881.3] | Yes |
| feature_fixed → full | -2.0471 | -76.63 | < 1e-300 | **-1.714** | Very Large | 61.75% | [58.9, 64.7] | Yes |
| full_fixed → feature | 0.3894 | 12.67 | 1.90e-35 | **0.283** | Small | -8.44% | [-9.7, -7.1] | Yes |
| feature_fixed → spatial | 0.5045 | 20.23 | 6.36e-83 | **0.452** | Small | -15.22% | [-16.6, -13.7] | Yes |
| full_fixed → spatial | 1.8032 | 55.67 | < 1e-300 | **1.245** | Very Large | -39.08% | [-40.3, -37.9] | Yes |

---

## SSIM (Structural Similarity Index) Comparisons
*Higher SSIM is better*

*positive % means increase in similarity*

| Comparison | Mean Diff | t-statistic | p-value | Cohen's d | Effect Size | Improvement % | 95% CI | Significant |
|------------|-----------|-------------|---------|-----------|-------------|---------------|--------|-------------|
| **Architecture-Only Comparisons (Fixed α=1)** | | | | | | | | |
| spatial_fixed → full_fixed | -0.5525 | -73.64 | < 1e-300 | **-1.647** | Very Large | 806.33% | [766.8, 851.4] | Yes |
| spatial_fixed → feature_fixed | -0.5198 | -79.13 | < 1e-300 | **-1.770** | Very Large | 758.52% | [716.0, 806.0] | Yes |
| baseline → full_fixed | -0.5346 | -69.32 | < 1e-300 | **-1.550** | Very Large | 618.60% | [588.6, 649.9] | Yes |
| baseline → feature_fixed | -0.5019 | -77.31 | < 1e-300 | **-1.729** | Very Large | 580.69% | [552.6, 608.9] | Yes |
| feature_fixed → full_fixed | -0.0328 | -4.02 | 6.04e-05 | **-0.090** | Negligible | 5.57% | [2.9, 8.4] | Yes |
| baseline → spatial_fixed | 0.0179 | 12.30 | 1.39e-33 | **0.275** | Small | -20.71% | [-23.7, -17.7] | Yes |
| **BBRE Enhancement Effects** | | | | | | | | |
| spatial_fixed → spatial | -0.4159 | -54.06 | < 1e-300 | **-1.209** | Very Large | 606.91% | [572.8, 641.9] | Yes |
| full_fixed → full | -0.0931 | -11.91 | 1.19e-31 | **-0.266** | Small | 14.99% | [12.4, 17.6] | Yes |
| feature_fixed → feature | -0.0699 | -9.82 | 2.94e-22 | **-0.220** | Small | 11.88% | [9.4, 14.1] | Yes |
| **Enhanced Model Comparisons (α∈[1,3])** | | | | | | | | |
| baseline → full | -0.6277 | -98.34 | < 1e-300 | **-2.200** | Huge | 726.29% | [693.7, 759.9] | Yes |
| baseline → feature | -0.5717 | -88.45 | < 1e-300 | **-1.978** | Very Large | 661.56% | [632.7, 692.8] | Yes |
| baseline → spatial | -0.3980 | -51.10 | < 1e-300 | **-1.143** | Large | 460.48% | [437.1, 486.4] | Yes |
| spatial → full | -0.2297 | -27.33 | < 1e-139 | **-0.611** | Medium | 47.42% | [43.1, 52.7] | Yes |
| spatial → feature | -0.1738 | -21.33 | 3.82e-91 | **-0.477** | Small | 35.88% | [32.1, 40.0] | Yes |
| feature → full | -0.0559 | -8.24 | 3.11e-16 | **-0.184** | Negligible | 8.50% | [6.4, 10.6] | Yes |
| **Cross-Type Comparisons** | | | | | | | | |
| spatial_fixed → full | -0.6456 | -101.59 | < 1e-300 | **-2.272** | Huge | 942.15% | [895.0, 996.1] | Yes |
| spatial_fixed → feature | -0.5896 | -90.57 | < 1e-300 | **-2.026** | Huge | 860.52% | [814.8, 909.2] | Yes |
| feature_fixed → full | -0.1258 | -17.86 | 2.57e-66 | **-0.399** | Small | 21.39% | [18.8, 23.9] | Yes |
| full_fixed → feature | -0.0371 | -4.73 | 2.42e-06 | **-0.106** | Negligible | 5.98% | [3.6, 8.4] | Yes |
| feature_fixed → spatial | 0.1039 | 12.65 | 2.47e-35 | **0.283** | Small | -17.66% | [-20.3, -15.1] | Yes |
| full_fixed → spatial | 0.1366 | 16.16 | 2.52e-55 | **0.362** | Small | -22.00% | [-24.3, -19.4] | Yes |

---

## L1 Loss (Mean Absolute Error) Comparisons
*Lower L1 is better*

*positive % means reduction in error*

| Comparison | Mean Diff | t-statistic | p-value | Cohen's d | Effect Size | Improvement % | 95% CI | Significant |
|------------|-----------|-------------|---------|-----------|-------------|---------------|--------|-------------|
| **Architecture-Only Comparisons (Fixed α=1)** | | | | | | | | |
| baseline → full_fixed | 0.3533 | 119.50 | < 1e-300 | **2.673** | Huge | 62.37% | [62.0, 62.7] | Yes |
| spatial_fixed → full_fixed | 0.3226 | 116.07 | < 1e-300 | **2.596** | Huge | 60.22% | [59.8, 60.6] | Yes |
| baseline → feature_fixed | 0.3001 | 109.63 | < 1e-300 | **2.452** | Huge | 52.99% | [52.5, 53.5] | Yes |
| spatial_fixed → feature_fixed | 0.2695 | 104.29 | < 1e-300 | **2.333** | Huge | 50.30% | [49.8, 50.8] | Yes |
| feature_fixed → full_fixed | 0.0531 | 47.50 | < 1e-300 | **1.062** | Large | 19.95% | [19.3, 20.7] | Yes |
| baseline → spatial_fixed | 0.0307 | 50.87 | < 1e-300 | **1.138** | Large | 5.41% | [5.2, 5.6] | Yes |
| **BBRE Enhancement Effects** | | | | | | | | |
| spatial_fixed → spatial | 0.2239 | 90.71 | < 1e-300 | **2.029** | Huge | 41.80% | [41.3, 42.3] | Yes |
| feature_fixed → feature | 0.0444 | 46.72 | < 1e-300 | **1.045** | Large | 16.66% | [16.1, 17.2] | Yes |
| full_fixed → full | 0.0229 | 22.10 | 4.84e-97 | **0.494** | Small | 10.75% | [10.0, 11.6] | Yes |
| **Enhanced Model Comparisons (α∈[1,3])** | | | | | | | | |
| baseline → full | 0.3762 | 114.44 | < 1e-300 | **2.560** | Huge | 66.42% | [66.1, 66.7] | Yes |
| baseline → feature | 0.3445 | 114.80 | < 1e-300 | **2.568** | Huge | 60.83% | [60.4, 61.2] | Yes |
| baseline → spatial | 0.2546 | 97.68 | < 1e-300 | **2.185** | Huge | 44.95% | [44.5, 45.4] | Yes |
| spatial → full | 0.1216 | 84.81 | < 1e-300 | **1.897** | Very Large | 39.00% | [38.5, 39.5] | Yes |
| spatial → feature | 0.0899 | 79.22 | < 1e-300 | **1.772** | Very Large | 28.84% | [28.3, 29.4] | Yes |
| feature → full | 0.0317 | 37.38 | < 1e-231 | **0.836** | Large | 14.28% | [13.7, 14.9] | Yes |
| **Cross-Type Comparisons** | | | | | | | | |
| spatial_fixed → full | 0.3455 | 110.85 | < 1e-300 | **2.479** | Huge | 64.50% | [64.2, 64.8] | Yes |
| spatial_fixed → feature | 0.3138 | 110.54 | < 1e-300 | **2.472** | Huge | 58.58% | [58.2, 59.0] | Yes |
| feature_fixed → full | 0.0760 | 60.11 | < 1e-300 | **1.344** | Very Large | 28.56% | [27.9, 29.2] | Yes |
| full_fixed → feature | -0.0088 | -9.30 | 3.72e-20 | **-0.208** | Small | -4.11% | [-5.0, -3.3] | Yes |
| feature_fixed → spatial | -0.0455 | -42.58 | < 1e-281 | **-0.952** | Large | -17.11% | [-18.0, -16.2] | Yes |
| full_fixed → spatial | -0.0987 | -78.46 | < 1e-300 | **-1.755** | Very Large | -46.29% | [-47.5, -44.9] | Yes |

---

## Effect Size Interpretation (Cohen's d)

| Range | Interpretation | Practical Meaning |
|-------|----------------|-------------------|
| < 0.2 | Negligible | No practical difference |
| 0.2-0.5 | Small | Subtle but potentially meaningful |
| 0.5-0.8 | Medium | Clear practical improvement |
| 0.8-1.2 | Large | Substantial improvement |
| 1.2-2.0 | Very Large | Major improvement |
| > 2.0 | Huge | Dramatic transformation |