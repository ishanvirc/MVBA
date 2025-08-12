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
| **MSE** | Error metric | Lower is better | Positive = Model 1 worse<br>Negative = Model 1 better | Positive = Model 2 improves<br>Negative = Model 2 degrades |

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

---

## MSE (Mean Squared Error) Comparisons
*MSE: Lower is better* 

*positive % means reduction in error*

| Comparison | Mean Diff | t-statistic | p-value | Cohen's d | Effect Size | Improvement % | 95% CI | Significant |
|------------|-----------|-------------|---------|-----------|-------------|---------------|--------|-------------|
| **Architecture-Only Comparisons (Fixed α=1)** | | | | | | | | |
| spatial_fixed → full_fixed | 0.7650 | 104.10 | < 1e-300 | **2.328** | Huge | 65.06% | [64.5, 65.7] | Yes ✓ |
| baseline → full_fixed | 0.7628 | 103.87 | < 1e-300 | **2.323** | Huge | 65.00% | [64.4, 65.6] | Yes ✓ |
| spatial_fixed → feature_fixed | 0.6320 | 93.19 | < 1e-300 | **2.084** | Huge | 53.75% | [53.0, 54.4] | Yes ✓ |
| baseline → feature_fixed | 0.6298 | 93.13 | < 1e-300 | **2.083** | Huge | 53.66% | [53.0, 54.4] | Yes ✓ |
| feature_fixed → full_fixed | 0.1330 | 34.00 | < 1e-199 | **0.760** | Medium | 24.46% | [23.3, 25.5] | Yes ✓ |
| baseline → spatial_fixed | -0.0022 | -6.27 | 4.49e-10 | **-0.140** | Negligible | -0.19% | [-0.2, -0.1] | Yes ✓ |
| **BBRE Enhancement Effects** | | | | | | | | |
| spatial_fixed → spatial | 0.5755 | 89.24 | < 1e-300 | **1.996** | Very Large | 48.95% | [48.2, 49.6] | Yes ✓ |
| feature_fixed → feature | 0.1014 | 30.94 | < 1e-171 | **0.692** | Medium | 18.65% | [17.7, 19.6] | Yes ✓ |
| full_fixed → full | 0.0755 | 22.07 | 8.15e-97 | **0.494** | Small | 18.38% | [17.0, 19.8] | Yes ✓ |
| **Enhanced Model Comparisons (α∈[1,3])** | | | | | | | | |
| baseline → full | 0.8383 | 105.80 | < 1e-300 | **2.366** | Huge | 71.43% | [71.0, 71.9] | Yes ✓ |
| baseline → feature | 0.7312 | 99.58 | < 1e-300 | **2.227** | Huge | 62.30% | [61.7, 62.9] | Yes ✓ |
| baseline → spatial | 0.5733 | 89.19 | < 1e-300 | **1.995** | Very Large | 48.85% | [48.1, 49.5] | Yes ✓ |
| spatial → full | 0.2650 | 60.39 | < 1e-300 | **1.351** | Very Large | 44.15% | [43.3, 44.9] | Yes ✓ |
| spatial → feature | 0.1579 | 43.66 | < 1e-292 | **0.976** | Large | 26.30% | [25.3, 27.3] | Yes ✓ |
| feature → full | 0.1071 | 36.27 | < 1e-221 | **0.811** | Large | 24.21% | [23.2, 25.2] | Yes ✓ |
| **Cross-Type Comparisons** | | | | | | | | |
| spatial_fixed → full | 0.8405 | 105.88 | < 1e-300 | **2.368** | Huge | 71.48% | [71.0, 72.0] | Yes ✓ |
| spatial_fixed → feature | 0.7334 | 99.69 | < 1e-300 | **2.230** | Huge | 62.37% | [61.7, 63.0] | Yes ✓ |
| feature_fixed → full | 0.2085 | 50.84 | < 1e-300 | **1.137** | Large | 38.34% | [37.5, 39.2] | Yes ✓ |
| full_fixed → feature | -0.0316 | -9.37 | 1.93e-20 | **-0.210** | Small | -7.69% | [-9.5, -6.0] | Yes ✓ |
| feature_fixed → spatial | -0.0565 | -15.83 | 3.01e-53 | **-0.354** | Small | -10.39% | [-11.7, -9.0] | Yes ✓ |
| full_fixed → spatial | -0.1895 | -46.28 | < 1e-300 | **-1.035** | Large | -46.13% | [-48.3, -43.9] | Yes ✓ |

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