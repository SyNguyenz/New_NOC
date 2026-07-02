# Paper Tables - PROVEDIt 45-class STR Contributor Identification
Generated from: C:\Tailieu\TinSinh\Project\new_NOC\results

## Table 1: Main results (test n=1,325, GF29cycles)

| Model           | Macro F1 | Micro F1 | Exact Match | Hamming | Reject AUROC |
| --------------- | -------- | -------- | ----------- | ------- | ------------ |
| Set Transformer | 0.9813   | 0.9821   | 0.9653      | 0.0011  | 1.0000       |
| XGBoost         | 0.9736   | 0.9779   | 0.9449      | 0.0013  | —            |
| MLP             | 0.9731   | 0.9756   | 0.9426      | 0.0015  | —            |
| LR              | 0.9622   | 0.9669   | 0.9238      | 0.0020  | —            |
| CNN             | 0.9622   | 0.9648   | 0.9253      | 0.0022  | —            |
| kNN             | 0.8832   | 0.9000   | 0.7766      | 0.0057  | —            |

## Table 2: Ablation study (60 epochs, same split)

| Model                    | Macro F1 | Micro F1 | Exact Match | Hamming | Reject AUROC |
| ------------------------ | -------- | -------- | ----------- | ------- | ------------ |
| Full (ISAB+PMA, 3 heads) | 0.9532   | 0.9586   | 0.9215      | 0.0026  | 1.0000       |
| Deep Sets (mean pool)    | 0.1453   | 0.1131   | 0.0015      | 0.2448  | 0.6644       |
| MLP encoder (flat 590)   | 0.9726   | 0.9752   | 0.9472      | 0.0015  | —            |
| No NOC head              | 0.9357   | 0.9414   | 0.8913      | 0.0037  | 0.9999       |
| No reject head           | 0.9624   | 0.9696   | 0.9396      | 0.0019  | —            |

## Table 3: Open-set AUROC comparison (1,325 closed vs 1,366 open)

| Method                              | AUROC  |
| ----------------------------------- | ------ |
| Trained reject head (sigmoid)       | 1.0000 |
| Mahalanobis distance (z_mix)        | 0.9934 |
| Openmax (Weibull, closest centroid) | 0.9914 |
| Max Sigmoid Probability (MSP)       | 0.9103 |
| Energy score (-logsumexp)           | 0.9006 |

## Table 4: Set Transformer - Exact Match by NOC

| NOC | Exact Match | n samples |
| --- | ----------- | --------- |
| 1   | 0.9660      | 1119      |
| 2   | 1.0000      | 46        |
| 3   | 0.9841      | 63        |
| 4   | 0.9394      | 33        |
| 5   | 0.9219      | 64        |

## Table 5: NOC head accuracy (vs deepNoC PROVEDIt baseline = 0.90)

Overall NOC accuracy: **0.9955** (deepNoC: 0.90)

| NOC | Accuracy | n |
| --- | -------- | - |
| 1   | 1.0000   | 1119 |
| 2   | 1.0000   | 46 |
| 3   | 1.0000   | 63 |
| 4   | 0.9091   | 33 |
| 5   | 0.9531   | 64 |

## Table 6: Calibration (ECE)

| Head | ECE |
| ---- | --- |
| Reject head | 0.0013 |
| Donor head (top-1) | 0.0096 |

## Table 7: Robustness - Filtered vs UnFiltered (same test samples)

| Metric | Filtered (train+test) | UnFiltered (test) |
| ------ | --------------------- | ----------------- |
| macro_f1     | 0.9813                | 0.9774            |
| micro_f1     | 0.9821                | 0.9765            |
| exact_match  | 0.9653                | 0.9540            |
| hamming      | 0.0011                | 0.0014            |
| reject_auroc | 1.0000                | 0.9998            |

## Table 8: Stratified accuracy by template DNA

| Template DNA | n | Exact Match | Macro F1 |
| ------------ | - | ----------- | -------- |
| <=0.05 ng (very low) | 454   | 0.9185 | 0.9479 |
| 0.05-0.15 ng (low)   | 394   | 0.9822 | 0.9896 |
| 0.15-0.50 ng (med)   | 250   | 0.9920 | 0.9541 |
| >0.50 ng (high)      | 227   | 1.0000 | 0.9333 |

## Table 9: Multi-seed confidence intervals

| Metric | Mean ± Std |
| ------ | ---------- |
| macro_f1 | 0.9412 ± 0.0183 |
| micro_f1 | 0.9510 ± 0.0142 |
| exact_match | 0.9131 ± 0.0158 |
| hamming | 0.0031 ± 0.0009 |
| precision | 0.9208 ± 0.0232 |
| recall | 0.9674 ± 0.0124 |
| reject_auroc | 1.0000 ± 0.0000 |

## Table 10a: Sweep — m_inducing (inducing points)

| Config | Macro F1 | Exact Match | Reject AUROC |
| ------ | -------- | ----------- | ------------ |
| sweep_m8             | 0.9508 | 0.9275 | 1.0000 |
| sweep_m16            | 0.9633 | 0.9389 | 1.0000 |
| sweep_m32            | 0.9565 | 0.9381 | 0.9999 |
| sweep_m64            | 0.9596 | 0.9336 | 0.9999 |

## Table 10b: Sweep — n_isab (ISAB stack depth)

| Config | Macro F1 | Exact Match | Reject AUROC |
| ------ | -------- | ----------- | ------------ |
| sweep_isab1          | 0.9310 | 0.9087 | 1.0000 |
| sweep_isab2          | 0.9505 | 0.9328 | 1.0000 |
| sweep_isab3          | 0.9526 | 0.9268 | 1.0000 |

## Table 10c: Sweep — n_heads (attention heads)

| Config | Macro F1 | Exact Match | Reject AUROC |
| ------ | -------- | ----------- | ------------ |
| sweep_h2             | 0.9615 | 0.9351 | 1.0000 |
| sweep_h4             | 0.9485 | 0.9238 | 1.0000 |
| sweep_h8             | 0.9432 | 0.9215 | 0.9999 |

## Attention: Top-10 loci by PMA attention weight

| Rank | Locus | Mean Attention |
| ---- | ----- | -------------- |
| 1    | TH01            | 0.1641         |
| 2    | D22S1045        | 0.1596         |
| 3    | CSF1PO          | 0.0979         |
| 4    | D5S818          | 0.0941         |
| 5    | D10S1248        | 0.0789         |
| 6    | D2S441          | 0.0736         |
| 7    | D7S820          | 0.0732         |
| 8    | D13S317         | 0.0630         |
| 9    | TPOX            | 0.0379         |
| 10   | D16S539         | 0.0285         |