# Benchmark data

The package includes the processed cell-line split and precomputed features distributed by HyperSynergy. The original DrugComb downloads are available through the data portal linked below. [Upstream benchmark data](https://github.com/NWPU-903PR/HyperSynergy/tree/f7e23e2b381b0120e88222124f0b06baac432630/data)

```text
data/
├── drugcomb_few_shot/
│   ├── train/                 # 85 base cell lines
│   └── test/                  # 21 held-out cell lines
└── sample_features/
    ├── cell_feature_900.p
    ├── codes_cell.p
    └── drug_feature_cell.p
```

Each task file is a tab-separated table containing `CELL_ID`, `PUB_IDA`, `PUB_IDB`, and `SCORE_BLISS`. The distributed split contains 427,710 rows referencing 1,928 drug IDs and 106 cell lines. Stage 1 pretrains on the 85 base cell lines with a row-level validation split. Stage 2 holds out 17 of those base cell lines for validation and uses the other 68 for meta-training. The 21 held-out test cell lines are used for final evaluation. [Published paper](https://doi.org/10.1145/3770855.3818930)

The original feature preparation instructions reference PubChem and the CCLE/GDSC expression sources. Drug synergy data come from DrugComb. Use the linked upstream terms for the data and feature sources. The MIT notice in this repository covers the inherited code. [HyperSynergy preparation instructions](https://github.com/NWPU-903PR/HyperSynergy#0-create-sample-feature-dictionaries-required-when-running-on-other-datasets) · [DrugComb data portal](https://researchportal.helsinki.fi/en/datasets/drugcomb/)
