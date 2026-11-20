# Ballon d'Or Predictor

MSE 446 final project investigating how accurately player performance statistics can predict Ballon d'Or vote shares.

## Repository Structure

```text
BallonDor_Predictor/
├── Data/
│   ├── raw/               # Original source workbook
│   └── processed/         # Generated cleaned datasets
├── Project_Proposal/      # Proposal PDF and feedback
├── notebooks/             # Exploratory analysis
├── src/                   # Reusable data, preprocessing, and evaluation code
├── scripts/               # Training and other runnable scripts
├── models/                # Saved model pipelines
├── reports/               # Evaluation metrics and predictions
│   └── figures/           # Plots for analysis and reporting
├── .gitignore
└── README.md
```

## Data

The source workbook is stored at `Data/raw/Ballon_dOr_2015_2025.xlsx`. Keep source data unchanged and write generated datasets to `Data/processed/`.

The planned target is `target_vote_share`, with linear regression as the initial model. Training and preprocessing code will be added in a later step.

## Generated Files

Processed datasets and saved models are excluded from Git. Empty folders contain `.gitkeep` files so the repository structure is preserved in Git.
