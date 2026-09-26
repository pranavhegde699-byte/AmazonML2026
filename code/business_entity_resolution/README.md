# Business Entity Resolution Pipeline

This repository contains our end-to-end entity resolution pipeline for the Amazon ML Challenge 2026.

## Requirements
Ensure you are using Python 3.10+ and install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Pipeline
To reproduce our final `matching_results.tsv` and `candidate_pairs.tsv` from the dataset:
1. Ensure the `dataset/` directory is in the root (parent directory of this code package) as standard.
2. Run the baseline script:
```bash
python src/baseline.py
```

The script will automatically:
1. Preprocess the datasets (normalize names and addresses).
2. Generate candidate pairs via 2-token blocking.
3. Extract features and train a logistic regression threshold model locally.
4. Output the validation-ready TSV files in the `output/` directory.
