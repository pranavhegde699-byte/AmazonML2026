# Requirements Understanding

## Project Goal
The goal of this project is to build an Entity Resolution (ER) system to match business records across three independent data sources. Specifically, given a deduplicated reference source (Source 1), the system must identify all corresponding records from Source 2 and Source 3 that represent the same real-world business entity.

## Data Sources
- **Source 1**: Deduplicated reference source.
- **Source 2**: Target matching source.
- **Source 3**: Target matching source.
- **Format**: All files are strictly tab-separated (`.tsv`).
- **Columns**: `entity_id`, `business_name`, `business_address`, `country`.

## Key Challenges & Constraints
1. **Noise Patterns**: The datasets contain noisy business names (abbreviations, typos, legal suffixes) and addresses (missing components, landmarks, different formats).
2. **Missing Information**: Fields may be incomplete.
3. **No External Data**: We are strictly prohibited from using external databases, geocoding APIs, or internet-based lookup services. All learning must come from the provided dataset.
4. **F0.5 Evaluation**: The primary evaluation metric is macro-averaged F0.5. This metric weights precision higher than recall, meaning false merges (predicting a match where none exists) are penalized more severely than missed matches.
5. **Zero Matches (Singletons)**: A Source -1 entity may have zero matches. Accurately predicting an empty list yields a perfect score (1.0) for that entity, whereas predicting a false match ruins it (0.0).

## System Outputs
The final system must produce two exact files in the `output/` directory:
1. `matching_results.tsv`: The final predicted matches for S1 entities. (Scored on leaderboard).
2. `candidate_pairs.tsv`: The exact candidate set fed into the final ML matching model, prior to applying a threshold.

## Pipeline Requirements
1. **Conservative Normalization**: Clean up strings without aggressively deleting potentially useful info (e.g., numbers).
2. **Candidate Generation (Blocking)**: Must effectively reduce the search space to a manageable subset of candidate pairs while preserving a very high "candidate recall" (ensuring true matches aren't missed early).
3. **Similarity Features**: Extract features like TF-IDF, Jaccard, Token Overlap, etc.
4. **Matching Model**: Start with a simple Logistic Regression baseline and tune the decision threshold based on validation F0.5 scores, not just accuracy.
5. **Validation**: Use `utils/validate_submission.py` to ensure formatting and structural integrity of the final TSVs before concluding the process.
