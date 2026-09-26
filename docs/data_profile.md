# Data Profile

## General Overview
The dataset contains business entity records across three sources, split into `train` and `test` directories.

- **Source 1** serves as the deduplicated reference source.
- **Source 2** and **Source 3** are the target matching sources.
- No duplicate `entity_id` values exist within any single file.

## Dataset Sizes
| File | Total Rows |
|------|-----------|
| `train_source1.tsv` | 2,206,821 |
| `train_source2.tsv` | 5,034,616 |
| `train_source3.tsv` | 5,285,603 |
| `test_source1.tsv` | 1,732,544 |
| `test_source2.tsv` | 4,887,273 |
| `test_source3.tsv` | 5,082,316 |

## Ground Truth Stats (`train_ground_truth.tsv`)
- **Total S1 Entities**: 2,206,821
- **Singletons (Zero Matches)**: 123,247
- **Entities with Exactly 1 Match**: 119,157
- **Entities with Multiple Matches**: 1,964,417
- **Total Positive Matches**: 7,638,365
- **Average Matches per S1**: 3.46

*Observation: The vast majority of S1 entities have multiple matches across S2 and S3.*

## Missing Values
- **Source 1 (Train)**: No missing values.
- **Source 2 (Train)**: 2 missing `business_name`s, 168,967 missing `business_address`es.
- **Source 3 (Train)**: 13 missing `business_name`s, 175,916 missing `business_address`es.

## Country Distribution
The `country` field is an open set string label.

**Train Set Distribution**:
- `US`: ~7.5 million total records (S1, S2, S3 combined)
- `India`: ~5.0 million total records
- No other countries in train.

**Test Set Distribution**:
- `India`: ~5.5 million total records
- `US`: ~4.4 million total records
- `France`: ~1.7 million total records (Unseen in train)

*Observation: The matching logic/models must not be explicitly hardcoded to just "US" and "India", as "France" appears exclusively in the test set.*

## Text Field Characteristics
- **Average Name Length**: ~24-25 characters across sources.
- **Average Address Length**: ~48-52 characters across sources.
