# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Antigravity  
**Team Members:** User & Agent  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary
Our baseline solution utilizes a two-token block strategy based on normalized business names and country to rapidly filter millions of candidates into a precise subset. A Logistic Regression model, tuned using Jaccard token overlap and RapidFuzz string distances on both names and addresses, is then used to threshold the most confident business entity matches, maximizing the F0.5 score locally.

---

## 2. Methodology

### 2.1 Problem Analysis
The datasets are massive, with a test candidate pool of ~10 million records. Initial attempts with generic string-prefix blocking caused memory exhaustion due to the Cartesian product explosion on common business prefixes (e.g. "national", "state bank"). A strict blocking strategy was strictly necessary as a foundation. Missing addresses were observed in ~170k rows of the candidate pool, while business names were almost uniformly present.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier
**Core Innovation:** To overcome memory constraints and execute inference efficiently on local hardware within minutes, we used strict deterministic blocking followed by simple, highly interpretable similarity features processed through an F0.5-optimized Logistic Regression model.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Up to the first two valid tokens (excluding stopwords) of the `business_name` + lowercase `country` (e.g. `taj_mahal_in`).
- **Candidate pairs generated:** Evaluated millions of candidates, restricting each S1 entity's candidate pool using bounded dictionary lookups to prevent Cartesian explosion.
- **How you ensured true matches were not lost:** Using the first two tokens (or single token/prefix if fewer exist) is more lenient than exact string matching, capturing variations and typos in the suffixes of business names while avoiding the OOM problems of single-word blocking. We bounded candidates per block key to ensure memory safety.

---

## 4. Matching Model

**Features used:**
- Name features: Jaccard Token Overlap, RapidFuzz string ratio
- Address features: Jaccard Token Overlap, RapidFuzz string ratio

**Model type:** Logistic Regression (`class_weight='balanced'`)
**Threshold selection method:** Empirical evaluation on candidate pairs generated from a 100k S1 training subset to directly maximize the F0.5 metric, finding `threshold = 0.9` as optimal.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9597 (on local candidate pairwise training validation)
- **Common false positives (wrong merges):** Franchises or branch locations with identical names in the same country but different specific addresses that string-distance metrics failed to heavily penalize.
- **Common false negatives (missed matches):** Candidates that share the real-world entity but had typos in their `business_name`, resulting in them failing the strict exact-name blocking phase.

---

## 6. Conclusion
The baseline solution demonstrates that strict blocking combined with lightweight similarity features and an interpretable classifier yields a highly memory-efficient, fast, and valid entity resolution pipeline. Future iterations can safely relax the blocking criteria (e.g. token-blocking) using batch processing to improve candidate recall.

---

## Appendix

### A. Code Artefacts
Our pipeline is fully contained within `src/baseline.py`. 
To reproduce the results:
1. `pip install -r requirements.txt`
2. `python src/baseline.py`
The script performs the preprocessing, blocking, training, thresholding, and output generation automatically.
