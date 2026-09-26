import os
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from rapidfuzz import fuzz
from baseline import normalize_text, normalize_address, get_block_keys, extract_features

DATA_DIR = "../dataset" # Assuming run from src/
if not os.path.exists(DATA_DIR):
    DATA_DIR = "dataset" # If run from root

TRAIN_DIR = os.path.join(DATA_DIR, "train")

def compute_entity_metrics(y_true_dict, y_pred_dict):
    """
    Computes macro-averaged Precision, Recall, and F0.5 over source1_entity_ids.
    y_true_dict: dict of s1_id -> set of true matched pool_ids
    y_pred_dict: dict of s1_id -> set of predicted pool_ids
    """
    f05_scores = []
    p_scores = []
    r_scores = []
    
    singletons_f05 = []
    matched_f05 = []
    
    for s1_id in y_true_dict.keys():
        true_set = y_true_dict[s1_id]
        pred_set = y_pred_dict.get(s1_id, set())
        
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        if len(true_set) == 0:
            # Singleton
            if len(pred_set) == 0:
                p, r, f05 = 1.0, 1.0, 1.0
            else:
                p, r, f05 = 0.0, 0.0, 0.0
            singletons_f05.append(f05)
        else:
            # Matched entity
            if tp == 0:
                p, r, f05 = 0.0, 0.0, 0.0
            else:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f05 = (1.25 * p * r) / (0.25 * p + r)
            matched_f05.append(f05)
            
        f05_scores.append(f05)
        p_scores.append(p)
        r_scores.append(r)
        
    return {
        'macro_p': np.mean(p_scores),
        'macro_r': np.mean(r_scores),
        'macro_f05': np.mean(f05_scores),
        'singletons_f05': np.mean(singletons_f05) if singletons_f05 else 0.0,
        'matched_f05': np.mean(matched_f05) if matched_f05 else 0.0,
        'num_singletons': len(singletons_f05),
        'num_matched': len(matched_f05)
    }

def run_diagnostic():
    print("Loading data...")
    s1_train = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source1.tsv'), sep='\t', dtype=str, nrows=200000)
    s2_train = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source2.tsv'), sep='\t', dtype=str)
    s3_train = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source3.tsv'), sep='\t', dtype=str)
    gt = pd.read_csv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'), sep='\t', dtype=str)
    
    # Split S1 entities 80/20 train/val
    np.random.seed(42)
    s1_ids = list(s1_train['entity_id'])
    np.random.shuffle(s1_ids)
    split_idx = int(0.8 * len(s1_ids))
    train_s1_ids = set(s1_ids[:split_idx])
    val_s1_ids = set(s1_ids[split_idx:])
    
    print(f"Total S1: {len(s1_ids)}, Train S1: {len(train_s1_ids)}, Val S1: {len(val_s1_ids)}")
    
    # Preprocess
    print("Preprocessing...")
    for df in [s1_train, s2_train, s3_train]:
        df['norm_name'] = df['business_name'].apply(normalize_text)
        df['norm_addr'] = df['business_address'].apply(normalize_address)
        df['block_key'] = get_block_keys(df['norm_name'], df['country'])
        
    pool_train = pd.concat([s2_train, s3_train], ignore_index=True)
    
    gt_map = {row.source1_entity_id: set(str(row.matched_entity_ids).split(',')) if pd.notna(row.matched_entity_ids) and row.matched_entity_ids != "" else set() for row in gt.itertuples()}
    
    # STEP 1: Candidate Generation Diagnostic on Val
    print("\n--- STEP 1: Candidate Generation Diagnostic (VAL split) ---")
    val_s1_df = s1_train[s1_train['entity_id'].isin(val_s1_ids)]
    
    print("Generating candidates via TF-IDF blocking (Val)...")
    from generate_candidates import get_candidates_tfidf
    val_candidates = get_candidates_tfidf(val_s1_df, pool_train, top_k=30, sim_cutoff=0.3)
    
    true_matches_in_val = 0
    captured_true_matches = 0
    zero_candidate_count = 0
    total_candidates = 0
    
    for row in val_s1_df[['entity_id', 'norm_name', 'norm_addr']].itertuples(index=False):
        cands = val_candidates.get(row.entity_id, [])
        cand_ids = {c.entity_id for c in cands}
        
        if len(cands) == 0:
            zero_candidate_count += 1
            
        total_candidates += len(cands)
        
        true_set = gt_map.get(row.entity_id, set())
        true_matches_in_val += len(true_set)
        captured_true_matches += len(true_set.intersection(cand_ids))

    candidate_recall = captured_true_matches / true_matches_in_val if true_matches_in_val > 0 else 0
    cand_counts = [len(cands) for cands in val_candidates.values()]
    avg_cands = np.mean(cand_counts) if len(cand_counts) > 0 else 0
    median_cands = np.median(cand_counts) if len(cand_counts) > 0 else 0
    max_cands = np.max(cand_counts) if len(cand_counts) > 0 else 0
    zero_cand_rate = zero_candidate_count / len(val_s1_ids) if len(val_s1_ids) > 0 else 0
    
    total_possible_pairs = len(val_s1_ids) * len(pool_train)
    reduction_ratio = total_candidates / total_possible_pairs if total_possible_pairs > 0 else 0
    
    print(f"CANDIDATE RECALL: {candidate_recall:.4f} ({captured_true_matches}/{true_matches_in_val})")
    print(f"AVG/MEDIAN/MAX CANDIDATES PER S1: {avg_cands:.2f} / {median_cands} / {max_cands}")
    print(f"ZERO-CANDIDATE RATE: {zero_cand_rate:.4f} ({zero_candidate_count}/{len(val_s1_ids)})")
    print(f"REDUCTION RATIO: {reduction_ratio:.6f} ({total_candidates} vs {total_possible_pairs})")
    
    if candidate_recall < 0.90:
        print(">>> WARNING: Candidate recall is below 90%. This caps the max possible score. Fix this first.")
        
    # STEP 2: Matching Model Diagnostic
    print("\n--- STEP 2: Matching Model Diagnostic ---")
    print("Generating train candidates via TF-IDF blocking...")
    train_s1_df = s1_train[s1_train['entity_id'].isin(train_s1_ids)]
    train_candidates = get_candidates_tfidf(train_s1_df, pool_train, top_k=30, sim_cutoff=0.3)
    
    print("Generating train features...")
    X_train, y_train = [], []
    
    for row in train_s1_df[['entity_id', 'norm_name', 'norm_addr']].itertuples(index=False):
        cands = train_candidates.get(row.entity_id, [])
        true_set = gt_map.get(row.entity_id, set())
        for c in cands:
            feats = extract_features(row.norm_name, c.norm_name, row.norm_addr, c.norm_addr)
            X_train.append(feats)
            y_train.append(1 if c.entity_id in true_set else 0)
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Training LR model on {len(X_train)} candidate pairs...")
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    
    print("Extracting val features...")
    X_val = []
    val_pairs = []
    val_true_dict = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_ids}
    
    for row in val_s1_df[['entity_id', 'norm_name', 'norm_addr']].itertuples(index=False):
        cands = val_candidates[row.entity_id]
        for c in cands:
            feats = extract_features(row.norm_name, c.norm_name, row.norm_addr, c.norm_addr)
            X_val.append(feats)
            val_pairs.append((row.entity_id, c.entity_id))
            
    X_val = np.array(X_val)
    val_probs = model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([])
    
    print("THRESHOLD SWEEP RESULTS:")
    best_f05 = 0.0
    best_thresh = 0.5
    best_metrics = None
    
    for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        val_pred_dict = {}
        if len(val_probs) > 0:
            preds = (val_probs >= t).astype(int)
            for i, (s1_id, pool_id) in enumerate(val_pairs):
                if preds[i] == 1:
                    if s1_id not in val_pred_dict:
                        val_pred_dict[s1_id] = set()
                    val_pred_dict[s1_id].add(pool_id)
                    
        metrics = compute_entity_metrics(val_true_dict, val_pred_dict)
        print(f"  Threshold {t:.2f} -> Macro P: {metrics['macro_p']:.4f}, Macro R: {metrics['macro_r']:.4f}, Macro F0.5: {metrics['macro_f05']:.4f}")
        print(f"    Singletons F0.5: {metrics['singletons_f05']:.4f}, Matched F0.5: {metrics['matched_f05']:.4f}")
        
        if metrics['macro_f05'] > best_f05:
            best_f05 = metrics['macro_f05']
            best_thresh = t
            best_metrics = metrics
            
    print("\n--- STEP 3: Report findings clearly ---")
    print("STAGE: Diagnostic")
    print(f"CANDIDATE RECALL: {candidate_recall:.4f}")
    print(f"AVG/MEDIAN/MAX CANDIDATES PER S1: {avg_cands:.2f} / {median_cands} / {max_cands}")
    print(f"ZERO-CANDIDATE RATE: {zero_cand_rate:.4f}")
    print(f"VALIDATION PRECISION / RECALL / F0.5 (current best threshold {best_thresh:.2f}): {best_metrics['macro_p']:.4f} / {best_metrics['macro_r']:.4f} / {best_metrics['macro_f05']:.4f}")
    print(f"F0.5 ON SINGLETON ENTITIES vs MATCHED ENTITIES: Singletons={best_metrics['singletons_f05']:.4f} vs Matched={best_metrics['matched_f05']:.4f}")

if __name__ == "__main__":
    run_diagnostic()
