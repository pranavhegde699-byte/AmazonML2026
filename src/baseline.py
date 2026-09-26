import os
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from rapidfuzz import fuzz
import time

DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_address(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_block_keys(names, countries):
    keys = []
    stopwords = {'the', 'and', 'of', 'in', 'at', 'inc', 'ltd', 'llc', 'co', 'corp', 'company', 'corporation', 'private', 'pvt', 'limited'}
    for name, country in zip(names, countries):
        if not isinstance(name, str) or not name:
            keys.append("empty_" + str(country).lower())
            continue
        valid_tokens = [t for t in name.split() if t not in stopwords and len(t) > 1]
        if len(valid_tokens) >= 2:
            key = valid_tokens[0][:5] + "_" + valid_tokens[1][:5] + "_" + str(country).lower()
        elif len(valid_tokens) == 1:
            key = valid_tokens[0][:5] + "_" + str(country).lower()
        else:
            key = name.replace(" ", "")[:8] + "_" + str(country).lower()
        keys.append(key)
    return keys

def f0_5_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def extract_features(s1_name, c_name, s1_addr, c_addr):
    # Name features
    s1_tokens = set(s1_name.split())
    c_tokens = set(c_name.split())
    
    name_intersect = len(s1_tokens.intersection(c_tokens))
    name_union = len(s1_tokens.union(c_tokens))
    name_jaccard = name_intersect / name_union if name_union > 0 else 0.0
    
    name_fuzz = fuzz.ratio(s1_name, c_name) / 100.0
    
    # Address features
    s1_addr_tokens = set(s1_addr.split())
    c_addr_tokens = set(c_addr.split())
    
    addr_intersect = len(s1_addr_tokens.intersection(c_addr_tokens))
    addr_union = len(s1_addr_tokens.union(c_addr_tokens))
    addr_jaccard = addr_intersect / addr_union if addr_union > 0 else 0.0
    
    addr_fuzz = fuzz.ratio(s1_addr, c_addr) / 100.0
    
    return [name_jaccard, name_fuzz, addr_jaccard, addr_fuzz]

def run_baseline(sample_size=100000):
    print("--- PHASE B: BASELINE ---")
    
    print(f"Loading {sample_size} records from Train...")
    s1_train = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source1.tsv'), sep='\t', dtype=str, nrows=sample_size)
    s2_train = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source2.tsv'), sep='\t', dtype=str)
    s3_train = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source3.tsv'), sep='\t', dtype=str)
    gt = pd.read_csv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'), sep='\t', dtype=str)
    
    print("Preprocessing Train...")
    for df in [s1_train, s2_train, s3_train]:
        df['norm_name'] = df['business_name'].apply(normalize_text)
        df['norm_addr'] = df['business_address'].apply(normalize_address)
        df['block_key'] = get_block_keys(df['norm_name'], df['country'])
        
    pool_train = pd.concat([s2_train, s3_train], ignore_index=True)
    
    # Candidate Generation via dictionary lookup
    print("Generating candidates via blocking...")
    pool_dict = {}
    for row in pool_train[['entity_id', 'block_key', 'norm_name', 'norm_addr']].itertuples(index=False):
        if row.block_key not in pool_dict:
            pool_dict[row.block_key] = []
        if len(pool_dict[row.block_key]) < 20:  # limit pool members per block to prevent explosion
            pool_dict[row.block_key].append(row)
            
    cand_records = []
    X_train = []
    y_train = []
    
    print("Extracting features for training on the fly...")
    gt_map = dict(zip(gt['source1_entity_id'], gt['matched_entity_ids'].fillna('')))
    
    for row in s1_train[['entity_id', 'block_key', 'norm_name', 'norm_addr']].itertuples(index=False):
        cands = pool_dict.get(row.block_key, [])
        true_matches = set(gt_map.get(row.entity_id, '').split(','))
        for c in cands:
            cand_records.append((row.entity_id, c.entity_id))
            feats = extract_features(row.norm_name, c.norm_name, row.norm_addr, c.norm_addr)
            X_train.append(feats)
            y_train.append(1 if c.entity_id in true_matches else 0)
            
    candidates_df = pd.DataFrame(cand_records, columns=['entity_id_s1', 'entity_id_pool'])
    print(f"Total candidate pairs generated: {len(candidates_df)}")
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Training LR model on {len(X_train)} candidate pairs...")
    if len(np.unique(y_train)) > 1:
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_train)[:, 1]
        best_f05, best_thresh = 0, 0.5
        for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
            preds = (probs >= t).astype(int)
            p = precision_score(y_train, preds, zero_division=0)
            r = recall_score(y_train, preds, zero_division=0)
            f = f0_5_score(p, r)
            if f > best_f05:
                best_f05, best_thresh = f, t
        print(f"Best threshold: {best_thresh}, Train Pair-wise F0.5: {best_f05:.4f}")
    else:
        print("Warning: Only one class in training set. Using dummy model.")
        class DummyModel:
            def predict_proba(self, X): return np.zeros((len(X), 2))
        model = DummyModel()
        best_thresh = 0.5

    # Memory cleanup
    del s1_train, s2_train, s3_train, pool_train, candidates_df, X_train, y_train
    
    print("--- INFERENCE ON FULL TEST SET ---")
    s1_test = pd.read_csv(os.path.join(TEST_DIR, 'test_source1.tsv'), sep='\t', dtype=str)
    s2_test = pd.read_csv(os.path.join(TEST_DIR, 'test_source2.tsv'), sep='\t', dtype=str)
    s3_test = pd.read_csv(os.path.join(TEST_DIR, 'test_source3.tsv'), sep='\t', dtype=str)
    
    print("Preprocessing Test...")
    for df in [s1_test, s2_test, s3_test]:
        df['norm_name'] = df['business_name'].apply(normalize_text)
        df['norm_addr'] = df['business_address'].apply(normalize_address)
        df['block_key'] = get_block_keys(df['norm_name'], df['country'])
        
    pool_test = pd.concat([s2_test, s3_test], ignore_index=True)
    
    print("Blocking on Test...")
    pool_dict = {}
    for row in pool_test[['entity_id', 'block_key', 'norm_name', 'norm_addr']].itertuples(index=False):
        if row.block_key not in pool_dict:
            pool_dict[row.block_key] = []
        if len(pool_dict[row.block_key]) < 20:
            pool_dict[row.block_key].append(row)
            
    test_cands = []
    X_test = []
    
    print("Generating candidates & extracting test features on the fly...")
    for row in s1_test[['entity_id', 'block_key', 'norm_name', 'norm_addr']].itertuples(index=False):
        cands = pool_dict.get(row.block_key, [])
        for c in cands:
            test_cands.append((row.entity_id, c.entity_id))
            X_test.append(extract_features(row.norm_name, c.norm_name, row.norm_addr, c.norm_addr))
            
    test_cands_df = pd.DataFrame(test_cands, columns=['entity_id_s1', 'entity_id_pool'])
    print(f"Total test candidate pairs generated: {len(test_cands_df)}")
        
    if len(X_test) > 0:
        probs = model.predict_proba(np.array(X_test))[:, 1]
        test_cands_df['is_match'] = (probs >= best_thresh).astype(int)
    else:
        test_cands_df['is_match'] = 0
        
    print("Writing outputs...")
    # Get matches
    matches = test_cands_df[test_cands_df['is_match'] == 1].groupby('entity_id_s1')['entity_id_pool'].apply(list).to_dict()
    # Get all candidates
    all_cands = test_cands_df.groupby('entity_id_s1')['entity_id_pool'].apply(list).to_dict()
    
    cand_out = open(os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv'), 'w', encoding='utf-8')
    match_out = open(os.path.join(OUTPUT_DIR, 'matching_results.tsv'), 'w', encoding='utf-8')
    
    cand_out.write("source1_entity_id\tcandidate_entity_ids\n")
    match_out.write("source1_entity_id\tmatched_entity_ids\n")
    
    for s1_id in s1_test['entity_id']:
        c_list = all_cands.get(s1_id, [])
        m_list = matches.get(s1_id, [])
        
        cand_out.write(f"{s1_id}\t{','.join(c_list)}\n")
        match_out.write(f"{s1_id}\t{','.join(m_list)}\n")
        
    cand_out.close()
    match_out.close()
    print("Baseline pipeline completed successfully.")

if __name__ == "__main__":
    start_time = time.time()
    run_baseline()
    print(f"Total time: {time.time() - start_time:.2f} seconds")
