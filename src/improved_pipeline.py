import os
import re
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import gc

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("WARNING: LightGBM not found, falling back to GradientBoosting")

from rapidfuzz import fuzz

DATA_DIR   = "dataset"
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
TEST_DIR   = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Abbreviation expansion pairs (pattern -> replacement)
ABBREV_MAP = [
    (r'\bltd\b',    'limited'),
    (r'\bllc\b',    'llc'),
    (r'\binc\b',    'incorporated'),
    (r'\bcorp\b',   'corporation'),
    (r'\bco\b',     'company'),
    (r'\bpvt\b',    'private'),
    (r'\bplc\b',    'public limited company'),
    (r'\bllp\b',    'limited liability partnership'),
    (r'\bintl\b',   'international'),
    (r'\bmfg\b',    'manufacturing'),
    (r'\bmgmt\b',   'management'),
    (r'\bsvcs\b',   'services'),
    (r'\bsvc\b',    'service'),
    (r'\btech\b',   'technology'),
    (r'\bsol\b',    'solutions'),
    (r'\bsols\b',   'solutions'),
    (r'\bgrp\b',    'group'),
    (r'\bassoc\b',  'associates'),
    (r'\bmktg\b',   'marketing'),
    (r'\bdev\b',    'development'),
    (r'\bsys\b',    'systems'),
    (r'\bsyst\b',   'systems'),
    (r'\bengr\b',   'engineering'),
    (r'\beng\b',    'engineering'),
    (r'\bconstr\b', 'construction'),
    (r'\bconst\b',  'construction'),
    (r'\bdist\b',   'distribution'),
    (r'\bcomm\b',   'communications'),
    (r'\bfinl\b',   'financial'),
    (r'\bhosp\b',   'hospital'),
    (r'\bmed\b',    'medical'),
    (r'\bpharma\b', 'pharmaceutical'),
    (r'\bpharm\b',  'pharmaceutical'),
    (r'\bst\b',     'street'),
    (r'\brd\b',     'road'),
    (r'\bave\b',    'avenue'),
    (r'\bblvd\b',   'boulevard'),
]

def normalize_name(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace('&', ' and ')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    for pattern, replacement in ABBREV_MAP:
        text = re.sub(pattern, replacement, text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_address(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    for pattern, replacement in ABBREV_MAP[-6:]:
        text = re.sub(pattern, replacement, text)
    return re.sub(r'\s+', ' ', text).strip()

FEATURE_NAMES = [
    "name_jaccard", "name_fuzz_ratio", "name_token_sort", "name_partial",
    "name_token_set", "addr_jaccard", "addr_fuzz_ratio", "addr_token_sort",
    "addr_partial", "name_len_diff", "addr_len_diff"
]

def extract_features(s1_name, c_name, s1_addr, c_addr):
    s1_n_tok = set(s1_name.split())
    c_n_tok  = set(c_name.split())
    n_union  = s1_n_tok | c_n_tok
    n_inter  = s1_n_tok & c_n_tok
    name_jaccard    = len(n_inter) / len(n_union) if n_union else 0.0
    name_fuzz       = fuzz.ratio(s1_name, c_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    name_partial    = fuzz.partial_ratio(s1_name, c_name) / 100.0
    name_token_set  = fuzz.token_set_ratio(s1_name, c_name) / 100.0
    max_name_len    = max(len(s1_name), len(c_name), 1)
    name_len_diff   = abs(len(s1_name) - len(c_name)) / max_name_len

    s1_a_tok = set(s1_addr.split())
    c_a_tok  = set(c_addr.split())
    a_union  = s1_a_tok | c_a_tok
    a_inter  = s1_a_tok & c_a_tok
    addr_jaccard    = len(a_inter) / len(a_union) if a_union else 0.0
    addr_fuzz       = fuzz.ratio(s1_addr, c_addr) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
    addr_partial    = fuzz.partial_ratio(s1_addr, c_addr) / 100.0
    max_addr_len    = max(len(s1_addr), len(c_addr), 1)
    addr_len_diff   = abs(len(s1_addr) - len(c_addr)) / max_addr_len

    return [
        name_jaccard, name_fuzz, name_token_sort, name_partial, name_token_set,
        addr_jaccard, addr_fuzz, addr_token_sort, addr_partial,
        name_len_diff, addr_len_diff,
    ]

def build_tfidf_candidates(s1_df, pool_df, top_k=50, sim_cutoff=0.2,
                            text_col='norm_name', analyzer='char_wb',
                            ngram=(3, 3), max_features=150000, min_df=2):
    candidates = {}
    s1_df   = s1_df.copy()
    pool_df = pool_df.copy()
    s1_df[text_col]   = s1_df[text_col].fillna('')
    pool_df[text_col] = pool_df[text_col].fillna('')
    s1_df['country']   = s1_df['country'].fillna('unknown')
    pool_df['country'] = pool_df['country'].fillna('unknown')

    unique_countries = set(s1_df['country'].unique()) | set(pool_df['country'].unique())

    for country in unique_countries:
        s1_c   = s1_df[s1_df['country'] == country]
        pool_c = pool_df[pool_df['country'] == country]

        if len(s1_c) == 0:
            continue
        if len(pool_c) == 0:
            for s1_id in s1_c['entity_id']:
                if s1_id not in candidates:
                    candidates[s1_id] = set()
            continue

        pool_docs = pool_c[text_col].values
        s1_docs   = s1_c[text_col].values
        pool_ids  = pool_c['entity_id'].values
        s1_ids    = s1_c['entity_id'].values

        vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=ngram,
            min_df=min_df, max_features=max_features, max_df=0.5
        )
        try:
            pool_tfidf = vectorizer.fit_transform(pool_docs)
            s1_tfidf   = vectorizer.transform(s1_docs)
        except ValueError:
            for s1_id in s1_ids:
                if s1_id not in candidates:
                    candidates[s1_id] = set()
            continue

        batch_size = 200
        for i in range(0, s1_tfidf.shape[0], batch_size):
            s1_batch = s1_tfidf[i:i + batch_size]
            sims = s1_batch.dot(pool_tfidf.T).tocsr()
            sims.data[sims.data < sim_cutoff] = 0
            sims.eliminate_zeros()

            for j in range(sims.shape[0]):
                rs  = sims.indptr[j]
                re_ = sims.indptr[j + 1]
                data    = sims.data[rs:re_]
                indices = sims.indices[rs:re_]
                s1_id = s1_ids[i + j]
                if s1_id not in candidates:
                    candidates[s1_id] = set()
                if len(data) == 0:
                    continue
                if len(data) > top_k:
                    k_idx = np.argpartition(data, -top_k)[-top_k:]
                    top_indices = indices[k_idx]
                else:
                    top_indices = indices
                for idx in top_indices:
                    candidates[s1_id].add(pool_ids[idx])

    return candidates

def prefix_blocking_candidates(s1_df, pool_df, prefix_len=5, top_k=20):
    candidates = {}
    s1_df   = s1_df.copy()
    pool_df = pool_df.copy()
    s1_df['country']   = s1_df['country'].fillna('unknown')
    pool_df['country'] = pool_df['country'].fillna('unknown')
    s1_df['norm_name']   = s1_df['norm_name'].fillna('')
    pool_df['norm_name'] = pool_df['norm_name'].fillna('')

    for country in s1_df['country'].unique():
        s1_c   = s1_df[s1_df['country'] == country]
        pool_c = pool_df[pool_df['country'] == country]
        if len(pool_c) == 0:
            for s1_id in s1_c['entity_id']:
                candidates[s1_id] = set()
            continue
        prefix_index = {}
        for pid, pname in zip(pool_c['entity_id'], pool_c['norm_name']):
            prefix = pname[:prefix_len]
            if prefix not in prefix_index:
                prefix_index[prefix] = []
            if len(prefix_index[prefix]) < top_k:
                prefix_index[prefix].append(pid)
        for s1_id, s1_name in zip(s1_c['entity_id'], s1_c['norm_name']):
            prefix = s1_name[:prefix_len]
            candidates[s1_id] = set(prefix_index.get(prefix, []))
    return candidates

def get_multi_pass_candidates(s1_df, pool_df, top_k=50, sim_cutoff1=0.2):
    print("  [Blocking Pass 1] Char trigram TF-IDF on name...")
    t0 = time.time()
    cands1 = build_tfidf_candidates(
        s1_df, pool_df, top_k=top_k, sim_cutoff=sim_cutoff1,
        text_col='norm_name', analyzer='char_wb', ngram=(3, 3),
        max_features=150000, min_df=2
    )
    p1_time = time.time() - t0
    print(f"    Done in {p1_time:.1f}s | Covered: {len(cands1)}")

    print("  [Blocking Pass 2] Word bigram TF-IDF on name+address...")
    t0 = time.time()
    s1_tmp   = s1_df.copy()
    pool_tmp = pool_df.copy()
    s1_tmp['combined_text']   = s1_tmp['norm_name'].fillna('') + ' ' + s1_tmp['norm_addr'].fillna('')
    pool_tmp['combined_text'] = pool_tmp['norm_name'].fillna('') + ' ' + pool_tmp['norm_addr'].fillna('')
    cands2 = build_tfidf_candidates(
        s1_tmp, pool_tmp, top_k=top_k, sim_cutoff=0.15,
        text_col='combined_text', analyzer='word', ngram=(1, 2),
        max_features=100000, min_df=3
    )
    del s1_tmp, pool_tmp
    p2_time = time.time() - t0
    print(f"    Done in {p2_time:.1f}s | Covered: {len(cands2)}")

    merged = {}
    for s1_id in s1_df['entity_id']:
        merged[s1_id] = cands1.get(s1_id, set()) | cands2.get(s1_id, set())

    zero_cand_ids = [s1_id for s1_id, cands in merged.items() if len(cands) == 0]
    print(f"  [Blocking Pass 3] Prefix fallback for {len(zero_cand_ids)} zero-candidate entities...")
    if zero_cand_ids:
        t0 = time.time()
        zero_s1_df = s1_df[s1_df['entity_id'].isin(set(zero_cand_ids))]
        cands3 = prefix_blocking_candidates(zero_s1_df, pool_df, prefix_len=5, top_k=20)
        for s1_id, cand_set in cands3.items():
            merged[s1_id] = merged.get(s1_id, set()) | cand_set
        p3_time = time.time() - t0
        print(f"    Done in {p3_time:.1f}s")

    total_cands = sum(len(v) for v in merged.values())
    avg_cands = total_cands / len(merged) if merged else 0
    print(f"  Total candidates: {total_cands}, avg per S1: {avg_cands:.1f}")
    return merged

def compute_entity_f05(y_true_dict, y_pred_dict, all_s1_ids):
    f05_scores = []
    for s1_id in all_s1_ids:
        true_set = y_true_dict.get(s1_id, set())
        pred_set = y_pred_dict.get(s1_id, set())
        if len(true_set) == 0:
            f05_scores.append(1.0 if len(pred_set) == 0 else 0.0)
        else:
            tp = len(true_set & pred_set)
            if tp == 0:
                f05_scores.append(0.0)
            else:
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)
                p  = tp / (tp + fp)
                r  = tp / (tp + fn)
                f05_scores.append((1.25 * p * r) / (0.25 * p + r))
    return float(np.mean(f05_scores)) if f05_scores else 0.0

def train_model(X, y, pos_rate):
    if LGBM_AVAILABLE:
        spw = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            max_depth=-1, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, n_jobs=-1,
            random_state=42, verbose=-1
        )
        model.fit(X, y, callbacks=[lgb.log_evaluation(period=100)])
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
        model.fit(X, y)
    return model

def run_improved_pipeline():
    print("=" * 60)
    print("IMPROVED ENTITY RESOLUTION PIPELINE")
    print("=" * 60)

    print("\n[1] Loading training data...")
    t0 = time.time()
    s1_train = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source1.tsv'), sep='\t', dtype=str)
    s2_train = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source2.tsv'), sep='\t', dtype=str)
    s3_train = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source3.tsv'), sep='\t', dtype=str)
    gt       = pd.read_csv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'), sep='\t', dtype=str)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s | S1={len(s1_train)}, S2={len(s2_train)}, S3={len(s3_train)}")

    print("\n[2] Preprocessing...")
    t0 = time.time()
    for df in [s1_train, s2_train, s3_train]:
        df['norm_name'] = df['business_name'].apply(normalize_name)
        df['norm_addr'] = df['business_address'].apply(normalize_address)
    prep_time = time.time() - t0
    print(f"  Done in {prep_time:.1f}s")

    pool_train = pd.concat([s2_train, s3_train], ignore_index=True)

    gt_map = {}
    for row in gt.itertuples():
        matched = str(row.matched_entity_ids) if pd.notna(row.matched_entity_ids) else ''
        gt_map[row.source1_entity_id] = set(matched.split(',')) - {'', 'nan'} if matched else set()

    print("\n[3] Train/Val split (80/20)...")
    np.random.seed(42)
    s1_ids = s1_train['entity_id'].values.copy()
    np.random.shuffle(s1_ids)
    split_idx    = int(0.8 * len(s1_ids))
    train_s1_ids = set(s1_ids[:split_idx])
    val_s1_ids   = set(s1_ids[split_idx:])
    train_s1_df  = s1_train[s1_train['entity_id'].isin(train_s1_ids)].reset_index(drop=True)
    val_s1_df    = s1_train[s1_train['entity_id'].isin(val_s1_ids)].reset_index(drop=True)
    print(f"  Train S1: {len(train_s1_df)}, Val S1: {len(val_s1_df)}")

    pool_name_map = dict(zip(pool_train['entity_id'], pool_train['norm_name']))
    pool_addr_map = dict(zip(pool_train['entity_id'], pool_train['norm_addr']))
    s1_name_map   = dict(zip(s1_train['entity_id'],  s1_train['norm_name']))
    s1_addr_map   = dict(zip(s1_train['entity_id'],  s1_train['norm_addr']))

    print("\n[4] Multi-pass blocking (train split)...")
    train_candidates = get_multi_pass_candidates(train_s1_df, pool_train)

    print("\n[5] Extracting training features...")
    t0 = time.time()
    X_train, y_train = [], []
    for s1_id in train_s1_df['entity_id']:
        cands    = train_candidates.get(s1_id, set())
        true_set = gt_map.get(s1_id, set())
        s1_name  = s1_name_map.get(s1_id, '')
        s1_addr  = s1_addr_map.get(s1_id, '')
        for pool_id in cands:
            X_train.append(extract_features(
                s1_name, pool_name_map.get(pool_id, ''),
                s1_addr, pool_addr_map.get(pool_id, '')))
            y_train.append(1 if pool_id in true_set else 0)

    X_train  = np.array(X_train, dtype=np.float32)
    y_train  = np.array(y_train, dtype=np.int32)
    pos_rate = float(y_train.mean())
    feat_time = time.time() - t0
    print(f"  Pairs: {len(X_train)}, Pos rate: {pos_rate:.4f}  [{feat_time:.1f}s]")

    print("\n[6] Training LightGBM (val model)...")
    t0 = time.time()
    model = train_model(X_train, y_train, pos_rate)
    model_time = time.time() - t0
    print(f"  Done in {model_time:.1f}s")

    if LGBM_AVAILABLE:
        print("  Feature importances:")
        for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1]):
            print(f"    {name}: {imp}")

    del X_train, y_train
    gc.collect()

    print("\n[7] Multi-pass blocking (val split)...")
    val_candidates = get_multi_pass_candidates(val_s1_df, pool_train)

    true_matches_total, captured_total, zero_cand_count = 0, 0, 0
    for s1_id in val_s1_df['entity_id']:
        cand_ids = val_candidates.get(s1_id, set())
        if not cand_ids:
            zero_cand_count += 1
        true_set = gt_map.get(s1_id, set())
        true_matches_total += len(true_set)
        captured_total     += len(true_set & cand_ids)
    cand_recall = captured_total / true_matches_total if true_matches_total > 0 else 0
    print(f"  Candidate recall: {cand_recall:.4f} ({captured_total}/{true_matches_total})")
    n_val = max(len(val_s1_df), 1)
    print(f"  Zero-candidate rate: {zero_cand_count}/{len(val_s1_df)} ({zero_cand_count/n_val:.4f})")

    print("  Extracting val features...")
    t0 = time.time()
    X_val, val_pairs = [], []
    for s1_id in val_s1_df['entity_id']:
        cands   = val_candidates.get(s1_id, set())
        s1_name = s1_name_map.get(s1_id, '')
        s1_addr = s1_addr_map.get(s1_id, '')
        for pool_id in cands:
            X_val.append(extract_features(
                s1_name, pool_name_map.get(pool_id, ''),
                s1_addr, pool_addr_map.get(pool_id, '')))
            val_pairs.append((s1_id, pool_id))

    X_val     = np.array(X_val, dtype=np.float32)
    val_probs = model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([])
    val_feat_time = time.time() - t0
    print(f"  Val pairs: {len(X_val)}  [{val_feat_time:.1f}s]")

    print("\n  Threshold sweep (entity-level F0.5):")
    val_true_dict = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_df['entity_id']}
    all_val_ids   = list(val_s1_df['entity_id'])
    best_f05, best_thresh = 0.0, 0.5
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    for t in thresholds:
        val_pred_dict = {}
        if len(val_probs) > 0:
            preds = (val_probs >= t)
            for i, (s1_id, pool_id) in enumerate(val_pairs):
                if preds[i]:
                    val_pred_dict.setdefault(s1_id, set()).add(pool_id)
        f05 = compute_entity_f05(val_true_dict, val_pred_dict, all_val_ids)
        best_marker = " <<< BEST" if f05 > best_f05 else ""
        print(f"    t={t:.2f} -> F0.5={f05:.4f}{best_marker}")
        if f05 > best_f05:
            best_f05, best_thresh = f05, t

    print(f"\n  Best threshold: {best_thresh:.2f}, Best val F0.5: {best_f05:.4f}")

    del X_val, val_probs, val_pairs
    gc.collect()

    print("\n[8] Re-training on FULL training data...")
    print("  Multi-pass blocking on full train...")
    full_candidates = get_multi_pass_candidates(s1_train, pool_train)

    print("  Extracting full train features...")
    t0 = time.time()
    X_full, y_full = [], []
    for s1_id in s1_train['entity_id']:
        cands    = full_candidates.get(s1_id, set())
        true_set = gt_map.get(s1_id, set())
        s1_name  = s1_name_map.get(s1_id, '')
        s1_addr  = s1_addr_map.get(s1_id, '')
        for pool_id in cands:
            X_full.append(extract_features(
                s1_name, pool_name_map.get(pool_id, ''),
                s1_addr, pool_addr_map.get(pool_id, '')))
            y_full.append(1 if pool_id in true_set else 0)

    X_full = np.array(X_full, dtype=np.float32)
    y_full = np.array(y_full, dtype=np.int32)
    full_feat_time = time.time() - t0
    print(f"  Full pairs: {len(X_full)}, Pos rate: {y_full.mean():.4f}  [{full_feat_time:.1f}s]")

    print("  Training final model...")
    t0 = time.time()
    final_model = train_model(X_full, y_full, float(y_full.mean()))
    final_model_time = time.time() - t0
    print(f"  Final model trained in {final_model_time:.1f}s")

    del X_full, y_full
    gc.collect()

    print("\n[9] Loading and preprocessing test data...")
    t0 = time.time()
    s1_test = pd.read_csv(os.path.join(TEST_DIR, 'test_source1.tsv'), sep='\t', dtype=str)
    s2_test = pd.read_csv(os.path.join(TEST_DIR, 'test_source2.tsv'), sep='\t', dtype=str)
    s3_test = pd.read_csv(os.path.join(TEST_DIR, 'test_source3.tsv'), sep='\t', dtype=str)
    for df in [s1_test, s2_test, s3_test]:
        df['norm_name'] = df['business_name'].apply(normalize_name)
        df['norm_addr'] = df['business_address'].apply(normalize_address)
    test_load_time = time.time() - t0
    print(f"  Done in {test_load_time:.1f}s | S1={len(s1_test)}, S2={len(s2_test)}, S3={len(s3_test)}")

    pool_test     = pd.concat([s2_test, s3_test], ignore_index=True)
    test_name_map = dict(zip(pool_test['entity_id'], pool_test['norm_name']))
    test_addr_map = dict(zip(pool_test['entity_id'], pool_test['norm_addr']))
    s1t_name_map  = dict(zip(s1_test['entity_id'],  s1_test['norm_name']))
    s1t_addr_map  = dict(zip(s1_test['entity_id'],  s1_test['norm_addr']))

    print("\n[10] Multi-pass blocking on test set...")
    test_candidates = get_multi_pass_candidates(s1_test, pool_test)

    print("\n[11] Scoring test candidates...")
    t0 = time.time()
    test_pairs = []
    X_test_list = []
    for s1_id in s1_test['entity_id']:
        cands   = test_candidates.get(s1_id, set())
        s1_name = s1t_name_map.get(s1_id, '')
        s1_addr = s1t_addr_map.get(s1_id, '')
        for pool_id in cands:
            X_test_list.append(extract_features(
                s1_name, test_name_map.get(pool_id, ''),
                s1_addr, test_addr_map.get(pool_id, '')))
            test_pairs.append((s1_id, pool_id))

    score_time = time.time() - t0
    print(f"  Test pairs: {len(X_test_list)}  [{score_time:.1f}s]")
    if len(X_test_list) > 0:
        X_test_arr    = np.array(X_test_list, dtype=np.float32)
        test_probs    = final_model.predict_proba(X_test_arr)[:, 1]
        test_is_match = (test_probs >= best_thresh)
    else:
        test_is_match = np.array([], dtype=bool)

    print("\n[12] Writing output files...")
    all_candidates = {}
    all_matches    = {}
    for i, (s1_id, pool_id) in enumerate(test_pairs):
        all_candidates.setdefault(s1_id, []).append(pool_id)
        if len(test_is_match) > 0 and test_is_match[i]:
            all_matches.setdefault(s1_id, []).append(pool_id)

    cand_path  = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    match_path = os.path.join(OUTPUT_DIR, 'matching_results.tsv')

    with open(cand_path, 'w', encoding='utf-8') as cf, \
         open(match_path, 'w', encoding='utf-8') as mf:
        cf.write("source1_entity_id\tcandidate_entity_ids\n")
        mf.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_test['entity_id']:
            c_list = list(dict.fromkeys(all_candidates.get(s1_id, [])))
            m_list = list(dict.fromkeys(all_matches.get(s1_id, [])))
            cf.write(f"{s1_id}\t{','.join(c_list)}\n")
            mf.write(f"{s1_id}\t{','.join(m_list)}\n")

    matched_count = sum(1 for v in all_matches.values() if v)
    print(f"  Wrote {len(s1_test)} rows | With matches: {matched_count} | Singletons: {len(s1_test)-matched_count}")
    print(f"  Output: {match_path}")
    print(f"  Output: {cand_path}")

if __name__ == "__main__":
    start_time = time.time()
    run_improved_pipeline()
    total = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Total time: {total:.1f}s ({total/60:.1f} min)")
    print("DONE!")
