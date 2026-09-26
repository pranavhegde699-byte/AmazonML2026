import os
import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import gc

def test_tfidf():
    print("Loading data...")
    TRAIN_DIR = os.path.join("dataset", "train")
    
    # Load sample S1, S2, S3
    s1 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source1.tsv'), sep='\t', dtype=str, nrows=100000)
    s2 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source2.tsv'), sep='\t', dtype=str, nrows=100000)
    s3 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source3.tsv'), sep='\t', dtype=str, nrows=100000)
    
    pool = pd.concat([s2, s3], ignore_index=True)
    
    print("TF-IDF Vectorization...")
    t0 = time.time()
    s1['text'] = s1['business_name'].fillna('') + " " + s1['business_address'].fillna('')
    pool['text'] = pool['business_name'].fillna('') + " " + pool['business_address'].fillna('')
    
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=100000)
    
    # Fit on pool, transform both
    pool_tfidf = vectorizer.fit_transform(pool['text'])
    s1_tfidf = vectorizer.transform(s1['text'])
    print(f"Vectorization done in {time.time()-t0:.2f}s")
    
    print("Computing dot products in chunks...")
    t0 = time.time()
    
    batch_size = 10000
    top_k = 20
    
    all_s1_indices = []
    all_pool_indices = []
    all_scores = []
    
    # Pool transpose for fast dot product: s1_batch @ pool_tfidf.T
    pool_tfidf_T = pool_tfidf.T.tocsr()
    
    for start_idx in range(0, s1_tfidf.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, s1_tfidf.shape[0])
        s1_batch = s1_tfidf[start_idx:end_idx]
        
        # Dot product
        sim_matrix = s1_batch.dot(pool_tfidf_T)
        
        # Get top K per row
        for i in range(sim_matrix.shape[0]):
            row = sim_matrix.getrow(i)
            if row.nnz == 0:
                continue
            
            # Get indices and data
            data = row.data
            indices = row.indices
            
            # Sort by highest score
            if len(data) > top_k:
                top_idx = np.argpartition(data, -top_k)[-top_k:]
                top_indices = indices[top_idx]
                top_data = data[top_idx]
            else:
                top_indices = indices
                top_data = data
            
            # Optional: sort them descending
            sort_order = np.argsort(-top_data)
            top_indices = top_indices[sort_order]
            top_data = top_data[sort_order]
            
            # Save
            s1_real_idx = start_idx + i
            all_s1_indices.extend([s1_real_idx] * len(top_indices))
            all_pool_indices.extend(top_indices)
            all_scores.extend(top_data)
            
        if (start_idx // batch_size) % 2 == 0:
            print(f" Processed {end_idx} rows...")

    print(f"Dot product & Top K done in {time.time()-t0:.2f}s")
    print(f"Found {len(all_scores)} candidate pairs.")

if __name__ == '__main__':
    test_tfidf()
