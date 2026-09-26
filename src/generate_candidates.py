import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import namedtuple

Row = namedtuple('Row', ['entity_id', 'norm_name', 'norm_addr'])

def get_candidates_tfidf(s1_df, pool_df, top_k=30, sim_cutoff=0.3):
    """
    Returns a dict: s1_entity_id -> list of candidate tuples (from pool_df)
    """
    candidates = {}
    s1_df = s1_df.copy()
    pool_df = pool_df.copy()
    
    s1_df['norm_name'] = s1_df['norm_name'].fillna('')
    pool_df['norm_name'] = pool_df['norm_name'].fillna('')
    s1_df['country'] = s1_df['country'].fillna('')
    pool_df['country'] = pool_df['country'].fillna('')
    
    unique_countries = s1_df['country'].unique()
    
    for country in unique_countries:
        s1_c = s1_df[s1_df['country'] == country]
        pool_c = pool_df[pool_df['country'] == country]
        
        if len(pool_c) == 0 or len(s1_c) == 0:
            for s1_id in s1_c['entity_id']:
                candidates[s1_id] = []
            continue
            
        # We include some address text in TFIDF to disambiguate identical names? 
        # Let's just use name first, as it's the strongest signal and limits noise.
        pool_docs = pool_c['norm_name']
        s1_docs = s1_c['norm_name']
        
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=5, max_features=100000, max_df=0.3)
        pool_tfidf = vectorizer.fit_transform(pool_docs)
        s1_tfidf = vectorizer.transform(s1_docs)
        
        pool_c_list = [Row(*x) for x in pool_c[['entity_id', 'norm_name', 'norm_addr']].values]
        s1_ids = s1_c['entity_id'].values
        
        batch_size = 100
        for i in range(0, s1_tfidf.shape[0], batch_size):
            s1_batch = s1_tfidf[i:i+batch_size]
            sims = s1_batch.dot(pool_tfidf.T) # (batch, pool_size)
            
            # Filter out below threshold FIRST (fast in CSR)
            sims.data[sims.data < sim_cutoff] = 0
            sims.eliminate_zeros()
            
            for j in range(sims.shape[0]):
                row_start = sims.indptr[j]
                row_end = sims.indptr[j+1]
                data = sims.data[row_start:row_end]
                indices = sims.indices[row_start:row_end]
                
                cands_for_row = []
                if len(data) > 0:
                    if len(data) > top_k:
                        k_idx = np.argpartition(data, -top_k)[-top_k:]
                        top_k_data = data[k_idx]
                        top_k_indices = indices[k_idx]
                        
                        sort_idx = np.argsort(top_k_data)[::-1]
                        top_k_data = top_k_data[sort_idx]
                        top_k_indices = top_k_indices[sort_idx]
                    else:
                        sort_idx = np.argsort(data)[::-1]
                        top_k_data = data[sort_idx]
                        top_k_indices = indices[sort_idx]
                        
                    for sim, idx in zip(top_k_data, top_k_indices):
                        cands_for_row.append(pool_c_list[idx])
                            
                s1_id = s1_ids[i + j]
                candidates[s1_id] = cands_for_row
                
    return candidates
