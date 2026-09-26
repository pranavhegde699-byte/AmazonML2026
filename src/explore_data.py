import pandas as pd
import json
import os
import sys

def profile_dataset():
    data_dir = 'dataset'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    stats = {}

    # Load Ground Truth
    gt_path = os.path.join(train_dir, 'train_ground_truth.tsv')
    if os.path.exists(gt_path):
        gt_df = pd.read_csv(gt_path, sep='\t', dtype=str)
        # handle nan matches
        gt_df['matched_entity_ids'] = gt_df['matched_entity_ids'].fillna('')
        
        matches_per_s1 = gt_df['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0)
        
        stats['ground_truth'] = {
            'total_rows': len(gt_df),
            'singletons_zero_match': int((matches_per_s1 == 0).sum()),
            'one_match': int((matches_per_s1 == 1).sum()),
            'multiple_matches': int((matches_per_s1 > 1).sum()),
            'total_positive_matches': int(matches_per_s1.sum()),
            'avg_matches_per_s1': float(matches_per_s1.mean())
        }
    
    # Load and profile train datasets
    for src in ['train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv']:
        path = os.path.join(train_dir, src)
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t', dtype=str)
            
            # Basic stats
            source_stats = {
                'total_rows': len(df),
                'columns': list(df.columns),
                'missing_values': df.isna().sum().to_dict(),
                'empty_strings': (df == '').sum().to_dict(),
                'duplicate_ids': int(df['entity_id'].duplicated().sum()) if 'entity_id' in df.columns else 0,
                'country_distribution': df['country'].value_counts(dropna=False).to_dict() if 'country' in df.columns else {},
            }

            # Text lengths
            if 'business_name' in df.columns:
                lens = df['business_name'].dropna().apply(len)
                source_stats['avg_name_len'] = float(lens.mean()) if len(lens) > 0 else 0
            
            if 'business_address' in df.columns:
                lens = df['business_address'].dropna().apply(len)
                source_stats['avg_address_len'] = float(lens.mean()) if len(lens) > 0 else 0

            stats[src] = source_stats

    # Load and profile test datasets
    for src in ['test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv']:
        path = os.path.join(test_dir, src)
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t', dtype=str)
            stats[src] = {
                'total_rows': len(df),
                'country_distribution': df['country'].value_counts(dropna=False).to_dict() if 'country' in df.columns else {},
            }

    with open('docs/data_profile.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("Data profiling complete. Stats saved to docs/data_profile.json")

if __name__ == "__main__":
    profile_dataset()
