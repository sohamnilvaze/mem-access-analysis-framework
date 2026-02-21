'''
Window type: fixed number of memory accesses (5000 per window)

Extracted Feature Categories:

1) Delta-based features (delta, abs_delta):
   - mean_delta
   - std_delta
   - mean_abs_delta
   - std_abs_delta
   - max_abs_delta
   - delta_entropy
   - dominant_stride_ratio
   - max_consecutive_same_delta
   - stride_change_rate

2) Direction-based features (direction):
   - forward_ratio
   - backward_ratio
   - zero_ratio

3) Cache-based features (cache_line, cl_delta):
   - unique_cache_lines
   - mean_cl_delta
   - std_cl_delta
   - cache_line_reuse_ratio

4) Page-based features (page):
   - unique_pages
   - page_reuse_ratio

5) Access-type ratios:
   - read_ratio
   - write_ratio

6) Instruction diversity:
   - unique_ip_count
'''
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import entropy

WINDOW_SIZE = 20

def compute_entropy(series):
    counts = series.value_counts()
    probs = counts / counts.sum()
    return entropy(probs)

def max_consecutive_run(arr):
    max_run = 1
    current_run = 1
    
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
            
    return max_run

def extract_window_features(file_name):
    df = pd.read_csv(f"/home/soham/pin/mem_access_analysis_framework/traces_csv2/{file_name}.csv")
    features = []
    
    num_windows = len(df) // WINDOW_SIZE
    
    for i in range(num_windows):
        start = i * WINDOW_SIZE
        end = (i + 1) * WINDOW_SIZE
        window = df.iloc[start:end]
        
        feature = {}
        
        # ----- DELTA FEATURES -----
        deltas = window['delta'].dropna()
        abs_deltas = window['abs_delta'].dropna()
        
        feature['mean_delta'] = deltas.mean()
        feature['std_delta'] = deltas.std()
        feature['mean_abs_delta'] = abs_deltas.mean()
        feature['std_abs_delta'] = abs_deltas.std()
        feature['max_abs_delta'] = abs_deltas.max()
        
        # NEW: entropy of delta distribution
        feature['delta_entropy'] = compute_entropy(deltas)
        
        # NEW: dominant stride ratio
        stride_counts = deltas.value_counts()
        feature['dominant_stride_ratio'] = stride_counts.max() / len(deltas)
        
        # NEW: max consecutive same delta
        feature['max_consecutive_same_delta'] = max_consecutive_run(deltas.values)
        
        # NEW: stride change rate
        stride_changes = np.sum(deltas.values[1:] != deltas.values[:-1])
        feature['stride_change_rate'] = stride_changes / len(deltas)
        
        # ----- DIRECTION FEATURES -----
        feature['forward_ratio'] = (window['direction'] == 1).mean()
        feature['backward_ratio'] = (window['direction'] == -1).mean()
        feature['zero_ratio'] = (window['direction'] == 0).mean()
        
        # ----- CACHE FEATURES -----
        unique_cache_lines = window['cache_line'].nunique()
        feature['unique_cache_lines'] = unique_cache_lines
        feature['mean_cl_delta'] = window['cl_delta'].mean()
        feature['std_cl_delta'] = window['cl_delta'].std()
        
        # NEW: cache line reuse ratio
        feature['cache_line_reuse_ratio'] = 1 - (unique_cache_lines / WINDOW_SIZE)
        
        # ----- PAGE FEATURES -----
        unique_pages = window['page'].nunique()
        feature['unique_pages'] = unique_pages
        
        # NEW: page reuse ratio
        feature['page_reuse_ratio'] = 1 - (unique_pages / WINDOW_SIZE)
        
        # ----- ACCESS TYPE -----
        feature['read_ratio'] = (window['access_type'] == 'R').mean()
        feature['write_ratio'] = (window['access_type'] == 'W').mean()
        
        # ----- IP DIVERSITY -----
        feature['unique_ip_count'] = window['ip'].nunique()
        
        features.append(feature)
    
    df_fin = pd.DataFrame(features)
    df_fin.to_csv(
        f"/home/soham/pin/mem_access_analysis_framework/traces_csv3/{file_name}.csv",
        index=False
    )

folder = Path('/home/soham/pin/mem_access_analysis_framework/traces_csv2')
file_paths = [str(p.absolute()) for p in folder.rglob('*') if p.is_file()]

for path in file_paths:
    parts = path.split('/')
    file_name = parts[-1].removesuffix(".csv")
    print(f"For {file_name}")
    extract_window_features(file_name)