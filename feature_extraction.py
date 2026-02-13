'''
Window type is fixed number of memory accesses 
window type is : 5000 accesses per window
what do these do?
1) df["mem_addr"] = df["mem_addr"].apply(lambda x: int(x, 16))
2) strides = window["mem_addr"].diff().dropna()
3) explain what is being done in compute_spatial_locality(), compute_temporal_locality() 

delta-based features (delta, abs_delta) -> mean, std, mean_abs, std_abs, max_abs, min_abs
direction-based features (direction) -> percent_forward, percent_backward, percent_zero_deltas
cache-based features (cache_line, cl_delta) -> unique_cache_lines, mean_cl_delta, std_cl_delta, cache_line_reuse_ratio
page-based features (page) -> unique_pages, page_reuse_ratio
acess-based ratio (access_type) -> read_ratio, write_ratio
'''
from pathlib import Path
import pandas as pd
import numpy as np

WINDOW_SIZE = 500

def extract_window_features(file_name):
    df = pd.read_csv(f"/home/soham/pin/mem_access_analysis_framework/traces_csv2/{file_name}.csv")
    features = []
    
    num_windows = len(df) // WINDOW_SIZE
    
    for i in range(num_windows):
        start = i * WINDOW_SIZE
        end = (i + 1) * WINDOW_SIZE
        window = df.iloc[start:end]
        
        feature = {}
        
        # Delta stats
        feature['mean_delta'] = window['delta'].mean()
        feature['std_delta'] = window['delta'].std()
        feature['mean_abs_delta'] = window['abs_delta'].mean()
        feature['std_abs_delta'] = window['abs_delta'].std()
        feature['max_abs_delta'] = window['abs_delta'].max()
        
        # Direction ratios
        feature['forward_ratio'] = (window['direction'] == 1).mean()
        feature['backward_ratio'] = (window['direction'] == -1).mean()
        feature['zero_ratio'] = (window['direction'] == 0).mean()
        
        # Cache line features
        feature['unique_cache_lines'] = window['cache_line'].nunique()
        feature['mean_cl_delta'] = window['cl_delta'].mean()
        feature['std_cl_delta'] = window['cl_delta'].std()
        
        # Page locality
        feature['unique_pages'] = window['page'].nunique()
        
        # Access type
        feature['read_ratio'] = (window['access_type'] == 'R').mean()
        feature['write_ratio'] = (window['access_type'] == 'W').mean()
        
        # IP diversity
        feature['unique_ip_count'] = window['ip'].nunique()
        
        features.append(feature)
    
    df_fin =  pd.DataFrame(features)
    df_fin.to_csv(f"/home/soham/pin/mem_access_analysis_framework/traces_csv3/{file_name}.csv",index=False)

folder = Path('/home/soham/pin/mem_access_analysis_framework/traces_csv2')
file_paths = [str(p.absolute()) for p in folder.rglob('*') if p.is_file()]

for path in file_paths:
    parts = path.split('/')
    file_name = parts[-1].removesuffix(".csv")
    print(f"For {file_name}")
    extract_window_features(file_name)

