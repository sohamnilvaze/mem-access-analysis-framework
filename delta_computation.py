'''
Deltas are computed seperately per thread in execution order
'''

from pathlib import Path
import pandas as pd

def compute_deltas(file_name):
    df = pd.read_csv(f"/home/soham/pin/mem_access_analysis_framework/traces_csv/{file_name}")
    df["delta"] = df.groupby("thread_id")["mem_addr"].diff()
    df = df.dropna(subset=["delta"])
    df["abs_delta"] = df["delta"].abs()
    df["direction"] = df["delta"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df["cache_line"] = df["mem_addr"] // 64  # modern cpus have 64 byte cache line -> movement across cache blocks, momory locality behaviour 
    df["cl_delta"] = df.groupby("thread_id")["cache_line"].diff()
    df["page"] = df["mem_addr"] // 4096
    df.to_csv(f"/home/soham/pin/mem_access_analysis_framework/traces_csv2/{file_name}",index=False)

folder = Path('/home/soham/pin/mem_access_analysis_framework/traces_csv')
file_paths = [str(p.absolute()) for p in folder.rglob('*') if p.is_file()]

for path in file_paths:
    parts = path.split('/')
    file_name = parts[-1]
    print(f"For {file_name}")
    compute_deltas(file_name)
    






