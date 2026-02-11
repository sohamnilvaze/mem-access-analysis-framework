import pandas as pd

def parse_trace(file_path,output_path):
    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            thread_id = int(parts[0])
            ip = int(parts[1], 16)
            mem_addr = int(parts[2], 16)
            access_type = int(parts[3])  # 0=read, 1=write
            size = int(parts[4])
            
            data.append([thread_id, ip, mem_addr, access_type, size])
    
    df = pd.DataFrame(data, columns=[
        "thread_id", "ip", "mem_addr", "access_type", "size"
    ])
    df.to_csv(output_path,index=False)
    return df

from pathlib import Path

# Replace '.' with your folder path
folder = Path('/home/soham/pin/mem_access_analysis_framework/traces2')

# Recursively find all files
file_paths = [str(p.absolute()) for p in folder.rglob('*') if p.is_file()]

for path in file_paths:
    print(f"For {path}")
    parts = path.split('/')
    file_name = parts[-1].removesuffix(".txt")
    if file_name != "commands":
        csv_name = file_name + ".csv"
        output_path = f"/home/soham/pin/mem_access_analysis_framework/traces_csv/{csv_name}"
        parse_trace(path,output_path)

