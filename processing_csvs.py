import pandas as pd
import os

#1- block access yes
#2- sequantial access yes
#3- strided access yes
#4- column major yes
#5- row major yes
#6- indirect access yes
#7- reuse no #################
#8- reverse linked list no ############
#9- random access yes
#10- recursive no ###############
#11- linked list yes

import pandas as pd
import os

BASE_IN = "traces_csv3"
BASE_OUT = "traces_csv4"

os.makedirs(BASE_OUT, exist_ok=True)


def safe_read_csv(path):
    """Safely read CSV and handle empty/missing files."""
    if not os.path.exists(path):
        print(f"Skipping (not found): {path}")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"Skipping (empty df): {path}")
            return None
        return df
    except pd.errors.EmptyDataError:
        print(f"Skipping (completely empty file): {path}")
        return None
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def process_group(file_list, target_label, prefix):
    """Add target column safely."""
    i = 1
    valid_dfs = []

    for file in file_list:
        path = os.path.join(BASE_IN, file)
        df = safe_read_csv(path)

        if df is not None and "Target" not in df.columns:
            df["Target"] = target_label
            df["Fine_grained_Target"] = f"{prefix}_{i}"
            df.to_csv(path, index=False)
            print(f"Added target for {file}")
            valid_dfs.append(df)
            i += 1
        elif df is not None:
            valid_dfs.append(df)
    # print(f"Valud csvs:- {valid_dfs}")
    return valid_dfs


def merge_and_save(dfs, output_name):
    if len(dfs) == 0:
        print(f"No valid files for {output_name}")
        return None

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(os.path.join(BASE_OUT, output_name), index=False)
    print(f"Saved {output_name} with shape {merged.shape}")
    return merged


def main():

    all_dfs = []

    groups = {
        "ba": (["ba1.csv","ba2.csv","ba4.csv","ba8.csv","ba16.csv","ba32.csv","ba64.csv","ba128.csv"], 1),
        "seq": (["seqif400.csv","seqib400.csv","seqdf400.csv","seqdb400.csv",
                 "seqif200.csv","seqib200.csv","seqdf200.csv","seqdb200.csv",
                 "seqif100.csv","seqib100.csv","seqdf100.csv","seqdb100.csv"], 2),
        "std": (["std1.csv","std2.csv","std4.csv","std8.csv","std16.csv","std32.csv","std64.csv","std128.csv"], 3),
        "mc": (["mc_1_100.csv","mc_10_90.csv","mc_20_80.csv","mc_30_70.csv","mc_40_60.csv",
                "mc_50_50.csv","mc_60_40.csv","mc_70_30.csv","mc_80_20.csv","mc_90_10.csv","mc_100_1.csv"], 4),
        "mr": (["mr_1_100.csv","mr_10_90.csv","mr_20_80.csv","mr_30_70.csv","mr_40_60.csv",
                "mr_50_50.csv","mr_60_40.csv","mr_70_30.csv","mr_80_20.csv","mr_90_10.csv","mr_100_1.csv"], 5),
        "ia": (["ia3.csv","i10.csv","i50.csv","i100.csv","i250.csv","i500.csv","i750.csv","i1024.csv"], 6),
        "ra": (["ra10.csv","ra50.csv","ra100.csv","ra250.csv","ra500.csv","ra750.csv","ra1000.csv"], 7),
        "ll": (["ll2.csv","ll4.csv","ll8.csv","ll16.csv","ll32.csv","ll64.csv","ll128.csv","ll256.csv","ll512.csv","ll1024.csv"], 8),
    }

    for prefix, (files, label) in groups.items():
        dfs = process_group(files, label, prefix)
        merged = merge_and_save(dfs, f"{prefix}_merged.csv")
        if merged is not None:
            all_dfs.append(merged)

    # Final global merge
    if len(all_dfs) > 0:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(os.path.join(BASE_OUT, "merged.csv"), index=False)
        print("Final merged shape:", final_df.shape)
    else:
        print("No data to merge.")


if __name__ == "__main__":
    main()