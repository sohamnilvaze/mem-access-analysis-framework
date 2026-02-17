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

def add_target_col():
    i = 1
    for dfs in ["traces_csv3/ba1.csv","traces_csv3/ba2.csv","traces_csv3/ba4.csv","traces_csv3/ba8.csv","traces_csv3/ba16.csv","traces_csv3/ba32.csv","traces_csv3/ba64.csv","traces_csv3/ba128.csv"] : #8
        df = pd.read_csv(dfs)
        if "Target" not in df.columns:
            df["Target"] = 1
            df["Fine_grained_Target"] = f"ba_{i}"
            i = i + 1
            df.to_csv(dfs,index=False)
            print(f"Added for {dfs}")
    i = 1
    for dfs in ["traces_csv3/seqif400.csv","traces_csv3/seqib400.csv","traces_csv3/seqdf400.csv","traces_csv3/seqdb400.csv","traces_csv3/seqif200.csv","traces_csv3/seqib200.csv","traces_csv3/seqdf200.csv","traces_csv3/seqdb200.csv","traces_csv3/seqif100.csv","traces_csv3/seqib100.csv","traces_csv3/seqdf100.csv","traces_csv3/seqdb100.csv"] : #12
        df = pd.read_csv(dfs)
        if "Target" not in df.columns:
            df["Target"] = 2
            df["Fine_grained_Target"] = f"seq_{i}"
            i = i + 1
            df.to_csv(dfs,index=False)
            print(f"Added for {dfs}")   
    i = 1
    for dfs in ["traces_csv3/std1.csv","traces_csv3/std2.csv","traces_csv3/std4.csv","traces_csv3/std8.csv","traces_csv3/std16.csv","traces_csv3/std32.csv","traces_csv3/std64.csv","traces_csv3/std128.csv"] : #8
        df = pd.read_csv(dfs)
        if "Target" not in df.columns:
            df["Target"] = 3
            df["Fine_grained_Target"] = f"std_{i}"
            i = i + 1
            df.to_csv(dfs,index=False)
            print(f"Added for {dfs}")
    i = 1
    for dfs in ["traces_csv3/mc_1_100.csv","traces_csv3/mc_10_90.csv","traces_csv3/mc_20_80.csv","traces_csv3/mc_30_70.csv","traces_csv3/mc_40_60.csv","traces_csv3/mc_50_50.csv","traces_csv3/mc_60_40.csv","traces_csv3/mc_70_30.csv","traces_csv3/mc_80_20.csv","traces_csv3/mc_90_10.csv","traces_csv3/mc_100_1.csv"] : #11
        df = pd.read_csv(dfs)
        if "Target" not in df.columns:
            df["Target"] = 4
            df["Fine_grained_Target"] = f"mc_{i}"
            i = i + 1
            df.to_csv(dfs,index=False)
            print(f"Added for {dfs}")
    i = 1
    for dfs in ["traces_csv3/mr_1_100.csv","traces_csv3/mr_10_90.csv","traces_csv3/mr_20_80.csv","traces_csv3/mr_30_70.csv","traces_csv3/mr_40_60.csv","traces_csv3/mr_50_50.csv","traces_csv3/mr_60_40.csv","traces_csv3/mr_70_30.csv","traces_csv3/mr_80_20.csv","traces_csv3/mr_90_10.csv","traces_csv3/mr_100_1.csv"] : #11
        df = pd.read_csv(dfs)
        if "Target" not in df.columns:
            df["Target"] = 5
            df["Fine_grained_Target"] = f"mr_{i}"
            i = i + 1
            df.to_csv(dfs,index=False)
            print(f"Added for {dfs}")
    i = 1
    for dfs in ["traces_csv3/i3.csv","traces_csv3/i10.csv","traces_csv3/i50.csv","traces_csv3/i100.csv","traces_csv3/i250.csv","traces_csv3/i500.csv","traces_csv3/i750.csv","traces_csv3/i1024.csv"]: #8
        df = pd.read_csv(dfs)
        if "Target" not in df.columns:
            df["Target"] = 6
            df["Fine_grained_Target"] = f"ia_{i}"
            i = i + 1
            df.to_csv(dfs,index=False)
            print(f"Added for {dfs}")   
    # i = 1
    # for dfs in["traces_csv3/reu_10.csv","traces_csv3/reu_20.csv","traces_csv3/reu_30.csv","traces_csv3/reu_40.csv","traces_csv3/reu_50.csv","traces_csv3/reu_60.csv","traces_csv3/reu_70.csv","traces_csv3/reu_80.csv","traces_csv3/reu_90.csv"]: #9
    #     df = pd.read_csv(dfs)
    #     df["Target"] = 7
    #     df["Fine_grained_Target"] = i
    #     i = i + 1
    #     df.to_csv(dfs,index=False)
    #     print(f"Added for {dfs}")
    # i = 1
    # for dfs in ["traces_csv3/rll2.csv","traces_csv3/rll4.csv","traces_csv3/rll8.csv","traces_csv3/rll16.csv","traces_csv3/rll32.csv","traces_csv3/rll64.csv","traces_csv3/rll128.csv","traces_csv3/rll256.csv","traces_csv3/rll512.csv","traces_csv3/rll1024.csv"]: #10
    #     df = pd.read_csv(dfs)
    #     df["Target"] = 8
    #     df["Fine_grained_Target"] = i
    #     i = i + 1
    #     df.to_csv(dfs,index=False)
    #     print(f"Added for {dfs}")
    i = 1
    for dfs in ["traces_csv3/ra10.csv","traces_csv3/ra50.csv","traces_csv3/ra100.csv","traces_csv3/ra250.csv","traces_csv3/ra500.csv","traces_csv3/ra750.csv","traces_csv3/ra1000.csv"]: #7
        df = pd.read_csv(dfs)
        if "Target" not in df.columns:
            df["Target"] = 7
            df["Fine_grained_Target"] = f"ra_{i}"
            i = i + 1
            df.to_csv(dfs,index=False)
            print(f"Added for {dfs}")   
    # i = 1
    # for dfs in ["traces_csv3/rec_10.csv","traces_csv3/rec_50.csv","traces_csv3/rec_100.csv","traces_csv3/rec_250.csv","traces_csv3/rec_500.csv","traces_csv3/rec_750.csv","traces_csv3/rec_1000.csv"]: #7
    #     df = pd.read_csv(dfs)
    #     df["Target"] = 10
    #     df["Fine_grained_Target"] = i
    #     i = i + 1
    #     df.to_csv(dfs,index=False)
    #     print(f"Added for {dfs}")
    i = 1
    for dfs in ["traces_csv3/ll2.csv","traces_csv3/ll4.csv","traces_csv3/ll8.csv","traces_csv3/ll16.csv","traces_csv3/ll32.csv","traces_csv3/ll64.csv","traces_csv3/ll128.csv","traces_csv3/ll256.csv","traces_csv3/ll512.csv","traces_csv3/ll1024.csv"]: #10
        df = pd.read_csv(dfs)
        if "Target" not in df.columns:
            df["Target"] = 8
            df["Fine_grained_Target"] = f"ll_{i}"
            i = i + 1
            df.to_csv(dfs,index=False)
            print(f"Added for {dfs}")



        

def merge_dfs():
    #1
    df1 = pd.read_csv("traces_csv3/ba1.csv")
    df2 = pd.read_csv("traces_csv3/ba2.csv")
    df3 = pd.read_csv("traces_csv3/ba4.csv")
    df4 = pd.read_csv("traces_csv3/ba8.csv")
    df5 = pd.read_csv("traces_csv3/ba16.csv")
    df6 = pd.read_csv("traces_csv3/ba32.csv")
    df7 = pd.read_csv("traces_csv3/ba64.csv")
    df8 = pd.read_csv("traces_csv3/ba128.csv")

    ba_dfs = [df1,df2,df3,df4,df5,df6,df7,df8]
    ba_merged_df = pd.concat(ba_dfs, axis=0,ignore_index=True)
    ba_merged_df.to_csv("traces_csv4/ba_merged.csv",index=False)


    #2
    df9 = pd.read_csv("traces_csv3/seqif400.csv")
    df10 = pd.read_csv("traces_csv3/seqib400.csv")
    df11 = pd.read_csv("traces_csv3/seqdf400.csv")
    df12 = pd.read_csv("traces_csv3/seqdb400.csv")
    df13 = pd.read_csv("traces_csv3/seqif200.csv")
    df14 = pd.read_csv("traces_csv3/seqib200.csv")
    df15 = pd.read_csv("traces_csv3/seqdf200.csv")
    df16 = pd.read_csv("traces_csv3/seqdb200.csv")
    df17 = pd.read_csv("traces_csv3/seqif100.csv")
    df18 = pd.read_csv("traces_csv3/seqib100.csv")
    df19 = pd.read_csv("traces_csv3/seqdf100.csv")
    df20 = pd.read_csv("traces_csv3/seqdb100.csv")

    sa_dfs = [df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20]
    sa_merged_df = pd.concat(sa_dfs, axis=0,ignore_index=True)
    sa_merged_df.to_csv("traces_csv4/sa_merged.csv",index=False)

    #3
    df21 = pd.read_csv("traces_csv3/std1.csv")
    df22 = pd.read_csv("traces_csv3/std2.csv")
    df23 = pd.read_csv("traces_csv3/std4.csv")
    df24 = pd.read_csv("traces_csv3/std8.csv")
    df25 = pd.read_csv("traces_csv3/std16.csv")
    df26 = pd.read_csv("traces_csv3/std32.csv")
    df27 = pd.read_csv("traces_csv3/std64.csv")
    df28 = pd.read_csv("traces_csv3/std128.csv")

    stddfs = [df21,df22,df23,df24,df25,df26,df27,df28]
    stdmerged_df = pd.concat(stddfs, axis=0,ignore_index=True)
    stdmerged_df.to_csv("traces_csv4/std_merged.csv",index=False)

    #6
    df29 = pd.read_csv("traces_csv3/i3.csv")
    df30 = pd.read_csv("traces_csv3/i10.csv")
    df31 = pd.read_csv("traces_csv3/i50.csv")
    df32 = pd.read_csv("traces_csv3/i100.csv")
    df33 = pd.read_csv("traces_csv3/i250.csv")
    df34 = pd.read_csv("traces_csv3/i500.csv")
    df35 = pd.read_csv("traces_csv3/i750.csv")
    df36 = pd.read_csv("traces_csv3/i1024.csv")

    idfs = [df29,df30,df31,df32,df33,df34,df35,df36]
    imerged_df = pd.concat(idfs, axis=0,ignore_index=True)
    imerged_df.to_csv("traces_csv4/imerged.csv",index=False)

    #4
    df37 = pd.read_csv("traces_csv3/mc_1_100.csv")
    df38 = pd.read_csv("traces_csv3/mc_10_90.csv")
    df39 = pd.read_csv("traces_csv3/mc_20_80.csv")
    df40 = pd.read_csv("traces_csv3/mc_30_70.csv")
    df41 = pd.read_csv("traces_csv3/mc_40_60.csv")
    df42 = pd.read_csv("traces_csv3/mc_50_50.csv")
    df43 = pd.read_csv("traces_csv3/mc_60_40.csv")
    df44 = pd.read_csv("traces_csv3/mc_70_30.csv")
    df45 = pd.read_csv("traces_csv3/mc_80_20.csv")
    df46 = pd.read_csv("traces_csv3/mc_90_10.csv")
    df47 = pd.read_csv("traces_csv3/mc_100_1.csv")

    mc_dfs = [df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47]
    mc_merged_df = pd.concat(mc_dfs, axis=0,ignore_index=True)
    mc_merged_df.to_csv("traces_csv4/mc_merged.csv",index=False)

    #5
    df48 = pd.read_csv("traces_csv3/mr_1_100.csv")
    df49 = pd.read_csv("traces_csv3/mr_10_90.csv")
    df50 = pd.read_csv("traces_csv3/mr_20_80.csv")
    df51 = pd.read_csv("traces_csv3/mr_30_70.csv")
    df52 = pd.read_csv("traces_csv3/mr_40_60.csv")
    df53 = pd.read_csv("traces_csv3/mr_50_50.csv")
    df54 = pd.read_csv("traces_csv3/mr_60_40.csv")
    df55 = pd.read_csv("traces_csv3/mr_70_30.csv")
    df56 = pd.read_csv("traces_csv3/mr_80_20.csv")
    df57 = pd.read_csv("traces_csv3/mr_90_10.csv")
    df58 = pd.read_csv("traces_csv3/mr_100_1.csv")

    mr_dfs = [df48,df49,df49,df50,df51,df52,df53,df54,df55,df56,df57,df58]
    mr_merged_df = pd.concat(mr_dfs, axis=0,ignore_index=True)
    mr_merged_df.to_csv("traces_csv4/mr_merged.csv",index=False)

    #7
    # df59 = pd.read_csv("traces_csv3/reu_10.csv")
    # df60 = pd.read_csv("traces_csv3/reu_20.csv")
    # df61 = pd.read_csv("traces_csv3/reu_30.csv")
    # df62 = pd.read_csv("traces_csv3/reu_40.csv")
    # df63 = pd.read_csv("traces_csv3/reu_50.csv")
    # df64 = pd.read_csv("traces_csv3/reu_60.csv")
    # df65 = pd.read_csv("traces_csv3/reu_70.csv")
    # df66 = pd.read_csv("traces_csv3/reu_80.csv")
    # df67 = pd.read_csv("traces_csv3/reu_90.csv")

    # reu_dfs = [df59,df60,df61,df62,df63,df64,df65,df66,df67]
    # reu_merged_df = pd.concat(reu_dfs, axis=0,ignore_index=True)
    # reu_merged_df.to_csv("traces_csv3/reu_merged4.csv",index=False)

    #8
    # df68 = pd.read_csv("traces_csv3/rll2.csv")
    # df69 = pd.read_csv("traces_csv3/rll4.csv")
    # df70 = pd.read_csv("traces_csv3/rll8.csv")
    # df71 = pd.read_csv("traces_csv3/rll16.csv")
    # df72 = pd.read_csv("traces_csv3/rll32.csv")
    # df73 = pd.read_csv("traces_csv3/rll64.csv")
    # df74 = pd.read_csv("traces_csv3/rll128.csv")
    # df75 = pd.read_csv("traces_csv3/rll256.csv")
    # df76 = pd.read_csv("traces_csv3/rll512.csv")
    # df77 = pd.read_csv("traces_csv3/rll1024.csv")

    # rlldfs = [df68,df69,df70,df71,df72,df73,df74,df75,df76,df77]
    # rllmerged_df = pd.concat(rlldfs, axis=0,ignore_index=True)
    # rllmerged_df.to_csv("traces_csv3/rllmerged4.csv",index=False)

    #9
    df78 = pd.read_csv("traces_csv3/ra10.csv")
    df79 = pd.read_csv("traces_csv3/ra50.csv")
    df80 = pd.read_csv("traces_csv3/ra100.csv")
    df81 = pd.read_csv("traces_csv3/ra250.csv")
    df82 = pd.read_csv("traces_csv3/ra500.csv")
    df83 = pd.read_csv("traces_csv3/ra750.csv")
    df84 = pd.read_csv("traces_csv3/ra1000.csv")

    radfs = [df78,df79,df80,df81,df82,df83,df84]
    ramerged_df = pd.concat(radfs, axis=0,ignore_index=True)
    ramerged_df.to_csv("traces_csv4/ramerged.csv",index=False)

    #10
    # df85 = pd.read_csv("traces_csv3/rec_10.csv")
    # df86 = pd.read_csv("traces_csv3/rec_50.csv")
    # df87 = pd.read_csv("traces_csv3/rec_100.csv")
    # df88 = pd.read_csv("traces_csv3/rec_250.csv")
    # df89 = pd.read_csv("traces_csv3/rec_500.csv")
    # df90 = pd.read_csv("traces_csv3/rec_750.csv")
    # df91 = pd.read_csv("traces_csv3/rec_1000.csv")

    # rec_dfs = [df85,df86,df87,df88,df89,df6,df90,df91]
    # rec_merged_df = pd.concat(rec_dfs, axis=0,ignore_index=True)
    # rec_merged_df.to_csv("traces_csv3/rec_merged4.csv",index=False)

    #11
    df92 = pd.read_csv("traces_csv3/ll2.csv")
    df93 = pd.read_csv("traces_csv3/ll4.csv")
    df94 = pd.read_csv("traces_csv3/ll8.csv")
    df95 = pd.read_csv("traces_csv3/ll16.csv")
    df96 = pd.read_csv("traces_csv3/ll32.csv")
    df97 = pd.read_csv("traces_csv3/ll64.csv")
    df98 = pd.read_csv("traces_csv3/ll128.csv")
    df99 = pd.read_csv("traces_csv3/ll256.csv")
    df100 = pd.read_csv("traces_csv3/ll512.csv")
    df101 = pd.read_csv("traces_csv3/ll1024.csv")

    lldfs = [df92,df93,df94,df95,df96,df97,df98,df99,df100,df101]
    llmerged_df = pd.concat(lldfs, axis=0,ignore_index=True)
    llmerged_df.to_csv("traces_csv4/llmerged.csv",index=False)







    alldfs = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df26,df26,df27,df28,df29,df30,df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47,df48,df49,df50,df51,df52,df53,df54,df55,df56,df57,df58,df78,df79,df80,df81,df82,df83,df84,df92,df93,df94,df95,df96,df97,df98,df99,df100,df101]

    merged_df = pd.concat(alldfs, axis=0,ignore_index=True)
    print(merged_df.shape)
    
    merged_df.to_csv("traces_csv4/merged.csv",index=False)

add_target_col()
merge_dfs()