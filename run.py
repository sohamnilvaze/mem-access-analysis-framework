import sys
import os
import pandas as pd
import warnings
from framework import TraceProcessor, MemoryAccessModel

# Suppress sklearn and pandas warnings globally in the runner
warnings.filterwarnings("ignore")

# ==============================
# CONFIG (Shared between Train/Test)
# ==============================
benchmark = "custom"
TRAINING_TRACE_FOLDER = "training_traces2"
MODEL_SAVE_PATH = f"models/{benchmark}_model.pkl"

# CRITICAL: These MUST be the same for both training and testing
WINDOW_SIZE = 500
STEP_SIZE = 100

# ==============================
# TRAINING PIPELINE
# ==============================
def train_from_traces():
    processor = TraceProcessor(window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    model = MemoryAccessModel()
    all_features = []

    print("\nStarting training from raw trace files...\n")
    for file in os.listdir(TRAINING_TRACE_FOLDER):
        if not file.endswith(".txt"): continue
        trace_path = os.path.join(TRAINING_TRACE_FOLDER, file)
        
        df_raw = processor.parse_trace(trace_path)
        df_delta = processor.compute_deltas(df_raw)
        df_features = processor.extract_features(df_delta)

        # Labeling Logic (keeping your existing logic)
        label = None
        if file.startswith("ba"): label = 1
        elif file.startswith("seq"): label = 2
        elif file.startswith("std"): label = 3
        elif file.startswith("i"): label = 4
        elif file.startswith("ra"): label = 5
        elif file.startswith("ll"): label = 6
        elif file.startswith("mc"): label = 7
        elif file.startswith("mr"): label = 8
        elif any(file.startswith(p) for p in ["2mm", "cg", "bfs"]): label = 1
        elif any(file.startswith(p) for p in ["atax", "ep", "hotspot"]): label = 2
        elif any(file.startswith(p) for p in ["corr", "ft", "kmeans"]): label = 3
        elif any(file.startswith(p) for p in ["fdtd", "is", "particle"]): label = 4
        elif any(file.startswith(p) for p in ["gemm", "mg", "srad"]): label = 5
        elif file.startswith("jacobi"): label = 6
        elif file.startswith("mvt"): label = 7
        elif file.startswith("syr2k"): label = 8

        if label:
            df_features["Target"] = label
            df_features["Fine_grained_Target"] = file
            all_features.append(df_features)
        else:
            print(f"Skipping {file}: Unknown pattern")

    final_df = pd.concat(all_features, ignore_index=True)
    model.train(final_df)
    print("\nTraining complete.\n")

# ==============================
# TESTING PIPELINE
# ==============================
def test_from_trace(trace_path):
    # Use the SAME window/step size as training
    processor = TraceProcessor(window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    model = MemoryAccessModel()

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model not found at {MODEL_SAVE_PATH}. Please train first.")
        return

    model.load_model(MODEL_SAVE_PATH)
    df_raw = processor.parse_trace(trace_path)
    df_delta = processor.compute_deltas(df_raw)
    df_features = processor.extract_features(df_delta)
    
    model.predict_trace(df_features)

# ==============================
# MAIN ENTRY
# ==============================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py [train|test] [path]")
        sys.exit()

    mode = sys.argv[1]
    if mode == "train":
        train_from_traces()
    elif mode == "test":
        if len(sys.argv) < 3:
            print("Please provide trace file path.")
        else:
            test_from_trace(sys.argv[2])