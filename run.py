import sys
import os
import pandas as pd
import numpy as np
import warnings
from framework import TraceProcessor, MemoryAccessModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Suppress sklearn and pandas warnings globally in the runner
warnings.filterwarnings("ignore")

# ==============================
# CONFIG (Shared between Train/Test)
# ==============================
benchmark = "custom"
TRAINING_TRACE_FOLDER = "training_traces2"
# IGNORE_Features = "page_based"
# MODEL_SAVE_PATH = f"models/{benchmark}_decision_tree_classifier_model.pkl"
MODEL_SAVE_PATH = f"models/{benchmark}_model.pkl"


# CRITICAL: These MUST be the same for both training and testing
WINDOW_SIZE = 500
STEP_SIZE = 250

def assign_label(file):

    if file.startswith("ba"): return 1
    elif file.startswith("seq"): return 2
    elif file.startswith("std"): return 3
    elif file.startswith("i"): return 4
    elif file.startswith("ra"): return 5
    elif file.startswith("ll"): return 6
    elif file.startswith("mc"): return 7
    elif file.startswith("mr"): return 8

    elif any(file.startswith(p) for p in ["2mm", "cg", "bfs"]): return 1
    elif any(file.startswith(p) for p in ["atax", "ep", "hotspot"]): return 2
    elif any(file.startswith(p) for p in ["corr", "ft", "kmeans"]): return 3
    elif any(file.startswith(p) for p in ["fdtd", "is", "particle"]): return 4
    elif any(file.startswith(p) for p in ["gemm", "mg", "srad"]): return 5
    elif file.startswith("jacobi"): return 6
    elif file.startswith("mvt"): return 7
    elif file.startswith("syr2k"): return 8

    return None


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
    model.train_decision_tree_classifier(final_df)
    # model.train_random_forest_classifier(final_df)
    # model.train_svc_classifier(final_df)
    # model.train_logistic_regression_classifier(final_df)
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

def cross_dataset_experiment(train_folder, test_folder):

    processor = TraceProcessor(window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    model = MemoryAccessModel()

    print("\n===== CROSS-DATASET TRAINING =====\n")

    # -------- TRAIN DATA --------
    train_features = []

    for file in os.listdir(train_folder):
        if not file.endswith(".txt"): continue

        path = os.path.join(train_folder, file)

        df_raw = processor.parse_trace(path)
        df_delta = processor.compute_deltas(df_raw)
        df_features = processor.extract_features(df_delta)

        label = assign_label(file)  # <-- we will define this below

        if label:
            df_features["Target"] = label
            df_features["Fine_grained_Target"] = file
            train_features.append(df_features)

    train_df = pd.concat(train_features, ignore_index=True)

    # Train model
    model.train_decision_tree_classifier(train_df)

    print("\n===== CROSS-DATASET TESTING =====\n")

    # -------- TEST DATA --------
    processor = TraceProcessor(window_size=100, step_size=20)
    all_preds = []
    all_true = []

    for file in os.listdir(test_folder):
        if not file.endswith(".txt"): continue

        path = os.path.join(test_folder, file)

        df_raw = processor.parse_trace(path)
        df_delta = processor.compute_deltas(df_raw)
        df_features = processor.extract_features(df_delta)
        if df_features.empty:
            print(f"Skipping {file}: Not enough data for windowing")
            continue
        label = assign_label(file)

        if label:
            preds = model.model.predict(df_features)

            # Majority vote for trace-level prediction
            unique, counts = np.unique(preds, return_counts=True)
            final_pred = unique[np.argmax(counts)]

            all_preds.append(final_pred)
            all_true.append(label)

    # -------- METRICS --------
    print("\n===== CROSS-DATASET RESULTS =====")
    print("Accuracy:", accuracy_score(all_true, all_preds))
    print(f"F1 Macro:    {f1_score(all_true, all_preds, average='macro'):.4f}")
    print(f"F1 Weighted: {f1_score(all_true, all_preds, average='weighted'):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(all_true, all_preds))

def cross_dataset_test(model_path, test_folder):

    processor = TraceProcessor(window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    model = MemoryAccessModel()

    print(f"\nLoading model from: {model_path}")
    model.load_model(model_path)

    all_preds = []
    all_true = []

    print("\n===== CROSS-DATASET TESTING =====\n")

    for file in os.listdir(test_folder):
        if not file.endswith(".txt"): continue

        path = os.path.join(test_folder, file)

        df_raw = processor.parse_trace(path)
        df_delta = processor.compute_deltas(df_raw)
        df_features = processor.extract_features(df_delta)

        if df_features.empty:
            print(f"Skipping {file}: Not enough data")
            continue

        label = assign_label(file)

        if label:
            preds = model.model.predict(df_features)

            unique, counts = np.unique(preds, return_counts=True)
            final_pred = unique[np.argmax(counts)]

            all_preds.append(final_pred)
            all_true.append(label)

    print("\n===== RESULTS =====")
    print("Accuracy:", accuracy_score(all_true, all_preds))
    print(f"F1 Macro:    {f1_score(all_true, all_preds, average='macro'):.4f}")
    print(f"F1 Weighted: {f1_score(all_true, all_preds, average='weighted'):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(all_true, all_preds))

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
    elif mode == "cross-dataset":
        if len(sys.argv) < 4:
            print("Usage: python run.py cross-dataset <train_folder> <test_folder>")
        else:
            cross_dataset_experiment(sys.argv[2], sys.argv[3])
    elif mode == "cross-test":
        if len(sys.argv) < 4:
            print("Usage: python run.py cross-test <model_path> <test_folder>")
        else:
            cross_dataset_test(sys.argv[2], sys.argv[3])