import sys
import os
import pandas as pd
from framework import TraceProcessor, MemoryAccessModel

# ==============================
# CONFIG
# ==============================

TRAINING_TRACE_FOLDER = "training_traces2"
MODEL_SAVE_PATH = "models/dt_model.pkl"

# ==============================
# TRAINING PIPELINE
# ==============================

def train_from_traces():

    processor = TraceProcessor()
    model = MemoryAccessModel()

    all_features = []

    print("\nStarting training from raw trace files...\n")

    for file in os.listdir(TRAINING_TRACE_FOLDER):

        if not file.endswith(".txt"):
            continue

        trace_path = os.path.join(TRAINING_TRACE_FOLDER, file)

        print(f"Processing {file}")

        # 1️⃣ Parse
        df_raw = processor.parse_trace(trace_path)

        # 2️⃣ Compute deltas
        df_delta = processor.compute_deltas(df_raw)

        # 3️⃣ Extract features
        df_features = processor.extract_features(df_delta)

        # 4️⃣ Assign label from filename
        # Example naming: ba_1.txt, seq_1.txt
        if file.startswith("ba"):
            label = 1
        elif file.startswith("seq"):
            label = 2
        elif file.startswith("std"):
            label = 3
        elif file.startswith("mc"):
            label = 4
        elif file.startswith("mr"):
            label = 5
        elif file.startswith("i"):
            label = 6
        elif file.startswith("ra"):
            label = 7
        elif file.startswith("ll"):
            label = 8
        else:
            print("Unknown label pattern. Skipping.")
            continue

        df_features["Target"] = label
        df_features["Fine_grained_Target"] = file

        all_features.append(df_features)

    # Merge all
    final_df = pd.concat(all_features, ignore_index=True)

    print("\nFinal training dataset shape:", final_df.shape)

    # Train model
    model.train(final_df)

    print("\nTraining complete.\n")


# ==============================
# TESTING PIPELINE
# ==============================

def test_from_trace(trace_path):

    processor = TraceProcessor()
    model = MemoryAccessModel()

    print("\nTesting on trace:", trace_path)

    # Load trained model
    model.load_model(MODEL_SAVE_PATH)

    # 1️⃣ Parse
    df_raw = processor.parse_trace(trace_path)

    # 2️⃣ Compute deltas
    df_delta = processor.compute_deltas(df_raw)

    # 3️⃣ Extract features
    df_features = processor.extract_features(df_delta)

    # 4️⃣ Predict
    model.predict_trace(df_features)

    print("\nTesting complete.\n")


# ==============================
# MAIN ENTRY
# ==============================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run.py train")
        print("  python run.py test path/to/trace.txt")
        sys.exit()

    mode = sys.argv[1]

    if mode == "train":
        train_from_traces()

    elif mode == "test":

        if len(sys.argv) < 3:
            print("Please provide trace file path.")
            sys.exit()

        test_trace_path = sys.argv[2]
        test_from_trace(test_trace_path)

    else:
        print("Invalid mode. Use 'train' or 'test'.")