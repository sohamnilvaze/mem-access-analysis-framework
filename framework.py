import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import entropy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import seaborn as sns

class TraceProcessor:

    def __init__(self, window_size=100, step_size=50):
        self.WINDOW_SIZE = window_size
        self.STEP_SIZE = step_size

    def parse_trace(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                data.append([
                    int(parts[0]),
                    int(parts[1], 16),
                    int(parts[2], 16),
                    int(parts[3]),
                    int(parts[4])
                ])

        df = pd.DataFrame(data, columns=[
            "thread_id", "ip", "mem_addr", "access_type", "size"
        ])
        return df

    def compute_deltas(self, df):
        df["delta"] = df.groupby("thread_id")["mem_addr"].diff()
        df = df.dropna()
        df["abs_delta"] = df["delta"].abs()
        df["direction"] = np.sign(df["delta"])
        df["cache_line"] = df["mem_addr"] // 64
        df["cl_delta"] = df.groupby("thread_id")["cache_line"].diff()
        df["page"] = df["mem_addr"] // 4096
        return df

    def compute_entropy(self, series):
        counts = series.value_counts()
        probs = counts / counts.sum()
        return entropy(probs)

    def max_consecutive_run(self, arr):
        max_run, current = 1, 1
        for i in range(1, len(arr)):
            if arr[i] == arr[i-1]:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 1
        return max_run

    def extract_features(self, df):

        features = []
        total_len = len(df)

        for start in range(0, total_len - self.WINDOW_SIZE + 1, self.STEP_SIZE):
            window = df.iloc[start:start+self.WINDOW_SIZE]

            deltas = window["delta"]
            abs_deltas = window["abs_delta"]

            feature = {}

            feature["mean_delta"] = deltas.mean()
            feature["std_delta"] = deltas.std()
            feature["mean_abs_delta"] = abs_deltas.mean()
            feature["std_abs_delta"] = abs_deltas.std()
            feature["max_abs_delta"] = abs_deltas.max()
            feature["delta_entropy"] = self.compute_entropy(deltas)

            stride_counts = deltas.value_counts()
            feature["dominant_stride_ratio"] = stride_counts.max() / len(deltas)
            feature["abs_dominant_stride"] = abs(stride_counts.idxmax())

            feature["max_consecutive_same_delta"] = self.max_consecutive_run(deltas.values)
            feature["stride_change_rate"] = np.mean(deltas.values[1:] != deltas.values[:-1])

            feature["forward_ratio"] = (window["direction"] == 1).mean()
            feature["backward_ratio"] = (window["direction"] == -1).mean()

            unique_cache_lines = window["cache_line"].nunique()
            feature["unique_cache_lines"] = unique_cache_lines
            feature["mean_cl_delta"] = window["cl_delta"].mean()
            feature["std_cl_delta"] = window["cl_delta"].std()
            feature["cache_line_reuse_ratio"] = 1 - (unique_cache_lines / self.WINDOW_SIZE)
            feature["avg_accesses_per_cache_line"] = self.WINDOW_SIZE / unique_cache_lines

            cl_deltas = window["cache_line"].diff().dropna()
            feature["cl_small_jump_ratio"] = (cl_deltas.abs() <= 1).mean()

            feature["mean_stride_to_cl_ratio"] = abs_deltas.mean() / (window["cl_delta"].abs().mean() + 1e-6)

            feature["page_reuse_ratio"] = 1 - (window["page"].nunique() / self.WINDOW_SIZE)
            feature["unique_ip_count"] = window["ip"].nunique()

            features.append(feature)

        return pd.DataFrame(features)

class MemoryAccessModel:

    def __init__(self):
        self.model = None
        self.explainer = None

        # Create directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("rules", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

    def train(self, df):

        X = df.drop(["Target", "Fine_grained_Target"], axis=1)
        y = df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        param_grid = {
            "max_depth": [5, 10, 20],
            "criterion": ["gini", "entropy"]
        }

        grid = GridSearchCV(
            DecisionTreeClassifier(),
            param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_
        self.explainer = shap.TreeExplainer(self.model)

        print("Best Params:", grid.best_params_)

        # ============================
        # Metrics
        # ============================

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        roc_macro = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        roc_weighted = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")

        print("\n===== TRAIN METRICS =====")
        print("Accuracy:", acc)
        print("F1 Macro:", f1_macro)
        print("F1 Weighted:", f1_weighted)
        print("ROC-AUC Macro:", roc_macro)
        print("ROC-AUC Weighted:", roc_weighted)

        # Save metrics
        with open("models/train_metrics.txt", "w") as f:
            f.write(f"Accuracy: {acc}\n")
            f.write(f"F1 Macro: {f1_macro}\n")
            f.write(f"F1 Weighted: {f1_weighted}\n")
            f.write(f"ROC-AUC Macro: {roc_macro}\n")
            f.write(f"ROC-AUC Weighted: {roc_weighted}\n")

        # ============================
        # Save Model
        # ============================
        joblib.dump(self.model, "models/dt_model_custom.pkl")

        # ============================
        # Save Decision Tree Rules
        # ============================
        rules = export_text(self.model, feature_names=list(X.columns))
        with open("rules/decision_tree_rules_custom.txt", "w") as f:
            f.write(rules)

        print("\nDecision tree rules saved to rules/decision_tree_rules_custom.txt")

        # ============================
        # Confusion Matrix Plot
        # ============================
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix (Train)")
        plt.savefig("plots/confusion_matrix_train_custom.png")
        plt.close()

        # ============================
        # SHAP Global Importance
        # ============================
        shap_values = self.explainer.shap_values(X_test)

        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig("plots/shap_summary_train_custom.png")
        plt.close()

        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig("plots/shap_bar_train_custom.png")
        plt.close()

        print("SHAP plots saved to plots/")

        return self.model    

    def load_model(self, path="models/dt_model_custom.pkl"):
        self.model = joblib.load(path)
        self.explainer = shap.TreeExplainer(self.model)

    def predict_trace(self, feature_df, true_labels=None):

        prediction = self.model.predict(feature_df)
        probabilities = self.model.predict_proba(feature_df)

        print("\n===== TEST RESULTS =====")
        print("Predicted Class:", prediction[0])
        print("Confidence:", np.max(probabilities))

        # ============================
        # If true labels provided → compute metrics
        # ============================
        if true_labels is not None:

            acc = accuracy_score(true_labels, prediction)
            f1_macro = f1_score(true_labels, prediction, average="macro")
            f1_weighted = f1_score(true_labels, prediction, average="weighted")
            roc_macro = roc_auc_score(true_labels, probabilities, multi_class="ovr", average="macro")
            roc_weighted = roc_auc_score(true_labels, probabilities, multi_class="ovr", average="weighted")

            print("\n===== TEST METRICS =====")
            print("Accuracy:", acc)
            print("F1 Macro:", f1_macro)
            print("F1 Weighted:", f1_weighted)
            print("ROC-AUC Macro:", roc_macro)
            print("ROC-AUC Weighted:", roc_weighted)

        # ============================
        # SHAP Plots (Test)
        # ============================
        shap_values = self.explainer.shap_values(feature_df)

        shap.summary_plot(shap_values, feature_df, show=False)
        plt.savefig("plots/shap_summary_test_custom.png")
        plt.close()

        shap.summary_plot(shap_values, feature_df, plot_type="bar", show=False)
        plt.savefig("plots/shap_bar_test_custom.png")
        plt.close()

        print("Test SHAP plots saved to plots/")

        return prediction
    
class MemoryAccessFramework:

    def __init__(self):
        self.processor = TraceProcessor()
        self.model = MemoryAccessModel()

    def training_pipeline(self, merged_csv_path):

        df = pd.read_csv(merged_csv_path)
        self.model.train(df)

    def testing_pipeline(self, trace_file):

        raw_df = self.processor.parse_trace(trace_file)
        delta_df = self.processor.compute_deltas(raw_df)
        feature_df = self.processor.extract_features(delta_df)

        self.model.load_model()
        self.model.predict_trace(feature_df)


## training pipeline
framework = MemoryAccessFramework()
framework.training_pipeline("traces_csv4/merged.csv")

# ## testing phasse
# framework = MemoryAccessFramework()
# framework.testing_pipeline("unknown_trace.txt")