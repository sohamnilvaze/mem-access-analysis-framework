import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import seaborn as sns

benchmark = "rodinia"

# ============================================================
# TRACE PROCESSOR
# ============================================================

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

        return pd.DataFrame(data, columns=[
            "thread_id", "ip", "mem_addr", "access_type", "size"
        ])

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
            if arr[i] == arr[i - 1]:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 1
        return max_run

    def extract_features(self, df):
        features = []
        total_len = len(df)

        for start in range(0, total_len - self.WINDOW_SIZE + 1, self.STEP_SIZE):
            window = df.iloc[start:start + self.WINDOW_SIZE]

            deltas = window["delta"]
            abs_deltas = window["abs_delta"]

            feature = {}

            # Delta features
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

            # Direction
            feature["forward_ratio"] = (window["direction"] == 1).mean()
            feature["backward_ratio"] = (window["direction"] == -1).mean()

            # Cache-level
            unique_cache_lines = window["cache_line"].nunique()
            feature["unique_cache_lines"] = unique_cache_lines
            feature["mean_cl_delta"] = window["cl_delta"].mean()
            feature["std_cl_delta"] = window["cl_delta"].std()
            feature["cache_line_reuse_ratio"] = 1 - (unique_cache_lines / self.WINDOW_SIZE)
            feature["avg_accesses_per_cache_line"] = self.WINDOW_SIZE / max(unique_cache_lines, 1)

            cl_deltas = window["cache_line"].diff().dropna()
            feature["cl_small_jump_ratio"] = (cl_deltas.abs() <= 1).mean()

            feature["mean_stride_to_cl_ratio"] = abs_deltas.mean() / (
                window["cl_delta"].abs().mean() + 1e-6
            )

            # Page + IP
            feature["page_reuse_ratio"] = 1 - (window["page"].nunique() / self.WINDOW_SIZE)
            feature["unique_ip_count"] = window["ip"].nunique()

            features.append(feature)

        return pd.DataFrame(features)


# ============================================================
# MODEL
# ============================================================

class MemoryAccessModel:

    def __init__(self):
        self.model = None
        self.explainer = None

        os.makedirs("models", exist_ok=True)
        os.makedirs("rules", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

    # --------------------------------------------------------
    # SAFE ROC
    # --------------------------------------------------------
    # --------------------------------------------------------
# FEATURE IMPORTANCE EXPORT
# --------------------------------------------------------
    def export_feature_importance(self, X, shap_values):

        feature_names = list(X.columns)

        # ------------------------------
        # 1️⃣ Tree Importance
        # ------------------------------
        tree_importance = self.model.feature_importances_

        df_tree = pd.DataFrame({
            "Feature": feature_names,
            "Tree_Importance": tree_importance
        })

        # ------------------------------
        # 2️⃣ SHAP Importance (Global)
        # ------------------------------
        if isinstance(shap_values, list):
            # Old SHAP (list per class)
            shap_array = np.mean(
                [np.abs(class_vals) for class_vals in shap_values],
                axis=0
            )
        else:
            # New SHAP (samples, features, classes)
            shap_array = np.mean(
                np.abs(shap_values),
                axis=2
            )

        shap_mean = np.mean(shap_array, axis=0)

        df_shap = pd.DataFrame({
            "Feature": feature_names,
            "Mean_Abs_SHAP": shap_mean
        })

        # ------------------------------
        # 3️⃣ Per-Class SHAP
        # ------------------------------
        df_per_class = pd.DataFrame({"Feature": feature_names})

        if isinstance(shap_values, list):
            for idx, class_vals in enumerate(shap_values):
                df_per_class[f"Class_{self.model.classes_[idx]}_SHAP"] = \
                    np.mean(np.abs(class_vals), axis=0)
        else:
            for idx in range(shap_values.shape[2]):
                df_per_class[f"Class_{self.model.classes_[idx]}_SHAP"] = \
                    np.mean(np.abs(shap_values[:, :, idx]), axis=0)

        # ------------------------------
        # 4️⃣ Merge All
        # ------------------------------
        df_final = df_tree.merge(df_shap, on="Feature")
        df_final = df_final.merge(df_per_class, on="Feature")

        # Rank by SHAP importance
        df_final = df_final.sort_values(
            by="Mean_Abs_SHAP",
            ascending=False
        )

        df_final.to_csv(f"rules/{benchmark}_feature_importance_ranking.csv", index=False)

        print("\nFeature importance ranking saved to:")
        print(f"rules/{benchmark}_feature_importance_ranking.csv")

    def safe_roc_auc(self, y_true, y_proba):
        try:
            if len(np.unique(y_true)) < 2:
                return "N/A (single class)"
            return roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro"
            )
        except:
            return "N/A"

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    def train(self, df):

        X = df.drop(["Target", "Fine_grained_Target"], axis=1)
        y = df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            stratify=y,
            test_size=0.2,
            random_state=42
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

        # ================= Metrics =================
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        print("\n===== TEST METRICS =====")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("F1 Macro:", f1_score(y_test, y_pred, average="macro"))
        print("F1 Weighted:", f1_score(y_test, y_pred, average="weighted"))
        print("ROC-AUC Macro:", self.safe_roc_auc(y_test, y_proba))

        # ================= Save Model =================
        joblib.dump(self.model, f"models/{benchmark}_model.pkl")

        rules = export_text(self.model, feature_names=list(X.columns))
        with open(f"rules/{benchmark}_decision_tree_rules.txt", "w") as f:
            f.write(rules)

        # ================= Confusion Matrix =================
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix (Test)")
        plt.savefig(f"plots/{benchmark}_confusion_matrix_test.png")
        plt.close()

        # ================= SHAP =================
        shap_values = self.explainer.shap_values(X_test)
        self.export_feature_importance(X_test, shap_values)
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Old SHAP: list per class
            for class_idx, class_values in enumerate(shap_values):
                plt.figure()
                shap.summary_plot(class_values, X_test, show=False)
                plt.title(f"SHAP Summary - Class {self.model.classes_[class_idx]}")
                plt.savefig(f"plots/{benchmark}_shap_summary_class_{class_idx}.png")
                plt.close()

        else:
            # New SHAP: 3D array (samples, features, classes)
            for class_idx in range(shap_values.shape[2]):
                plt.figure()
                shap.summary_plot(
                    shap_values[:, :, class_idx],
                    X_test,
                    show=False
                )
                plt.title(f"SHAP Summary - Class {self.model.classes_[class_idx]}")
                plt.savefig(f"plots/{benchmark}_shap_summary_class_{class_idx}.png")
                plt.close()
                # Global bar
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                plt.savefig(f"plots/{benchmark}_shap_bar_global.png")
                plt.close()

        # 2️⃣ Feature Distribution Per Class
        self.plot_feature_distributions(X_test, y_test)

        # 3️⃣ Decision Path Example
        self.save_decision_path_example(X_test.iloc[0])

        return self.model

    # --------------------------------------------------------
    # FEATURE DISTRIBUTIONS
    # --------------------------------------------------------
    def plot_feature_distributions(self, X, y):

        df_plot = X.copy()
        df_plot["Target"] = y.values

        for feature in X.columns:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x="Target", y=feature, data=df_plot)
            plt.title(f"Feature Distribution per Class: {feature}")
            plt.savefig(f"plots/{benchmark}_feature_distribution_{feature}.png")
            plt.close()

    # --------------------------------------------------------
    # DECISION PATH (LOCAL EXPLANATION)
    # --------------------------------------------------------
    def save_decision_path_example(self, sample):

        node_indicator = self.model.decision_path([sample])
        leaf_id = self.model.apply([sample])

        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold

        rules = []

        for node_id in node_indicator.indices:
            if leaf_id[0] == node_id:
                continue

            if sample[feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            rules.append(
                f"{sample.index[feature[node_id]]} "
                f"{threshold_sign} "
                f"{threshold[node_id]:.4f}"
            )

        with open(f"rules/{benchmark}_decision_path_example.txt", "w") as f:
            f.write("\n".join(rules))

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------
    def load_model(self, path=f"models/{benchmark}_model.pkl"):
        self.model = joblib.load(path)
        self.explainer = shap.TreeExplainer(self.model)

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------
    def predict_trace(self, feature_df, true_labels=None):

        predictions = self.model.predict(feature_df)
        probabilities = self.model.predict_proba(feature_df)

        print("\n===== WINDOW-LEVEL PREDICTIONS =====")
        print("Total Windows:", len(predictions))

        unique, counts = np.unique(predictions, return_counts=True)
        distribution = dict(zip(unique, counts))
        print("Prediction Distribution:", distribution)

        final_prediction = unique[np.argmax(counts)]
        confidence = np.mean(np.max(probabilities, axis=1))

        print("\n===== TRACE-LEVEL SUMMARY =====")
        print("Majority Class:", final_prediction)
        print("Average Confidence:", confidence)

        # Per-class SHAP at inference
        shap_values = self.explainer.shap_values(feature_df)

        if isinstance(shap_values, list):
            for class_idx, class_values in enumerate(shap_values):
                plt.figure()
                shap.summary_plot(class_values, feature_df, show=False)
                plt.title(f"Inference SHAP - Class {self.model.classes_[class_idx]}")
                plt.savefig(f"plots/{benchmark}_shap_inference_class_{class_idx}.png")
                plt.close()
        else:
            for class_idx in range(shap_values.shape[2]):
                plt.figure()
                shap.summary_plot(
                    shap_values[:, :, class_idx],
                    feature_df,
                    show=False
                )
                plt.title(f"Inference SHAP - Class {self.model.classes_[class_idx]}")
                plt.savefig(f"plots/{benchmark}_shap_inference_class_{class_idx}.png")
                plt.close()
        return final_prediction

# ============================================================
# FRAMEWORK
# ============================================================

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
        return self.model.predict_trace(feature_df)


# ================= RUN =================

framework = MemoryAccessFramework()
framework.training_pipeline("traces_csv4/merged.csv")

# For testing:
# framework.testing_pipeline("unknown_trace.txt")