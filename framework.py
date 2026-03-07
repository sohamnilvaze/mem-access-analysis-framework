import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import warnings

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

benchmark = "custom"

class TraceProcessor:
    def __init__(self, window_size=500, step_size=250):
        print(f"Initializing TraceProcessor with window_size={window_size} and step_size={step_size}")
        self.WINDOW_SIZE = window_size
        self.STEP_SIZE = step_size

    def parse_trace(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                data.append([
                    int(parts[0]), int(parts[1], 16), int(parts[2], 16), int(parts[3]), int(parts[4])
                ])
        return pd.DataFrame(data, columns=["thread_id", "ip", "mem_addr", "access_type", "size"])

    def compute_deltas(self, df):
        # Use .copy() to avoid SettingWithCopyWarning
        df = df.copy()
        df["delta"] = df.groupby("thread_id")["mem_addr"].diff()
        df = df.dropna().copy() 
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
            feature["mean_stride_to_cl_ratio"] = abs_deltas.mean() / (window["cl_delta"].abs().mean() + 1e-6)

            # Page + IP
            feature["page_reuse_ratio"] = 1 - (window["page"].nunique() / self.WINDOW_SIZE)
            feature["unique_ip_count"] = window["ip"].nunique()
            feature["unique_address_ratio"] = window["mem_addr"].nunique() / self.WINDOW_SIZE

            delta_sign = np.sign(deltas.values)
            feature["delta_sign_change_rate"] = np.mean(delta_sign[1:] != delta_sign[:-1])
            feature["large_jump_ratio"] = (abs_deltas > 4096).mean()
            feature["delta_cv"] = deltas.std() / (abs(deltas.mean()) + 1e-6)
            feature["cl_delta_entropy"] = self.compute_entropy(window["cl_delta"].dropna())

            features.append(feature)
        return pd.DataFrame(features)

class MemoryAccessModel:
    def __init__(self):
        self.model = None
        self.explainer = None
        os.makedirs("models", exist_ok=True)
        os.makedirs("rules", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

    def export_feature_importance(self, X, shap_values):
        feature_names = list(X.columns)
        tree_importance = self.model.feature_importances_
        df_tree = pd.DataFrame({"Feature": feature_names, "Tree_Importance": tree_importance})

        if isinstance(shap_values, list):
            shap_array = np.mean([np.abs(class_vals) for class_vals in shap_values], axis=0)
        else:
            shap_array = np.mean(np.abs(shap_values), axis=2)
        
        shap_mean = np.mean(shap_array, axis=0)
        df_shap = pd.DataFrame({"Feature": feature_names, "Mean_Abs_SHAP": shap_mean})
        df_final = df_tree.merge(df_shap, on="Feature").sort_values(by="Mean_Abs_SHAP", ascending=False)
        df_final.to_csv(f"rules/{benchmark}_feature_importance_ranking.csv", index=False)

    def plot_feature_separability(self, X, y):

        df = X.copy()
        df["Target"] = y

        important_features = [
            "delta_entropy",
            "unique_address_ratio",
            "delta_sign_change_rate",
            "large_jump_ratio",
            "delta_cv"
        ]

        for feature in important_features:

            plt.figure(figsize=(7,5))

            for cls in sorted(df["Target"].unique()):
                sns.kdeplot(
                    df[df["Target"]==cls][feature],
                    label=f"class {cls}",
                    fill=True,
                    alpha=0.3
                )

            plt.title(f"Feature Separability: {feature}")
            plt.legend()
            plt.savefig(f"plots/{benchmark}_separability_{feature}.png")
            plt.close()

    def train(self, df):
        print(f"Starting training on {len(df)} samples with {len(df.columns)} features...")
        X = df.drop(["Target", "Fine_grained_Target"], axis=1)
        y = df["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        param_grid = {"max_depth": [10,20,30], "min_samples_leaf": [1,5,10], "criterion": ["gini","entropy"]}
        grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_
        self.explainer = shap.TreeExplainer(self.model)
        
        print("Best Params:", grid.best_params_)
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        print("\n===== TEST METRICS =====")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        # F1 Scores
        print(f"F1 Macro:         {f1_score(y_test, y_pred, average='macro'):.4f}")
        print(f"F1 Weighted:      {f1_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        # ROC-AUC Scores (Multi-class requires 'ovr' or 'ovo' and probability scores)
        # We use a try-except block in case some classes are missing in the test split
        try:
            macro_roc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
            weighted_roc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            print(f"ROC-AUC Macro:    {macro_roc:.4f}")
            print(f"ROC-AUC Weighted: {weighted_roc:.4f}")
        except Exception as e:
            print(f"ROC-AUC: Could not calculate (Error: {e})")

        self.plot_feature_separability(X_test, y_test) 
        
        joblib.dump(self.model, f"models/{benchmark}_model.pkl")
        return self.model

    def load_model(self, path=f"models/{benchmark}_model.pkl"):
        self.model = joblib.load(path)
        self.explainer = shap.TreeExplainer(self.model)
    
    def explain_prediction(self, sample):

        node_indicator = self.model.decision_path(sample)
        leaf_id = self.model.apply(sample)

        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold

        feature_names = sample.columns

        explanation = []

        for node_id in node_indicator.indices:

            if leaf_id[0] == node_id:
                continue

            feature_index = feature[node_id]
            feature_name = feature_names[feature_index]
            threshold_val = threshold[node_id]

            sample_val = sample.iloc[0, feature_index]

            if sample_val <= threshold_val:
                rule = f"{feature_name} ({sample_val:.3f}) <= {threshold_val:.3f}"
            else:
                rule = f"{feature_name} ({sample_val:.3f}) > {threshold_val:.3f}"

            explanation.append(rule)

        return explanation

    def predict_trace(self, feature_df):
        # Ensure we suppress sklearn feature name warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = self.model.predict(feature_df)
            probabilities = self.model.predict_proba(feature_df)

        unique, counts = np.unique(predictions, return_counts=True)
        final_prediction = unique[np.argmax(counts)]
        print("\n===== TRACE-LEVEL SUMMARY =====")
        print("Majority Class:", final_prediction)
        print("\n===== EXPLANATION =====")

        sample = feature_df.iloc[[0]]
        rules = self.explain_prediction(sample)

        for r in rules:
            print(" -", r)
        return final_prediction

    def save_decision_path_example(self, sample):
        node_indicator = self.model.decision_path([sample])
        leaf_id = self.model.apply([sample])
        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold
        rules = []
        for node_id in node_indicator.indices:
            if leaf_id[0] == node_id: continue
            # FIXED: Use .iloc for positional indexing to avoid FutureWarning
            val = sample.iloc[feature[node_id]]
            sign = "<=" if val <= threshold[node_id] else ">"
            rules.append(f"{sample.index[feature[node_id]]} {sign} {threshold[node_id]:.4f}")
        
        with open(f"rules/{benchmark}_decision_path_example.txt", "w") as f:
            f.write("\n".join(rules))

# PREVENTS UNWANTED EXECUTION ON IMPORT
if __name__ == "__main__":
    print("Framework module loaded directly.")