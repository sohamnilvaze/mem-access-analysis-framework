import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import shap
import numpy as np

# ==========================================================
# DOMAIN KNOWLEDGE LAYER (Hardware Meaning Mapping)
# ==========================================================

FEATURE_EXPLANATIONS = {
    "delta_entropy": {
        "high": "High entropy indicates irregular or random memory access patterns.",
        "low": "Low entropy indicates predictable and structured memory access behavior."
    },
    "unique_cache_lines": {
        "high": "High number of unique cache lines suggests low spatial locality.",
        "low": "Low number of unique cache lines suggests strong spatial locality."
    },
    "unique_ip_count": {
        "high": "Many instruction pointers indicate diverse memory access sources.",
        "low": "Few instruction pointers suggest loop-driven structured access."
    },
    "std_delta": {
        "high": "High delta variation suggests irregular stepping.",
        "low": "Low delta variation suggests consistent stride behavior."
    },
    "max_consecutive_same_delta": {
        "high": "Long sequences of identical deltas indicate strong stride repetition.",
        "low": "Few repeated deltas indicate unstable stepping."
    },
    "stride_change_rate": {
        "high": "Frequent stride changes indicate irregular traversal.",
        "low": "Stable stride suggests structured traversal."
    }
}

# ==========================================================
# DATA LOADING
# ==========================================================

df = pd.read_csv("traces_csv4/merged.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.drop(['Target', 'Fine_grained_Target'], axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================================
# MODEL TRAINING
# ==========================================================

dt = DecisionTreeClassifier(random_state=42)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [None, 'sqrt', 'log2']
}

grid = GridSearchCV(
    dt,
    param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)

best_dt = grid.best_estimator_

# ==========================================================
# EVALUATION METRICS
# ==========================================================

y_pred = best_dt.predict(X_test)
y_proba = best_dt.predict_proba(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("F1 Macro:", f1_score(y_test, y_pred, average="macro"))
print("F1 Weighted:", f1_score(y_test, y_pred, average="weighted"))
print("ROC-AUC Macro OVR:", roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

# ==========================================================
# SHAP EXPLAINABILITY
# ==========================================================

explainer = shap.TreeExplainer(best_dt)
shap_values = explainer.shap_values(X_test)

# Ensure 3D format: (samples, features, classes)
if isinstance(shap_values, list):
    shap_stack = np.stack(shap_values, axis=-1)
else:
    shap_stack = shap_values

print("SHAP shape:", shap_stack.shape)

# ==========================================================
# GLOBAL SHAP IMPORTANCE
# ==========================================================

shap_abs = np.abs(shap_stack)
mean_importance = shap_abs.mean(axis=(0, 2))  # avg over samples + classes

shap_importance = pd.Series(mean_importance, index=X.columns)

print("\n===== SHAP Global Feature Importance =====")
print(shap_importance.sort_values(ascending=False))

# Summary plots
shap.summary_plot(shap_stack, X_test)
shap.summary_plot(shap_stack, X_test, plot_type="bar")

# ==========================================================
# AUTOMATIC EXPLANATION GENERATOR
# ==========================================================

def generate_explanation(sample_index):
    
    predicted_class = best_dt.predict(X_test.iloc[[sample_index]])[0]
    class_index = list(best_dt.classes_).index(predicted_class)
    confidence = best_dt.predict_proba(X_test.iloc[[sample_index]])[0][class_index]

    sample_shap = shap_stack[sample_index, :, class_index]
    feature_values = X_test.iloc[sample_index]

    shap_series = pd.Series(sample_shap, index=X.columns)
    shap_series = shap_series.sort_values(key=abs, ascending=False)

    print("\n================================================")
    print(f"Predicted Class: {predicted_class}")
    print(f"True Class: {y_test.iloc[sample_index]}")
    print(f"Model Confidence: {confidence:.4f}")
    print("================================================\n")

    top_features = shap_series.head(5)

    for feature, shap_value in top_features.items():

        if feature not in FEATURE_EXPLANATIONS:
            continue

        value = feature_values[feature]
        median = X_test[feature].median()

        direction = "high" if value > median else "low"
        contribution = "increased" if shap_value > 0 else "decreased"

        print(f"{feature} = {value:.4f}")
        print(f"→ This {direction} value {contribution} probability of class {predicted_class}.")
        print(f"→ {FEATURE_EXPLANATIONS[feature][direction]}\n")

# ==========================================================
# RUN EXPLANATION FOR ONE SAMPLE
# ==========================================================

generate_explanation(sample_index=0)