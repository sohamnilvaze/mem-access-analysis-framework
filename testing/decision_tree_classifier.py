import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree, export_text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report


def explain_prediction(model, feature_names, sample):
    tree = model.tree_
    feature = tree.feature
    threshold = tree.threshold
    
    node_indicator = model.decision_path(sample)
    leaf_id = model.apply(sample)

    explanation = []

    for node_id in node_indicator.indices:
        if leaf_id[0] == node_id:
            continue
            
        if sample[0, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"
            
        explanation.append(
            f"{feature_names[feature[node_id]]} {threshold_sign} {threshold[node_id]:.3f}"
        )

    return explanation

df = pd.read_csv("traces_csv4/merged.csv")

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split features
X = df.drop(['Target', 'Fine_grained_Target'], axis=1)

# Targets
y_coarse = df['Target']
y_fine = df['Fine_grained_Target']

# Train-test split
X_train, X_test, y_coarse_train, y_coarse_test = train_test_split(
    X, y_coarse, test_size=0.2, random_state=42, stratify=y_coarse
)

# Scaling
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


dt = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [None, 'sqrt', 'log2']
}

grid_dt = GridSearchCV(
    dt,
    param_grid_dt,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)
grid_dt.fit(X_train, y_coarse_train)

print("Best params:", grid_dt.best_params_)
print("Best CV score:", grid_dt.best_score_)

best_dt = grid_dt.best_estimator_
y_pred = best_dt.predict(X_test)
y_proba = best_dt.predict_proba(X_test)
acc = accuracy_score(y_coarse_test,y_pred)
print(f"Accuracy:- {acc}")
f1m = f1_score(y_coarse_test,y_pred,average="macro")
print(f"F1 score macro:- {f1m}")
f1w = f1_score(y_coarse_test,y_pred,average="weighted")
print(f"F1 score weighted:- {f1w}")
# roc_auc_macro_ovo = roc_auc_score(y_coarse_test,y_pred,average="macro",multi_class="ovo")
roc_auc_macro_ovr = roc_auc_score(y_coarse_test,y_proba,average="macro",multi_class="ovr")
# roc_auc_weighted_ovo = roc_auc_score(y_coarse_test,y_pred,average="weighted",multi_class="ovo")
roc_auc_weighted_ovr = roc_auc_score(y_coarse_test,y_proba,average="weighted",multi_class="ovr")

# print(f"ROC-AUC macro-ovo:- {roc_auc_macro_ovo}")
print(f"ROC-AUC macro-ovr:- {roc_auc_macro_ovr}")
# print(f"ROC-AUC weighted-ovo:- {roc_auc_weighted_ovo}")
print(f"ROC-AUC weighted-ovr:- {roc_auc_weighted_ovr}")
print(f"Classification report:- {classification_report(y_coarse_test,y_pred)}")

importance = pd.Series(best_dt.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))

print(df['Target'].value_counts(normalize=True))

rules = export_text(best_dt, feature_names=list(X.columns))
with open("rules/decision_tree_classifier.txt","w") as f:
    f.write(rules)

cm = confusion_matrix(y_coarse_test,y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt="d")
plt.show()

'''
Best params: {'criterion': 'gini', 'max_depth': 10, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
Best CV score: 0.5851838931024547
Accuracy:- 0.6484787528287654
F1 score macro:- 0.635639539133205
F1 score weighted:- 0.606312012678665
ROC-AUC macro-ovr:- 0.8211965771045634
ROC-AUC weighted-ovr:- 0.7575557334407634
unique_cache_lines    0.228394
unique_ip_count       0.226375
mean_delta            0.221246
forward_ratio         0.212336
std_cl_delta          0.059169
backward_ratio        0.039150
max_abs_delta         0.006202
mean_cl_delta         0.003677
mean_abs_delta        0.001399
std_abs_delta         0.000843
unique_pages          0.000762
std_delta             0.000447
zero_ratio            0.000000
read_ratio            0.000000
write_ratio           0.000000
'''
'''
Best params: {'criterion': 'entropy', 'max_depth': 15, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
Best CV score: 0.7358952116923364
Accuracy:- 0.7806813172973994
F1 score macro:- 0.7350676606608086
F1 score weighted:- 0.7540471151750853
ROC-AUC macro-ovr:- 0.8846335683400941
ROC-AUC weighted-ovr:- 0.8588522709091659
Classification report:-               precision    recall  f1-score   support

           1       1.00      0.75      0.86      5242
           2       0.97      0.60      0.74       193
           3       0.65      1.00      0.79      6547
           4       0.98      0.50      0.67      1327
           5       0.85      0.05      0.10      1327
           6       0.99      0.75      0.85       440
           7       1.00      0.77      0.87       318
           8       1.00      1.00      1.00       487

    accuracy                           0.78     15881
   macro avg       0.93      0.68      0.74     15881
weighted avg       0.84      0.78      0.75     15881

max_consecutive_same_delta     0.437298
abs_dominant_stride            0.236825
stride_change_rate             0.077689
dominant_stride_ratio          0.055965
backward_ratio                 0.044793
unique_ip_count                0.040499
mean_abs_delta                 0.034667
mean_delta                     0.028052
unique_cache_lines             0.013486
delta_entropy                  0.007733
forward_ratio                  0.007258
std_abs_delta                  0.004586
max_abs_delta                  0.002760
mean_cl_delta                  0.002271
std_cl_delta                   0.001729
avg_accesses_per_cache_line    0.001383
cl_small_jump_ratio            0.000767
std_delta                      0.000757
cache_line_reuse_ratio         0.000709
page_reuse_ratio               0.000443
unique_pages                   0.000167
mean_stride_to_cl_ratio        0.000161
zero_ratio                     0.000000
read_ratio                     0.000000
write_ratio                    0.000000
dtype: float64
Target
3    0.412246
1    0.330059
4    0.083560
5    0.083560
8    0.030666
6    0.027706
7    0.020012
2    0.012191
Name: proportion, dtype: float64
'''

