import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


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
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [None, 'sqrt', 'log2']
}

grid_dt = GridSearchCV(
    dt,
    param_grid_dt,
    cv=3,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=2
)
grid_dt.fit(X_train, y_coarse_train)

print("Best params:", grid_dt.best_params_)
print("Best CV score:", grid_dt.best_score_)

best_dt = grid_dt.best_estimator_
y_pred = grid_dt.predict(X_test)
y_proba = grid_dt.predict_proba(X_test)
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

importance = pd.Series(best_dt.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))


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


