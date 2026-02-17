import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


rf = RandomForestClassifier(random_state=42)
# param_grid_rf = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 5],
#     'max_features': ['sqrt', 'log2']
# }

param_grid_rf = {
    'n_estimators': [300],
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}


grid_dt = GridSearchCV(
    rf,
    param_grid_rf,
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

# {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}


importance = pd.Series(best_dt.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))


cm = confusion_matrix(y_coarse_test,y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt="d")
plt.show()


'''
Best params: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
Best CV score: 0.5848657775311041
Accuracy:- 0.6489816444556198
F1 score macro:- 0.6374483984330201
F1 score weighted:- 0.6066747442913225
ROC-AUC macro-ovr:- 0.8223318050294055
ROC-AUC weighted-ovr:- 0.7580719346498409
mean_delta            0.307872
unique_cache_lines    0.186246
unique_ip_count       0.129420
backward_ratio        0.101095
forward_ratio         0.099155
max_abs_delta         0.032057
mean_abs_delta        0.029619
std_cl_delta          0.028707
std_delta             0.028080
std_abs_delta         0.026402
mean_cl_delta         0.015825
unique_pages          0.015509
zero_ratio            0.000013
read_ratio            0.000000
write_ratio           0.000000
'''