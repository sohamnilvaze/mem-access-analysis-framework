import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


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
acc = accuracy_score(y_coarse_test,y_pred)
print(f"Accuracy:- {acc}")
f1m = f1_score(y_coarse_test,y_pred,average="macro")
print(f"F1 score macro:- {f1m}")
f1w = f1_score(y_coarse_test,y_pred,average="weighted")
print(f"F1 score weighted:- {f1w}")
# roc_auc_macro_ovo = roc_auc_score(y_coarse_test,y_pred,average="macro",multi_class="ovo")
roc_auc_macro_ovr = roc_auc_score(y_coarse_test,y_pred,average="macro",multi_class="ovr")
# roc_auc_weighted_ovo = roc_auc_score(y_coarse_test,y_pred,average="weighted",multi_class="ovo")
roc_auc_weighted_ovr = roc_auc_score(y_coarse_test,y_pred,average="weighted",multi_class="ovr")

# print(f"ROC-AUC macro-ovo:- {roc_auc_macro_ovo}")
print(f"ROC-AUC macro-ovr:- {roc_auc_macro_ovr}")
# print(f"ROC-AUC weighted-ovo:- {roc_auc_weighted_ovo}")
print(f"ROC-AUC weighted-ovr:- {roc_auc_weighted_ovr}")


