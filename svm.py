import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import pickle

# Load data
df_train = pd.read_csv('glcm_features_train (3).csv')
df_test = pd.read_csv('glcm_features_test (1).csv')
df_valid = pd.read_csv('glcm_features_valid (1).csv')

# Cek NaN pada label
print("NaN count in y_train (before dropna):", df_train['label'].isnull().sum())
print("NaN count in y_valid (before dropna):", df_valid['label'].isnull().sum())
print("NaN count in y_test (before dropna):", df_test['label'].isnull().sum())

# Drop NaN pada label
df_train = df_train.dropna(subset=['label'])
df_valid = df_valid.dropna(subset=['label'])
df_test = df_test.dropna(subset=['label'])

# Pisahkan label
y_train = df_train['label']
y_valid = df_valid['label']
y_test = df_test['label']

# Fitur
feature_cols = [col for col in df_train.columns if col != 'label']
X_train = df_train[feature_cols].values
X_valid = df_valid[feature_cols].values
X_test = df_test[feature_cols].values

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Cek NaN
print("\nNaN count in X_train (after scaling):", np.isnan(X_train_scaled).sum())
print("NaN count in X_valid (after scaling):", np.isnan(X_valid_scaled).sum())
print("NaN count in X_test (after scaling):", np.isnan(X_test_scaled).sum())

print("NaN count in y_train (after dropna and filter):", pd.Series(y_train).isnull().sum())
print("NaN count in y_valid (after dropna and filter):", pd.Series(y_valid).isnull().sum())
print("NaN count in y_test (after dropna and filter):", pd.Series(y_test).isnull().sum())

# PCA
if X_train_scaled.shape[0] > 1:
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_valid_pca = pca.transform(X_valid_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"\nPCA applied. Reduced dimensions: {X_train_scaled.shape[1]} -> {X_train_pca.shape[1]}")
else:
    print("\nSkipping PCA due to insufficient samples.")
    X_train_pca = X_train_scaled
    X_valid_pca = X_valid_scaled
    X_test_pca = X_test_scaled

# Label Encoding
le = LabelEncoder()
all_labels = pd.concat([
    pd.Series(y_train),
    pd.Series(y_valid),
    pd.Series(y_test)
]).unique()
all_labels = [label for label in all_labels if isinstance(label, str) and label is not None]
if len(all_labels) == 0:
    raise ValueError("No valid class labels found.")
le.fit(all_labels)

y_train_encoded = le.transform(y_train)
y_valid_encoded = le.transform(y_valid)
y_test_encoded = le.transform(y_test)

# SMOTE
unique_classes, counts = np.unique(y_train_encoded, return_counts=True)
if len(unique_classes) > 1 and np.min(counts) > 1:
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_pca, y_train_encoded)
    print(f"Original training samples: {len(y_train_encoded)}")
    print(f"Training samples after SMOTE: {len(y_train_balanced)}")
else:
    print("\nSkipping SMOTE.")
    X_train_balanced = X_train_pca
    y_train_balanced = y_train_encoded

# >>> Tambahkan GridSearchCV untuk SVM
print("\nPerforming Grid Search for SVM hyperparameters...")

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced']
}

svm = SVC()
grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=2
)

grid_search.fit(X_train_balanced, y_train_balanced)

print("\nBest parameters from GridSearch:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Gunakan best estimator dari GridSearch
best_svm = grid_search.best_estimator_

# Validasi
y_pred_valid = best_svm.predict(X_valid_pca)
print("\nValidasi:")
print("Akurasi:", accuracy_score(y_valid_encoded, y_pred_valid))
print(classification_report(y_valid_encoded, y_pred_valid, target_names=le.classes_))
print(confusion_matrix(y_valid_encoded, y_pred_valid))

# Test
y_pred_test = best_svm.predict(X_test_pca)
print("\nTest:")
print("Akurasi:", accuracy_score(y_test_encoded, y_pred_test))
print(classification_report(y_test_encoded, y_pred_test, target_names=le.classes_))
print(confusion_matrix(y_test_encoded, y_pred_test))


model_artifacts = {
    'model': best_svm,       
    'scaler': scaler,
    'pca': pca,
    'label_encoder': le
}

with open('svm_fracture_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

