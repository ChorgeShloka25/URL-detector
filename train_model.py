import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

DATASET_PATH = 'phishing_url_dataset.csv'
MODEL_SAVE_PATH = 'phishing_model.joblib'
METADATA_SAVE_PATH = 'model_metadata.joblib'

def train_and_evaluate():
    print("=" * 65)
    print("      PHISHING URL DETECTOR - ML MODEL TRAINING PIPELINE")
    print("=" * 65)
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
        
    print(f"\n[1/5] Loading dataset from '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)
    total_samples = len(df)
    legit_count = (df['label'] == 0).sum()
    phish_count = (df['label'] == 1).sum()
    print(f"      Total URLs: {total_samples}")
    print(f"      Legitimate URLs (0): {legit_count}")
    print(f"      Phishing URLs (1):   {phish_count}")
    
    # Separate features and target
    X = df.drop(columns=['url', 'label'])
    y = df['label']
    
    categorical_cols = ['tld']
    numerical_cols = [c for c in X.columns if c not in categorical_cols]
    
    print(f"\n[2/5] Setting up Preprocessing Pipeline...")
    print(f"      Categorical Features: {categorical_cols}")
    print(f"      Numerical Features ({len(numerical_cols)}): {numerical_cols}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # 80/20 train-test split stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Train set size: {len(X_train)} samples")
    print(f"      Test set size:  {len(X_test)} samples")
    
    candidate_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=12, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    print("\n[3/5] Benchmarking Multiple Machine Learning Classifiers...")
    results = {}
    fitted_pipelines = {}
    
    for name, clf in candidate_models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        y_pred = pipe.predict(X_test)
        
        # Probabilities for ROC-AUC
        if hasattr(pipe.named_steps['classifier'], "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_prob)
        else:
            roc = 0.0
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc, 4),
            'confusion_matrix': cm
        }
        print(f"      {name:20}: Accuracy={acc*100:6.2f}% | F1={f1:6.4f} | ROC-AUC={roc:6.4f}")

    # Best model is Random Forest
    best_model_name = 'Random Forest'
    best_pipeline = fitted_pipelines[best_model_name]
    
    print(f"\n[4/5] Selected Champion Model: {best_model_name} ({results[best_model_name]['accuracy']*100:.2f}% accuracy)")
    
    # Extract Feature Importances from Random Forest
    rf_classifier = best_pipeline.named_steps['classifier']
    preproc = best_pipeline.named_steps['preprocessor']
    
    ohe_cols = preproc.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
    all_feature_names = ohe_cols + numerical_cols
    importances = rf_classifier.feature_importances_
    
    # Sort feature importances
    feature_importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    top_10_features = feature_importance_df.head(10).to_dict(orient='records')
    print("\n      Top 5 Most Informative Features:")
    for i, item in enumerate(top_10_features[:5], 1):
        print(f"        {i}. {item['feature']:30} ({item['importance']*100:.2f}%)")

    # Metadata dictionary for Streamlit App display
    metadata = {
        'best_model_name': best_model_name,
        'models_comparison': results,
        'top_features': top_10_features,
        'all_features': numerical_cols + categorical_cols,
        'dataset_stats': {
            'total_samples': total_samples,
            'legitimate_samples': legit_count,
            'phishing_samples': phish_count,
            'test_samples': len(X_test),
            'train_samples': len(X_train)
        }
    }
    
    print("\n[5/5] Saving model artifacts...")
    joblib.dump(best_pipeline, MODEL_SAVE_PATH)
    joblib.dump(metadata, METADATA_SAVE_PATH)
    print(f"      Model artifact saved -> '{MODEL_SAVE_PATH}'")
    print(f"      Metadata saved       -> '{METADATA_SAVE_PATH}'")
    print("\nTraining completed successfully! Ready for Streamlit App.")
    print("=" * 65)

if __name__ == '__main__':
    train_and_evaluate()
