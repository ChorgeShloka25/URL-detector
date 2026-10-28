import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# File paths
DATASET_PATH    = 'phishing_url_dataset.csv'
MODEL_SAVE_PATH = 'phishing_model.joblib'
METADATA_SAVE_PATH = 'model_metadata.joblib'


def train_and_evaluate():
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    total   = len(df)
    legit   = (df['label'] == 0).sum()
    phishing = (df['label'] == 1).sum()
    print(f"  Total: {total} | Legitimate: {legit} | Phishing: {phishing}")

    X = df.drop(columns=['url', 'label'])
    y = df['label']


    cat_cols = ['tld']
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(transformers=[
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ], remainder='passthrough')

    # 80% training, 20% testing (stratify keeps class ratio equal in both)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    models = {
        'Random Forest':     RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'Decision Tree':     DecisionTreeClassifier(max_depth=12, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    }

    results, pipelines = {}, {}
    print("\nTraining models...")

    for name, clf in models.items():
        # Each pipeline: preprocess → classify
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred).tolist()

        roc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]) \
              if hasattr(clf, 'predict_proba') else 0.0

        results[name] = {
            'accuracy': round(acc, 4), 'precision': round(prec, 4),
            'recall': round(rec, 4),   'f1_score': round(f1, 4),
            'roc_auc': round(roc, 4),  'confusion_matrix': cm
        }
        print(f"  {name:22}: Acc={acc*100:.2f}% | F1={f1:.4f} | ROC-AUC={roc:.4f}")

   
    best_name = 'Random Forest'
    best_pipe = pipelines[best_name]
    print(f"\nBest model: {best_name} ({results[best_name]['accuracy']*100:.2f}% accuracy)")

    # Get feature importances from Random Forest
    rf   = best_pipe.named_steps['classifier']
    ohe  = best_pipe.named_steps['preprocessor'].named_transformers_['ohe']
    all_features = ohe.get_feature_names_out(cat_cols).tolist() + num_cols

    feat_df = pd.DataFrame({'feature': all_features, 'importance': rf.feature_importances_})
    feat_df = feat_df.sort_values('importance', ascending=False)
    top10 = feat_df.head(10).to_dict(orient='records')

    print("\nTop 5 features:")
    for i, row in enumerate(top10[:5], 1):
        print(f"  {i}. {row['feature']:30} ({row['importance']*100:.2f}%)")

    # Save model + metadata 
    metadata = {
        'best_model_name':   best_name,
        'models_comparison': results,
        'top_features':      top10,
        'all_features':      num_cols + cat_cols,
        'dataset_stats': {
            'total_samples':      total,
            'legitimate_samples': legit,
            'phishing_samples':   phishing,
            'train_samples':      len(X_train),
            'test_samples':       len(X_test),
        }
    }

    joblib.dump(best_pipe, MODEL_SAVE_PATH)
    joblib.dump(metadata,  METADATA_SAVE_PATH)
    print(f"\nModel saved  → {MODEL_SAVE_PATH}")
    print(f"Metadata saved → {METADATA_SAVE_PATH}")
    print("Done! Run `streamlit run app.py` to launch the app.")


if __name__ == '__main__':
    train_and_evaluate()
