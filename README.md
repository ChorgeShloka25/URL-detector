# 🛡️ AI/ML Phishing URL Detector

An intelligent machine learning micro-project developed in **Python** and **Streamlit** to detect malicious and phishing URLs in real time. The model is trained on a balanced dataset of 1,600 URLs and achieves **100% test accuracy** (well above the 70–80% goal), paired with an explainable AI dashboard.

---

## 🌟 Key Features

1. **High-Accuracy ML Model**:
   - Uses an ensemble **Random Forest Classifier** achieving **100.00% accuracy** on test data.
   - Also evaluated against **Gradient Boosting (99.69%)**, **Decision Tree (100.00%)**, and **Logistic Regression (99.69%)**.
2. **20-Dimensional Feature Extraction**:
   - **Lexical**: URL length, hostname length, path length, dots, hyphens, underscores, digits, special characters.
   - **Structural**: Subdomain depth, path segments, query parameters, `@` userinfo delimiter, TLD.
   - **Security Heuristics**: Raw IP address detection, HTTPS encryption status, shortening services (`bit.ly`, `tinyurl`, `goo.gl`, etc.), Shannon character entropy, and numeric ratio.
   - **Targeted Brand Spoofing & Phishing Triggers**: Detects spoofed brand names (`netflix`, `google`, `microsoft`, `apple`, `chase`, `paypal`, `sbi`, etc.) and sensitive attack keywords (`login`, `signin`, `verify`, `banking`, `confirm`, `password`, `suspend`, `unlock`, etc.).
3. **Interactive Streamlit Web Interface**:
   - **Single URL Scanner**: Live URL input with instant risk scoring (0–100%) and threat breakdown.
   - **Preloaded Demo URLs**: 1-click test buttons for both authentic and malicious links.
   - **Explainable AI (XAI)**: Explicitly flags red flags (e.g. brand impersonation, unencrypted connection, suspicious keywords, excessive subdomains).
   - **Batch URL Scanner**: Scan multiple URLs simultaneously or upload a CSV file and download analyzed results.
   - **Model Analytics & Metrics**: Interactive confusion matrix, comparative model benchmark table, and top feature importance graph.

---

## 📁 Project Structure

```
Phishing URL Detector/
│
├── phishing_url_dataset.csv     # Training dataset (1,600 balanced URLs)
├── feature_extractor.py         # 20-feature extraction engine
├── train_model.py               # Model training, evaluation & persistence script
├── phishing_model.joblib        # Saved champion Random Forest pipeline
├── model_metadata.joblib        # Saved evaluation metrics & feature importances
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Install Dependencies
Open your terminal / command prompt in this project folder and run:
```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-train the Machine Learning Model
The model artifacts (`phishing_model.joblib` and `model_metadata.joblib`) are already trained and saved. To re-train or inspect terminal output:
```bash
python train_model.py
```

### 3. Launch the Streamlit Web Application
Run the following command in terminal:
```bash
streamlit run app.py
```
This will automatically launch the web application in your default browser at `http://localhost:8501`.

---

## 📊 Model Evaluation Results

| Algorithm | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Decision Tree | 100.00% | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gradient Boosting | 99.69% | 0.9938 | 1.0000 | 0.9969 | 1.0000 |
| Logistic Regression | 99.69% | 0.9938 | 1.0000 | 0.9969 | 1.0000 |

### Top 5 Predictive Features
1. `num_suspicious_keywords` (16.58%)
2. `num_subdomains` (10.03%)
3. `url_length` (10.00%)
4. `num_digits` (8.32%)
5. `has_https` (7.85%)

---

## 🛡️ License & Academic Usage
This project is suitable as an Academic AI/ML Micro-Project, Mini-Project, or Cybersecurity Portfolio Demonstration.
