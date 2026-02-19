# 🧪 ML Training Studio

A cloud-style machine learning GUI built with Streamlit. Upload your data, explore it visually, train models, and track experiments — all without writing a single line of code.

> Built as a portfolio project for CIS432 — inspired by platforms like Google Cloud Vertex AI and AWS SageMaker.

---

## 🚀 Live Demo

<!-- After deploying, replace this with your Streamlit Cloud link -->
🔗 [Launch App](https://ml-training-studio.streamlit.app)

---

## ✨ Features

### 01 · Upload Data
- Upload any CSV file
- Instant preview of rows, columns, and missing value count
- Data type inspection and missing value summary
- Descriptive statistics table

### 02 · Visualize Data
- **Distribution plots** — histogram for any numeric column
- **Correlation heatmap** — see relationships between all numeric features
- **Categorical counts** — bar chart for categorical columns

### 03 · Configure & Train
- Select target column and feature columns
- Choose task type: **Classification** or **Regression**
- Supported models:
  - Logistic Regression / Linear Regression
  - Decision Tree
  - Random Forest
- Validation methods: **Train/Test Split** or **5-Fold Cross-Validation**
- Configurable test size and random seed for reproducibility

### 04 · Evaluate Results
- **Classification**: Accuracy, Confusion Matrix, Classification Report
- **Regression**: RMSE, MAE, R², Actual vs Predicted scatter plot
- **Feature Importance** chart (for tree-based models)

### 05 · Experiment Tracking
- Automatically logs each training run with name, model, and metrics
- Compare experiments side by side in a table
- Download experiment log as CSV

---

## 🛠️ Installation

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ml-training-studio.git
cd ml-training-studio

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web GUI framework |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | ML models and evaluation metrics |

---

## 📁 Project Structure

```
ml-training-studio/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
---

## 🗺️ Roadmap

Planned features for future versions:

- [ ] More preprocessing options (StandardScaler, One-Hot Encoding)
- [ ] Hyperparameter tuning interface
- [ ] SHAP explainability plots
- [ ] Export trained model as `.pkl` file
- [ ] Support for time-based train/test splits
- [ ] Stratified cross-validation

---

## 🙏 Acknowledgements

- Inspired by [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai) and [AWS SageMaker](https://aws.amazon.com/sagemaker/)
- Example educational Streamlit app: [flexibility.streamlit.app](https://flexibility.streamlit.app/)
- Built with [Streamlit](https://streamlit.io/), [scikit-learn](https://scikit-learn.org/)
