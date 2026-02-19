# ML-Training-Studio

A Streamlit-based GUI application that mimics cloud platform workflows for tabular machine learning model development. Users can upload datasets, select features/targets, choose algorithms, configure hyperparameters, train models, and inspect metrics through a low-code interface.

## Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the app locally**:
   ```bash
   streamlit run app.py
   ```
You can use the included `sample_data.csv` to try out the interface or upload your own tabular dataset.3. **Deploy** on [Streamlit Community Cloud](https://streamlit.io/cloud) or any other hosting provider. Ensure the repository contains the `app.py` and `requirements.txt`.

## Running Tests

To run the included unit tests:

```bash
PYTHONPATH=. pytest
```

## Features

* CSV dataset upload
* Dynamic feature & target selection
* Multiple algorithms (Logistic Regression, Decision Tree, Random Forest, SVC)
* Hyperparameter tuning
* Train/test split configuration
* Metrics display (accuracy, classification report, confusion matrix)
* Downloadable trained model artifact

## License

This project is released under the MIT License.
