# Customer Churn Data Prediction

Predicting customer churn is a common and valuable task for subscription- and contract-based businesses. This repository contains code, exploratory analysis, preprocessing steps, and example machine learning models to predict whether a customer will churn based on historical features.

Contents
- Overview
- Repository structure
- Dataset
- Features & preprocessing
- Models included
- How to run
- Evaluation & expected outputs
- Reproducibility
- Contributing
- License

Overview
--------
This project demonstrates a typical churn-prediction workflow:
1. Load and inspect customer data.
2. Clean and preprocess features (encoding, scaling, handling missing values).
3. Perform exploratory data analysis (EDA) and feature engineering.
4. Train baseline and advanced models (e.g., Logistic Regression, Random Forest, XGBoost).
5. Evaluate models using metrics suitable for classification (accuracy, precision, recall, F1, ROC-AUC).
6. Save the best model and report findings.

Repository structure (example)
------------------------------
The repository is organized to keep analysis, code, and outputs separate and reproducible.

- data/
  - raw/                # original datasets (do not commit sensitive data)
  - processed/          # cleaned / preprocessed datasets used for modeling
- notebooks/
  - 01_EDA.ipynb        # exploratory data analysis and visualizations
  - 02_Feature_Engineering.ipynb
  - 03_Modeling.ipynb   # training and evaluation in notebook form
- src/
  - data_processing.py  # helper functions for loading and preprocessing data
  - features.py         # feature engineering utilities
  - train.py            # script to train models (CLI-friendly)
  - evaluate.py         # evaluation and metrics reporting
- models/
  - best_model.pkl      # serialized trained model (example)
- reports/
  - figures/            # saved plots from EDA and evaluation
  - metrics.md          # summary of model performance
- requirements.txt      # Python dependencies
- README.md

Dataset
-------
This repository expects a tabular dataset containing one row per customer and a target column indicating churn (commonly called `Churn`, `Exited`, `is_churn`, or similar). Typical features include customer demographics, account information, service usage, tenure, and billing/payment details.

If the dataset is not included due to privacy or size, create a `data/raw/` folder and put the CSV there (e.g., `customer_churn.csv`). Make sure to add the dataset name and path to any configuration used by scripts or notebooks.

Features & preprocessing
------------------------
Common preprocessing steps included in this project:
- Imputation of missing values (median, mode, or domain-specific rules).
- Encoding categorical variables (one-hot or ordinal encoding).
- Scaling numeric features (StandardScaler/MinMaxScaler) for algorithms that require it.
- Handling class imbalance (resampling techniques or class-weighted models).
- Splitting data into train/validation/test sets with a reproducible random seed.

Models included
---------------
This repo contains code to train and compare multiple classification models:
- Baseline: Logistic Regression
- Tree-based: Decision Tree, Random Forest
- Gradient boosting: XGBoost or LightGBM (if installed)
- Optionally: hyperparameter tuning with GridSearchCV or RandomizedSearchCV

How to run
----------
1. Clone the repository
   git clone https://github.com/Maheswaran101/Customer-Churn-Data-prediction.git
   cd Customer-Churn-Data-prediction

2. Create a Python environment and install dependencies
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows
   pip install -r requirements.txt

3. Place your dataset
   - Put your CSV into `data/raw/` and update any file-path variables in notebooks/scripts if necessary.

4. Run notebooks (recommended for exploration)
   - Launch Jupyter:
     jupyter notebook
   - Open and run `notebooks/01_EDA.ipynb`, `02_Feature_Engineering.ipynb`, `03_Modeling.ipynb`

5. Run scripts (example)
   - Train a model from the command line:
     python src/train.py --data-path data/processed/train.csv --output-dir models/
   - Evaluate a saved model:
     python src/evaluate.py --model-path models/best_model.pkl --test-data data/processed/test.csv

(If these scripts are not present, use the notebooks to run the pipeline interactively.)

Evaluation & expected outputs
-----------------------------
Key evaluation metrics for churn prediction:
- Accuracy
- Precision / Recall
- F1 score
- ROC-AUC score
- Confusion matrix

A practical deployment consideration is to prioritize recall (or a suitable business-oriented metric) for the churn class if the cost of missing churners is high.

Reproducibility
---------------
- Use the same random seeds across preprocessing, train/test splitting and model training.
- Save preprocessing artifacts (encoders, scalers) alongside models so inference uses the same transformations.
- Store model hyperparameters and training metrics in a small metadata file (e.g., JSON) when saving models.

Contributing
------------
Contributions are welcome! Suggested ways to contribute:
- Improve preprocessing and feature engineering.
- Add new models or hyperparameter tuning pipelines.
- Add tests for data processing functions.
- Improve documentation and example notebooks.

Please follow standard GitHub flow:
- Fork the repository
- Create a feature branch
- Open a PR describing your changes

License
-------
Specify the license you want to use (e.g., MIT). If none is specified in this repo, add a LICENSE file before redistribution.

Acknowledgements
----------------
- This project is a template/workflow for churn prediction problems and can be adapted to other binary classification tasks.
- If you used a public dataset, list it here and include licensing/attribution information.

Contact
-------
Created by Maheswaran101. For questions or collaboration, open an issue on the repository or contact via your GitHub profile.
