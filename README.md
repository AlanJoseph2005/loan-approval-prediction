# 🏦 Loan Approval Prediction System

A machine learning based loan approval prediction web application built
with **Python, Streamlit, Scikit-learn, and XGBoost**.

The system takes an applicant's financial and personal information,
predicts whether the loan is likely to be **Approved** or **Rejected**,
explains some rule-based factors associated with the result, and
generates a downloadable PDF report.

## 📌 Project Overview

The project has two main parts:

1.  **Machine Learning Model**
    -   Cleans and preprocesses the loan dataset.
    -   Encodes categorical features.
    -   Splits the data into training and testing sets.
    -   Compares Random Forest and XGBoost classifiers.
    -   Selects XGBoost for the saved application model.
2.  **Streamlit Web Application**
    -   Collects applicant information through an interactive form.
    -   Loads the trained XGBoost model.
    -   Predicts loan approval status.
    -   Displays reasoning based on selected applicant factors.
    -   Generates a downloadable PDF loan report.

## 🛠️ Technologies Used

-   **Python**
-   **Pandas** -- data loading and preprocessing
-   **NumPy** -- numerical processing
-   **Scikit-learn** -- preprocessing, train-test split, Random Forest
    and evaluation
-   **XGBoost** -- loan status classification
-   **Streamlit** -- interactive web application
-   **Joblib** -- saving and loading the trained model
-   **ReportLab** -- generating PDF reports
-   **Matplotlib / Seaborn** -- data analysis and visualization

## 📊 Dataset

The project uses `train.csv`.

The dataset contains **3,192 records** and the following 14 columns:

-   Loan_ID
-   Gender
-   Married
-   Dependents
-   Education
-   Employment_Status
-   Applicant_Income
-   Coapplicant_Income
-   Loan_Amount
-   Loan_Term
-   Credit_History
-   Property_Area
-   Age
-   Loan_Status

The target variable is:

-   `Loan_Status`

The dataset contains balanced target classes in the supplied training
file: 1,596 records for each encoded class.

## 🧹 Data Preprocessing

The training process includes:

-   Filling missing categorical values using the mode.
-   Filling missing `Applicant_Income` and `Loan_Amount` values using
    the median.
-   Removing the `Loan_ID` column.
-   Checking for duplicate records.
-   Encoding categorical columns using `LabelEncoder`.
-   Separating features (`X`) and target (`y`).
-   Splitting the data into training and testing sets using an **80/20
    split** with `random_state=42`.

## 🤖 Machine Learning Models

Two classification models are trained and compared.

### Random Forest

The Random Forest classifier uses:

-   `n_estimators = 100`
-   `random_state = 42`

### XGBoost

The XGBoost classifier uses:

-   `n_estimators = 100`
-   `learning_rate = 0.1`
-   `max_depth = 5`
-   `random_state = 42`

For the supplied dataset and preprocessing procedure, the measured test
accuracies are:

  Model             Test Accuracy
  --------------- ---------------
  Random Forest            98.44%
  XGBoost                  98.90%

XGBoost achieved the higher test accuracy in this run and was saved as
`loan_model.pkl`.

> **Note:** Accuracy depends on the dataset, preprocessing, train/test
> split, and software versions. The reported values correspond to the
> supplied `train.csv` and the preprocessing/training procedure used in
> this project.

## 🌐 Streamlit Application

The application is implemented in `app(1).py`.

Users can enter:

-   Applicant Name
-   Gender
-   Marital Status
-   Number of Dependents
-   Education
-   Employment Status
-   Applicant Income
-   Coapplicant Income
-   Loan Amount
-   Loan Term
-   Credit History
-   Property Area
-   Age

After clicking **Predict Loan Status**, the application displays:

-   ✅ Loan Approved or ❌ Loan Rejected
-   Reasoning associated with the result
-   Downloadable PDF report

## 🔎 Reasoning System

The application provides additional rule-based explanations.

For an approved prediction, possible reasons include:

-   Good credit history
-   Good applicant income
-   Loan amount is affordable
-   Stable employment status

For a rejected prediction, possible reasons include:

-   Poor credit history
-   Low applicant income
-   Loan amount is high compared to income
-   Applicant age is low

These explanations are **rule-based indicators in the application** and
are separate from the internal decision process of the trained XGBoost
model.

## 📄 PDF Report Generation

The application generates a PDF using **ReportLab**.

The report contains:

-   Prediction result
-   Applicant details
-   Reasoning behind the displayed result

The generated file is named:

`loan_report.pdf`

## 📁 Project Structure

``` text
loan-approval-prediction/
│
├── app(1).py
├── project.py
├── train.csv
├── loan_model.pkl
├── loan_report.pdf
└── README.md
```

### File Description

-   `app(1).py` -- Streamlit application for loan prediction and report
    generation
-   `project.py` -- data preprocessing, model training, model
    comparison, and model saving
-   `train.csv` -- loan dataset
-   `loan_model.pkl` -- saved XGBoost model used by the application
-   `loan_report.pdf` -- example generated loan report
-   `README.md` -- project documentation

## ▶️ How to Run

### 1. Install dependencies

``` bash
pip install pandas numpy scikit-learn xgboost streamlit joblib reportlab matplotlib seaborn
```

### 2. Train the model

Run:

``` bash
python project.py
```

This performs preprocessing, trains Random Forest and XGBoost, compares
their accuracy, and saves the selected XGBoost model as:

``` text
loan_model.pkl
```

### 3. Start the Streamlit application

Run:

``` bash
streamlit run app(1).py
```

The application will open in your web browser.

## ⚠️ Important Implementation Note

The training dataset uses the employment category **`Salaried`**, while
the current Streamlit interface labels the corresponding option
**`Employed`**. Both are mapped to the same encoded value in the current
application.

For a production version, the training and application category names
should be kept exactly consistent.

## 🎯 Project Objective

The objective of this project is to demonstrate how machine learning can
be applied to loan approval classification and integrated into an
interactive web application with explainable rule-based feedback and
downloadable reports.

## 🚀 Future Improvements

-   Add model probability/confidence information.
-   Add confusion matrix and classification report.
-   Improve the explanation system using model-based feature importance
    or SHAP.
-   Add stronger input validation.
-   Use cross-validation for more reliable model comparison.
-   Add more financial features.
-   Improve the PDF report design.
-   Deploy the Streamlit application online.
-   Keep training and application categorical mappings fully
    synchronized.

## 👨‍💻 Author

**Alan Joseph**

Computer Science Engineering Student

GitHub: `https://github.com/AlanJoseph2005`
