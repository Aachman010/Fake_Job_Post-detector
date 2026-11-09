# Fake Job Posting Detector

An NLP-based Machine Learning application that identifies potentially fraudulent job postings using Natural Language Processing and classification algorithms.

The project combines text analysis and structured employment attributes to estimate the probability of a job posting being fraudulent. The trained model is integrated into a FastAPI backend with an interactive frontend built using HTML, CSS, and JavaScript.

## Project Overview

Online job platforms can contain fraudulent listings designed to mislead job seekers through fake employment opportunities, unrealistic salaries, registration fees, and requests for sensitive personal information.

This project aims to identify potentially fraudulent job postings by analyzing their textual content and employment-related attributes.

The application generates a fraud probability and classifies the posting using a configurable decision threshold.

## Features

- NLP-based job description analysis.
- TF-IDF text feature extraction.
- Structured employment attribute processing.
- Handling of imbalanced classification data.
- Comparison of Logistic Regression and Linear Support Vector Machine.
- Probability-based fraud classification.
- Configurable classification threshold.
- FastAPI backend for model inference.
- Interactive web interface.
- Input validation for insufficient job descriptions.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| NLP | TF-IDF Vectorizer |
| Classification | Logistic Regression, LinearSVC |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Model Serialization | Joblib |
| Development | Jupyter Notebook, VS Code |

## Dataset

The project uses a dataset containing approximately 17,880 job postings.

The target variable is `fraudulent`:

- `0` — Genuine job posting
- `1` — Fraudulent job posting

The dataset contains an imbalanced distribution:

| Class | Number of Records |
|---|---:|
| Genuine | 17,014 |
| Fraudulent | 866 |
| **Total** | **17,880** |

To address class imbalance, `class_weight='balanced'` was used during model training.

## Machine Learning Workflow

### 1. Data Preprocessing

- Handled missing values.
- Combined relevant textual fields into a single text feature.
- Removed irrelevant columns and identifiers.
- Separated text, categorical, binary, and numerical features.

### 2. Feature Engineering

The model uses three categories of features.

**Text features**
- Job title
- Company profile
- Job description
- Requirements
- Benefits

**Categorical features**
- Location
- Employment type
- Required experience
- Required education
- Industry
- Function

**Additional features**
- Telecommuting
- Company logo presence
- Screening questions
- Text length
- Word count

### 3. Feature Transformation

A Scikit-learn `ColumnTransformer` was used to process different feature types:

- `TfidfVectorizer(max_features=5000)` for textual data.
- `OneHotEncoder(handle_unknown='ignore')` for categorical features.
- `StandardScaler` for numerical and binary features.

### 4. Model Training

Two classification algorithms were implemented and evaluated using the same preprocessing pipeline.

## Model Comparison

A comparative evaluation was conducted to identify the most suitable classification algorithm for the project.

### Logistic Regression

- Used balanced class weights to address class imbalance.
- Evaluated at multiple probability thresholds.
- Supports probability estimation through `predict_proba()`.
- Threshold tuning was performed to balance fraud precision and recall.

### Linear Support Vector Machine (LinearSVC)

- Implemented using Scikit-learn's `LinearSVC`.
- Used the same preprocessing pipeline.
- Applied balanced class weights.
- Evaluated using classification metrics and a confusion matrix.

### Comparative Performance

The following results were obtained during model evaluation:

| Evaluation Metric | Logistic Regression (0.7 threshold) | Linear SVM |
|---|---:|---:|
| Accuracy | 98.74% | 98.66% |
| Fraud Precision | 87.21% | ~85% |
| Fraud Recall | 86.71% | ~88% |
| Fraud F1-score | 86.96% | ~86% |

*Linear SVM metrics are approximate. LinearSVC was evaluated using its standard decision predictions, not a probability threshold.*

### Threshold Optimization — Logistic Regression

Different classification thresholds were tested to understand the trade-off between fraud precision and recall.

| Threshold | Fraud Precision | Fraud Recall | Fraud F1-score |
|---|---:|---:|---:|
| 0.4 | 57.60% | 94.22% | 71.49% |
| 0.5 | 66.53% | 91.91% | 77.18% |
| 0.6 | 76.73% | 89.60% | 82.67% |
| 0.7 | 87.21% | 86.71% | 86.96% |

Increasing the threshold improved precision while reducing recall.

A threshold of 0.7 was selected as a practical balance between identifying fraudulent listings and reducing false fraud alerts.

### Final Model Selection

**Logistic Regression was selected as the final deployed model.**

Reasons:

1. Higher fraud precision than LinearSVC in the recorded comparison.
2. Better fraud F1-score.
3. Supports probability estimation through `predict_proba()`.
4. Allows probability-based threshold tuning.
5. Integrates naturally with the application's risk estimation interface.

**Final model:** Logistic Regression with a fraud probability threshold of 0.7.

*Evaluation note: Threshold selection was informed by evaluation results. An independent untouched test set is recommended for an unbiased final performance estimate.*

## Application Architecture

The trained ML pipeline is integrated into a FastAPI application.

```text
              User
                |
                v
       HTML / CSS / JavaScript
                |
                v
          FastAPI Backend
                |
                v
      Input Validation & Feature
             Preparation
                |
                v
      Saved Scikit-learn Pipeline
                |
                v
       Fraud Probability Output
                |
                v
       Threshold-Based Decision
                |
                v
       Prediction Result Display
```

## Project Structure

```text
fake_job_post_detector/
│
├── main.py
├── index.html
├── modeltraining.ipynb
├── fake_job_post_detector_model.pkl
├── requirements.txt
└── .gitignore
```

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version.
- Git.

### 1. Clone the Repository

```bash
git clone https://github.com/Aachman010/fake_job_post_detector.git
cd fake_job_post_detector
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
uvicorn main:app --reload
```

### 6. Open the Application

**Frontend:**

```text
http://127.0.0.1:8000/
```

**FastAPI Interactive Documentation:**

```text
http://127.0.0.1:8000/docs
```

## API Usage

**Endpoint:** `POST /predict`

The API accepts job listing text and employment-related attributes.

### Example Request

```json
{
  "text": "Urgent hiring! Work from home...",
  "location": "India",
  "employment_type": "Full-time",
  "required_experience": "Entry level",
  "required_education": "Bachelor's Degree",
  "industry": "Information Technology",
  "function": "Engineering",
  "telecommuting": 1,
  "has_company_logo": 0,
  "has_questions": 0
}
```

### Example Response

```json
{
  "prediction": 1,
  "result": "Fraudulent",
  "fraud_probability": 0.9489,
  "threshold": 0.7
}
```

*The response is an illustrative example.*

## Limitations

- Model performance depends on the quality and distribution of the training dataset.
- Unseen job posting patterns may affect prediction accuracy.
- A high fraud probability does not prove that a listing is fraudulent.
- The application should be treated as a screening tool, not a definitive fraud verification system.
- Empty or insufficient job descriptions should be rejected rather than classified.

## Future Improvements

- Evaluate the model on independent real-world job postings.
- Experiment with additional machine learning algorithms.
- Improve generalization on unseen job listings.
- Add explainability to highlight suspicious listing attributes.
- Deploy the application to a cloud hosting platform.
- Expand the dataset with more recent job postings.

## Author

**Aachman Bhardwaj**

GitHub: [Aachman010](https://github.com/Aachman010)

---

If you find this project useful, consider giving the repository a ⭐.
