# MailShield | Email Spam Classifier

**A machine learning system that separates unwanted messages from legitimate communication.**

MailShield is a Python-based NLP project that learns patterns in text messages to identify spam. It compares Logistic Regression and Multinomial Naive Bayes, evaluates their performance, selects a model based on spam-class F1 score, and serves predictions through a Flask API.

The goal is simple: build a complete machine learning workflow, from raw text to a usable prediction service.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Machine_Learning-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/NLP-TF--IDF-7C3AED?style=flat-square" alt="NLP"/>
  <img src="https://img.shields.io/badge/API-Flask-222222?style=flat-square&logo=flask&logoColor=white" alt="Flask"/>
</p>

---

## 01 — The Idea

Spam detection is a binary classification problem, but getting it right involves more than achieving high accuracy. A useful classifier should detect unwanted messages without incorrectly marking legitimate communication as spam.

MailShield explores this trade-off through two traditional machine learning algorithms and a shared evaluation pipeline.

**The system can:**

- Classify a message as `spam` or `ham`.
- Compare two models using the same dataset split.
- Select the model with the higher spam-class F1 score.
- Generate visual reports to understand classification errors.
- Expose predictions through a JSON-based REST API.

## 02 — Inside the Pipeline

```text
        SMS Spam Collection
                 |
                 v
         Data Preparation
                 |
                 v
          Text Cleaning
                 |
                 v
          TF-IDF Features
                 |
          +------+------+
          |             |
          v             v
     Logistic        Multinomial
    Regression      Naive Bayes
          |             |
          +------+------+
                 |
                 v
        Evaluation & Comparison
                 |
                 v
       Best Model by Spam F1
                 |
                 v
        Saved Model Artifacts
                 |
                 v
            Flask API
                 |
                 v
          Spam / Ham Result
```

### How a prediction happens

1. A message is received by the API.
2. The text is cleaned using the project's preprocessing rules.
3. The fitted TF-IDF vectorizer converts the message into numerical features.
4. The selected classifier predicts the message category.
5. The API returns the prediction and, where supported, a confidence score.

The same fitted vectorizer must be used during training and inference to maintain consistent feature representations.

## 03 — Technology Choices

| Technology | Role in the project |
|---|---|
| Python | Core implementation |
| Pandas | Dataset loading and manipulation |
| Scikit-learn | TF-IDF, training, and evaluation |
| Logistic Regression | Linear text classification |
| Multinomial Naive Bayes | Probabilistic text classification |
| Flask | Prediction API |
| Matplotlib | Performance visualizations |
| Pickle | Saving and loading model artifacts |

## 04 — Repository Layout

```text
email-spam-classifier/
│
├── data/
│   └── SMSSpamCollection
│
├── models/
│   ├── lr_model.pkl
│   ├── nb_model.pkl
│   ├── best_model.pkl
│   ├── best_model_info.txt
│   ├── vectorizer.pkl
│   └── test_data.pkl
│
├── reports/
│   ├── confusion_matrices.png
│   └── precision_recall_curves.png
│
├── data.py
├── utils.py
├── vectorizer.py
├── model.py
├── evaluate.py
├── main.py
├── app.py
├── requirements.txt
└── README.md
```

*This layout reflects the intended project organization. Keep only files and directories that exist in your actual repository.*

## 05 — Run It Locally

### Requirements

- Python 3.9 or later, compatible with the installed dependencies
- pip
- Git

### Step 1: Get the code

```bash
git clone https://github.com/10pateladitya/email-spam-classifier.git
cd email-spam-classifier
```

### Step 2: Set up the environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Train the classifiers

```bash
python model.py
```

This should train both classifiers and save the artifacts configured by the training pipeline.

### Step 5: Generate evaluation reports

```bash
python evaluate.py
```

The evaluation script generates the comparison metrics and visualizations in `reports/`, assuming the required model artifacts are available.

### Step 6: Start the API

```bash
python app.py
```

The API is expected to run at:

`http://localhost:5000`

Run the training and evaluation steps first if your API requires previously saved models.

## 06 — Try the API

**Endpoint:** `POST /predict`

Send a message as JSON:

```json
{
  "text": "Congratulations! You have won a prize. Call now to claim."
}
```

Example response:

```json
{
  "prediction": "spam",
  "confidence": 0.98,
  "model_used": "Logistic Regression"
}
```

The response is illustrative; actual predictions, confidence values, and field names depend on the implementation.

### Request using cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Congratulations! Claim your free prize now.\"}"
```

| Response field | Description |
|---|---|
| `prediction` | Predicted category: `spam` or `ham` |
| `confidence` | Probability associated with the prediction, if returned |
| `model_used` | Classifier used for inference |

## 07 — Understanding the Text

Before classification, the raw message passes through a lightweight preprocessing pipeline.

```text
Raw message
     ↓
Lowercase conversion
     ↓
Special-character cleaning
     ↓
Whitespace normalization
     ↓
TF-IDF transformation
     ↓
Numerical feature vector
```

The current configuration uses TF-IDF with a maximum of 5,000 features and unigram tokens, according to the supplied project description.

TF-IDF represents words numerically while reducing the influence of terms that occur frequently across many messages.

## 08 — Dataset

The project uses the **UCI SMS Spam Collection**, a labelled dataset containing 5,574 SMS messages.

| Category | Messages | Approx. share |
|---|---:|---:|
| Legitimate (`ham`) | 4,827 | 86.6% |
| Spam | 747 | 13.4% |
| **Total** | **5,574** | **100%** |

The uneven class distribution makes evaluation particularly important. A model that predicts every message as legitimate could still achieve approximately 86.6% accuracy while detecting no spam at all.

## 09 — Measuring Performance

MailShield compares the models using multiple metrics rather than treating accuracy as the only measure of success.

| Metric | Why it matters |
|---|---|
| Accuracy | Measures overall correctness |
| Precision | Measures how many predicted spam messages are actually spam |
| Recall | Measures how much actual spam is detected |
| F1 score | Balances spam precision and recall |
| Average Precision | Summarizes performance across precision-recall thresholds |

### Reported baseline

The initial project notes report the following approximate results. These figures should be regenerated and verified against the current implementation before being treated as final benchmark results.

| Metric | Logistic Regression | Multinomial Naive Bayes |
|---|---:|---:|
| Accuracy | ~98% | ~97% |
| Spam precision | ~96% | ~100% |
| Spam recall | ~94% | ~88% |
| Spam F1 | ~95% | ~93% |

**Model selection:** The classifier with the higher spam-class F1 score is selected automatically.

Logistic Regression offers a balance between precision and recall in the reported results. Naive Bayes has higher reported precision but lower recall, illustrating the trade-off between false alarms and missed spam.

### Visual reports

After evaluation, the project generates:

- `reports/confusion_matrices.png` — compares true positives, true negatives, false positives, and false negatives.
- `reports/precision_recall_curves.png` — visualizes the precision-recall trade-off and Average Precision for each classifier.

These plots help reveal errors that a single metric cannot explain.

## 10 — Engineering Lessons

This project helped me explore several important machine learning concepts:

- Turning unstructured text into numerical features.
- Comparing linear and probabilistic classification approaches.
- Evaluating imbalanced datasets with appropriate metrics.
- Keeping training and inference preprocessing consistent.
- Saving trained models for reuse without retraining on every request.
- Integrating a machine learning model with a REST API.
- Interpreting model errors using diagnostic visualizations.

## 11 — Next Improvements

- [ ] Experiment with bigrams and alternative text-cleaning strategies.
- [ ] Add cross-validation and hyperparameter tuning.
- [ ] Introduce automated tests for preprocessing and API responses.
- [ ] Add a confidence threshold for uncertain predictions.
- [ ] Compare the baseline with a transformer-based text classifier.
- [ ] Containerize the application using Docker.
- [ ] Build a minimal web interface for interactive predictions.
- [ ] Explore FastAPI for API validation and interactive documentation.

## About

**Aditya Patel**

GitHub: [@10pateladitya](https://github.com/10pateladitya)

This project is part of my exploration of machine learning, NLP, and practical model deployment.

---

<p align="center">
  <sub>MailShield · A hands-on exploration of text classification and applied machine learning.</sub>
</p>
