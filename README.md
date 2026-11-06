# Datheon Machine Learning & Data Science Internship

Projects completed for the Datheon Machine Learning & Data Science Internship.

## Task 1: Iris Species Classification
**Notebook:** `task1_iris_classification/iris_classification.ipynb`

Supervised classification on the Iris dataset, comparing Logistic Regression and a Decision Tree.

- Exploratory pairplot (Seaborn)
- 80/20 stratified train/test split
- Feature scaling with StandardScaler (fitted on training data only)
- Accuracy, precision, recall and F1-score benchmarks
- Confusion matrix heatmaps and decision boundary plots

| Model | Accuracy | F1 (macro) |
|---|---|---|
| Logistic Regression | 0.933 | 0.933 |
| Decision Tree | 0.967 | 0.967 |

Setosa is classified perfectly by both models. All errors are between versicolor and virginica. With only 30 test samples, the gap between the two models is a single flower.

## Task 2: California Housing Price Prediction
**Notebook:** `task2_california_housing/housing_regression.ipynb`

Predicting median house value and comparing OLS, Ridge (L2) and Lasso (L1) regression.

- Median imputation of missing `total_bedrooms` values
- Correlation heatmap and VIF analysis to detect and reduce multicollinearity
- Engineered ratio features and mean-centred interaction features
- Standardisation of numeric features and one-hot encoding of `ocean_proximity`
- Ridge and Lasso alpha tuning with 5-fold GridSearchCV
- Predicted vs actual plots and residual distribution plots

| Model | RMSE | MAE | R² |
|---|---|---|---|
| OLS | $71,558 | $51,255 | 0.609 |
| Ridge | $71,563 | $51,260 | 0.609 |
| Lasso | $71,558 | $51,255 | 0.609 |

All three models perform almost identically once the features are engineered and multicollinearity is reduced. The remaining error comes from non-linear location effects and the $500,001 cap on the target.

## Task 3: Telco Customer Churn Classifier (Random Forest & XGBoost)
**Notebook:** `task3_customer_churn/churn_classifier.ipynb`

Predicting which telecom customers will churn, with a focus on handling class imbalance (26.5% churn) and optimising ROC-AUC.

- Converted `TotalCharges` to numeric; the 11 blank rows are new customers with zero tenure, so they were set to 0
- Exploratory analysis of churn by contract, internet service, tenure and charges
- `OneHotEncoder` for categorical features inside the pipeline
- Imbalance handled two ways and compared: class weighting (`class_weight`, `scale_pos_weight`) and SMOTE (applied inside cross-validation folds only)
- Random Forest and XGBoost with 5-fold stratified cross-validation
- ROC curves, confusion matrices and top-10 feature importances

| Model | ROC-AUC | Recall (churn) | Precision (churn) |
|---|---|---|---|
| RF + class weight | 0.842 | 0.762 | 0.548 |
| RF + SMOTE | 0.837 | 0.610 | 0.571 |
| XGBoost + scale_pos_weight | 0.846 | 0.789 | 0.522 |
| XGBoost + SMOTE | 0.846 | 0.570 | 0.632 |

All four models exceed the 0.80 ROC-AUC target. Month-to-month contracts, short tenure, no online security or tech support, and fibre optic internet are the strongest churn signals. Class weighting catches far more churners than SMOTE at the cost of more false alarms.

## Task 4: NLP Sentiment Classification (TF-IDF)
**Notebook:** `task4_sentiment_analysis/sentiment_classifier.ipynb`

Classifying movie reviews as positive or negative using TF-IDF features and Multinomial Naive Bayes / Logistic Regression.

- Text cleaning: lowercasing, HTML and punctuation removal, tokenization, stopword removal (negations such as "not" are deliberately kept), WordNet lemmatization
- `TfidfVectorizer(max_features=5000)` inside a pipeline, so the vocabulary is learned from training data only
- Hyperparameters tuned with 5-fold cross-validation
- Accuracy, precision, recall, F1, ROC-AUC and confusion matrices
- Extension comparing unigram and bigram features
- `predict_sentiment()` function and an interactive helper for testing custom phrases

| Model | CV accuracy | Test accuracy | Test F1 |
|---|---|---|---|
| Multinomial Naive Bayes | 0.803 | 0.813 | 0.800 |
| Logistic Regression | 0.846 | 0.848 | 0.848 |
| Multinomial NB + bigrams | 0.814 | 0.803 | 0.793 |
| Logistic Regression + bigrams | 0.849 | 0.863 | 0.862 |

Logistic Regression outperforms Naive Bayes on this dataset. Both struggle with negation such as "not bad", a known limitation of bag-of-words models.

## Project structure
```
.
├── README.md
├── requirements.txt
├── task1_iris_classification/
│   ├── iris_classification.ipynb
│   └── iris.csv
├── task2_california_housing/
│   ├── housing_regression.ipynb
│   └── california_housing.csv
├── task3_customer_churn/
│   ├── churn_classifier.ipynb
│   └── telco_customer_churn.csv
└── task4_sentiment_analysis/
    ├── sentiment_classifier.ipynb
    └── movie_reviews.csv
```

## How to run
```
pip install -r requirements.txt
jupyter notebook
```
Open a notebook and choose Run All. Each notebook reads its dataset from its own folder.

## Tools
Python, Pandas, NumPy, Scikit-Learn, XGBoost, imbalanced-learn (SMOTE), NLTK, Seaborn, Matplotlib
