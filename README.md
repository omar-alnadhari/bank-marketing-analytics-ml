# Bank Marketing Analytics & Subscription Prediction

![Python](https://img.shields.io/badge/Python-3.14.6-blue)
![Tests](https://img.shields.io/badge/tests-33%20passed-brightgreen)
![Dataset](https://img.shields.io/badge/dataset-UCI%20Bank%20Marketing-orange)
![Dataset License](https://img.shields.io/badge/dataset%20license-CC%20BY%204.0-lightgrey)

An end-to-end data science project that analyzes bank marketing campaigns, identifies customer segments associated with term-deposit subscription, and develops a leakage-safe machine-learning model for pre-campaign client prioritization.

The project combines:

- Python data analysis
- Data validation and cleaning
- Exploratory data analysis
- SQL business analysis
- Statistical hypothesis testing
- Feature engineering
- Machine-learning pipelines
- Hyperparameter tuning
- Classification-threshold selection
- Business-oriented model evaluation
- Automated testing

---

## Business Problem

Direct marketing campaigns can require substantial operational effort because banks may contact many clients who are unlikely to subscribe.

The business objective of this project is to help a bank:

1. Understand customer and campaign characteristics associated with term-deposit subscription.
2. Identify customer segments with higher historical subscription rates.
3. Rank clients by predicted subscription probability.
4. Prioritize campaign resources toward higher-potential clients.
5. Reduce unnecessary contacts while preserving useful subscriber coverage.

The machine-learning objective is binary classification:

> Predict whether a client will subscribe to a bank term deposit.

The target variable is:

```text
y = yes / no
```

During data preparation, the target is transformed into:

```text
subscribed = 1 / 0
```

---

## Prediction-Time Contract

The primary model is designed for **pre-campaign client prioritization**.

Predictions are assumed to be generated before any contact in the current campaign begins.

The model may use:

- Customer demographic information
- Account balance information
- Existing loan obligations
- Previous campaign-contact history
- Previous campaign outcome

The model excludes information generated during the current campaign:

```text
contact
day
month
duration
campaign
```

This strict prediction contract ensures that every feature used by the model is available at the intended decision point.

---

## Data-Leakage Decision

The `duration` variable records the duration of the last campaign call.

It is highly predictive, but it is only known after the call has finished. Using it to decide which clients should be contacted before the campaign would create temporal data leakage.

Therefore:

- `duration` is analyzed during exploratory data analysis.
- `duration` is excluded from the primary predictive model.
- Final model performance represents a realistic pre-campaign use case.
- Model selection does not depend on unavailable future information.

The project intentionally prefers operational validity over artificially inflated predictive performance.

---

## Dataset

The project uses the full `bank-full.csv` version of the **UCI Bank Marketing Dataset**.

| Property | Value |
|---|---:|
| Records | 45,211 |
| Input variables | 16 |
| Target | `y` |
| Target type | Binary classification |
| Positive class | `yes` |
| Negative class | `no` |
| Overall subscription rate | 11.70% |

The dataset contains information about:

- Customer demographics
- Employment and education
- Account balance
- Credit default
- Housing and personal loans
- Current campaign activity
- Previous campaign history
- Term-deposit subscription

The dataset contains no conventional null values in its original form. However, several categorical variables use the explicit value `unknown`, which is analyzed and retained rather than silently converted into missing values.

### Official source

- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing)
- [Dataset DOI: 10.24432/C5K306](https://doi.org/10.24432/C5K306)
- [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

### Dataset citation

```text
Moro, S., Rita, P., & Cortez, P. (2014).
Bank Marketing [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C5K306
```

---

## Project Objectives

The project was designed to demonstrate a complete and reproducible data science workflow:

1. Download data from the official source.
2. Validate file structure, schema, and target distribution.
3. Assess data quality, duplicates, unknown values, and outliers.
4. Clean and document the dataset.
5. Perform univariate and bivariate exploratory analysis.
6. Translate analytical questions into SQL.
7. Conduct statistical hypothesis tests with effect-size interpretation.
8. Engineer leakage-safe predictive features.
9. Create a stratified train/test split.
10. Build preprocessing pipelines with scikit-learn.
11. Compare baseline, linear, and tree-based classifiers.
12. Tune the strongest candidate using training data only.
13. Select a classification threshold using out-of-fold predictions.
14. Evaluate the final model once on an untouched test set.
15. Interpret ranking value, lift, and feature importance.
16. Validate the workflow using automated pytest tests.

---

## Analytical Workflow

```text
Official UCI data
        │
        ▼
Download and schema validation
        │
        ▼
Data-quality assessment and cleaning
        │
        ▼
Exploratory data analysis
        │
        ├──────────────► SQLite and SQL business analysis
        │
        ├──────────────► Statistical hypothesis testing
        │
        ▼
Leakage-safe feature engineering
        │
        ▼
Stratified train/test split
        │
        ▼
Cross-validation model comparison
        │
        ▼
Random Forest hyperparameter tuning
        │
        ▼
Training-only threshold selection
        │
        ▼
One-time held-out test evaluation
        │
        ▼
Business lift and model interpretation
```

---

## Machine-Learning Methodology

### Data split

The cleaned data is divided using an 80/20 stratified split:

| Split | Records | Subscribers | Subscription rate |
|---|---:|---:|---:|
| Training | 36,168 | 4,231 | 11.70% |
| Test | 9,043 | 1,058 | 11.70% |

The test set remains isolated during:

- Feature-processing decisions
- Candidate-model comparison
- Hyperparameter tuning
- Classification-threshold selection

### Preprocessing

Numerical features use:

- Median imputation
- Missing-value indicators
- Standard scaling

Categorical features use:

- Most-frequent imputation
- One-hot encoding
- Unknown-category handling

All learned preprocessing operations are included inside scikit-learn pipelines and fitted only on training folds.

### Candidate models

The project compares:

- Dummy prior classifier
- Logistic Regression
- Class-balanced Logistic Regression
- Class-balanced Random Forest

Five-fold stratified cross-validation is used for candidate comparison.

Because the positive class represents only approximately 11.7% of the data, the primary model-selection metric is **Average Precision / PR-AUC**, rather than accuracy alone.

---

## Candidate-Model Results

| Model | ROC-AUC | Average Precision | Balanced Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Dummy Prior | 0.500 | 0.117 | 0.500 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.713 | 0.347 | 0.576 | 0.667 | 0.162 | 0.261 |
| Balanced Logistic Regression | 0.714 | 0.345 | 0.662 | 0.224 | 0.598 | 0.326 |
| Balanced Random Forest | **0.734** | **0.378** | **0.679** | 0.297 | 0.520 | **0.378** |

The balanced Random Forest produced the strongest overall cross-validation performance and was selected for reasonable hyperparameter tuning.

---

## Final Model

The final model is a tuned, class-balanced Random Forest contained inside a complete preprocessing pipeline.

The classification threshold was selected using five-fold out-of-fold probabilities from the training set.

The threshold-selection rule was:

```text
Select the threshold that maximizes out-of-fold F1-score.
```

The selected threshold was approximately:

```text
0.5951
```

The test set was not used to choose this threshold.

---

## Final Held-Out Test Results

| Metric | Result |
|---|---:|
| Test records | 9,043 |
| Test subscribers | 1,058 |
| Positive-class prevalence | 0.1170 |
| ROC-AUC | **0.7382** |
| Average Precision / PR-AUC | **0.3902** |
| Balanced Accuracy | **0.6606** |
| Precision | **0.4004** |
| Recall | **0.4008** |
| F1-score | **0.4006** |
| Selected threshold | **0.5951** |
| Lift over baseline | **3.42×** |

Average Precision is substantially higher than the positive-class prevalence, indicating useful ranking performance for the minority subscriber class.

At the selected threshold, approximately 40% of clients predicted as subscribers actually subscribed, compared with an overall subscription rate of approximately 11.7%.

---

## Threshold Trade-Off

The training-selected threshold and default threshold provide different operating policies:

| Policy | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Training-selected maximum-F1 threshold | 0.595 | **0.400** | 0.401 | **0.401** |
| Default threshold | 0.500 | 0.291 | **0.533** | 0.377 |

The selected threshold:

- Produces higher precision.
- Reduces unnecessary positive classifications.
- Provides a more balanced precision-recall relationship.
- Achieves a higher F1-score.

The default threshold:

- Captures a larger percentage of subscribers.
- Requires contacting more clients.
- Produces more false-positive campaign contacts.

A production threshold should ultimately incorporate campaign cost, expected subscription value, available contact capacity, and customer-contact fatigue.

---

## Key Exploratory Findings

The exploratory analysis identified substantial variation in historical subscription rates across customer and campaign segments.

| Segment | Historical subscription rate |
|---|---:|
| Overall dataset | 11.70% |
| Previous campaign outcome = success | 64.73% |
| Contact month = March | 51.99% |
| Age 70 or older | 42.42% |
| Student occupation | 28.68% |
| Balance above 5,000 | 15.50% |
| One campaign contact | 14.60% |
| Cellular contact | 14.92% |

These findings are descriptive associations from historical data. They do not demonstrate that changing a customer characteristic or campaign condition would cause a subscription.

### Target imbalance

The target distribution is:

| Target | Records | Percentage |
|---|---:|---:|
| Non-subscriber | 39,922 | 88.30% |
| Subscriber | 5,289 | 11.70% |

Because the target is imbalanced:

- Accuracy is not used as the primary model-selection metric.
- Stratified splitting and cross-validation are used.
- Average Precision, ROC-AUC, balanced accuracy, precision, recall, and F1 are reported.
- Class-balanced models are evaluated.

### Campaign-contact pattern

Historical subscription rates decreased as the number of campaign contacts increased.

For example:

| Campaign contacts | Historical subscription rate |
|---|---:|
| One contact | 14.60% |
| Eleven or more contacts | 3.93% |

This may indicate diminishing returns, difficult-to-convert clients receiving repeated contacts, or operational targeting patterns. It should not be interpreted as proof that additional calls directly reduce subscription probability.

---

## SQL Business Analysis

The cleaned dataset is loaded into a local SQLite database for reproducible business analysis.

The SQL workflow demonstrates:

- `GROUP BY`
- Conditional aggregation
- `CASE WHEN`
- Common Table Expressions
- `UNION ALL`
- `DENSE_RANK`
- `ROW_NUMBER`
- Window functions
- Segment-level conversion analysis

Example business questions include:

- What is the overall subscription rate?
- Which job groups generate the largest number of subscribers?
- Which customer groups have the highest historical subscription rates?
- How does previous campaign outcome relate to subscription?
- How does contact frequency relate to conversion?
- Which segments combine useful scale and above-average conversion?

### Selected SQL findings

- The overall subscription rate was approximately **11.70%**.
- Management clients contributed the largest share of all subscribers, approximately **24.6%**.
- Clients with a previous successful campaign outcome subscribed at approximately **64.73%**.
- Clients contacted once subscribed at approximately **14.60%**.
- Clients contacted eleven or more times subscribed at approximately **3.93%**.

The SQL results were cross-checked against equivalent Pandas calculations.

---

## Statistical Analysis

The project uses statistical inference to evaluate whether selected historical differences are likely to be larger than expected from sampling variation.

### Methods

Categorical relationships are evaluated using:

- Chi-square tests of independence
- Cramér's V effect size

Numeric variables are evaluated using:

- Mann–Whitney U tests
- Rank-biserial effect size

Subscription proportions are accompanied by:

- Wilson confidence intervals
- Confidence intervals for differences in proportions

Multiple comparisons are controlled using:

- Benjamini–Hochberg false-discovery-rate correction

### Selected statistical findings

| Comparison | Estimated difference |
|---|---:|
| Previous success versus other previous outcomes | +54.86 percentage points |
| Cellular contact versus unknown contact type | +10.85 percentage points |
| One contact versus multiple contacts | +4.74 percentage points |

After multiple-testing correction, age did not provide strong evidence of a meaningful distribution difference in the selected numeric tests.

Campaign-contact count showed statistical evidence of a difference, but its estimated effect size was small.

### Important interpretation

The dataset is observational.

The statistical analyses identify historical associations and distribution differences. They do not estimate causal treatment effects and are not equivalent to randomized A/B tests.

---

## Business Targeting Value

The final model produces subscription probabilities that can be used to rank clients.

Instead of contacting every client, a bank may contact only the highest-ranked percentage based on:

- Campaign capacity
- Cost per contact
- Expected subscription value
- Desired subscriber coverage
- Customer-contact fatigue

The targeting-lift analysis compares model-based ranking with random targeting.

The final model provides meaningful lift at smaller targeting percentages and converges to a lift of `1.0` when all clients are contacted.

At the selected binary threshold:

```text
Precision ≈ 40.0%
Overall subscription rate ≈ 11.7%
Lift ≈ 3.42×
```

Therefore, the clients classified as likely subscribers convert at approximately 3.4 times the overall historical rate.

This result describes predictive prioritization value. It does not prove that contacting a high-ranked client causes subscription.

---

## Model Interpretation

Held-out permutation importance is calculated using Average Precision.

The most important original model features were:

1. `poutcome`
2. `housing`
3. `pdays_since_previous_contact`
4. `age`
5. `balance`
6. `previously_contacted`
7. `loan`
8. `marital`
9. `previous`
10. `education`

Previous campaign outcome was the strongest source of predictive information.

Permutation importance measures predictive reliance:

- It does not establish causality.
- Importance may be distributed across correlated features.
- A low importance value does not necessarily mean a variable has no business relevance.
- Importance values can change when the model or data distribution changes.

---

## Project Structure

```text
bank-marketing-analytics-ml/
│
├── data/
│   ├── raw/                    # Original downloaded data
│   ├── processed/              # Cleaned analytical data
│   └── interim/                # Generated train/test and prediction files
│
├── models/
│   └── *.joblib                # Local generated model artifact, ignored by Git
│
├── notebooks/
│   ├── 01_data_quality_and_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_sql_business_analysis.ipynb
│   ├── 04_statistical_analysis.ipynb
│   └── 05_machine_learning.ipynb
│
├── reports/
│   ├── modeling/               # Model comparison and evaluation reports
│   ├── sql/                    # SQL query outputs
│   └── statistics/             # Statistical-test outputs
│
├── sql/
│   ├── create_tables.sql
│   └── business_analysis.sql
│
├── src/
│   ├── download_data.py
│   ├── validate_raw_data.py
│   ├── data_cleaning.py
│   ├── database.py
│   ├── sql_analysis.py
│   ├── statistical_analysis.py
│   ├── features.py
│   ├── train.py
│   ├── tune.py
│   ├── select_threshold.py
│   └── evaluate.py
│
├── tests/
│   ├── conftest.py
│   ├── test_features.py
│   ├── test_database.py
│   ├── test_statistical_analysis.py
│   └── test_modeling_reports.py
│
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

Some generated data and binary artifacts are excluded from version control and can be rebuilt using the documented scripts.

---

## Notebooks

### [`01_data_quality_and_cleaning.ipynb`](notebooks/01_data_quality_and_cleaning.ipynb)

Covers:

- Dataset shape and schema
- Target transformation
- Missing-value checks
- Explicit `unknown` categories
- Duplicate detection
- Numeric distribution checks
- Outlier reporting
- Cleaning decisions

### [`02_exploratory_data_analysis.ipynb`](notebooks/02_exploratory_data_analysis.ipynb)

Covers:

- Target imbalance
- Numeric distributions
- Categorical subscription rates
- Customer-segment analysis
- Campaign behavior
- Previous campaign outcomes
- Data-leakage discussion
- Business-oriented EDA findings

### [`03_sql_business_analysis.ipynb`](notebooks/03_sql_business_analysis.ipynb)

Covers:

- SQLite database creation
- Business queries
- Segment ranking
- Conditional aggregation
- CTEs and window functions
- SQL and Pandas consistency checks

### [`04_statistical_analysis.ipynb`](notebooks/04_statistical_analysis.ipynb)

Covers:

- Confidence intervals
- Chi-square tests
- Cramér's V
- Mann–Whitney U tests
- Rank-biserial effect sizes
- Difference-in-proportion intervals
- Benjamini–Hochberg correction
- Observational versus experimental interpretation

### [`05_machine_learning.ipynb`](notebooks/05_machine_learning.ipynb)
Covers:

- Prediction contract
- Feature contract
- Preprocessing pipeline
- Train/test split
- Candidate-model comparison
- Random Forest tuning
- Threshold selection
- Final test evaluation
- ROC and Precision–Recall curves
- Confusion matrices
- Targeting lift
- Permutation importance
- Model metadata
- Responsible-use considerations

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/omar-alnadhari/bank-marketing-analytics-ml.git
cd bank-marketing-analytics-ml
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Reproducing the Workflow

Run all commands from the repository root.

### 1. Download the dataset

```bash
python src/download_data.py
```

### 2. Validate the raw dataset

```bash
python src/validate_raw_data.py
```

### 3. Clean and prepare the analytical dataset

```bash
python src/data_cleaning.py
```

### 4. Create the SQLite database

```bash
python src/database.py
```

### 5. Run SQL business analysis

```bash
python src/sql_analysis.py
```

### 6. Run statistical analysis

```bash
python src/statistical_analysis.py
```

### 7. Create leakage-safe features and train/test splits

```bash
python src/features.py
```

### 8. Compare candidate models

```bash
python src/train.py
```

### 9. Tune the Random Forest

```bash
python src/tune.py
```

### 10. Select the classification threshold

```bash
python src/select_threshold.py
```

### 11. Train and evaluate the final model

```bash
python src/evaluate.py
```

After the final evaluation, the test set is considered consumed and should not be used for additional model or threshold optimization.

---

## Running the Automated Tests

Run the complete test suite:

```bash
python -m pytest -v
```

Expected result:

```text
33 passed
```

The automated tests cover:

- Leakage-feature exclusion
- Feature-group definitions
- Feature engineering
- Train/test schemas and row counts
- Stratified target distribution
- SQL column conversion
- SQLite table creation and validation
- Statistical utility functions
- Confidence intervals and corrected test results
- Model-comparison reports
- Final test metrics
- Threshold consistency
- Confusion matrices
- Targeting lift
- Permutation importance
- Local model-artifact loading

The project configures pytest to use a temporary directory inside the repository to avoid Windows temporary-directory permission problems.

---

## Generated Outputs

Important generated outputs include:

### Data-quality reports

```text
reports/data_quality_overview.csv
reports/unknown_values_summary.csv
reports/numeric_outlier_summary.csv
```

### SQL reports

```text
reports/sql/
```

### Statistical reports

```text
reports/statistics/
```

### Modeling reports

```text
reports/modeling/model_cv_comparison.csv
reports/modeling/random_forest_tuning_results.csv
reports/modeling/random_forest_best_parameters.json
reports/modeling/selected_threshold.json
reports/modeling/final_test_probability_metrics.csv
reports/modeling/final_test_threshold_metrics.csv
reports/modeling/final_test_confusion_matrices.csv
reports/modeling/final_targeting_lift_table.csv
reports/modeling/test_permutation_importance.csv
reports/modeling/final_model_metadata.json
```

### Local model artifact

```text
models/bank_marketing_pre_campaign_random_forest.joblib
```

The binary model artifact is ignored by Git because serialized models are environment-dependent. It can be rebuilt by running:

```bash
python src/evaluate.py
```

---

## Reproducibility

The project uses:

- Fixed random seed: `42`
- Stratified train/test splitting
- Stratified cross-validation
- Pipeline-based preprocessing
- Training-only model selection
- Training-only threshold selection
- One-time held-out test evaluation
- Saved model parameters and metadata
- Automated validation tests

The final metadata report records:

- Python version
- Operating system
- NumPy version
- Pandas version
- scikit-learn version
- joblib version
- Model parameters
- Feature contract
- Classification threshold
- Model-file checksum
- Final test metrics

---

## Limitations

### Observational data

The dataset contains historical campaign records and is not a randomized experiment.

The analysis identifies associations, not causal effects.

### Historical targeting bias

The model may learn patterns created by previous campaign strategies, customer selection, and operational processes.

### Dataset age and context

The data reflects a specific banking institution and historical period. Performance may not transfer directly to another country, bank, product, or time period.

### Probability calibration

The model is primarily evaluated for ranking and classification performance. Predicted probabilities may require additional calibration before financial decision-making.

### Business-cost assumptions

A final production threshold should consider:

- Contact cost
- Expected subscription value
- Campaign capacity
- Customer-contact fatigue
- Compliance constraints

### Fairness

Features such as age and marital status may require policy and fairness review.

Before production deployment, performance should be evaluated across relevant customer groups.

### Model drift

Customer behavior, market conditions, and campaign operations may change. Production use would require monitoring for:

- Data drift
- Prediction drift
- Performance drift
- Calibration drift
- Segment-level performance changes

---

## Responsible Use

This model is designed only for marketing prioritization.

It should not be used for:

- Credit approval
- Loan eligibility
- Pricing decisions
- Customer exclusion
- Automated decisions with significant legal or financial effects

Predictions should support operational planning rather than replace appropriate human, legal, compliance, and fairness review.

---

## Business Recommendations

Based on the analytical results:

1. Use probability ranking to prioritize clients rather than contacting the complete customer list without differentiation.
2. Give particular analytical attention to previous campaign outcome and previous-contact history.
3. Treat repeated campaign contacts cautiously because historical conversion decreased among heavily contacted clients.
4. Use threshold selection as a business decision linked to cost and capacity, not only as a technical default.
5. Monitor high-performing segments for both scale and conversion rate.
6. Run randomized campaign experiments before making causal claims about contact strategies.
7. Audit performance across age and other sensitive or policy-relevant customer segments.
8. Revalidate and retrain the model when customer behavior or campaign design changes.

---

## Future Improvements

Possible future extensions include:

- Probability calibration
- Cost-sensitive threshold optimization
- SHAP-based model explanations
- Fairness and segment-level error analysis
- Temporal validation
- External validation on newer campaign data
- Experiment-design recommendations
- Uplift modeling
- Customer-contact fatigue modeling
- Streamlit demonstration interface
- Automated continuous-integration testing
- Model monitoring and drift dashboards

A causal or uplift model would require suitable experimental or treatment-assignment data and cannot be built reliably from the current observational dataset alone.

---

## Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- statsmodels
- scikit-learn
- SQLite
- SQL
- Jupyter Notebook
- pytest
- Git
- GitHub

---

## Author

**Omar Al-Nadhari**

MSc Computer Science — Artificial Intelligence  
University of Pisa

Background in:

- Data analysis
- Information technology
- Banking systems
- SQL and database analysis
- Python
- Machine learning
- Academic teaching

GitHub: [github.com/omar-alnadhari](https://github.com/omar-alnadhari)

---

## License and Attribution

The project source code is provided for educational and portfolio purposes.

The dataset is distributed separately by the UCI Machine Learning Repository under the Creative Commons Attribution 4.0 International license.

Dataset citation:

```text
Moro, S., Rita, P., & Cortez, P. (2014).
Bank Marketing [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C5K306
```