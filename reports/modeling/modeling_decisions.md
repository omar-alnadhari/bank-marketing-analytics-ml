# Machine-Learning Modeling Decisions

## Primary Business Objective

The primary model prioritizes clients for a term-deposit marketing campaign before current-campaign contacts begin.

The prediction target is:

- `subscribed = 1`: the client eventually subscribed.
- `subscribed = 0`: the client did not subscribe.

## Prediction Time

Predictions are assumed to be generated before the client receives any contact from the current marketing campaign.

This strict prediction-time definition prevents the model from using information created during or after the campaign interaction.

## Excluded Current-Campaign Features

The following features are excluded from the primary model:

- `contact`
- `day`
- `month`
- `duration`
- `campaign`

### Reasons

- `duration` is only known after the call ends and would create temporal data leakage.
- `campaign` records current-campaign contact intensity and may contain information accumulated after the original targeting decision.
- `contact`, `day`, and `month` describe the execution of the current campaign and are not treated as available at campaign-entry prediction time.

These features may still be used for retrospective exploratory analysis, SQL reporting, and educational leakage comparisons.

## Included Information

The primary model uses:

- Customer demographic attributes.
- Account-balance information.
- Existing financial obligations.
- Previous campaign-contact history.
- Previous campaign outcome.

## Feature Engineering

The following deterministic features are created:

- `previously_contacted`
- `pdays_since_previous_contact`
- `negative_balance`

No aggregate statistics or target information are used during feature engineering.

## Train/Test Strategy

- Train size: 80%
- Test size: 20%
- Stratified by the binary target.
- Random state: 42
- The test set remains untouched during model selection and hyperparameter tuning.

Cross-validation and model selection will use only the training set.

## Fairness Considerations

Age and marital status are retained for analytical benchmarking, but they may raise fairness or policy concerns.

A later sensitivity analysis will compare the main model with a reduced feature set that excludes potentially sensitive demographic variables.

The model should support campaign prioritization rather than automatically exclude customers or make credit decisions.

## Classification-Threshold Policy

The classification threshold is selected using out-of-fold predicted probabilities generated exclusively from the training set.

The primary threshold-selection rule is:

- Select the threshold that maximizes the F1-score.
- If multiple thresholds produce the same F1-score, prefer higher recall and then higher precision.

The test set is not used during threshold selection.

The default threshold of 0.50 is retained as a benchmark, but it is not automatically assumed to be optimal for the imbalanced campaign-targeting problem.

## Final Test-Set Evaluation Policy

The held-out test set is opened only after:

- The prediction contract was defined.
- Features were selected.
- Candidate models were compared.
- Random Forest hyperparameters were selected.
- The classification threshold was selected from training-only out-of-fold predictions.

The final model is trained on the complete training set and evaluated once on the untouched test set.

Test-set results are used for final reporting and interpretation, not for additional hyperparameter tuning or threshold optimization.