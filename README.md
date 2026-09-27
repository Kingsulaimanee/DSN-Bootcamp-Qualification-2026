# DSN Hackathon 2026 - ML Track: Retail Sales Prediction

This repository contains the end-to-end machine learning pipeline used for the sales prediction challenge. The solution focuses on leak-free target encoding, domain feature engineering, multi-seed bagging, and ensembling.

---

## 1. Problem Formulation & Validation Strategy

- **Target Variable**: `total_sales` (trained on `log1p(total_sales)` to handle skewness and stabilize variance).
- **Evaluation Metric**: Root Mean Squared Error (RMSE).
- **Validation Scheme**: 
  - Stratified/Group-aware **5-Fold Cross-Validation** (`KFold(n_splits=5, shuffle=True)`).
  - Bagging over **3 distinct seeds** (`[42, 202, 777]`) across folds to suppress prediction variance.

---

## 2. Feature Engineering & Preprocessing

- **Missing Data Handling**:
  - `product_weight_kg`: Hierarchically imputed using mean by `product_code`, fallback to mean by `product_category`, and finally global mean.
  - `shelf_visibility`: Zeroes treated as missing values and imputed hierarchically by product and category means.
  - `store_size`: Categorized with an explicit `'Unknown'` class to capture structural missingness.
- **Engineered Signals**:
  - Interaction terms: `price_per_kg`, `size_x_tier`, `store_x_category`, `format_x_category`, `tier_x_category`.
  - Store & Assortment metrics: Product store count, assortment breadth, and relative price ranks within categories.
- **Out-of-Fold Target Encoding**:
  - Implemented leak-free K-Fold target encoding on high-cardinality categorical features.
  - Applied dual additive Laplace smoothing (weights of 10 and 50 averaged) to prevent overfitting on low-frequency categories.

---

## 3. Modeling & Ensembling

The solution leverages an ensemble of diverse estimators:
1. **Regularized Linear**: Ridge Regression ($\alpha = 5.0$) on smooth target encodings and interaction terms.
2. **Gradient Boosted Trees**: LightGBM, XGBoost, and HistGradientBoosting with early stopping on validation folds.
3. **Randomized Trees**: ExtraTreesRegressor for high variance suppression.

### Model Ensembling & Weighting
- Out-of-Fold (OOF) predictions from all base estimators were ensembled using **Nelder-Mead numerical optimization** to directly minimize OOF RMSE.
- **Final Validation Performance**:
  - Naive Baseline RMSE: `~1697.77`
  - Final Blended OOF RMSE: `1102.71` (~35% relative improvement over baseline)

---

## 4. Repository Structure

```text
├── notebooks/
│   ├── best_blend_submission.ipynb   # Final multi-model pipeline with Nelder-Mead blend
│   └── optuna_tuned_models.ipynb     # Hyperparameter optimization using Optuna
├── README.md
