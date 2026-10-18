# Airbnb Price Classification — ML Model Comparison

An individual capstone project from the **Break Through Tech AI** program, comparing logistic regression and a feedforward neural network for classifying New York City Airbnb listings as high-priced or low-priced.

## Problem

Predict whether a listing belongs to the **high-price category** (at or above the dataset's 75th percentile) using listing, location, host, availability, and review characteristics. The original dataset contains **28,022 listings**.

## Methods

- Explored the class distribution, missing values, outliers, and feature relationships.
- Removed price from the predictors to avoid directly leaking the target; excluded free-text features.
- Imputed missing values, encoded categorical variables, and standardized numerical inputs.
- Used a stratified train/test split (80/20) and five-fold grid search to tune logistic regression (`C=100`).
- Built a Keras feedforward neural network with **64 → 32 → 16 ReLU hidden units**, a sigmoid output, SGD optimization, and binary cross-entropy loss.
- Evaluated models using held-out accuracy and F1 score.

## Results

| Model | Test accuracy | Test F1 score |
|---|---:|---:|
| Logistic regression | 0.8278 | 0.6092 |
| Neural network | 0.8282 | 0.6367 |

The neural network improved F1 score while accuracy remained nearly unchanged. Its validation loss increased as training continued, suggesting overfitting. These results are from the original notebook run, not a fresh reproduction.

## Files

- [`airbnb_price_classification.ipynb`](airbnb_price_classification.ipynb) — code, explanations, saved results, and training plots.

## Running the notebook

1. Obtain the original BTT-provided `airbnbListingsData.csv` dataset, subject to its distribution terms.
2. Place it at `data_capstone/airbnbListingsData.csv` relative to the notebook's working directory.
3. Install dependencies using `pip install -r requirements.txt` and run the notebook in Jupyter.

**Note:** The dataset is not included. Dataset versions may differ and lead to different results.

## Limitations and future work

- High-priced listings are the minority class, making F1 important alongside accuracy.
- In the original notebook, imputation and categorical encoding were applied **before** splitting the data. A stronger implementation would fit all learned preprocessing steps on training data only using a scikit-learn pipeline.
- The neural network showed signs of overfitting; early stopping and regularization could help.
- Results reflect NYC listings and may not generalize to other markets.
- Location features may reflect socioeconomic differences in historical prices; any real-world use would require fairness evaluation.

## Author

Sofia Popova — Case Western Reserve University  
Break Through Tech AI capstone project
