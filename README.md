# Mushroom Classification: Model Evaluation and Robustness

A machine learning study of edible/poisonous classification from mushroom measurements and categorical attributes. The project explores preprocessing, model comparison, hyperparameter search, model sanity checks, feature importance, and robustness under degraded inputs.


## What the project explores

- Comparing Logistic Regression, RBF SVM, a multilayer perceptron, Random Forest, and XGBoost, with an additional CatBoost experiment.
- Evaluating accuracy, precision, poisonous-class recall, F1, ROC AUC, and confusion matrices.
- Checking label distributions, missing values, duplicate features, prediction consistency, dummy baselines, and shuffled-label performance.
- Interpreting a Random Forest using permutation importance and partial dependence / individual conditional expectation plots.
- Investigating missing inputs, feature perturbations, and feature removal.

Inputs are structured attributes such as cap diameter, gill attachment, stem width, habitat, and season. 
## Dataset

The main input is [`secondary_data.csv`](secondary_data.csv), a semicolon-separated copy of the **Secondary Mushroom** dataset.

| Property | Repository data |
| --- | --- |
| Rows | 61,069 before deduplication |
| Input features | 20: 17 categorical and 3 numerical |
| Target | `class`: `e` → 0, `p` → 1 |
| Class counts | 27,181 edible; 33,888 poisonous/unknown |
| Features retained by Part 1 | 16 |
| Current split | Stratified 80% training / 20% test, seed 42 |

The data was simulated from descriptions of 173 mushroom species. The poisonous label also includes mushrooms whose edibility is unknown or not recommended. See [the included metadata](secondary_data_meta.txt) and the [UCI dataset page](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset).

Part 1 drops `veil-type`, `spore-print-color`, `veil-color`, and `stem-root` because of extensive missingness. It fits categorical mode imputation and ordinal encoding, plus numerical median imputation and standard scaling, on the training partition. It exports the transformed arrays and feature names to `mushroom_train_test.npz`.

The current export contains 48,855 training rows and 12,214 test rows. 

## Notebook guide

| Notebook | Contents |
| --- | --- |
| [Part 1 — Initial exploration](Part%201%20-%20inital%20exploration.ipynb) | Missingness, preprocessing, split export, baseline models, and metric plots |
| [Part 2 — Model analysis](Part%202%20-%20Model%20analysis.ipynb) | Hyperparameter experiments and model ranking; currently requires execution fixes |
| [Part 3 — Sanity and features](Part%203%20-%20Sanity%20and%20Features.ipynb) | Dummy baselines, shuffled labels, split checks, permutation importance, and PDP/ICE plots |
| [Part 4 — Model robustness](Part%204%20-%20Model%20robustness.ipynb) | CatBoost, missing-input experiments, drift experiments, and feature ablation |

`primary_data.csv` and its metadata are also included. `secondary_data2.csv` duplicates the main CSV.

## Recorded results and interpretation

Part 3 reports the following Random Forest baseline: 200 trees, unrestricted depth, and seed 42. A targeted rerun of the current Part 1 preprocessing and this baseline reproduced these figures.

| Metric | Current split |
| --- | ---: |
| Accuracy | 99.9918% |
| Poisonous-class recall | 99.9852% |
| F1, poisonous class | 0.999926 |
| ROC AUC | 0.99999995 |
| Poisonous rows predicted edible | 1 of 6,778 |

The saved Part 3 outputs also show:

- A majority-class baseline accuracy of about 55.49%.
- Mean ROC AUC of about 0.499 after shuffling training labels across three runs.
- Stem width and gill attachment as the two highest ranked features by permutation importance.

The saved Part 4 experiment drops to about 68.83% accuracy with 50% randomly missing entries. 

## Dataset attribution

Wagner, D., Heider, D., & Hattab, G. (2021). *Secondary Mushroom* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5FP5Q](https://doi.org/10.24432/C5FP5Q).
