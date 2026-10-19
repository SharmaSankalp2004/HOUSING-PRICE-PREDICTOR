# 🏠 Housing Price Predictor

A machine learning project that predicts **median house values** in California using regression models. It walks through the full workflow: loading data, cleaning, exploring, transforming, feature engineering, training, and evaluating two models (Linear Regression and Random Forest).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Tech Stack](#tech-stack)
4. [Installation and Usage](#installation-and-usage)
5. [Code Walkthrough](#code-walkthrough)
6. [Results](#results)
7. [Important Notes](#important-notes)
8. [Future Improvements](#future-improvements)
9. [Project Structure](#project-structure)
10. [License](#license)

---

## Project Overview

**Goal:** Given information about a block of houses (location, age, number of rooms, population, income, distance to the ocean), predict the **median house value** of that block.

**Type of problem:** Supervised learning, specifically **regression** (the target is a continuous number).

**Approach:**
1. Load and clean the data
2. Split it into training and testing sets
3. Explore it visually and statistically
4. Fix skewed features and create new ones
5. Train Linear Regression and Random Forest models
6. Compare them using the R² score on unseen data

---

## Dataset

The file `housing.csv` is the California housing dataset. It has **20,640 rows** (one per census block group) and **10 columns**.

| Column | Type | Description |
|---|---|---|
| `longitude` | float | How far west the block is (more negative = further west) |
| `latitude` | float | How far north the block is |
| `housing_median_age` | float | Median age of the houses in the block |
| `total_rooms` | float | Total number of rooms in all houses of the block |
| `total_bedrooms` | float | Total number of bedrooms in the block (has 207 missing values) |
| `population` | float | Number of people living in the block |
| `households` | float | Number of households (families or living units) in the block |
| `median_income` | float | Median income of households in the block (in tens of thousands of dollars) |
| `ocean_proximity` | text | Category: `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN` |
| `median_house_value` | float | **Target:** median house value in the block (in dollars) |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Programming language |
| NumPy | Numerical operations (log transform) |
| pandas | Loading, cleaning, and manipulating tabular data |
| Matplotlib | Plotting |
| Seaborn | Statistical visualizations (heatmaps, scatter plots) |
| scikit-learn | Train/test split and machine learning models |
| Jupyter Notebook / Google Colab | Running the code interactively |

---

## Installation and Usage

```bash
# 1. Clone the repository
git clone https://github.com/your-username/housing-price-predictor.git
cd housing-price-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook HOUSING_PRICE_PREDICTOR.ipynb
```

Make sure `housing.csv` is in the same folder as the notebook, then run the cells from top to bottom.

---

## Code Walkthrough

This section explains every step of the code in the order it appears.

### Step 1: Importing libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

- `numpy` is used for math functions such as the logarithm.
- `pandas` provides the DataFrame, a table-like structure for the data.
- `matplotlib.pyplot` is the basic plotting library.
- `seaborn` builds nicer statistical charts on top of matplotlib.

### Step 2: Loading the data

```python
data = pd.read_csv('housing.csv')
data
data.info()
```

- `pd.read_csv` reads the CSV file into a DataFrame called `data`.
- Writing `data` on its own line displays the table (20,640 rows × 10 columns).
- `data.info()` prints a summary of each column: its name, data type, and how many non-null values it has. Here it shows that `total_bedrooms` has only 20,433 non-null values, meaning **207 values are missing**. All other columns are complete. It also shows that `ocean_proximity` is text (object type) while the rest are numbers.

### Step 3: Handling missing values

```python
data.dropna(inplace=True)
```

Removes every row that has a missing value. Since only 207 of 20,640 rows (about 1%) are affected, dropping them loses very little data. `inplace=True` modifies `data` directly instead of creating a copy.

### Step 4: Separating features and target

```python
from sklearn.model_selection import train_test_split

X = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']
```

- `X` holds the **features** (inputs): every column except the price.
- `y` holds the **target** (output): `median_house_value`, the value we want to predict.
- `axis=1` tells pandas to drop a column rather than a row.

### Step 5: Train/test split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

Splits the data randomly into:
- **70% training set**: used to teach the model
- **30% test set**: kept hidden and used only to evaluate the model on data it has never seen

This shows how well the model generalizes instead of just memorizing the training data.

### Step 6: Rebuilding a training DataFrame

```python
train_data = X_train.join(y_train)
train_data
```

`join` puts the features and the target back together in one table, `train_data`. This makes it easy to analyze correlations between features and price, and to apply transformations to the training set.

### Step 7: Exploratory Data Analysis (EDA)

```python
train_data.hist()
```

Draws a **histogram for every numeric column**. This shows the shape of each distribution. You will notice that `total_rooms`, `total_bedrooms`, `population`, and `households` are **heavily right-skewed**: most blocks are small, but a few are extremely large.

```python
train_data.corr(numeric_only=True)
```

Computes the **correlation matrix**, which measures how strongly each pair of numeric columns moves together (from -1 to +1). `numeric_only=True` skips the text column `ocean_proximity`.

```python
plt.figure(figsize=(15, 8))
sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap="YlGnBu")
plt.show()
```

Visualizes the correlation matrix as a **heatmap**:
- `figsize=(15, 8)` makes the figure large enough to read.
- `annot=True` prints the correlation number inside each cell.
- `cmap="YlGnBu"` sets the yellow-green-blue color scheme.

Key observations:
- `median_income` is the feature most strongly correlated with house value.
- `total_rooms`, `total_bedrooms`, `population`, and `households` are very strongly correlated with each other (above 0.85), because bigger blocks have more of everything.
- `longitude` and `latitude` are strongly negatively correlated (about -0.92), because of the shape of California's coastline.

### Step 8: Fixing skewed features with a log transform

```python
train_data['total_rooms']      = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms']   = np.log(train_data['total_bedrooms'] + 1)
train_data['population']       = np.log(train_data['population'] + 1)
train_data['households']       = np.log(train_data['households'] + 1)
```

Applies `log(x + 1)` to the four skewed columns.
- The **log** compresses large values, which turns the long-tailed distributions into more bell-shaped ones.
- The **+1** avoids taking the log of zero, which is undefined.

```python
train_data.hist(figsize=(15, 8))
```

Plots the histograms again to confirm the distributions are now more symmetric.

### Step 9: Encoding the categorical column

```python
train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)
```

Machine learning models need numbers, not text. `pd.get_dummies` performs **one-hot encoding**: it creates one 0/1 column for each category (`<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN`). The original text column is then dropped.

In the notebook this line is only used to *display* the result; the actual encoding used for training happens in Step 12.

### Step 10: More visualizations

```python
plt.figure(figsize=(15, 8))
sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap="YlGnBu")
plt.show()
```

A new heatmap after the log transform, to see how the correlations changed.

```python
plt.figure(figsize=(15, 8))
sns.scatterplot(x='latitude', y='longitude', data=train_data,
                hue="median_house_value", palette='coolwarm')
```

A **geographic scatter plot**: each point is a block placed by its latitude and longitude, colored by house value (blue = cheaper, red = more expensive). This reveals that expensive houses cluster along the coast, especially around the San Francisco Bay Area and Los Angeles.

### Step 11: Feature engineering

```python
train_data['households_ratio']    = train_data['households'] / train_data['population']
train_data['rooms_per_household'] = train_data['total_rooms'] / train_data['households']
```

Creates two new features that may carry more meaning than the raw numbers:
- `households_ratio`: households relative to population (related to household size)
- `rooms_per_household`: how spacious the homes are

Another heatmap is then drawn to see how these new features correlate with the price.

### Step 12: Training the Linear Regression model

```python
from sklearn.linear_model import LinearRegression

X_train = train_data.drop('median_house_value', axis=1)
y_train = train_data['median_house_value']

X_train = pd.get_dummies(X_train, columns=['ocean_proximity'], dtype=int)

reg = LinearRegression()
reg.fit(X_train, y_train)
```

1. Separates the transformed features (`X_train`) from the target (`y_train`).
2. One-hot encodes `ocean_proximity`, using `dtype=int` so the new columns hold 0/1 integers.
3. Creates a `LinearRegression` model and calls `.fit()` to **train** it. It finds the best straight-line (linear) combination of the features that predicts price.

### Step 13: Preparing the test data

The test set must go through **exactly the same transformations** as the training set, otherwise the model would receive inputs in a format it was not trained on.

```python
test_data = X_test.join(y_test)

test_data['total_rooms']    = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population']     = np.log(test_data['population'] + 1)
test_data['households']     = np.log(test_data['households'] + 1)

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)

test_data['households_ratio']    = test_data['households'] / test_data['population']
test_data['rooms_per_household'] = test_data['total_rooms'] / test_data['households']

X_test, y_test = test_data.drop('median_house_value', axis=1), test_data['median_house_value']
```

This repeats the log transform, one-hot encoding, and the two engineered features on the test data, then separates features from the target again.

```python
cols = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
test_data[cols] = test_data[cols].astype(int)
```

Converts the dummy columns from True/False to 1/0 so they match the training data.

### Step 14: Aligning test columns with training columns

```python
X_test = test_data.drop('median_house_value', axis=1)

X_test = X_test.rename(columns={
    '<1H OCEAN': 'ocean_proximity_<1H OCEAN',
    'INLAND': 'ocean_proximity_INLAND',
    'ISLAND': 'ocean_proximity_ISLAND',
    'NEAR BAY': 'ocean_proximity_NEAR BAY',
    'NEAR OCEAN': 'ocean_proximity_NEAR OCEAN'
})

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
```

The training data's dummy columns are named `ocean_proximity_<category>`, while the test data's are named just `<category>`. This block:
1. **Renames** the test columns to match the training names.
2. **Reindexes** the test data so it has exactly the same columns in exactly the same order as the training data. If a column is missing from the test set (for example, the rare `ISLAND` category may not appear in it), it is filled with 0.

Scikit-learn requires the training and test features to have identical columns in the same order.

### Step 15: Evaluating Linear Regression

```python
reg.score(X_test, y_test)
```

Returns the **R² score** on the test set. R² measures how much of the variation in house prices the model explains:
- 1.0 means perfect predictions
- 0.0 means no better than always predicting the average price

**Result: about 0.665.**

### Step 16: Training and evaluating Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()
forest.fit(X_train, y_train)
forest.score(X_test, y_test)
```

A **Random Forest** builds many decision trees on random subsets of the data and averages their predictions. Unlike Linear Regression, it can capture **non-linear relationships** and interactions between features (for example, how location and income combine to affect price).

**Result: about 0.923.**

---

## Results

| Model | R² Score (test set) |
|---|---|
| Linear Regression | 0.665 |
| Random Forest Regressor | **0.923** |

Random Forest performs much better, which indicates that the relationship between the features and house prices is non-linear and too complex for a straight line to capture.

---

## Important Notes

- **Results change between runs.** `train_test_split` and `RandomForestRegressor` are not given a `random_state`, so each run uses a different random split and a different forest. Your scores may differ slightly from those above. Add `random_state=42` to both for reproducible results.
- **Ratio features are built from logged values.** `households_ratio` and `rooms_per_household` are calculated *after* the log transform, so they are ratios of log values rather than ratios of the original counts. This still works as a feature, but it is a design choice you may want to revisit.
- **Why the dummy-encoding steps look repetitive.** Training and test data are transformed separately by hand. A scikit-learn `Pipeline` with `ColumnTransformer` would do this more cleanly and avoid column-alignment fixes.
- **Missing rows are dropped before splitting.** This is fine for 1% of the data, but imputation (for example, filling with the median) is an alternative.

---

## Future Improvements

- Set `random_state` for reproducibility
- Use cross-validation for a more reliable score
- Tune hyperparameters with `GridSearchCV` or `RandomizedSearchCV`
- Try Gradient Boosting and XGBoost
- Report RMSE and MAE alongside R²
- Plot feature importances from the Random Forest
- Use a scikit-learn `Pipeline` to simplify preprocessing
- Deploy the model as a web app with Streamlit or Flask

---

## Project Structure

```
housing-price-predictor/
├── HOUSING_PRICE_PREDICTOR.ipynb   # Main notebook with all code
├── housing.csv                     # Dataset
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## License

This project is licensed under the MIT License.
