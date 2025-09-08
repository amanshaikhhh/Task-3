import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Install imbalanced-learn if not already installed
# !pip install imbalanced-learn

sns.set_theme(context='notebook', style='whitegrid', palette='muted')

# Load data
data = pd.read_csv("C:\\Users\\91740\\Desktop\\tasks\\bank-full.csv", sep=';')

print("Data Info:")
data.info()

print("\nObject columns description:")
print(data.describe(include='object'))

print(f"\nDuplicates: {data.duplicated().sum()}")

# Rename target column and map values
data = data.rename(columns={'y': 'subscribed'})
data['subscribed'] = data['subscribed'].map({'yes': 'Subscribed', 'no': 'Not Subscribed'})

# Process categorical columns
categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
data[categorical_cols] = (data[categorical_cols].apply(lambda x: x.str.title())
                         .astype('category'))

# Process binary columns
binary_cols = ['default', 'housing', 'loan']
data[binary_cols] = data[binary_cols] == 'yes'

# Identify columns with outliers
cols_with_outliers = ['age', 'balance', 'duration', 'campaign']

# Plot distributions before outlier removal
fig, axes = plt.subplots(4, 2, figsize=(15, 10))
for i, col in enumerate(cols_with_outliers):
    hist_ax, box_ax = axes[i, :]
    sns.histplot(data=data, x=col, bins=100, ax=hist_ax)
    hist_ax.set_title(f'Histogram of {col.title()}')
    hist_ax.set_xlabel('')
    hist_ax.set_ylabel('')
    sns.boxplot(data=data, x=col, ax=box_ax)
    box_ax.set_title(f'Boxplot of {col.title()}')
    box_ax.set_xlabel('')
    box_ax.set_ylabel('')
plt.tight_layout()
plt.show()

# Function to remove outliers
def remove_outliers(df, columns):
    df_outliers_removed = df.copy()
    for col in columns:
        Q1 = df_outliers_removed[col].quantile(0.25)
        Q3 = df_outliers_removed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_outliers_removed = df_outliers_removed[
            (df_outliers_removed[col] >= lower_bound) &
            (df_outliers_removed[col] <= upper_bound)
        ]
    return df_outliers_removed

# Remove outliers
print(f"Data shape before outlier removal: {data.shape}")
data = remove_outliers(data, cols_with_outliers)
print(f"Data shape after outlier removal: {data.shape}")

# Plot distributions after outlier removal
fig, axes = plt.subplots(4, 2, figsize=(15, 10))
for i, col in enumerate(cols_with_outliers):
    hist_ax, box_ax = axes[i, :]
    sns.histplot(data=data, x=col, bins=100, ax=hist_ax)
    hist_ax.set_title(f'Histogram of {col.title()} (After Outlier Removal)')
    hist_ax.set_xlabel('')
    hist_ax.set_ylabel('')
    sns.boxplot(data=data, x=col, ax=box_ax)
    box_ax.set_title(f'Boxplot of {col.title()} (After Outlier Removal)')
    box_ax.set_xlabel('')
    box_ax.set_ylabel('')
plt.tight_layout()
plt.show()

# Identify column types
num_cols = data.select_dtypes('number').columns.tolist()
bool_cols = data.select_dtypes(bool).columns.tolist()
cat_cols = data.select_dtypes('category').columns.tolist()

print(f"Numeric columns: {num_cols}")
print(f"Boolean columns: {bool_cols}")
print(f"Categorical columns: {cat_cols}")

# Plot target distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='subscribed')
plt.title('Count of Subscribed Term Deposits')
plt.xlabel('')
plt.ylabel('')
plt.show()

# Plot categorical features
fig, axes = plt.subplots(3, 2, figsize=(25, 15))
for feature, ax in zip(cat_cols, axes.flatten()):
    sns.countplot(data=data, x=feature, ax=ax)
    ax.set_title(f'Count of {feature.title()}')
    ax.set_xlabel('')
    ax.set_ylabel('')
plt.tight_layout()
plt.show()

# Import required libraries for modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler

# Prepare features and target
X = data.drop(columns='subscribed')
y = data['subscribed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    stratify=y, 
                                                    random_state=42)

# Define preprocessing pipeline
num_vars = data.select_dtypes('number').columns.tolist()
cat_vars = data.select_dtypes('category').columns.tolist()

preprocessing_pipeline = ColumnTransformer([
    ('numerical', StandardScaler(), num_vars),
    ('categorical', OneHotEncoder(handle_unknown='ignore'), cat_vars),
])

# Apply preprocessing
X_train = preprocessing_pipeline.fit_transform(X_train)
X_test = preprocessing_pipeline.transform(X_test)

# Apply oversampling
sampler = RandomOverSampler(random_state=42)
X_train, y_train = sampler.fit_resample(X_train, y_train)

print(f"Training set shape after oversampling: {X_train.shape}")
print(f"Class distribution after oversampling:")
print(pd.Series(y_train).value_counts())

# Build basic model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

print("\n" + "="*50)
print("Building Basic Decision Tree Model")
print("="*50)

start_time = time.time()
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
end_time = time.time()

print(f"Training time: {end_time - start_time:.2f} seconds")

# Evaluate basic model
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
report = classification_report(y_test, y_pred)
print(f'Basic Model Accuracy: {accuracy:.2%}')
print(f'Classification Report:\n{report}')

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

print("\n" + "="*50)
print("Hyperparameter Tuning with GridSearchCV")
print("="*50)

param_grid = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

scorer = make_scorer(f1_score, pos_label='Subscribed')

base_model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(estimator=base_model,
                          param_grid=param_grid,
                          cv=5,
                          scoring=scorer,
                          verbose=1,
                          n_jobs=-1)

start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

print(f"Grid search time: {end_time - start_time:.2f} seconds")

# Best model results
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f'Best Model Accuracy: {accuracy:.2%}')
print(f'Best Parameters: {best_params}')

# Final evaluation
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred)
print(f'Final Classification Report:\n{report}')

# Confusion Matrix
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)
labels = best_model.classes_

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            xticklabels=labels,
            yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Analysis Complete!")