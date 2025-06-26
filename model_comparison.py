import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the preprocessed data
df = pd.read_csv("final_data.csv")

# --- DASS Baseline Implementation ---

# Define the DASS Depression items
depression_items = [f'Q{i}A' for i in [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]]

# The original data was 1-4, but DASS scores are 0-3.
# The notebook doesn't show the original question format, but the DASS website says:
# 0 = Did not apply to me at all
# 1 = Applied to me to some degree, or some of the time
# 2 = Applied to me to a considerable degree, or a good part of the time
# 3 = Applied to me very much, or most of the time
# The data in final_data.csv is 1-4 for the Q*A columns. I will assume this corresponds to 0-3 scoring.
# So, I will subtract 1 from each of the question responses.
df_dass = df.copy()
for item in depression_items:
    # Check if the column exists before trying to subtract
    if item in df_dass.columns:
        df_dass[item] = df_dass[item] - 1

# Calculate the DASS Depression score
# The DASS-42 score is the sum of the item scores.
df_dass['dass_depression_score'] = df_dass[depression_items].sum(axis=1)

# Define the severity levels
def get_dass_severity(score):
    if score <= 9:
        return 'Normal'
    elif score <= 13:
        return 'Mild'
    elif score <= 20:
        return 'Moderate'
    elif score <= 27:
        return 'Severe'
    else:
        return 'Extremely Severe'

df_dass['dass_severity'] = df_dass['dass_depression_score'].apply(get_dass_severity)

# --- Model Training and Evaluation ---

# Find all columns that start with 'Q'
question_cols = [col for col in df.columns if col.startswith('Q')]

# Prepare the data for the ML models
X = df.drop(['target', 'total_count', 'Unnamed: 0'], axis=1, errors='ignore')
y = df['target']

# Ensure the order of severity levels is consistent for reports
labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- DASS Baseline Evaluation ---
# The DASS baseline is evaluated on the test set index
y_true_baseline = y_test
y_pred_baseline = df_dass.loc[y_test.index]['dass_severity']

print("--- DASS Official Scoring (Baseline) ---")
print(f"Accuracy: {accuracy_score(y_true_baseline, y_pred_baseline):.4f}")
print(classification_report(y_true_baseline, y_pred_baseline, labels=labels, zero_division=0))
conf_matrix_baseline = confusion_matrix(y_true_baseline, y_pred_baseline, labels=labels)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_baseline, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('DASS Baseline Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_baseline.png')
plt.close()


# --- Machine Learning Model Evaluation ---

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(10,20,30,20,10), max_iter=1000, random_state=42)
}

for name, model in models.items():
    print(f"--- {name} ---")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
    
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png')
    plt.close()

print("Model comparison script finished. Confusion matrix images have been saved.")
