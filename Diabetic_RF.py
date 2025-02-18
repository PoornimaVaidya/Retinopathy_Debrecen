import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/poornimavaidya/Desktop/MLP/Retinopathy_Debrecen.csv'
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop(columns=['class'])  # Features
y = data['class']  # Target variable

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Set up RandomForest with GridSearchCV to tune the hyperparameters
param_grid = {
    'n_estimators': [50, 100, 150, 200],  # Different number of trees to test
    'max_depth': [None, 10, 20, 30],  # Different maximum depths
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2']  # Different strategies for the maximum number of features
}

# Initialize RandomForest
rf = RandomForestClassifier(random_state=42)

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters found: ", grid_search.best_params_)

# Use the best model from GridSearchCV
best_rf = grid_search.best_estimator_

# Predict the class for the test set
y_pred = best_rf.predict(X_test)
y_pred_prob = best_rf.predict_proba(X_test)[:, 1]  # Probabilities for class=1 (diabetic retinopathy)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_pred_prob):.2f}", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label="Precision-Recall Curve", color="green")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall (Sensitivity)")
plt.ylabel("Precision")
plt.legend()
plt.grid()
plt.show()
