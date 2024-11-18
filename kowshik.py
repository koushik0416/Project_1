import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_algorithms(X, y):
    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define algorithms to test
    algorithms = {
        'XGBoost': XGBClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    # Store results
    results = []
    
    # Evaluate each algorithm
    for name, model in algorithms.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results.append({
            'Algorithm': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC-ROC': auc,
            'CV Mean': cv_mean,
            'CV Std': cv_std
        })
    
    return pd.DataFrame(results)

# Load and prepare data
df = pd.read_csv(r'C:\Users\lenovo\Desktop\hello\survey_lung_data.csv')
X = df.drop('LUNG_CANCER', axis=1)
y = (df['LUNG_CANCER'] == 'YES').astype(int)

# Convert categorical variables
X['GENDER'] = (X['GENDER'] == 'M').astype(int)

# Evaluate algorithms
results_df = evaluate_algorithms(X, y)

# Sort by AUC-ROC score
results_df = results_df.sort_values('AUC-ROC', ascending=False)

# Display results
print("\nAlgorithm Comparison Results:")
print(results_df.round(4))

# Plot performance comparison
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
results_plot = results_df.melt(id_vars=['Algorithm'], 
                             value_vars=metrics, 
                             var_name='Metric', 
                             value_name='Score')

sns.barplot(data=results_plot, x='Algorithm', y='Score', hue='Metric')
plt.xticks(rotation=45)
plt.title('Algorithm Performance Comparison')
plt.tight_layout()
plt.show()

# Plot cross-validation results
plt.figure(figsize=(10, 6))
plt.errorbar(range(len(results_df)), 
            results_df['CV Mean'], 
            yerr=results_df['CV Std'], 
            fmt='o')
plt.xticks(range(len(results_df)), results_df['Algorithm'], rotation=45)
plt.title('Cross-validation Results with Standard Deviation')
plt.ylabel('Score')
plt.tight_layout()
plt.show()