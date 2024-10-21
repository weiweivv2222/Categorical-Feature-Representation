import numpy as np
import pandas as pd
import random
import time
import config_cat_embedding

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from tqdm.notebook import tqdm
from scipy import stats  # For confidence intervals
from NodeTransformer import NodeTransformer  # Import your NodeTransformer
from data_prep import bank_data_prep, adult_data_prep
from embedding_helper import create_network


# Set the random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load and preprocess data
data_path = config_cat_embedding.paths['data']
data_path_out = config_cat_embedding.paths['data_output']
bank_data = pd.read_csv(data_path+'adult.csv', sep=',')

df_bank, cat_cols = adult_data_prep(bank_data)

X = df_bank.iloc[:, :-1]
y = df_bank.y  # Assuming y is already numeric and doesn't need mapping

# Define the classifiers
seed = 42

models = [
    #('LR', LogisticRegression(solver='lbfgs', random_state=seed, max_iter=1000)),
    #('DT', DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=seed)),
    #('RF', RandomForestClassifier(n_estimators=200, max_depth=5, random_state=seed, min_samples_leaf=3)),
    #('KNN', KNeighborsClassifier(n_neighbors=3))
    #('XGB', XGBClassifier(eval_metric='logloss', random_state=seed))
    # ('SVM', SVC(gamma='scale', random_state=seed, probability=True)),
    ('MLP', KerasClassifier(model=create_network,epochs=100, batch_size=100, verbose=0, random_state=seed))]

# Cross-validation setup
cv = StratifiedKFold(n_splits=20, shuffle=True, random_state=seed)  # Adjust n_splits as needed

# Function to calculate confidence intervals
def confidence_interval(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return m, m - h, m + h

# Main loop over models
for name, classifier in models:
    print(f"Classifier: {name}")
    # Lists to store metrics for each fold
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    roc_aucs = []
    computation_times = []

    fold = 1
    for train_index, test_index in cv.split(X, y):
        # Split data into training and test sets for this fold
    	
        X_train_fold = X.iloc[train_index].copy()
        X_test_fold = X.iloc[test_index].copy()
        y_train_fold = y.iloc[train_index].reset_index(drop=True)
        y_test_fold = y.iloc[test_index].reset_index(drop=True)

        # Apply pd.get_dummies to training data
        X_train_fold = pd.get_dummies(X_train_fold, prefix_sep='_', drop_first=True)
        X_train_fold.columns = X_train_fold.columns.astype(str)

        # Get the columns after encoding
        encoded_columns = X_train_fold.columns

        # Apply the same encoding to test data, ensuring same columns
        X_test_fold = pd.get_dummies(X_test_fold, prefix_sep='_', drop_first=True)
        X_test_fold.columns = X_test_fold.columns.astype(str)
        X_test_fold = X_test_fold.reindex(columns=encoded_columns, fill_value=0)

        # Ensure that the order of columns in test set matches the training set
        X_test_fold = X_test_fold[encoded_columns]

        # Fit the NodeTransformer on the training data
        node_transformer = NodeTransformer(mode='Extended', n_estimators=200, impurity_reweight=True,
                                           max_depth=5, walk_length=5, n_walks=50, window=5, dimension=10,
                                           random_state=seed)
        node_transformer.fit(X_train_fold.values, y_train_fold.values)

        # Transform both training and test data
        X_train_transformed = node_transformer.transform(X_train_fold.values)
        X_test_transformed = node_transformer.transform(X_test_fold.values)

        # Convert transformed data to DataFrame and ensure column names are strings
        X_train_transformed = pd.DataFrame(X_train_transformed)
        X_test_transformed = pd.DataFrame(X_test_transformed)
        X_train_transformed.columns = X_train_transformed.columns.astype(str)
        X_test_transformed.columns = X_test_transformed.columns.astype(str)

        # Standard scaling
        stc = StandardScaler()
        X_train_scaled = stc.fit_transform(X_train_transformed)
        X_test_scaled = stc.transform(X_test_transformed)

        # Update number_of_features for MLP
        number_of_features = X_train_scaled.shape[1]
        if name == 'MLP':
            classifier.set_params(model__number_of_features=number_of_features)

        # Start timing
        start_time = time.time()

        # Fit the classifier
        classifier.fit(X_train_scaled, y_train_fold)

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        computation_times.append(elapsed_time)

        # Predict on test data
        y_pred_fold = classifier.predict(X_test_scaled)
        if hasattr(classifier, "predict_proba"):
            y_pred_prob_fold = classifier.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_scores = classifier.decision_function(X_test_scaled)
            # Normalize scores to [0,1]
            y_pred_prob_fold = (y_pred_scores - y_pred_scores.min()) / (y_pred_scores.max() - y_pred_scores.min())

        # Collect performance metrics
        accuracies.append(accuracy_score(y_test_fold, y_pred_fold))
        precisions.append(precision_score(y_test_fold, y_pred_fold, zero_division=0))
        recalls.append(recall_score(y_test_fold, y_pred_fold))
        f1s.append(f1_score(y_test_fold, y_pred_fold))
        roc_aucs.append(roc_auc_score(y_test_fold, y_pred_prob_fold))

        # Optionally, print confusion matrix and classification report for each fold
        # print(f"Fold {fold} Confusion Matrix:")
        # print(confusion_matrix(y_test_fold, y_pred_fold))
        # print(classification_report(y_test_fold, y_pred_fold, digits=4))

        fold += 1
        print(f"########fold number: {fold}")

    # Calculate mean and confidence intervals
    acc_mean, acc_ci_lower, acc_ci_upper = confidence_interval(accuracies)
    prec_mean, prec_ci_lower, prec_ci_upper = confidence_interval(precisions)
    rec_mean, rec_ci_lower, rec_ci_upper = confidence_interval(recalls)
    f1_mean, f1_ci_lower, f1_ci_upper = confidence_interval(f1s)
    roc_mean, roc_ci_lower, roc_ci_upper = confidence_interval(roc_aucs)
    time_mean = np.mean(computation_times)

    # Print results
    print(f"Accuracy: {acc_mean:.4f} (95% CI: {acc_ci_lower:.4f} - {acc_ci_upper:.4f})")
    print(f"Precision: {prec_mean:.4f} (95% CI: {prec_ci_lower:.4f} - {prec_ci_upper:.4f})")
    print(f"Recall: {rec_mean:.4f} (95% CI: {rec_ci_lower:.4f} - {rec_ci_upper:.4f})")
    print(f"F1 Score: {f1_mean:.4f} (95% CI: {f1_ci_lower:.4f} - {f1_ci_upper:.4f})")
    print(f"ROC AUC: {roc_mean:.4f} (95% CI: {roc_ci_lower:.4f} - {roc_ci_upper:.4f})")
    print(f"Average Computation Time per Fold: {time_mean:.4f} seconds\n")

import pickle

# Dictionary of results
result_pkl = {
    'accuracies': accuracies,
    'precisions': precisions,
    'recalls': recalls,
    'f1s': f1s,
    'rocs': roc_aucs
}

# Save to a pickle file
with open('results.pkl', 'wb') as f:
    pickle.dump(result_pkl, f)
