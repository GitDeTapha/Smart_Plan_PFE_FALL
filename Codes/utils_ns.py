# utils_exploration.py
# Contains common routines and variable definitions
# Allows to lighten the notebook prediction_ns.ipynb

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.utils import resample, shuffle

############################################################
#                                                          #
#        ------      General            -------            #
#                                                          #
############################################################ 

def save_dataframe(df, name):
    """
    Save a DataFrame as a CSV file in the 'data_ns' directory.
    
    Args:
        df (DataFrame): The DataFrame to save.
        name (str): The name of the CSV file (without extension).
    """
    filename = f'data_ns/{name}.csv'
    df.to_csv(filename, index=False)

def save_results_to_csv(filename, dataframes):
    """
    Save multiple DataFrames into a single CSV file with model identifiers.
    
    Args:
        filename (str): The name of the output CSV file.
        dataframes (list): A list of DataFrames to combine and save.
    """
    # Combine DataFrames with keys to identify each model
    combined_df = pd.concat(dataframes, keys=['AdaBoost', 'XGBoost', 'Random Forest'])
    combined_df.reset_index(level=0, inplace=True)
    combined_df.rename(columns={'level_0': 'Model'}, inplace=True)
    combined_df.to_csv(filename, index=False)

############################################################
#                                                          #
#        ------      Modeling           -------            #
#                                                          #
############################################################ 

# (No code under this section)

############################################################
#                                                          #
#        ------          Evaluation     -------            #
#                                                          #
############################################################ 

#=====================
#  Tree Ensemble
#=====================

def evaluate_model(X_train, y_train, X_test, y_test, y_test_proba, y_train_proba, y_test_pred, y_train_pred):
    """
    Evaluate model performance by calculating metrics and displaying confusion matrices.
    
    Args:
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        y_test_proba, y_train_proba: Predicted probabilities for test and train sets.
        y_test_pred, y_train_pred: Predicted classes for test and train sets.
    Returns:
        DataFrame containing evaluation metrics.
    """
    # Confusion matrices for train and test sets
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    # Display confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_train, annot=True, fmt="d", ax=ax[0], cmap="Blues")
    ax[0].set_title("Confusion Matrix (Train)")
    ax[0].set_xlabel("Predicted Labels")
    ax[0].set_ylabel("True Labels")
    
    sns.heatmap(cm_test, annot=True, fmt="d", ax=ax[1], cmap="Blues")
    ax[1].set_title("Confusion Matrix (Test)")
    ax[1].set_xlabel("Predicted Labels")
    ax[1].set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    metrics = {
        'Metric': ['Recall', 'AUC', 'Accuracy', 'Precision', 'F1 Score'],
        'Train': [
            recall_score(y_train, y_train_pred, average='macro'),
            roc_auc_score(y_train, y_train_proba),
            accuracy_score(y_train, y_train_pred),
            precision_score(y_train, y_train_pred, average='macro'),
            f1_score(y_train, y_train_pred, average='macro')
        ],
        'Test': [
            recall_score(y_test, y_test_pred, average='macro'),
            roc_auc_score(y_test, y_test_proba),
            accuracy_score(y_test, y_test_pred),
            precision_score(y_test, y_test_pred, average='macro'),
            f1_score(y_test, y_test_pred, average='macro')
        ]
    }

    # Create a DataFrame for displaying
    results_df = pd.DataFrame(metrics)
    print(results_df.set_index('Metric'))
    return results_df

def evaluate_tree_ensemble_model(model, X_train, y_train, X_test, y_test, y_train_proba, y_test_proba):
    """
    Evaluate a tree ensemble model by calculating metrics and displaying confusion matrices.
    
    Args:
        model: Trained tree ensemble model.
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        y_train_proba, y_test_proba: Predicted probabilities for train and test sets.
    """
    # Predict classes
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    # Display confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_train, annot=True, fmt="d", ax=ax[0], cmap="Blues")
    ax[0].set_title("Confusion Matrix (Train)")
    ax[0].set_xlabel("Predicted Labels")
    ax[0].set_ylabel("True Labels")
    
    sns.heatmap(cm_test, annot=True, fmt="d", ax=ax[1], cmap="Blues")
    ax[1].set_title("Confusion Matrix (Test)")
    ax[1].set_xlabel("Predicted Labels")
    ax[1].set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    metrics = {
        'Metric': ['Recall', 'AUC', 'Accuracy', 'Precision', 'F1 Score'],
        'Train': [
            recall_score(y_train, y_train_pred, average='macro'),
            roc_auc_score(y_train, y_train_proba),
            accuracy_score(y_train, y_train_pred),
            precision_score(y_train, y_train_pred, average='macro'),
            f1_score(y_train, y_train_pred, average='macro')
        ],
        'Test': [
            recall_score(y_test, y_test_pred, average='macro'),
            roc_auc_score(y_test, y_test_proba),
            accuracy_score(y_test, y_test_pred),
            precision_score(y_test, y_test_pred, average='macro'),
            f1_score(y_test, y_test_pred, average='macro')
        ]
    }

    # Create a DataFrame for displaying
    results_df = pd.DataFrame(metrics)
    print(results_df.set_index('Metric'))

def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Plot ROC curve for model's predicted probabilities.
    
    Args:
        y_true (array): True labels.
        y_pred_proba (array): Predicted probabilities for the positive class.
        model_name (str): Name of the model for the plot title.
    """
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_pr_curve(y_test, y_test_proba, model_name='Model'):
    """
    Plot Precision-Recall curve for the model.
    
    Args:
        y_test (array): True labels.
        y_test_proba (array): Predicted probabilities.
        model_name (str): Name of the model for the plot title.
    """
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    average_precision = average_precision_score(y_test, y_test_proba)
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()

def plot_feature_importances(model, feature_names, title, top_n=16):
    """
    Plot the top n feature importances from the model.
    
    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names (list): Names of the features.
        title (str): Title of the plot.
        top_n (int): Number of top features to display.
    """
    # Extract feature importances
    importances = model.feature_importances_
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importances = feature_importances.sort_values(by='importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances)
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()


def plot_feature_importances_bootstrap(model, X_train, y_train, feature_names, title, top_n=15, n_iterations=10):
    """
    Plot feature importances using bootstrapping.
    
    Args:
        model: Trained model with feature_importances_ attribute.
        X_train (DataFrame or ndarray): Training data.
        y_train (Series or ndarray): Training labels.
        feature_names (list): Names of the features.
        title (str): Title of the plot.
        top_n (int): Number of top features to display.
        n_iterations (int): Number of bootstrap iterations.
    """
    # Initialize array to store importances
    all_importances = np.zeros((n_iterations + 1, X_train.shape[1]))
    all_importances[0] = model.feature_importances_
    
    for i in range(1, n_iterations + 1):
        # Bootstrap sample
        X_resampled, y_resampled = resample(X_train, y_train)
        model.fit(X_resampled, y_resampled)
        all_importances[i] = model.feature_importances_
    
    # Compute mean and std
    mean_importances = np.mean(all_importances, axis=0)
    std_importances = np.std(all_importances, axis=0)
    
    # Create DataFrame
    feature_importances = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': mean_importances,
        'importance_std': std_importances
    })
    feature_importances = feature_importances.sort_values(by='importance_mean', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance_mean', y='feature', data=feature_importances, xerr=feature_importances['importance_std'])
    plt.title(title)
    plt.xlabel('Importance (mean)')
    plt.ylabel('Features')
    plt.show()

def permutation_importance(model, X, y, metric=accuracy_score):
    """
    Calculate permutation importance for features in a classification model.
    
    Args:
        model: Trained model.
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.
        metric (function): Metric function to evaluate model performance.
    
    Returns:
        list: Importance scores for each feature.
    """
    baseline = model.evaluate(X, y, verbose=1)
    baseline_metric = baseline[1] 
    importance_scores = []
    
    for i in range(X.shape[1]):
        X_permuted = X.copy()
        X_permuted[:, i] = shuffle(X_permuted[:, i])
        metric_score = model.evaluate(X_permuted, y, verbose=0)[1]
        importance_scores.append(baseline_metric - metric_score)
    
    return importance_scores

def plot_boxplot_of_probabilities(model, X_test, y_test, y_test_prob, title, threshold=0.5):
    """
    Plot boxplots of predicted probabilities for false positives and false negatives.
    
    Args:
        model: Trained model.
        X_test (DataFrame or ndarray): Test features.
        y_test (Series or ndarray): True labels.
        y_test_prob (ndarray): Predicted probabilities.
        title (str): Title of the plot.
        threshold (float): Threshold for classification.
    """
    # Ensure y_test is a Series
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze()
    elif isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)
    
    # Check lengths
    if len(y_test) != len(y_test_prob):
        raise ValueError("Length of y_test_prob must equal length of y_test")
    
    # Determine false positives and false negatives
    fp_probs = y_test_prob[(y_test_prob >= threshold) & (y_test == 0)]
    fn_probs = y_test_prob[(y_test_prob <= threshold) & (y_test == 1)]
    
    # Create boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[fp_probs, fn_probs], notch=True, palette=['red', 'blue'])
    plt.xticks([0, 1], ['False Positives', 'False Negatives'])
    plt.ylabel('Predicted Probabilities')
    plt.title(title)
    plt.show()

def plot_confidence_histogram(model, X_test, y_test, y_test_prob, title, threshold=0.5):
    """
    Plot histogram of predicted probabilities for false positives and false negatives.
    
    Args:
        model: Trained model.
        X_test (DataFrame or ndarray): Test features.
        y_test (Series or ndarray): True labels.
        y_test_prob (ndarray): Predicted probabilities.
        title (str): Title of the plot.
        threshold (float): Threshold for classification.
    """
    # Ensure y_test is a Series
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze()
    elif isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)
    
    # Check lengths
    if len(y_test) != len(y_test_prob):
        raise ValueError("Length of y_test_prob must equal length of y_test")
    
    # Determine false positives and false negatives
    fp_probs = y_test_prob[(y_test_prob >= threshold) & (y_test == 0)]
    fn_probs = y_test_prob[(y_test_prob <= threshold) & (y_test == 1)]
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    plt.hist(fp_probs, bins=30, alpha=0.5, label='False Positives', color='red')
    plt.hist(fn_probs, bins=30, alpha=0.5, label='False Negatives', color='blue')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Number of Observations')
    plt.title(title)
    plt.xlim(0, 1) 
    plt.legend()
    plt.show()

def permutation_importance_bootstrap(model, X, y, metric=r2_score, test_score=None, num_rounds=10, n_iterations=100):
    """
    Calculate permutation importance using bootstrapping.
    
    Args:
        model: Trained model.
        X (DataFrame or ndarray): Feature matrix.
        y (Series or ndarray): Target vector.
        metric (function): Metric function to evaluate model performance.
        test_score (float): Baseline performance score.
        num_rounds (int): Number of times to permute a feature.
        n_iterations (int): Number of bootstrap iterations.
    
    Returns:
        DataFrame: Mean and std of importance scores for each feature.
    """
    # Ensure X and y are DataFrame and Series
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    # Get baseline metric
    baseline_metric = test_score if test_score is not None else model.evaluate(X.values, y.values, verbose=0)[1]
    all_importances = np.zeros((n_iterations, X.shape[1]))
    
    for n in range(n_iterations):
        X_resampled, y_resampled = resample(X, y)
        for i in range(X.shape[1]):
            score_changes = []
            for _ in range(num_rounds):
                X_permuted = X_resampled.copy()
                X_permuted.iloc[:, i] = shuffle(X_permuted.iloc[:, i])
                score = model.evaluate(X_permuted.values, y_resampled.values, verbose=0)
                permuted_score = score if isinstance(score, float) else score[1]
                score_changes.append(baseline_metric - permuted_score)
            all_importances[n, i] = np.mean(score_changes)
    
    mean_importances = np.mean(all_importances, axis=0)
    std_importances = np.std(all_importances, axis=0)
    
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': mean_importances,
        'importance_std': std_importances
    })
    
    return feature_importances

#===============
#           ANN
#===============

def permutation_importance_(model, X, y, metric=accuracy_score):
    """
    Calculate permutation importance for features in a classification model (ANN).
    
    Args:
        model: Trained Keras model.
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.
        metric (function): Metric function to evaluate model performance.
    
    Returns:
        list: Importance scores for each feature.
    """
    baseline = model.evaluate(X, y, verbose=0)
    baseline_metric = baseline[1]
    importance_scores = []
    
    for i in range(X.shape[1]):
        X_permuted = X.copy()
        X_permuted[:, i] = shuffle(X_permuted[:, i])
        metric_score = model.evaluate(X_permuted, y, verbose=0)[1]
        importance_scores.append(baseline_metric - metric_score)
    
    return importance_scores

def evaluate_model_(model, X_train, y_train, X_test, y_test, y_test_proba, y_train_proba, y_test_pred, y_train_pred):
    """
    Evaluate an ANN model by calculating metrics and displaying confusion matrices.
    
    Args:
        model: Trained Keras model.
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        y_test_proba, y_train_proba: Predicted probabilities for test and train sets.
        y_test_pred, y_train_pred: Predicted classes for test and train sets.
    Returns:
        DataFrame containing evaluation metrics.
    """
    # Confusion matrices for train and test sets
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    # Display confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_train, annot=True, fmt="d", ax=ax[0], cmap="Blues")
    ax[0].set_title("Confusion Matrix (Train)")
    ax[0].set_xlabel("Predicted Labels")
    ax[0].set_ylabel("True Labels")
    
    sns.heatmap(cm_test, annot=True, fmt="d", ax=ax[1], cmap="Blues")
    ax[1].set_title("Confusion Matrix (Test)")
    ax[1].set_xlabel("Predicted Labels")
    ax[1].set_ylabel("True Labels")
    plt.show()
    
    # Calculate metrics
    metrics = {
        'Metric': ['Recall', 'AUC', 'Accuracy', 'Precision', 'F1 Score'],
        'Train': [
            recall_score(y_train, y_train_pred, average='macro'),
            roc_auc_score(y_train, y_train_proba),
            accuracy_score(y_train, y_train_pred),
            precision_score(y_train, y_train_pred, average='macro'),
            f1_score(y_train, y_train_pred, average='macro')
        ],
        'Test': [
            recall_score(y_test, y_test_pred, average='macro'),
            roc_auc_score(y_test, y_test_proba),
            accuracy_score(y_test, y_test_pred),
            precision_score(y_test, y_test_pred, average='macro'),
            f1_score(y_test, y_test_pred, average='macro')
        ]
    }

    # Create a DataFrame for displaying
    results_df = pd.DataFrame(metrics)
    print(results_df.set_index('Metric'))
    return results_df

def plot_learning_curve_ann(history):
    """
    Plot training and validation accuracy and loss curves for an ANN.
    
    Args:
        history: History object returned by model.fit().
    """
    # Accuracy plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.show()    


def permutation_importance__(model, X, y, metric, test_score, num_rounds=1):
    """
    Calculate permutation importance for features in a regression model.
    
    Args:
        model: Trained model.
        X (DataFrame): Feature matrix.
        y (Series or DataFrame): Target vector.
        metric (function): Metric function to evaluate model performance.
        test_score (float): Baseline performance score.
        num_rounds (int): Number of times to permute a feature.
    
    Returns:
        list: Importance scores for each feature.
    """
    baseline_metric = test_score if isinstance(test_score, float) else test_score[1]
    importance_scores = []

    for i in range(X.shape[1]):
        score_changes = []
        for _ in range(num_rounds):
            X_permuted = X.copy()
            X_permuted.iloc[:, i] = shuffle(X_permuted.iloc[:, i])
            score = model.evaluate(X_permuted, y, verbose=0)
            permuted_score = score if isinstance(score, float) else score[1]
            score_changes.append(baseline_metric - permuted_score)
        importance_scores.append(np.mean(score_changes))
    
    return importance_scores

def plot_permutation_importance(importance_scores, feature_names):
    """
    Plot a bar chart of feature importances based on permutation importance.
    
    Args:
        importance_scores (list): Importance scores for each feature.
        feature_names (list): Names of the features.
    """
    # Create DataFrame
    feature_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(importance_scores)
    })
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importances)
    plt.title('ANN')
    plt.xlabel('Absolute Importance')
    plt.ylabel('Features')
    plt.grid(True)
    plt.show()

def plot_permutation_importance_(importance_scores, title):
    """
    Plot a bar chart of feature importances with error bars.
    
    Args:
        importance_scores (DataFrame): DataFrame containing mean and std of importance scores.
        title (str): Title of the plot.
    """
    # Sort
    feature_importances = importance_scores.sort_values(by='importance_mean', ascending=True)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importances['feature'], feature_importances['importance_mean'], xerr=feature_importances['importance_std'])
    plt.title(title)
    plt.xlabel('Importance (mean)')
    plt.ylabel('Features')
    plt.grid(True)
    plt.show()
        
#===============
# Common for both ann and tree ensemble
#===============   

def plot_confidence_histogram_(model, X_test, y_test, y_prob):
    """
    Plot histogram of predicted probabilities for false positives and false negatives.
    
    Args:
        model: Trained model.
        X_test (DataFrame or ndarray): Test features.
        y_test (Series or ndarray): True labels.
        y_prob (ndarray): Predicted probabilities.
    """
    # Determine false positives and false negatives
    y_pred = (y_prob > 0.5).astype(int)
    fp_probs = y_prob[(y_pred == 1) & (y_test == 0)]
    fn_probs = y_prob[(y_pred == 0) & (y_test == 1)]

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(fp_probs, bins=30, alpha=0.5, label='False Positives', color='red')
    plt.hist(fn_probs, bins=30, alpha=0.5, label='False Negatives', color='blue')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Number of Observations')
    plt.title('ANN')
    plt.legend()
    plt.show()
        
def plot_boxplot_of_probabilities_(model, X_test, y_test, y_prob):
    """
    Plot boxplots of predicted probabilities for false positives and false negatives.
    
    Args:
        model: Trained model.
        X_test (DataFrame or ndarray): Test features.
        y_test (Series or ndarray): True labels.
        y_prob (ndarray): Predicted probabilities.
    """
    # Determine false positives and false negatives
    y_pred = (y_prob > 0.5).astype(int)
    fp_probs = y_prob[(y_pred == 1) & (y_test == 0)]
    fn_probs = y_prob[(y_pred == 0) & (y_test == 1)]

    # Create boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[fp_probs, fn_probs], notch=True, palette=["red", "blue"])
    plt.xticks([0, 1], ['False Positives', 'False Negatives'])
    plt.ylabel('Predicted Probability')
    plt.title('Probability of False Positives and False Negatives')
    plt.show()
