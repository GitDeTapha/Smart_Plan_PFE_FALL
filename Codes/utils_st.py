# utils_st.py
# Contains common routines and variable definitions
# Prevents overloading the Prediction_Service_Time.ipynb notebook

# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle

############################################################
#                                                          #
#        ------      General             -------           #
#                                                          #
############################################################ 

def plot_service_time_distribution(df, column):
    """
    Plot a boxplot and histogram for a specified DataFrame column, and print its variance.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the column to plot.
    """
    variance = df[column].var()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Boxplot
    sns.boxplot(ax=axes[0], x=df[column])
    axes[0].set_title(f'Boxplot of {column}')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Values')
    axes[0].set_xlim(-10, 375)
    
    # Histogram
    sns.histplot(ax=axes[1], data=df, x=column, kde=True, bins=30)
    axes[1].set_title(f'Histogram of {column}')
    axes[1].set_xlabel(column)
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim(-10, 375)
    
    plt.tight_layout()
    plt.show()
    print(f"The variance of {column} is: {variance}")

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

def plot_residuals_histogram(residuals):
    """
    Plot a histogram of residuals.

    Args:
        residuals (array-like): Residuals (true values - predicted values).
    """
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, alpha=0.5, color='skyblue')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Residuals')
    plt.grid(True)
    plt.show()

def plot_residuals_boxplot(residuals):
    """
    Plot a boxplot of residuals.

    Args:
        residuals (array-like): Residuals (true values - predicted values).
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=residuals)
    plt.xlabel('Residuals')
    plt.title('Boxplot of Prediction Residuals')
    plt.grid(True)
    plt.show()
    
#===========================
#  Tree Ensemble Routine
#===========================

def evaluate_model_performance(y_train, y_test, y_pred_train, y_pred_test):
    """
    Evaluate and print regression metrics for training and test sets.

    Args:
        y_train (array-like): True target values for the training set.
        y_test (array-like): True target values for the test set.
        y_pred_train (array-like): Predicted values for the training set.
        y_pred_test (array-like): Predicted values for the test set.
    """
    # Training metrics
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    # Test metrics
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print("Training Metrics:")
    print(f"R-squared: {r2_train}")
    print(f"MAE: {mae_train}")
    print(f"MSE: {mse_train}")
    print(f"RMSE: {rmse_train}\n")

    print("Testing Metrics:")
    print(f"R-squared: {r2_test}")
    print(f"MAE: {mae_test}")
    print(f"MSE: {mse_test}")
    print(f"RMSE: {rmse_test}")

def plot_feature_importances(model, feature_names, title, top_n=15):
    """
    Plot the top n feature importances of the model.

    Args:
        model: The trained model with attribute feature_importances_.
        feature_names (list): Names of the features.
        title (str): Title of the plot.
        top_n (int): Number of top features to display.
    """
    importances = model.feature_importances_
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importances = feature_importances.sort_values(by='importance', ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances)
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()
    
#===============
#  ANN Routine
#===============

def permutation_importance(model, X, y, metric=None, num_rounds=1):
    """
    Calculate permutation importance for features in a Keras regression model.

    Args:
        model: The trained Keras model.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        metric (function, optional): Metric function to evaluate model performance. If None, use model.evaluate.
        num_rounds (int): Number of times to permute each feature.

    Returns:
        pd.DataFrame: DataFrame with features and their importance scores.
    """
    # Evaluate baseline performance
    if metric is None:
        baseline_metric = model.evaluate(X, y, verbose=0)
    else:
        y_pred = model.predict(X)
        baseline_metric = metric(y, y_pred)
    importance_scores = []

    for i in range(X.shape[1]):
        score_changes = []
        for _ in range(num_rounds):
            X_permuted = X.copy()
            X_permuted.iloc[:, i] = shuffle(X_permuted.iloc[:, i]).values
            if metric is None:
                permuted_metric = model.evaluate(X_permuted, y, verbose=0)
            else:
                y_pred_permuted = model.predict(X_permuted)
                permuted_metric = metric(y, y_pred_permuted)
            score_changes.append(permuted_metric - baseline_metric)
        importance_scores.append(np.mean(score_changes))
    
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importance_scores})
    return feature_importances.sort_values(by='importance', ascending=False)

def plot_permutation_importance(feature_importances, title='Feature Importances'):
    """
    Plot a bar chart of feature importances.

    Args:
        feature_importances (pd.DataFrame): DataFrame with 'feature' and 'importance' columns.
        title (str): Title of the plot.
    """
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importances)
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.grid(True)
    plt.show()

### Quenrataine

def permutation_importance_(model, X, y, metric, baseline_metric, num_rounds=1):
    """
    Calculate permutation importance for features in a regression model.

    Args:
        model: The trained regression model.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or pd.DataFrame): Target vector.
        metric (function): Metric function to evaluate model performance.
        baseline_metric (float): Baseline performance score of the model.
        num_rounds (int): Number of times to permute each feature.

    Returns:
        pd.DataFrame: DataFrame with features and their importance scores.
    """
    importance_scores = []

    for i in range(X.shape[1]):
        score_changes = []
        for _ in range(num_rounds):
            X_permuted = X.copy()
            X_permuted.iloc[:, i] = shuffle(X_permuted.iloc[:, i]).values
            y_pred_permuted = model.predict(X_permuted)
            permuted_metric = metric(y, y_pred_permuted)
            score_changes.append(permuted_metric - baseline_metric)
        importance_scores.append(np.mean(score_changes))
    
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importance_scores})
    return feature_importances.sort_values(by='importance', ascending=False)
