import numpy as np
import pandas as pd
# task 1
def read_data(filename):
    """
    Reads a CSV file into a pandas DataFrame.
    
    Args:
    filename (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data.
    """
    return pd.read_csv(filename)

data = read_data('breast-cancer.csv')

#task 2

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def clean_and_split_data(data):
    """
    Cleans the data by removing rows with any empty cells and splits it into training and testing sets.
    
    Args:
    data (pd.DataFrame): The data to be cleaned and split.

    Returns:
    tuple: A tuple containing the training and testing data.
    """
    # Removing rows with any empty cells
    data_cleaned = data.dropna()

    # Splitting the dataset into training (80%) and testing (20%)
    train_data, test_data = train_test_split(data_cleaned, test_size=0.2, random_state=42)
    return train_data, test_data

train_data, test_data = clean_and_split_data(data)

#task 3
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

def train_decision_tree(train_data, test_data, target_column):
    """
    Trains a Decision Tree Classifier on the training data and evaluates it using the testing data.
    
    Args:
    train_data (pd.DataFrame): The training data.
    test_data (pd.DataFrame): The testing data.
    target_column (str): The name of the target column.

    Returns:
    dict: A dictionary containing the model, training time, and performance metrics.
    """
    # Separating features and target
    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]
    
    # Training the Decision Tree
    start_time = time.time()
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Evaluating the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, pos_label='M')  # assuming 'M' is malignant
    specificity = recall_score(y_test, y_pred, pos_label='B')  # assuming 'B' is benign

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    # Visualizing the Decision Tree
    plt.figure(figsize=(20,10))
    plot_tree(model, filled=True, feature_names=X_train.columns, class_names=['B', 'M'])
    plt.title('Decision Tree')
    plt.show()

    # Visualizing the Confusion Matrix
    disp.plot()
    plt.show()
    
    return {
        "model": model,
        "training_time": training_time,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity
    }

# Example usage
results_dt = train_decision_tree(train_data, test_data, 'diagnosis')

#task 4

from sklearn.svm import SVC

def train_svm_rbf(train_data, test_data, target_column):
    """
    Trains a Support Vector Machine (RBF kernel) on the training data and evaluates it using the testing data.
    
    Args:
    train_data (pd.DataFrame): The training data.
    test_data (pd.DataFrame): The testing data.
    target_column (str): The name of the target column.

    Returns:
    dict: A dictionary containing the model, training time, and performance metrics.
    """
    # Separating features and target
    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]

    # Training the SVM
    start_time = time.time()
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Evaluating the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, pos_label='M')  # assuming 'M' is malignant
    specificity = recall_score(y_test, y_pred, pos_label='B')  # assuming 'B' is benign

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    return {
        "model": model,
        "training_time": training_time,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity
    }

# Example usage
results_svm = train_svm_rbf(train_data, test_data, 'diagnosis')

#task 6
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import time

def feature_importance_and_model_evaluation(train_data, test_data, target_column):
    """
    Uses Random Forest to determine feature importance, visualizes the top two features, and retrains the 
    Decision Tree model after sequentially removing the least important features.
    
    Args:
    train_data (pd.DataFrame): The training data.
    test_data (pd.DataFrame): The testing data.
    target_column (str): The name of the target column.

    Returns:
    dict: A dictionary containing evaluations for models with features removed.
    """
    # Separate features and target
    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]

    # Train Random Forest to get feature importances
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    
    # Mapping 'B' and 'M' to numbers for plotting
    color_map = {'B': 0, 'M': 1}  # Mapping benign to 0 and malignant to 1
    y_numeric = y_train.map(color_map)  # Convert y_train to numeric for coloring in the scatter plot
    
    # Plotting the top two features in an x-y coordinate system
    indices = np.argsort(importances)[-2:]  # Indices of the top two features
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train.iloc[:, indices[0]], X_train.iloc[:, indices[1]], c=y_numeric, cmap='coolwarm', edgecolor='k', s=20, label='Samples')
    plt.xlabel(X_train.columns[indices[0]])
    plt.ylabel(X_train.columns[indices[1]])
    plt.title('Top Two Features by Importance')
    plt.colorbar(label='Malignancy Label')
    plt.legend()
    plt.show()

    results = {}
    for count, features_to_remove in enumerate([1, 4, 10]):
        # Remove the least important features
        features_to_drop = np.argsort(importances)[:features_to_remove]
        X_train_reduced = X_train.drop(X_train.columns[features_to_drop], axis=1)
        X_test_reduced = X_test.drop(X_test.columns[features_to_drop], axis=1)

        # Train Decision Tree with reduced feature set
        model_reduced = DecisionTreeClassifier(random_state=42)
        start_time = time.time()
        model_reduced.fit(X_train_reduced, y_train)
        training_time = time.time() - start_time

        # Evaluate the reduced model
        y_pred_reduced = model_reduced.predict(X_test_reduced)
        accuracy_reduced = accuracy_score(y_test, y_pred_reduced)
        
        # Visualize the Decision Tree
        plt.figure(figsize=(20, 10))
        plot_tree(model_reduced, filled=True, feature_names=X_train_reduced.columns, class_names=['Benign', 'Malignant'])
        plt.title(f'Decision Tree after removing {features_to_remove} least important features')
        plt.show()
        
        # Adding results to the dictionary
        results[f'Remove {features_to_remove} features'] = {
            "model": model_reduced,
            "training_time": training_time,
            "accuracy": accuracy_reduced
        }

    return results

# Example usage
feature_results = feature_importance_and_model_evaluation(train_data, test_data, 'diagnosis')
