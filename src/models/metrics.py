from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import numpy as np
import pandas as pd


# Function to compute regression metrics for HuggingFace Trainer
def reg_metrics(eval_pred):
    # Unpacking
    preds, labels = eval_pred

    # Dimension control
    preds = preds.squeeze()
    labels = labels.squeeze()

    # Metrics
    mse = mean_squared_error(labels, preds)
    rmse = root_mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    mape = mean_absolute_percentage_error(labels, preds) * 100
    r2 = r2_score(labels, preds)

    correct_05 = np.sum(np.abs(labels - preds) <= 0.05)
    correct_10 = np.sum(np.abs(labels - preds) <= 0.10)
    correct_15 = np.sum(np.abs(labels - preds) <= 0.15)
    n_sets = labels.shape[0]

    accuracy_05 = correct_05 / n_sets
    accuracy_10 = correct_10 / n_sets
    accuracy_15 = correct_15 / n_sets

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "accuracy_05": accuracy_05,
        "accuracy_10": accuracy_10,
        "accuracy_15": accuracy_15,
    }

    return metrics


# Function to compute classification metrics for HuggingFace Trainer
def cls_metrics(eval_pred):
    # Unpacking
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    # Dimension control
    preds = preds.squeeze()
    labels = labels.squeeze()

    # Metrics
    accuracy = accuracy_score(labels, preds)
    precision_macro = precision_score(labels, preds, average="macro")
    precision_weighted = precision_score(labels, preds, average="weighted")
    recall_macro = recall_score(labels, preds, average="macro")
    recall_weighted = recall_score(labels, preds, average="weighted")
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")

    # cm = confusion_matrix(labels, preds)
    # report_dict = classification_report(labels, preds, output_dict=True)
    # report_df = pd.DataFrame(report_dict).transpose()

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

    return metrics


# Function to evaluate regression task with ML models
def eval_with_reg_ML(train_set, test_set, model):
    # Unpacking
    X_train, y_train = train_set
    X_test, y_test = test_set

    # Model fitting
    model.fit(X_train, y_train)

    # Evaluate with test data
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)

    correct_05 = np.sum(np.abs(y_test - y_pred) <= 0.05)
    correct_10 = np.sum(np.abs(y_test - y_pred) <= 0.10)
    correct_15 = np.sum(np.abs(y_test - y_pred) <= 0.15)
    n_sets = y_test.shape[0]

    accuracy_05 = correct_05 / n_sets
    accuracy_10 = correct_10 / n_sets
    accuracy_15 = correct_15 / n_sets

    metrics = [mse, rmse, mae, mape, r2, accuracy_05, accuracy_10, accuracy_15]

    return metrics


# Function to print result (Regression Task)
def print_reg_result(metric_list, name_list):
    metrics_name = [
        "MSE",
        "RMSE",
        "MAE",
        "MAPE(%)",
        "R²",
        "±5 Acc",
        "±10 Acc",
        "±15 Acc",
    ]

    print("               ", end="")
    for name in metrics_name:
        print(f"{name:<10s}", end="")
    print()
    print("Model")
    for i in range(len(name_list)):
        print(f"{name_list[i]:<15s}", end="")
        for metric in metric_list[i]:
            print(f"{metric:<10.4f}", end="")
        print()


# Function to evaluate regression task with ML models
def eval_with_cls_ML(train_set, test_set, model):
    # Unpacking
    X_train, y_train = train_set
    X_test, y_test = test_set

    # Model fitting
    model.fit(X_train, y_train)

    # Evaluate with test data
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro")
    precision_weighted = precision_score(y_test, y_pred, average="weighted")
    recall_macro = recall_score(y_test, y_pred, average="macro")
    recall_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    cm = confusion_matrix(y_test, y_pred)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    metrics = [
        accuracy,
        precision_macro,
        precision_weighted,
        recall_macro,
        recall_weighted,
        f1_macro,
        f1_weighted,
        cm,
        report_df,
    ]

    return metrics


# Function to print result (Classification Task)
def print_cls_result(metric_list, name_list):
    metrics_name = [
        "Accuracy",
        "Precision_macro",
        "Precision_weighted",
        "Recall_macro",
        "Recall_weighted",
        "F1_macro",
        "F1_weighted",
    ]

    print("               ", end="")
    for name in metrics_name:
        print(f"{name:<20s}", end="")
    print()
    print("Model")
    for i in range(len(name_list)):
        print(f"{name_list[i]:<15s}", end="")
        for j in range(len(metric_list[i]) - 2):
            print(f"{metric_list[i][j]:<20.4f}", end="")
        print()
    print("\n")

    print()
    print("========================================================")
    for i in range(len(name_list)):
        print(f"( {name_list[i]} )")

        print("Confusion Matrix:")
        print(metric_list[i][-2])
        print()

        print("Classification Report:")
        print(metric_list[i][-1].to_string(justify="left", float_format="%.4f".__mod__))
        print("========================================================")
