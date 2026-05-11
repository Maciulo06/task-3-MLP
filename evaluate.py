import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def model_metrics(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)
    return acc, report, matrix


def plot_confusion_matrix(matrix, class_names, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.title('Confusion matrix for the best MLP model')
    plt.tight_layout()
    plt.show()