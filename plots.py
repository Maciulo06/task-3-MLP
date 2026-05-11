import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_class_distribution(class_counts, class_names):
    x = np.arange(len(class_names))
    y = [class_counts.get(i, 0) for i in range(len(class_names))]
    plt.figure(figsize=(10, 5))
    plt.bar(x, y, color='steelblue')
    plt.xticks(x, class_names, rotation=45)
    plt.ylabel('Number of images')
    plt.title('Class distribution in CIFAR-10 training set')
    plt.tight_layout()
    plt.show()


def show_sample_images(dataset, class_names, n=10):
    plt.figure(figsize=(15, 3))
    for i in range(n):
        image, label = dataset[i]
        image = image.permute(1, 2, 0).numpy()
        plt.subplot(2, 5, i + 1)
        plt.imshow(image)
        plt.title(class_names[label])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_series_histories(results_df, metric, title):
    plt.figure(figsize=(9, 5))
    for _, row in results_df.iterrows():
        plt.plot(row[metric], label=str(row['label']))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_misclassified(images, true_labels, pred_labels, class_names, max_images=12):
    count = min(len(images), max_images)
    cols = 4
    rows = math.ceil(count / cols)
    plt.figure(figsize=(12, 3 * rows))
    for i in range(count):
        image = images[i].transpose(1, 2, 0)
        image = np.clip(image, 0, 1)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.title(f'True: {class_names[true_labels[i]]}\nPred: {class_names[pred_labels[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
