"""
helper.py
---------
Utility functions for training visualization, model evaluation,
and error analysis used across the MNIST classification project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix, classification_report


# ---------------------------------------------------------------------------
# 1. Plot Training History
# ---------------------------------------------------------------------------

def plot_training_history(df: pd.DataFrame, model_name: str) -> None:
    """
    Plot training and validation loss and accuracy over epochs.

    The function visualizes model performance during training using two
    side-by-side plots: loss and accuracy. It assumes the DataFrame
    comes from a Keras CSVLogger output.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the columns: 'loss', 'val_loss',
        'accuracy', and 'val_accuracy'.
    model_name : str
        Model name used in plot titles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    axes[0].plot(df['loss'], label='Train', linewidth=2)
    axes[0].plot(df['val_loss'], label='Val', linewidth=2)
    axes[0].set_title(f'{model_name} – Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(df['accuracy'], label='Train', linewidth=2)
    axes[1].plot(df['val_accuracy'], label='Val', linewidth=2)
    axes[1].set_title(f'{model_name} – Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. Evaluate Classifier
# ---------------------------------------------------------------------------

def evaluate_classifier(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    class_names: list = None
) -> dict:
    """
    Evaluate a trained Keras classifier on test data.

    Computes test loss, accuracy, weighted F1-score, and a detailed
    classification report. Predictions are obtained using argmax
    over model outputs.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained classification model.
    x_test : numpy.ndarray
        Test input samples.
    y_test : numpy.ndarray
        True labels for test data (integer-encoded).
    batch_size : int
        Batch size used during evaluation and prediction.
    class_names : list of str, optional
        Names of target classes in label order.

    Returns
    -------
    dict
        Dictionary containing loss, accuracy, F1 score,
        predicted labels, and classification report.
    """
    # 1. Loss & accuracy
    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        batch_size=batch_size,
        verbose=1
    )

    # 2. Predictions
    preds = model.predict(x_test, batch_size=batch_size)
    y_pred = np.argmax(preds, axis=1)

    # 3. F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 4. Classification report
    if class_names is None:
        class_names = [str(i) for i in np.unique(y_test)]

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4
    )

    print(report)

    return {
        "loss": test_loss,
        "accuracy": test_accuracy,
        "f1_score": f1,
        "y_pred": y_pred,
        "classification_report": report
    }


# ---------------------------------------------------------------------------
# 3. Plot Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    title: str = "Confusion Matrix",
    save_path: str = None
) -> dict:
    """
    Plot a confusion matrix and compute per-class accuracy.

    Visualizes the confusion matrix as a heatmap and reports
    accuracy for each class based on true positives over
    total samples per class.

    Parameters
    ----------
    y_true : numpy.ndarray
        True class labels.
    y_pred : numpy.ndarray
        Predicted class labels.
    class_names : list of str, optional
        Names of classes in label order.
    title : str, optional
        Title for the confusion matrix plot.
    save_path : str, optional
        File path to save the plot image.

    Returns
    -------
    dict
        Contains the confusion matrix and per-class accuracy.
    """
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    # Per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    print("Per-class accuracy:")
    for label, acc in zip(class_names, class_accuracy):
        print(f"  {label}: {acc * 100:.2f}%")

    return {
        "confusion_matrix": cm,
        "per_class_accuracy": class_accuracy
    }


# ---------------------------------------------------------------------------
# 4. Analyze Misclassifications
# ---------------------------------------------------------------------------

def analyze_misclassifications(
    x_data: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    image_shape: tuple = (28, 28),
    num_samples: int = 12,
    title: str = "Misclassified Samples",
    save_path: str = None,
    seed: int = None          # FIX: was `seed: int = SEED` — global default
                              # evaluated at definition time, breaks on import.
                              # Now defaults to None and falls back to 42 internally.
) -> dict:
    """
    Analyze and visualize misclassified predictions.

    Identifies samples where predictions differ from true labels,
    reports the overall error rate, and displays a random subset
    of misclassified images with true vs predicted labels.

    Parameters
    ----------
    x_data : numpy.ndarray
        Input samples (flattened or reshaped images).
    y_true : numpy.ndarray
        True class labels.
    y_pred : numpy.ndarray
        Predicted class labels.
    image_shape : tuple, optional
        Shape used to reshape each image for visualization.
    num_samples : int, optional
        Number of misclassified samples to display.
    title : str, optional
        Title for the visualization.
    save_path : str, optional
        File path to save the plotted figure.
    seed : int, optional
        Random seed for reproducible sampling. Defaults to 21.

    Returns
    -------
    dict
        Contains misclassified indices and error rate.
    """
    if seed is None:
        seed = 21

    mis_idx = np.where(y_pred != y_true)[0]
    total = len(y_true)
    errors = len(mis_idx)

    print(f"Total misclassifications: {errors} / {total}")
    print(f"Error rate: {errors / total * 100:.2f}%")

    if errors == 0:
        print("No misclassifications to display.")
        return {
            "misclassified_indices": mis_idx,
            "error_rate": 0.0
        }

    np.random.seed(seed)
    sample_idx = np.random.choice(mis_idx, min(num_samples, errors), replace=False)

    cols = 4
    rows = int(np.ceil(len(sample_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, sample_idx):
        img = x_data[idx].reshape(image_shape)
        ax.imshow(img, cmap='gray')
        ax.set_title(
            f"True: {y_true[idx]} | Pred: {y_pred[idx]}",
            color='red',
            fontsize=11
        )
        ax.axis('off')

    for ax in axes[len(sample_idx):]:
        ax.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return {
        "misclassified_indices": mis_idx,
        "error_rate": errors / total
    }

# ---------------------------------------------------------------------------
# 5. Plot Sample Images
# ---------------------------------------------------------------------------

def plot_sample_images(
    x_data: np.ndarray,
    y_data: np.ndarray,
    num_samples: int = 10
) -> None:
    """
    Display one sample image per class from the dataset.

    Shows a grid of grayscale images, one for each digit class,
    using the first matching sample found in the dataset.

    Parameters
    ----------
    x_data : numpy.ndarray
        Image data array of shape (N, H, W) or (N, H, W, 1).
    y_data : numpy.ndarray
        Integer class labels of shape (N,).
    num_samples : int, optional
        Number of classes to display. Defaults to 10.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(num_samples):
        idx = np.where(y_data == i)[0][0]
        axes[i].imshow(x_data[idx], cmap='gray')
        axes[i].set_title(f'Digit: {i}')
        axes[i].axis('off')

    plt.suptitle('Sample Images from Each Class', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()
