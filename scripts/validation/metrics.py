"""Enhanced validation metrics for classification evaluation.

This module provides comprehensive metrics beyond basic accuracy,
including per-class metrics, ROC curves, and learning curves.

Key Features:
-------------
- Per-class precision, recall, F1 scores
- ROC curves for binary and multiclass
- AUC computation
- Learning curves for overfitting detection
- Confusion matrix improvements
- Classification reports

Example Usage:
--------------
    >>> from scripts.validation.metrics import ValidationMetrics
    >>>
    >>> metrics = ValidationMetrics()
    >>> report = metrics.compute_per_class_metrics(y_true, y_pred)
    >>> print(report)
    >>>
    >>> # Generate ROC curves
    >>> metrics.plot_roc_curves(y_true, y_proba, ['Class0', 'Class1'], 'roc.png')

Author:
-------
Nicolas Karasiak <karasiak.nicolas@gmail.com>

License:
--------
GNU General Public License v2.0 or later

"""
import importlib.util
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import sklearn
try:
    from sklearn.metrics import (
        accuracy_score,
        auc,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import learning_curve

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import matplotlib for plotting
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Try to import pandas for dataframes
PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


class ValidationMetrics:
    """Enhanced validation metrics for classification.

    Provides comprehensive metrics beyond basic accuracy, including
    per-class performance, ROC curves, and learning curves.

    Example:
    -------
    >>> metrics = ValidationMetrics()
    >>> report = metrics.compute_per_class_metrics(y_true, y_pred)
    >>> metrics.plot_roc_curves(y_true, y_proba, class_names, 'roc.png')

    """

    @staticmethod
    def compute_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute precision, recall, F1 per class.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        class_names : list, optional
            Class names for readable output

        Returns
        -------
        metrics : dict
            - 'overall': Overall metrics (accuracy, kappa, macro F1)
            - 'per_class': Per-class metrics (precision, recall, F1, support)
            - 'confusion_matrix': Confusion matrix
            - 'classification_report': Sklearn classification report

        Example
        -------
        >>> metrics = ValidationMetrics.compute_per_class_metrics(y_true, y_pred)
        >>> print(f"Overall accuracy: {metrics['overall']['accuracy']:.3f}")
        >>> print(f"Class 0 F1: {metrics['per_class'][0]['f1']:.3f}")

        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for metrics computation")

        # Overall metrics
        overall = {
            "accuracy": accuracy_score(y_true, y_pred),
            "kappa": cohen_kappa_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        }

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
        )

        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        per_class = {}

        for i, cls in enumerate(unique_classes):
            class_name = class_names[i] if class_names and i < len(class_names) else f"Class {cls}"
            per_class[class_name] = {
                "precision": float(precision[i]) if i < len(precision) else 0.0,
                "recall": float(recall[i]) if i < len(recall) else 0.0,
                "f1": float(f1[i]) if i < len(f1) else 0.0,
                "support": int(support[i]) if i < len(support) else 0,
            }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Classification report
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names if class_names else None,
            output_dict=True,
        )

        return {
            "overall": overall,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    @staticmethod
    def plot_roc_curves(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        output_path: str,
        figsize: Tuple[int, int] = (10, 8),
    ) -> Dict[int, float]:
        """Generate ROC curve plots for binary or multiclass.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_proba : np.ndarray
            Predicted probabilities, shape (n_samples, n_classes)
        class_names : list
            Class names for legend
        output_path : str
            Path to save plot
        figsize : tuple, default=(10, 8)
            Figure size

        Returns
        -------
        auc_scores : dict
            {class_id: AUC score} for each class

        Example
        -------
        >>> auc_scores = ValidationMetrics.plot_roc_curves(
        ...     y_true, y_proba, ['Water', 'Forest', 'Urban'], 'roc.png'
        ... )

        """
        if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            raise ImportError("sklearn and matplotlib required for ROC curves")

        n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 2

        plt.figure(figsize=figsize)
        auc_scores = {}

        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            auc_scores[1] = roc_auc

            plt.plot(fpr, tpr, lw=2, label=f"{class_names[1]} (AUC = {roc_auc:.3f})")

        else:
            # Multiclass classification (one-vs-rest)
            from sklearn.preprocessing import label_binarize

            y_true_bin = label_binarize(y_true, classes=range(n_classes))

            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                auc_scores[i] = roc_auc

                class_name = class_names[i] if i < len(class_names) else f"Class {i}"
                plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.3f})")

        # Diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves", fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return auc_scores

    @staticmethod
    def plot_learning_curves(
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        output_path: str,
        cv: int = 5,
        train_sizes: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> Dict[str, np.ndarray]:
        """Generate learning curves for overfitting detection.

        Learning curves show how training and validation scores change
        with dataset size. Useful for diagnosing overfitting/underfitting.

        Parameters
        ----------
        model : estimator
            Sklearn-compatible model
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels
        output_path : str
            Path to save plot
        cv : int, default=5
            Cross-validation folds
        train_sizes : np.ndarray, optional
            Training set sizes to evaluate
            Default: np.linspace(0.1, 1.0, 10)
        figsize : tuple, default=(10, 6)
            Figure size

        Returns
        -------
        curves : dict
            - 'train_sizes': Training set sizes used
            - 'train_scores_mean': Mean training scores
            - 'train_scores_std': Std of training scores
            - 'val_scores_mean': Mean validation scores
            - 'val_scores_std': Std of validation scores

        Example
        -------
        >>> curves = ValidationMetrics.plot_learning_curves(
        ...     model, X, y, 'learning_curves.png'
        ... )
        >>> print(f"Final validation score: {curves['val_scores_mean'][-1]:.3f}")

        """
        if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            raise ImportError("sklearn and matplotlib required for learning curves")

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        # Compute learning curves
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model,
            X,
            y,
            cv=cv,
            train_sizes=train_sizes,
            n_jobs=-1,
            shuffle=True,
        )

        # Compute means and stds
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        # Plot
        plt.figure(figsize=figsize)

        # Training score
        plt.plot(train_sizes_abs, train_scores_mean, "o-", color="r", label="Training score", linewidth=2)
        plt.fill_between(
            train_sizes_abs,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )

        # Validation score
        plt.plot(train_sizes_abs, val_scores_mean, "o-", color="g", label="Validation score", linewidth=2)
        plt.fill_between(
            train_sizes_abs,
            val_scores_mean - val_scores_std,
            val_scores_mean + val_scores_std,
            alpha=0.1,
            color="g",
        )

        plt.xlabel("Training Set Size", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Learning Curves", fontsize=14)
        plt.legend(loc="best", fontsize=10)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return {
            "train_sizes": train_sizes_abs.tolist(),
            "train_scores_mean": train_scores_mean.tolist(),
            "train_scores_std": train_scores_std.tolist(),
            "val_scores_mean": val_scores_mean.tolist(),
            "val_scores_std": val_scores_std.tolist(),
        }

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]],
        output_path: str,
        normalize: bool = False,
        figsize: Tuple[int, int] = (8, 6),
    ) -> np.ndarray:
        """Plot confusion matrix with nice formatting.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        class_names : list, optional
            Class names for axes
        output_path : str
            Path to save plot
        normalize : bool, default=False
            Normalize confusion matrix (show percentages)
        figsize : tuple, default=(8, 6)
            Figure size

        Returns
        -------
        cm : np.ndarray
            Confusion matrix

        """
        if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            raise ImportError("sklearn and matplotlib required for confusion matrix plot")

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=figsize)

        if MATPLOTLIB_AVAILABLE and sns:
            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f" if normalize else "d",
                cmap="Blues",
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto",
                cbar_kws={"label": "Proportion" if normalize else "Count"},
            )
        else:
            plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.colorbar(label="Proportion" if normalize else "Count")

        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=14)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return cm


def compute_multiclass_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    average: str = "macro",
) -> float:
    """Compute ROC AUC for multiclass classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    average : str, default='macro'
        Averaging strategy: 'macro', 'weighted', 'micro'

    Returns
    -------
    auc_score : float
        ROC AUC score

    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn required for ROC AUC computation")

    return roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)


def create_classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> str:
    """Create comprehensive text summary of classification results.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray, optional
        Predicted probabilities (for AUC computation)
    class_names : list, optional
        Class names

    Returns
    -------
    summary : str
        Formatted text summary

    """
    metrics = ValidationMetrics()
    results = metrics.compute_per_class_metrics(y_true, y_pred, class_names)

    summary = "=" * 60 + "\n"
    summary += "CLASSIFICATION PERFORMANCE SUMMARY\n"
    summary += "=" * 60 + "\n\n"

    # Overall metrics
    summary += "Overall Metrics:\n"
    summary += f"  Accuracy:    {results['overall']['accuracy']:.4f}\n"
    summary += f"  Kappa:       {results['overall']['kappa']:.4f}\n"
    summary += f"  Macro F1:    {results['overall']['macro_f1']:.4f}\n"
    summary += f"  Weighted F1: {results['overall']['weighted_f1']:.4f}\n"

    if y_proba is not None:
        try:
            auc = compute_multiclass_roc_auc(y_true, y_proba)
            summary += f"  ROC AUC:     {auc:.4f}\n"
        except Exception:
            summary += "  ROC AUC:     unavailable\n"

    # Per-class metrics
    summary += "\nPer-Class Metrics:\n"
    for class_name, metrics_dict in results["per_class"].items():
        summary += f"\n  {class_name}:\n"
        summary += f"    Precision: {metrics_dict['precision']:.4f}\n"
        summary += f"    Recall:    {metrics_dict['recall']:.4f}\n"
        summary += f"    F1:        {metrics_dict['f1']:.4f}\n"
        summary += f"    Support:   {metrics_dict['support']}\n"

    summary += "\n" + "=" * 60 + "\n"

    return summary
