"""Accuracy assessment metrics for classification evaluation.

This module provides classes for computing classification accuracy metrics
including confusion matrices, overall accuracy, Kappa coefficient, and F1 scores.

Author:
    Mathieu Fauvel

Modified by:
    Nicolas Karasiak for dzetsaka integration
"""

import numpy as np


class ConfusionMatrix:
    """Compute and store confusion matrix statistics.

    This class calculates various accuracy metrics from predicted and reference
    classifications, including confusion matrix, overall accuracy, Kappa coefficient,
    and F1 scores.

    Attributes
    ----------
    confusion_matrix : numpy.ndarray or None
        The confusion matrix as a 2D array
    OA : float or None
        Overall accuracy (0-1)
    Kappa : float or None
        Kappa coefficient (-1 to 1)
    F1mean : float or None
        Mean F1 score across all classes

    Examples
    --------
    >>> cm = ConfusionMatrix()
    >>> cm.compute_confusion_matrix(predicted_labels, reference_labels)
    >>> print(f"Overall Accuracy: {cm.OA:.3f}")
    >>> print(f"Kappa: {cm.Kappa:.3f}")

    """

    def __init__(self):
        """Initialize confusion matrix object with None values."""
        self.confusion_matrix = None
        self.OA = None
        self.Kappa = None
        self.F1mean = None

    def compute_confusion_matrix(self, yp, yr):
        """Compute confusion matrix and associated accuracy metrics.

        Parameters
        ----------
        yp : numpy.ndarray
            Predicted class labels (1-based indexing)
        yr : numpy.ndarray
            Reference/true class labels (1-based indexing)

        Notes
        -----
        This method assumes class labels start from 1 (not 0). The confusion matrix
        rows represent predicted classes and columns represent reference classes.

        The method computes:
        - Confusion matrix
        - Overall Accuracy (OA)
        - Kappa coefficient
        - Mean F1 score

        """
        # Initialization
        n = yp.size
        C = int(yr.max())
        self.confusion_matrix = np.zeros((C, C))

        # Compute confusion matrix
        for i in range(n):
            self.confusion_matrix[yp[i].astype(int) - 1, yr[i].astype(int) - 1] += 1

        # Compute overall accuracy
        self.OA = np.sum(np.diag(self.confusion_matrix)) / n

        # Compute Kappa
        nl = np.sum(self.confusion_matrix, axis=1)
        nc = np.sum(self.confusion_matrix, axis=0)
        self.Kappa = ((n**2) * self.OA - np.sum(nc * nl)) / (n**2 - np.sum(nc * nl))

        #
        try:
            nl = np.sum(self.confusion_matrix, axis=1, dtype=float)
            nc = np.sum(self.confusion_matrix, axis=0, dtype=float)
            self.F1mean = 2 * np.mean(np.divide(np.diag(self.confusion_matrix), (nl + nc)))
        except BaseException:
            self.F1mean = 0

        # TBD Variance du Kappa


class StatsFromConfusionMatrix:
    """Compute statistics from an existing confusion matrix.

    This class takes a pre-computed confusion matrix and calculates various
    accuracy metrics including overall accuracy, Kappa coefficient, and F1 scores.

    Parameters
    ----------
    confusionMatrix : numpy.ndarray
        A square confusion matrix where rows are predicted classes and
        columns are reference classes

    Attributes
    ----------
    confusionMatrix : numpy.ndarray
        The input confusion matrix
    n : int
        Total number of samples in the confusion matrix
    OA : float
        Overall accuracy (0-1)
    kappa : float
        Kappa coefficient (-1 to 1)
    F1mean : float
        Mean F1 score across all classes
    F1 : list
        F1 score for each individual class

    Examples
    --------
    >>> cm = np.array([[50, 3, 2], [4, 45, 1], [1, 2, 47]])
    >>> stats = StatsFromConfusionMatrix(cm)
    >>> print(f"Overall Accuracy: {stats.OA:.3f}")
    >>> print(f"Kappa: {stats.kappa:.3f}")

    """

    def __init__(self, confusionMatrix):
        """Initialize with confusion matrix and compute all statistics."""
        self.confusionMatrix = confusionMatrix
        self.n = np.sum(self.confusionMatrix)
        self.OA = self.__get_OA()
        self.kappa = self.__get_kappa()
        self.F1mean = self.__get_F1Mean()
        self.F1 = self.__get_F1()

    def __get_OA(self):
        """Compute overall accuracy.

        Returns
        -------
        float
            Overall accuracy as the ratio of correctly classified samples
            to total samples (0-1)

        """
        return np.sum(np.diag(self.confusionMatrix)) / float(self.n)

    def __get_kappa(self):
        """Compute Cohen's Kappa coefficient.

        Cohen's Kappa measures inter-rater agreement for categorical items.
        It is generally thought to be a more robust measure than simple
        percent agreement calculation, as it takes into account the possibility
        of the agreement occurring by chance.

        Returns
        -------
        float
            Kappa coefficient ranging from -1 (perfect disagreement) to 1
            (perfect agreement). 0 indicates agreement equivalent to chance.

        """
        nl = np.sum(self.confusionMatrix, axis=1)
        nc = np.sum(self.confusionMatrix, axis=0)
        OA = np.sum(np.diag(self.confusionMatrix)) / float(self.n)
        return ((self.n**2) * OA - np.sum(nc * nl)) / (self.n**2 - np.sum(nc * nl))

    def __get_F1Mean(self):
        """Compute mean F1 score across all classes.

        F1 score is the harmonic mean of precision and recall. This method
        computes the F1 score for each class and returns the mean.

        Returns
        -------
        float
            Mean F1 score across all classes (0-1)

        """
        nl = np.sum(self.confusionMatrix, axis=1, dtype=float)
        nc = np.sum(self.confusionMatrix, axis=0, dtype=float)
        return 2 * np.mean(np.divide(np.diag(self.confusionMatrix), (nl + nc)))

    def __get_F1(self):
        """Compute F1 score for each individual class.

        F1 score is the harmonic mean of precision and recall, calculated as:
        F1 = 2 * (precision * recall) / (precision + recall)
        or equivalently: F1 = 2 * TP / (2 * TP + FP + FN)

        Returns
        -------
        list
            F1 score for each class (0-1)

        """
        f1 = []
        for label in range(self.confusionMatrix.shape[0]):
            # True Positives: diagonal element
            TP = self.confusionMatrix[label, label]
            # False Negatives: sum of column minus diagonal
            FN = np.sum(self.confusionMatrix[:, label]) - self.confusionMatrix[label, label]
            # False Positives: sum of row minus diagonal
            FP = np.sum(self.confusionMatrix[label, :]) - self.confusionMatrix[label, label]

            # Calculate F1 score: 2*TP / (2*TP + FP + FN)
            f1.append(2 * TP / (2 * TP + FP + FN))
        return f1
