"""!@brief Script by Mathieu Fauvel which performs Gaussian Mixture Model."""  # -*- coding: utf-8 -*-

import multiprocessing as mp

import numpy as np
from numpy import linalg

# Temporary predict function


def predict(tau, model, xT, yT):
    """Predict using GMM model with different tau values."""
    err = np.zeros(tau.size)
    for j, t in enumerate(tau):
        yp = model.predict(xT, tau=t)[0]
        eq = np.where(yp.ravel() == yT.ravel())[0]
        err[j] = eq.size * 100.0 / yT.size
    return err


class CV:
    """Implements the generation of several folds for cross validation."""

    def __init__(self):
        self.it = []
        self.iT = []

    def split_data(self, n, v=5):
        """Split the data into v folds.

        Parameters
        ----------
        n : int
            The number of samples.
        v : int, optional
            The number of folds (default 5).

        """
        step = n // v
        np.random.seed(1)
        t = np.random.permutation(n)

        # Create fold boundaries
        indices = []
        for i in range(v - 1):
            indices.append(t[i * step : (i + 1) * step])
        indices.append(t[(v - 1) * step : n])

        # Build train/test splits efficiently using list concatenation
        for i in range(v):
            self.iT.append(np.asarray(indices[i]))
            # Collect all other folds' indices, then concatenate once
            train_folds = [indices[j] for j in range(v) if j != i]
            self.it.append(np.concatenate(train_folds))

    def split_data_class(self, y, v=5):
        """Split the data into v folds with stratified class-based splitting.

        Parameters
        ----------
        y : np.ndarray
            Class labels for each sample.
        v : int, optional
            The number of folds (default 5).

        """
        C = y.max().astype("int")

        for j in range(v):
            train_indices = []
            test_indices = []

            for i in range(C):
                t = np.where(y == (i + 1))[0]
                nc = t.size
                stepc = nc // v
                if stepc == 0:
                    print("Not enough sample to build " + str(v) + " folds in class " + str(i))

                np.random.seed(i)
                tc = t[np.random.permutation(nc)]

                # Test indices for this fold
                if j < (v - 1):
                    start, end = j * stepc, (j + 1) * stepc
                else:
                    start, end = j * stepc, nc
                test_indices.append(tc[start:end])

                # Training indices: all other folds
                for fold_idx in range(v):
                    if fold_idx != j:
                        if fold_idx < (v - 1):
                            start, end = fold_idx * stepc, (fold_idx + 1) * stepc
                        else:
                            start, end = fold_idx * stepc, nc
                        train_indices.append(tc[start:end])

            # Concatenate all indices at once instead of repeated extend
            self.it.append(np.concatenate(train_indices))
            self.iT.append(np.concatenate(test_indices))


class GMMR:
    """Gaussian Mixture Model Ridge regression implementation."""

    def __init__(self):
        self.ni = []
        self.prop = []
        self.mean = []
        self.cov = []
        self.Q = []
        self.L = []
        self.classnum = []  # to keep right labels
        self.classes_ = []
        self.tau = 0.0

    def learn(self, x, y):
        """Learn the GMM with ridge regularization from training samples.

        Input:
            x : the training samples
            y :  the labels
        Output:
            the mean, covariance and proportion of each class, as well as the spectral decomposition of the covariance matrix.
        """
        # Get information from the data
        C = np.unique(y).shape[0]
        # C = int(y.max(0))  # Number of classes
        n = x.shape[0]  # Number of samples
        d = x.shape[1]  # Number of variables

        # Initialization
        # Vector of number of samples for each class
        self.ni = np.empty((C, 1))
        self.prop = np.empty((C, 1))  # Vector of proportion
        self.mean = np.empty((C, d))  # Vector of means
        self.cov = np.empty((C, d, d))  # Matrix of covariance
        self.Q = np.empty((C, d, d))  # Matrix of eigenvectors
        self.L = np.empty((C, d))  # Vector of eigenvalues
        self.classnum = np.empty(C).astype("uint16")
        self.classes_ = self.classnum
        # Learn the parameter of the model for each class
        for c, cR in enumerate(np.unique(y)):
            j = np.where(y == (cR))[0]

            self.classnum[c] = cR  # Save the right label
            self.ni[c] = float(j.size)
            self.prop[c] = self.ni[c] / n
            self.mean[c, :] = np.mean(x[j, :], axis=0)
            # Normalize by ni to be consistent with the update formulae
            self.cov[c, :, :] = np.cov(x[j, :], bias=1, rowvar=0)

            # Spectral decomposition
            L, Q = linalg.eigh(self.cov[c, :, :])
            idx = L.argsort()[::-1]
            self.L[c, :] = L[idx]
            self.Q[c, :, :] = Q[:, idx]

    def predict(self, xt, tau=None, confidenceMap=None):
        """Predict the label for sample xt using the learned model.

        Parameters
        ----------
        xt : np.ndarray
            The samples to be classified, shape (n_samples, n_features).
        tau : float, optional
            Regularization parameter. If None, uses self.tau.
        confidenceMap : bool, optional
            If True, returns confidence values along with predictions.

        Returns
        -------
        yp : np.ndarray
            Predicted class labels.
        K : np.ndarray, optional
            Confidence values (only if confidenceMap is True).

        """
        MAX = np.finfo(np.float64).max
        E_MAX = np.log(MAX)

        nt = xt.shape[0]  # Number of testing samples
        C = self.ni.shape[0]  # Number of classes
        d = xt.shape[1]  # Number of features

        TAU = self.tau if tau is None else tau

        # Precompute all inverse covariances, log determinants, and constants
        invCovs = np.empty((C, d, d))
        csts = np.empty(C)

        for c in range(C):
            Lr = self.L[c, :] + TAU
            temp = self.Q[c, :, :] * (1.0 / Lr)
            invCovs[c] = np.dot(temp, self.Q[c, :, :].T)
            logdet = np.sum(np.log(Lr))
            csts[c] = logdet - 2.0 * np.log(self.prop[c])

        # Vectorized discriminant computation using einsum
        # xtc shape: (nt, C, d) - centered data for each class
        xtc = xt[:, np.newaxis, :] - self.mean  # Broadcasting: (nt, 1, d) - (C, d)

        # Compute quadratic form: sum_ij(xtc_i * invCov_ij * xtc_j) for each sample and class
        # Using einsum: 'nci,cij,ncj->nc' means:
        # n=samples, c=classes, i,j=features
        K = np.einsum('nci,cij,ncj->nc', xtc, invCovs, xtc) + csts

        yp = np.argmin(K, 1)

        if confidenceMap is None:
            # Assign the label save in classnum to the minimum value of K
            yp = self.classnum[yp]

            return yp

        else:
            K *= -0.5
            K[K > E_MAX], K[K < -E_MAX] = E_MAX, -E_MAX
            np.exp(K, out=K)
            K /= K.sum(axis=1).reshape(nt, 1)
            K = K[np.arange(len(K)), yp]
            # K = sp.diag(K[:,yp])

            yp = self.classnum[yp]

            return yp, K

    def compute_inverse_logdet(self, c, tau):
        """Compute inverse covariance matrix and log determinant."""
        Lr = self.L[c, :] + tau  # Regularized eigenvalues
        temp = self.Q[c, :, :] * (1 / Lr)
        invCov = np.dot(temp, self.Q[c, :, :].T)  # Pre compute the inverse
        logdet = np.sum(np.log(Lr))  # Compute the log determinant
        return invCov, logdet

    def BIC(self, x, y, tau=None):
        """Computes the Bayesian Information Criterion of the model."""
        # Get information from the data
        C, d = self.mean.shape
        n = x.shape[0]

        # Initialization
        TAU = self.tau if tau is None else tau

        # Penalization
        P = C * (d * (d + 3) / 2) + (C - 1)
        P *= np.log(n)

        # Compute the log-likelihood
        L = 0
        for c in range(C):
            j = np.where(y == (c + 1))[0]
            xi = x[j, :]
            invCov, logdet = self.compute_inverse_logdet(c, TAU)
            cst = logdet - 2 * np.log(self.prop[c])  # Pre compute the constant
            xi -= self.mean[c, :]
            temp = np.dot(invCov, xi.T).T
            K = np.sum(xi * temp, axis=1) + cst
            L += np.sum(K)
            del K, xi

        return L + P

    def cross_validation(self, x, y, tau, v=5):
        """Compute the cross validation accuracy for the value tau of the regularization.

        Input:
            x : the training samples
            y : the labels
            tau : a range of values to be tested
            v : the number of fold
        Output:
            err : the estimated error with cross validation for all tau's value.
        """
        # Initialization
        np = tau.size  # Number of parameters to test
        cv = CV()  # Initialization of the indices for the cross validation
        cv.split_data_class(y)
        err = np.zeros(np)  # Initialization of the errors

        # Create GMM model for each fold
        model_cv = []
        for i in range(v):
            model_cv.append(GMMR())
            model_cv[i].learn(x[cv.it[i], :], y[cv.it[i]])

        # Initialization of the pool of processes
        pool = mp.Pool()
        processes = [pool.apply_async(predict, args=(tau, model_cv[i], x[cv.iT[i], :], y[cv.iT[i]])) for i in range(v)]
        pool.close()
        pool.join()
        for p in processes:
            err += p.get()
        err /= v

        # Free memory
        for model in model_cv:
            del model

        del processes, pool, model_cv

        return tau[err.argmax()], err
