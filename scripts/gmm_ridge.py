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

        Whatever the number of sample per class
        Input:
            n : the number of samples
            v : the number of folds
        Output: None.
        """
        step = n // v  # Compute the number of samples in each fold
        # Set the random generator to the same initial state
        np.random.seed(1)
        # Generate random sampling of the indices
        t = np.random.permutation(n)

        indices = []
        for i in range(v - 1):  # group in v fold
            indices.append(t[i * step : (i + 1) * step])
        indices.append(t[(v - 1) * step : n])

        for i in range(v):
            self.iT.append(np.asarray(indices[i]))
            indices_list = list(range(v))
            indices_list.remove(i)
            temp = np.empty(0, dtype=np.int64)
            for j in indices_list:
                temp = np.concatenate((temp, np.asarray(indices[j])))
            self.it.append(temp)

    def split_data_class(self, y, v=5):
        """Split the data into v folds with class-based splitting.

        The samples of each class are split approximately in v folds
        Input:
            n : the number of samples
            v : the number of folds
        Output: None.
        """
        # Get parameters
        C = y.max().astype("int")

        # Get the step for each class
        tc = []
        for j in range(v):
            tempit = []
            tempiT = []
            for i in range(C):
                # Get all samples for each class
                t = np.where(y == (i + 1))[0]
                nc = t.size
                stepc = nc // v  # Step size for each class
                if stepc == 0:
                    # fix_print_with_import
                    print("Not enough sample to build " + str(v) + " folds in class " + str(i))
                # Set the random generator to the same initial state
                np.random.seed(i)
                # Random sampling of indices of samples for class i
                tc = t[np.random.permutation(nc)]

                # Set testing and training samples
                if j < (v - 1):
                    start, end = j * stepc, (j + 1) * stepc
                else:
                    start, end = j * stepc, nc
                tempiT.extend(np.asarray(tc[start:end]))  # Testing
                k = list(range(v))
                k.remove(j)
                for fold_idx in k:
                    if fold_idx < (v - 1):
                        start, end = fold_idx * stepc, (fold_idx + 1) * stepc
                    else:
                        start, end = fold_idx * stepc, nc
                    tempit.extend(np.asarray(tc[start:end]))  # Training

            self.it.append(tempit)
            self.iT.append(tempiT)


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

        Inputs:
            xt: the samples to be classified
        Outputs:
            y: the class
            K: the decision value for each class.
        """
        MAX = np.finfo(np.float64).max
        # Maximum value that is possible to compute with sp.exp
        E_MAX = np.log(MAX)

        # Get information from the data
        nt = xt.shape[0]  # Number of testing samples
        C = self.ni.shape[0]  # Number of classes

        # Initialization
        K = np.empty((nt, C))

        TAU = self.tau if tau is None else tau

        for c in range(C):
            invCov, logdet = self.compute_inverse_logdet(c, TAU)
            cst = logdet - 2 * np.log(self.prop[c])  # Pre compute the constant

            xtc = xt - self.mean[c, :]
            temp = np.dot(invCov, xtc.T).T
            K[:, c] = np.sum(xtc * temp, axis=1) + cst
            del temp, xtc

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
