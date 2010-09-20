# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Probability density functions"""

import numpy as np

class Pdf:
    """Base class for all (TODO: unconditional?) multivariate pdfs"""

    def mean(self):
        """Return mean value (a vector) of the pdf"""
        raise NotImplementedError("Derived classes must implement this function")

    def variance(self):
        """Return variance (TODO: matrix? vector?) of the pdf"""
        raise NotImplementedError("Derived classes must implement this function")

    def eval_log(self, x):
        """Returning logarithm of likelihood function in point x"""
        raise NotImplementedError("Derived classes must implement this function")

    def sample(self):
        """Return one random sample from this density"""
        raise NotImplementedError("Derived classes must implement this function")

class GaussPdf(Pdf):
    """Unconditional Gaussian (normal)probability density function

    .. math: f(x|\mu,b) \propto \exp(-(x-\mu)'R^{-1}(x-\mu))
    """

    def __init__(self, mean=np.array([1]), variance=np.array([[1]])):
        """Initialise Gaussian pdf with mean value mu and variance R

        mu % mean values
        R  % variance
        """
        mean = np.asarray(mean)
        variance = np.asarray(variance)
        if mean.ndim != 1:
            raise ValueError("mean must be one-dimensional (" + str(mean.ndim) + " encountered)")
        if variance.ndim != 2:
            raise ValueError("variance must be 2-dimensional")
        if variance.shape[0] != variance.shape[1]:
            raise ValueError("variance must be rectangular")
        if mean.shape[0] != variance.shape[0]:
            raise ValueError("mean and variance must have equal shape")
        if np.any(variance != variance.T):
            raise ValueError("variance must be symmetric (complex variance not supported)")
        # TODO: variance must be positive definite
        self.mu = mean
        self.R = variance

    def mean(self):
        return self.mu

    def dimension(self):
        return self.mu.shape[0]

    def variance(self):
        return np.diag(self.R) # TODO: inconsistent naming? (vector vs. matrix)

#    def eval_log(self, x):
#        return -log(2*self.b)-abs(x-self.mu)/self.b

    def sample(self):
        z = np.random.normal(size=self.mu.shape[0]);
        # NumPy's chol(R) is equivalent to Matlab's chol(R).transpose()
        s = self.mu + np.dot(np.linalg.cholesky(self.R), z);
        return s
