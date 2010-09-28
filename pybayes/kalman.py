# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Kalman filter"""

from numpywrap import *

from pybayes.pdfs import GaussPdf


class Kalman(object):
    """Kalman filter"""

    def __init__(self, A, B, C, D, Q, R, state_pdf):
        """TODO: documentation"""

        # check type of pdf
        if not isinstance(state_pdf, GaussPdf):
            raise TypeError("state_pdf must be (a subclass of) GaussPdf")

        # check type of input arrays
        matrices = {"A":A, "B":B, "C":C, "D":D, "Q":Q, "R":R}
        for name in matrices:
            matrix = matrices[name]
            if type(matrix) != ndarray:
                raise TypeError(name + " must be (exactly) numpy.ndarray, " +
                                str(type(matrix)) + " given")
            if matrix.ndim != 2:
                raise ValueError(name + " must have 2 dimensions (thus forming a matrix), " +
                                 str(matrix.ndim) + " given")

        # remember vector shapes
        self.n = state_pdf.shape()[0]  # dimension of state vector
        self.k = B.shape[1]  # dimension of control vector
        self.j = C.shape[0]  # dimension of observation vector

        # dict of required matrice shapes (sizes)
        shapes = {
            "A":(self.n, self.n),
            "B":(self.n, self.k),
            "C":(self.j, self.n),
            "D":(self.j, self.k),
            "Q":(self.n, self.n),
            "R":(self.j, self.j)
        }
        # check input matrix sizes
        for name in matrices:
            matrix = matrices[name]
            if matrix.shape != shapes[name]:
                raise ValueError("Given shapes of state_pdf, B and C, matrix " + name +
                                 " must have shape " + str(shapes[name]) + ", " +
                                 str(matrix.shape) + " given")

        self.A, self.B, self.C, self.D, self.Q, self.R = A, B, C, D, Q, R

        self.P = state_pdf
        self.S = GaussPdf()  # observation probability density function

        self._bayes_type_check = True  # whether to check arguments in bayes() method


    def bayes(self, yt, ut):
        """Approximate Bayes rule"""
        if self._bayes_type_check:
            if type(yt) != ndarray or type(ut) != ndarray:
                raise TypeError("Both yt and ut must be numpy.ndarray. " +
                                str(type(yt)) + " and " + str(type(ut)) + " given")
            if yt.ndim != 1 or yt.shape[0] != self.j:
                raise ValueError("yt must have shape " + str((self.j,)) + ". (" +
                                str(yt.shape[0]) + ",) given")  # TODO
            if ut.ndim != 1 or ut.shape[0] != self.k:
                raise ValueError("yt must have shape " + str((self.k,)) + ". (" +
                                str(ut.shape[0]) + ",) given")  # TODO
        else:
            self._bayes_type_check = False  # for performance reasons check only first time

        # predict
        self.P.mu = dot(self.A, self.P.mu) + dot(self.B, ut)  # a priori estimate
        self.P.R  = dot(dot(self.A, self.P.R), self.A.T) + self.Q  # a priori variance

        # data update
        self.S.mu = dot(self.C, self.P.mu) + dot(self.D, ut)
        self.S.R = dot(dot(self.C, self.P.R), self.C.T) + self.R

        # kalman gain
        K = dot(dot(self.P.R, self.C.T), inv(self.S.R))

        self.P.mu = self.P.mu + dot(K, (yt - self.S.mu))  # a posteriori estimate
        self.P.R = self.P.R - dot(dot(K, self.C), self.P.R)  # a posteriori variance

        return self.P.mu
