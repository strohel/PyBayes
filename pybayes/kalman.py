# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Kalman filter"""

from numpy import dot, asarray
from numpy.linalg import inv

from pybayes.pdfs import GaussPdf

class Kalman:
    """Kalman filter"""

    def __init__(self, A, B, C, D, Q, R, state_pdf):
        if not isinstance(state_pdf, GaussPdf):
            raise TypeException("state_pdf must be (a subclass of) GaussPdf")

        self.A = asarray(A)
        self.B = asarray(B)
        self.C = asarray(C)
        self.D = asarray(D)
        self.Q = asarray(Q)
        self.R = asarray(R)

        if False:
            print
            dict = {"A":A, "B":B, "C":C, "D":D, "Q":Q, "R":R}
            for key in dict:
                print key + ":"
                print repr(dict[key])


        self.P = state_pdf
        self.S = GaussPdf()


    def bayes(self, yt, ut):
        """Aproximate Bayes rule"""
        yt = asarray(yt)
        ut = asarray(ut)

        # predict
        self.P.mu = dot(self.A, self.P.mu) + dot(self.B, ut)  # a priori estimate
        self.P.R  = dot(dot(self.A, self.P.R), self.A.T) + self.Q  # a priori variance

        # data update
        self.S.mu = dot(self.C, self.P.mu) + dot(self.D, ut)
        self.S.R = dot(dot(self.C, self.P.R), self.C.T) + self.R

        K = dot(dot(self.P.R, self.C.T), inv(self.S.R))

        self.P.mu = self.P.mu + dot(K, (yt - self.S.mu))  # a posteriori estimate
        self.P.R = self.P.R - dot(dot(K, self.C), self.P.R)  # a posteriori variance

        return self.P.mu
