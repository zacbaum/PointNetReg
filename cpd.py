import numpy as np
import scipy
from scipy.spatial.distance import cdist
from builtins import super

class expectation_maximization_registration(object):
    def __init__(self, X, Y, sigma2=None, max_iterations=100, tolerance=0.001, w=0, *args, **kwargs):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError("The target point cloud (X) must be at a 2D numpy array.")
        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError("The source point cloud (Y) must be a 2D numpy array.")
        if X.shape[1] != Y.shape[1]:
            raise ValueError("Both point clouds need to have the same number of dimensions.")

        self.X              = X
        self.Y              = Y
        self.sigma2         = sigma2
        (self.N, self.D)    = self.X.shape
        (self.M, _)         = self.Y.shape
        self.tolerance      = tolerance
        self.w              = w
        self.max_iterations = max_iterations
        self.iteration      = 0
        self.err            = self.tolerance + 1
        self.P              = np.zeros((self.M, self.N))
        self.Pt1            = np.zeros((self.N, ))
        self.P1             = np.zeros((self.M, ))
        self.Np             = 0

    def register(self, callback=lambda **kwargs: None):
        self.transform_point_cloud()
        if self.sigma2 is None:
            self.sigma2 = initialize_sigma2(self.X, self.TY)
        self.q = -self.err - self.N * self.D/2 * np.log(self.sigma2)
        while self.iteration < self.max_iterations and self.err > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration, 'error': self.err, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

        return self.TY, self.get_registration_parameters()

    def get_registration_parameters(self):
        raise NotImplementedError("Registration parameters should be defined in child classes.")

    def iterate(self):
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        P = np.zeros((self.M, self.N))

        for i in range(0, self.M):
            diff     = self.X - np.tile(self.TY[i, :], (self.N, 1))
            diff     = np.multiply(diff, diff)
            P[i, :]  = P[i, :] + np.sum(diff, axis=1)

        c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        c = c * self.w / (1 - self.w)
        c = c * self.M / self.N

        P = np.exp(-P / (2 * self.sigma2))
        den = np.sum(P, axis=0)
        den = np.tile(den, (self.M, 1))
        den[den==0] = np.finfo(float).eps
        den += c

        self.P   = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1  = np.sum(self.P, axis=1)
        self.Np  = np.sum(self.P1)

    def maximization(self):
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()

def gaussian_kernel(X, Y, beta=2):
    D = cdist(X, Y, 'sqeuclidean')
    return scipy.exp(-D.T / (2 * beta))     

class deformable_registration(expectation_maximization_registration):
    def __init__(self, alpha=2, beta=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha         = 2 if alpha is None else alpha
        self.beta          = 2 if alpha is None else beta
        self.W             = np.zeros((self.M, self.D))
        self.G             = gaussian_kernel(self.Y, self.Y, self.beta)

    def update_transform(self):
        A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(self.M)
        B = np.dot(self.P, self.X) - np.dot(np.diag(self.P1), self.Y)
        self.W = np.linalg.solve(A, B)

    def transform_point_cloud(self, Y=None):
        if Y is None:
            self.TY = self.Y + np.dot(self.G, self.W)
            return
        else:
            return Y + np.dot(self.G, self.W)

    def update_variance(self):
        qprev = self.sigma2

        xPx      = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.X, self.X), axis=1))
        yPy      = np.dot(np.transpose(self.P1),  np.sum(np.multiply(self.TY, self.TY), axis=1))
        trPXY    = np.sum(np.multiply(self.TY, np.dot(self.P, self.X)))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10
        self.err = np.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        return self.G, self.W

def initialize_sigma2(X, Y):
    (N, D) = X.shape
    (M, _) = Y.shape
    XX = np.reshape(X, (1, N, D))
    YY = np.reshape(Y, (M, 1, D))
    XX = np.tile(XX, (M, 1, 1))
    YY = np.tile(YY, (1, N, 1))
    diff = XX - YY
    err  = np.multiply(diff, diff)
    return np.sum(err) / (D * M * N)