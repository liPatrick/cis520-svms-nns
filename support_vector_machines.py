import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


class SVM():
    def __init__(self, kernel, C=1):
        """
        Constructor for SVM classifier
        NOTE: do NOT modify this function
        
        Args:
            kernel (function):
                The kernel function, it needs to take
                TWO vector input and returns a float

            C (float): Regularization parameter

        Returns:
            An initialized SVM object

        Example usage:
            >>> kernel = lambda x, y: numpy.sum(x * y)
            >>> C = 0.1
            >>> model = SVM(kernel, C) 

        ---
        """
        self.kernel = kernel 
        self.C = C
    
    def fit(self, X, y):
        N, d = X.shape
        #precomputed kernel multiplications of yiyjxixj
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                K[i][j] = y[i]*y[j]*self.kernel(X[i], X[j])
        
        P = cvxopt_matrix(K, tc='d')
        q = cvxopt_matrix(-1*np.ones((N,1)), tc='d') #this is because we are minimizing instead of maximizing
        G = cvxopt_matrix(np.vstack((-1*np.eye(N), np.eye(N))), tc='d')
        h = cvxopt_matrix(np.vstack((np.zeros((N,1)), self.C*np.ones((N,1)))), tc='d')
        A = cvxopt_matrix(y.reshape(1, N), tc='d')
        b = cvxopt_matrix(np.zeros(1), tc='d')

        sol = cvxopt_solvers.qp(P,q,G,h,A,b)
        sol = np.array(sol['x']).reshape(1,-1)[0]

        
        sv1 = []
        svLength = 0
        alphas = []
        ys = []
        xs = []

        for i in range(N):

            if sol[i] > (self.C * (10**(-6))):
                svLength += 1
                alphas.append(sol[i])
                ys.append(y[i])
                xs.append(X[i])
                if sol[i] < self.C*(1-10**(-6)):
                    sv1.append(i) 

        biasTot = 0
        for i in sv1:
            val = 0
            for j in range(svLength):
                 val += alphas[j]*ys[j]*self.kernel(xs[j], X[i])
            biasTot = biasTot + y[i] - val

        bias = biasTot/len(sv1)
        self.bias = bias
        self.svLength = svLength 
        self.alphas = alphas
        self.ys = ys 
        self.xs = xs
        return
    
    def predict(self, X):
        N, d = X.shape
        y_hats = []
        # Your code goes here
        for i in range(N):
            prediction = 0
            for j in range(self.svLength):
                prediction += self.alphas[j]*self.ys[j]*self.kernel(self.xs[j],X[i])
            prediction += self.bias 
            y_hats.append(np.sign(prediction))

        return np.array(y_hats)

        