import numpy as np

"""Some helper functions"""

def sigmoid(g):
    return 1./(1. + np.exp(-g))

def sigmoid_derivative(g):
    return sigmoid(g) * (1 - sigmoid(g))

def relu(g):
    return np.maximum(0, g)

def relu_derivative(g):
    derivatives = np.heaviside(g, 0)
    return derivatives


class NeuralNetworkBase():
    def __init__(self, input_dim, num_hidden=10, activation="sigmoid", W1=None, b1=None, W2=None, b2=None):
        """ Base Neural Network class
        You are to implement NeuralNetworkClassification which
        inherit from NeuralNetworkBase.

        NOTE: do NOT modify this class

        Args:
            input_dim (int): Number of input features

            num_hidden (int): Hidden dimension (number of hidden nodes)
                Default value is 10

            activation (str): Activation function name. Default is "sigmoid"

            W1 (numpy.array): 
                Initial value for the weight matrix of the 1st layer
                W1 should have shape (num_hidden, input_dim)
                Default is None

            b1 (numpy.array): 
                Initial value for the bias of the 1st layer
                b1 should have shape (num_hidden, )
                Default is None

            W2 (numpy.array): 
                Initial value for the weight matrix of the 2nd layer
                W2 should have shape (1, num_hidden)
                Default is None

            b2 (numpy.float): 
                Initial value for the bias of the 1st layer
                b2 should be a scalar
                Default is None

        Example usage:
            >>> init_param = utils.load_initial_weights("P3/Synthetic-Dataset/InitParams/relu/5")
            >>> nnet__classification = NeuralNetworkClassification(input_dim=10, num_hidden=20)
            >>> nnet__classification.fit(X_train, y_train, step_size=0.01)

        """
        self.d  = input_dim # Remember the input dimension and hidden dimension
        self.d1 = num_hidden
        
        assert(activation in ["sigmoid", "relu"])
        self.g = sigmoid if activation == "sigmoid" else relu
        self.g_prime = sigmoid_derivative if activation == "sigmoid" else relu_derivative
        
        # If the parameters are not specified, initialize them randomly
        self.layers = {
            "W1": W1 if W1 is not None else np.random.rand(self.d1, self.d),
            "b1": b1 if b1 is not None else np.random.rand(self.d1),
            "W2": W2 if W2 is not None else np.random.rand(1, self.d1),
            "b2": b2 if b2 is not None else np.random.rand(), 
        }
        
        # NOTE the shape of the vectors
        assert(self.layers["b1"].shape == (self.d1,))
        assert(self.layers["W1"].shape == (self.d1, self.d))
        assert(self.layers["W2"].shape == (1, self.d1))
    
    def fit(self, X, y, num_iters=8000, step_size=0.1):
        """ Train the neural network on given data via gradient descent

        Args:
            X (numpy.array):
                Input data matrix. X should have shape (num_instances, input_dim)
            
            y (numpy.array):
                Input label vector. y should have shape (num_instances,)

        Returns:
            None

        """
        def mapTo01(value):
            value = (value+1)/2
            return value
        #convert y -> y_hat, basically map {-1,+1} -> {0,1}
        y_hat = mapTo01(y)

        print("fitting...")
        for it in range(num_iters):
            if it == 4000: 
                print(it)
            derivatives = self.back_propagate(X, y_hat)
            max_deriv_norm = 0.

            for parameter, value in derivatives.items():
                self.layers[parameter] = self.layers[parameter] - step_size * value
                max_deriv_norm = max(max_deriv_norm, np.sum(value ** 2))

    def back_propagate(self, X, y):
        """ Compute the derivative of the loss function with respect to
        the parameters of the neural networks

        Args:
            X (numpy.array):
                Input data matrix. X should have shape (num_instances, input_dim)
            
            y (numpy.array):
                Input label vector. y should have shape (num_instances,)

        Returns:
            derivative (dictionary):
                A dictionary that contains the derivative of the loss function
                with respect to the parameters of the neural networks.
                For example,
                derivative["W1"] should contain a numpy array that is the derivative
                of the loss function with respect to W1. It should have the same shape
                as W1

        """
        raise NotImplementedError

    def predict(self, X):
        """ Do prediction on given input data
        Arg:
            X (numpy.array):
                Input data matrix. X should have shape (num_instances, input_dim)
        
        Returns:
            y (numpy.array):
                Label vector. y should have shape (num_instances,)

        """
        raise NotImplementedError


class NeuralNetworkClassification(NeuralNetworkBase):
    def __init__(self, input_dim, num_hidden=10, activation="sigmoid", W1=None, b1=None, W2=None, b2=None):
        """ Neural network for classification.
        This simply calls the constructor for NeuralNetworkBase

        NOTE: do NOT modify this constructor
        NOTE: do NOT modify the signature of any class functions

        Args:
            input_dim (int): Number of input features -> k (dimensions of features)

            num_hidden (int): Hidden dimension (number of hidden nodes) -> j (number of units in hidden layer)
                Default value is 10

            activation (str): Activation function name. Default is "sigmoid" -> g

            W1 (numpy.array): -> first layer shape: (|j|, |k|)
                Initial value for the weight matrix of the 1st layer
                W1 should have shape (num_hidden, input_dim)
                Default is None

            b1 (numpy.array): -> shape: (|j|)
                Initial value for the bias of the 1st layer
                b1 should have shape (num_hidden, )
                Default is None

            W2 (numpy.array): shape: (1, |j|)
                Initial value for the weight matrix of the 2nd layer
                W2 should have shape (1, num_hidden)
                Default is None

            b2 (numpy.float): (1)
                Initial value for the bias of the 1st layer
                b2 should be a scalar
                Default is None

        Example usage:
            >>> init_param = utils.load_initial_weights("P3/Synthetic-Dataset/InitParams/relu/5")
            >>> nnet_classification = NeuralNetworkClassification(input_dim=10, num_hidden=5)
            >>> nnet_classification.fit(X_train, y_train, step_size=0.01)

        """
        super(NeuralNetworkClassification, self).__init__(
            input_dim, num_hidden, activation, W1, b1, W2, b2
        )

    def back_propagate(self, X, y):
        """ Perform back propagation

        Args:
            X (numpy.array):
                Input data matrix. X should have shape (num_instances, input_dim)
            
            y (numpy.array):
                Binary {-1, +1} label vector. y should have shape (num_instances,)

        Returns:
            derivative (dictionary):
                A dictionary that contains the derivative of the loss function
                with respect to the parameters of the neural networks.
                For example,
                derivative["W1"] should contain a numpy array that is the derivative
                of the loss function with respect to W1. It should have the same shape
                as W1
        """
        
        
        
        N, d = X.shape
        m = N # i -> (1...m)


        #|j| = self.d1, |k| = self.d

                               # X is m,d1
        w1 = self.layers["W1"] # d1xd
        b1 = self.layers["b1"]
        w2 = self.layers["W2"] # 1xd1
        b2 = self.layers["b2"]
        # ------  w1 derivatives --------
        #calculating z
        wTimesx = np.matmul(w1,X.T) 
        bExtended = b1.reshape(-1,1)
        z = wTimesx + bExtended 
        a = self.g(z) 

        #calculating gfwb
        multaWithw2 = np.matmul(w2,a)
        fwb = multaWithw2+b2 
        gfwb = self.g(fwb) #1xm


        #calculating loss (1)
        loss = gfwb-y 

        gprimez = self.g_prime(z) #d1xm
        w2gz = w2.T * gprimez #mxd1


        #------ tot derivative w1 --------
        lossw2gz = loss*w2gz #m x d1
        w1Ds = (1.0/m) * np.matmul(lossw2gz, X)

        
        #------ b1 derivatives -------
        b1Ds = (1.0/m) * np.dot(loss,w2gz.T)

        #----- w2 derivatives -------
        w2Ds = (1.0/m) * np.matmul(loss, a.T)

        #------ b2 derivatives 
        b2D = (1.0/m) * np.dot(loss, np.ones(m))


        w1_deriv = w1Ds 
        b1_deriv = b1Ds 
        w2_deriv = w2Ds 
        b2_deriv = b2D

        
        '''
        # Your code should go here
        N, d = X.shape
        m = N # i -> (1...m)


        #|j| = self.d1, |k| = self.d

                               # X is m,d1
        w1 = self.layers["W1"] # d1xd
        b1 = self.layers["b1"]
        w2 = self.layers["W2"] # 1xd1
        b2 = self.layers["b2"]

        #calculating z
        wTimesx = np.matmul(w1,X.T) #d1xm
        bExtended = np.transpose(np.tile(b1, (m,1))) #d1xm
        z = wTimesx + bExtended 
        a = self.g(z) #d1 x m

        #calculating gfwb
        multaWithw2 = np.matmul(w2,a)
        fwb = multaWithw2+b2 
        gfwb = self.g(fwb) #1xm


        #calculating loss (1)
        loss = np.subtract(gfwb.T, y.reshape(m,1)) #mx1
        lossSquash = loss.sum(axis=0)

        #calculate w2 element wise mult g(z) (2)
        gprimez = self.g_prime(z) #d1xm
        gprimzSquash = gprimez.T.sum(axis=0) #1xd1
        w2gz = (w2 * gprimzSquash).T #d1x1

        #tot derivative 
        squashX = np.reshape(X.sum(axis=0), (1,self.d))
        w1Ds = np.matmul(lossSquash*w2gz, squashX)/m
        #print("w1Ds shape", w1Ds.shape)
    
        
        #------ b1 derivatives -------
        b1Ds = lossSquash*w2gz
        #print("b1ds shape", b1Ds.shape)

        #----- w2 derivatives -------
        aSquash = np.reshape(a.T.sum(axis=0), (1,self.d1))
        w2Ds = lossSquash*aSquash/m
        #print("w2Ds shape", w2Ds.shape)

        #------ b2 derivatives 
        b2D = lossSquash
        #print("b2d shape,", b2D.shape)


        w1_deriv = w1Ds 
        b1_deriv = b1Ds 
        w2_deriv = w2Ds 
        b2_deriv = b2D
        '''
        return {
                # the keys here are selected to match those in self.layers
                # (initialized on line 70)
                "W1": w1_deriv,
                "b1": b1_deriv,
                "W2": w2_deriv,
                "b2": b2_deriv
                }        

        """
        #w1 derivatives
        for j in range(len(w1Ds)):
            for k in range(len(w1Ds[0])):
                sum = 0
                for i in range(m):
                    z = np.matmul(self.layers["W1"],(X[i]))+self.layers["b1"] #this is a |j|x1 matrix
                    a = self.applyFunctionOverRowValues(self.vectorizedg, z)
                    f_wbx = self.layers["W2"].dot(a)+self.layers["b2"]
                    n = self.g(f_wbx)

                    sum += (n-y[i])*self.layers["W2"][0][j]*self.g_prime(z[j])*X[i][k]
                derivative = sum/m 
                w1Ds[j][k] = derivative

        #b1 derivatives
        for j in range(len(b1Ds)):
            sum = 0
            for i in range(m):
                z = np.matmul(self.layers["W1"],(X[i]))+self.layers["b1"] #this is a |j|x1 matrix
                a = self.applyFunctionOverRowValues(self.vectorizedg, z)
                f_wbx = self.layers["W2"].dot(a)+self.layers["b2"]
                n = self.g(f_wbx)

                sum += (n-y[i])*self.layers["W2"][0][j]*self.g_prime(z[j])
            derivative = sum/m 
            b1Ds[j] = derivative


        #w2 derivatives 
        for j in range(len(w2Ds[0])):
            sum = 0
            for i in range(m):
                z = np.matmul(self.layers["W1"],(X[i]))+self.layers["b1"] #this is a |j|x1 matrix
                a = self.applyFunctionOverRowValues(self.vectorizedg, z)
                f_wbx = self.layers["W2"].dot(a)+self.layers["b2"]
                n = self.g(f_wbx)

                sum += (n-y[i])*a[j]

            derivative = sum/m 
            w2Ds[0][j] = derivative

        #b2 derivative 
        sum = 0
        for i in range(m):
            z = np.matmul(self.layers["W1"],(X[i]))+self.layers["b1"] #this is a |j|x1 matrix
            a = self.applyFunctionOverRowValues(self.vectorizedg, z)
            f_wbx = self.layers["W2"].dot(a)+self.layers["b2"]
            n = self.g(f_wbx)

            sum += (n-y[i])

        derivative = sum/m 
        b2D = derivative





        """

    
    def predict(self, X):
        """ Do prediction on given input data
        Arg:
            X (numpy.array):
                Input data matrix. X should have shape (num_instances, input_dim)
        
        Returns:
            y (numpy.array):
                Binary {-1, +1} label vector. y should have shape (num_instances,)

        """
        print("predicting...")
        # Your code should go here
        N, d = X.shape
        m=N

        w1 = self.layers["W1"]
        b1 = self.layers["b1"]
        w2 = self.layers["W2"]
        b2 = self.layers["b2"]

        #calculating z
        wTimesx = np.matmul(w1,X.T)
        #bExtended = (np.tile(b1, (m,1))).T
        bExtended = b1.reshape(-1,1)

        z = wTimesx + bExtended 
        a = self.g(z)

        #calculating gfwb
        multaWithw2 = np.matmul(w2,a)
        fwb = multaWithw2+b2 
        gfwb = self.g(fwb)
        #gfwb = gfwb[0]
        #print("gfwb shape", gfwb.shape)

        

        return np.sign(gfwb-.5).reshape(-1)

