import numpy as np
from numpy.core.fromnumeric import size
from numpy.core.numeric import indices
from numpy.lib.function_base import gradient

class MLPClassifier():
    """
    My implementation of a Neural Network Classifier.
    """

    acti_fns = ['relu', 'sigmoid', 'tanh']
    weight_inits = ['random', 'he', 'xavier']
    optimizers = ['gradient_descent','gradient_descent_with_momentum', 'NAG', 'AdaGrad', 'RMSProp', 'Adam']

    def __init__(self, layers, num_epochs, dropouts=0, learning_rate=1e-5, activation_function='relu', optimizer='gradient_descent', weight_init='random', regularization='l2', batch_size=64, **kwargs):
        """
        Initializing a new MyNeuralNetwork object

        Parameters
        ----------
        - learning_rate: Learning rate of the neural network. Default value = 1e-5.
        
        - activation_function: A string containing the name of the activation function to be
        used in the hidden layers. For the output layer use Softmax activation function. Default
        value = “relu”.
        
        - optimizer: A string containing the name of the optimizer to be used by the network.
        Default value = “gradient_descent”.
        
        - Weight_init: “random”, “he” or “xavier”: String defining type of weight initialization
        used by the network. Default value = “random”.
        
        - Regularization: A string containing the type of regularization. The accepted values
        can be “l1”, “l2”, “batch_norm”, and “layer_norm” . The default value is “l2”.
        
        - Batch_size: An integer specifying the mini batch size. By default the value is 64.
        
        - Num_epochs: An integer with a number of epochs the model should be trained for.
        
        - dropout: An integer between 0 to 1 describing the percentage of input neurons to be
        randomly masked.
        
        - **kwargs: A dictionary of additional parameters required for different optimizers.
        """

        if activation_function not in self.acti_fns:
            raise Exception('Unsupported Activation Function')

        if weight_init not in self.weight_inits:
            raise Exception('Unsupported Weight Initialization Function')

        if optimizer not in self.optimizers:
            raise Exception('Unsupported Optimizer')
        
        # class attributes init
        self.w={}
        self.n_layers = len(layers)
        self.layers = layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = optimizer

        # setting activation to be used
        if activation_function == 'relu':
            self.act_fn = self.relu
            self.act_grad = self.relu_grad
        if activation_function == 'sigmoid':
            self.act_fn = self.sigmoid
            self.act_grad = self.sigmoid_grad
        if activation_function == 'tanh':
            self.act_fn = self.tanh
            self.act_grad = self.tanh_grad
        
        # init weights
        for i in range(1,self.n_layers):
            if weight_init=='random':
                self.w[i] = self.random_init((layers[i], 1+layers[i-1]))    # one row represents one neuron, each neuron with w1...wn as n columns and b as 1 bias
            elif weight_init=='he':
                pass
            elif weight_init=='xavier':
                pass
        
        # optimizer parameters init
        if optimizer == "gradient_descent_with_momentum":
            self.update = {}
            for i in range(1,self.n_layers):
                self.update[i] = np.zeros(self.w[i].shape)
        if optimizer == "NAG":
            self.update = {}
            for i in range(1,self.n_layers):
                self.update[i] = np.zeros(self.w[i].shape)
        if optimizer == "AdaGrad":
            self.Alpha = {}
            for i in range(1,self.n_layers):
                self.Alpha[i] = np.zeros(self.w[i].shape)
        if optimizer == "RMSProp":
            self.E = {}
            for i in range(1,self.n_layers):
                self.E[i] = np.zeros(self.w[i].shape)
        if optimizer == "Adam":
            self.v = {}
            self.update = {}
            for i in range(1,self.n_layers):
                self.v[i] = np.zeros(self.w[i].shape)
                self.update[i] = np.zeros(self.w[i].shape)


# ACTIVATION FUNCTIONS

    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        X = np.clip( X, -1000, 1000)
        return (X>0)*X + 0.00001*(X<=0)

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return (X>0)*1 + 0.00001*(X<=0)

    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1.0 / (1 + np.exp(-X))

    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        sig = self.sigmoid(X)
        return sig * (1-sig)

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.tanh(X)

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1 - np.power(self.tanh(X), 2)

    def softmax(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        # print(np.unique(X))
        X = np.clip( X, -1000, 1000)
        exps=np.exp(X-X.max())
        return exps/np.sum(exps, axis=0)

    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        soft = self.softmax(X)
        return soft * (1-soft)


# INIT FUNCTIONS

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        np.random.seed(1234)
        return 0.01*np.random.normal(size=shape)    #0.01*np.random.rand(shape[0],shape[1])


# HELPER FUNCTIONS

    def CrossEntropyLoss(self, y_, ypred):
        """
        Cross Entropy Loss

        Parameters
        ----------
        y_ : 2-dimensional numpy array of shape (n_samples, n_classes) which acts as binarized true labels.

        ypred : 2-dimensional numpy array of shape (n_samples, n_classes) which acts as predicted probabilities.
        
        Returns
        -------
        cross entropy loss averaged over the entire batch supplied
        """
        ypred += (ypred==0)*0.0000001 - (ypred==1)*0.0000001
        return -np.sum( y_*np.log(ypred) + (1-y_)*np.log(1-ypred) )/y_.shape[0]


# OPTIMIZERS

    def gradient_descent(self, D, lr):
        for i in range(1,self.n_layers):
            self.w[i] -= D[i]*lr

    def gradient_descent_with_momentum(self, D, lr, beta=0.9):
        for i in range(1,self.n_layers):
            self.update[i] = beta*self.update[i] + D[i]*lr
            self.w[i] = self.w[i] - self.update[i]

    def NAG(self, D, lr, X,y, beta=0.9):
        w_look_ahead = {}
        
        for i in range(1,self.n_layers):
            w_look_ahead[i] = self.w[i] - beta*self.update[i]
            
        a,z = self.forward(X, w_look_ahead)
        D = self.backward(y, a,z)
        for i in range(1,self.n_layers):
            D[i] /= self.batch_size

        for i in range(1,self.n_layers):
            self.update[i] = beta*self.update[i] + lr*D[i]
            self.w[i] = self.w[i] - self.update[i]

    def AdaGrad(self, D, lr):
        for i in range(1,self.n_layers):
            self.Alpha[i] += (D[i])**2
            lr_ = lr/np.sqrt(self.Alpha[i] + 1e-8)
            self.w[i] = self.w[i] - lr_*D[i]

    def RMSProp(self, D, lr, gamma=0.9):
        for i in range(1,self.n_layers):
            self.E[i] = gamma*self.E[i] + (1-gamma)*(D[i])**2
            lr_ = lr/np.sqrt(self.E[i] + 1e-8)
            self.w[i] = self.w[i] - lr_*D[i]

    def Adam(self, D, lr, gamma=0.9, beta=0.9):
        for i in range(1,self.n_layers):
            self.update[i] = beta*self.v[i] + (1-beta)*D[i]
            self.v[i] = gamma*self.v[i] + (1-gamma)*(D[i])**2
            lr_ = lr/np.sqrt(self.v[i] + 1e-8)
            self.w[i] = self.w[i] - lr_*self.update[i]
            # self.update[i] = beta*self.update[i] - lr*D[i]
            # self.v[i] = gamma*self.v[i] + (1-gamma)*(D[i])**2
            # lr_ = lr/np.sqrt(self.v[i] + 1e-8)
            # self.w[i] = self.w[i] + lr_*self.update[i]


# TRAIN-TEST-EVAL FUNCTIONS

    def forward(self, X, w):
        a = {}
        z = {}
        a[1] = X

        for layer in range(1,self.n_layers):
            a[layer] = np.insert(a[layer], 0, np.ones(a[layer].shape[1]), axis=0)   # insert bias nodes
            z[layer+1] = np.dot(w[layer], a[layer])    # Z_l+1 = W_l.A_l
            if 1+layer==self.n_layers:                      # A_l+1 = act_fn( Z_l+1 )
                a[layer+1] = self.softmax(z[layer+1])
            else:
                a[layer+1] = self.act_fn(z[layer+1]) 
        return a,z
    
    def backward(self, y, a,z):
        D = {}
        delta = {}
        delta[self.n_layers] = a[self.n_layers] - y

        # backward
        for layer in range(self.n_layers-1,0,-1):
            if layer>1:
                delta[layer] = (np.dot( self.w[layer].T , delta[layer+1] )[1:,:])*self.act_grad(z[layer])  #slicing to remove extra calculated error for bias
            D[layer] = np.dot(delta[layer+1], a[layer].T)
        
        return D


    def fit(self, X, y, Xtest=None, ytest=None, save_error=False, shuffle=True):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Xtest : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data for plotting purposes.

        ytest : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels for plotting purposes.

        save_error : boolean, whether to save per iteration train and test loss.

        shuffle : boolean, whether to shuffle the batches at every iteration.
        
        Returns
        -------
        self : an instance of self
        """
        # X is n(samples) x m(features)
        # y is n(samples)
        # y_ is n(samples) x c(classes)

        if save_error:
            self.train_CE=[]
            self.test_CE=[]

        y_ = np.eye(y.max()+1)[y]
        ytest_ = None
        if ytest is not None:
            ytest_ = np.eye(ytest.max()+1)[ytest]
        
        samples = X.shape[0]
        idx=np.arange(samples)

        for epoch in range(self.num_epochs):

            print("Iteration",epoch+1)

            if(shuffle):
                np.random.shuffle(idx)
                X=X[idx]
                y=y[idx]
                y_=y_[idx]

            for batch in range(0, samples, self.batch_size):
                print("\tBatch",1+(batch//self.batch_size), end=': ')

                a,z = self.forward(X[batch:batch+self.batch_size].T, self.w)
                D = self.backward(y_[batch:batch+self.batch_size].T, a,z)
    
                for i in range(1,self.n_layers):
                    D[i] /= min(self.batch_size,samples-batch)

                if batch%10==0:
                    print("Validation Acc:",self.score(Xtest,ytest))
                    print(np.unique(self.predict(Xtest)))

                
                if self.optimizer == 'gradient_descent':
                    self.gradient_descent(D, self.learning_rate)
                if self.optimizer == 'gradient_descent_with_momentum':
                    self.gradient_descent_with_momentum(D, self.learning_rate)
                if self.optimizer == 'NAG':
                    self.NAG(D, self.learning_rate, X[batch:batch+self.batch_size].T, y_[batch:batch+self.batch_size].T)
                if self.optimizer == 'AdaGrad':
                    self.AdaGrad(D, self.learning_rate)
                if self.optimizer == 'RMSProp':
                    self.RMSProp(D, self.learning_rate)
                if self.optimizer == 'Adam':
                    self.Adam(D, self.learning_rate)
            

            if save_error:
                self.train_CE.append(self.CrossEntropyLoss(y_,self.predict_proba(X)))
                if (Xtest is not None) and (ytest is not None):
                    self.test_CE.append(self.CrossEntropyLoss(ytest_,self.predict_proba(Xtest)))
                    print("train CE:",self.train_CE[-1],"test CE:",self.test_CE[-1])

            

        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def predict_proba(self, X):
        """
        Predicting probabilities using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
            class wise prediction probabilities.

        """
        a={}
        z={}
        a[1] = X.T #column-vector

        max_layers=self.n_layers
        
        for layer in range(1,max_layers):
            a[layer] = np.insert(a[layer], 0, np.ones(a[layer].shape[1]), axis=0)   # insert bias nodes
            z[layer+1] = np.dot(self.w[layer], a[layer])    # Z_l+1 = W_l.A_l
            if 1+layer==self.n_layers:                      # A_l+1 = act_fn( Z_l+1 )
                a[layer+1] = self.softmax(z[layer+1])
            else:
                a[layer+1] = self.act_fn(z[layer+1])
        
        ypred = a[max_layers].T

        # return the numpy array ypred which contains the predicted values
        return ypred

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """
        # return the numpy array y which contains the predicted values
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

        Returns
        -------
        acc : float value specifying the accuracy of the model on the provided testing set
        """
        ypred = self.predict(X)
        # return the numpy array y which contains the predicted values
        return np.mean(ypred==y)

    def get_params(self):
        """
        Returns an array of 2d numpy arrays. This array contains the weights of the model.
        """
        weights = []
        for i in range(1,self.n_layers):
            weights.append(self.w[i])
        
        return weights