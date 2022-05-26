import numpy as np
from matplotlib import pyplot as plt

class MyPerceptron:
    """
    MyPerceptron class:
        - plot: (function) for plotting decision boundary and data points along the training process
        - fit: (function) for training a perceptron model on the training data given using the perceptron training algorithm.
        - predict: (function) for predicting class labels for input data features provided using current weight values. (throws error if called before fit)
    """

    def __init__(self, max_iter=10):
        """
        Initializes a perceptron model

        Parameters:
            max_iter (optional, default=10): maximum iterations to perform while seeking for convergence.

        Returns:
            None
        """
        super().__init__()
        self.epochs = max_iter


    def plot(self, X, y, msg="Plots"):
        """
        Takes X and y as input data and plots these data points along with the calculated line of fit using w.
        
        Works for 2D and 1D feature space only.

        Parameters:
            X : input data features, , shape = (n_samples, n_features)
            y : input data labels ( 0 or 1 ), shape = (n_samples,)
            msg : text to print on the plot as title, default value=Plots

        Returns:
            None
        """
        if X.shape[1]==2: # for 2D case
            
            # plot data points
            plt.scatter(X[y==0][:,0], X[y==0][:,1], c='red', label='negative samples')
            plt.scatter(X[y==1][:,0], X[y==1][:,1], c='blue', label='positive samples')
            
            # plot calculated line
            x1 = np.linspace(-2,2)
            x2 = -(self.w[0] + self.w[1]*x1)/self.w[2]
            plt.plot(x1,x2, '--', c='green', label='decision boundary')
            
            plt.legend()
            plt.suptitle(msg)
            plt.show()
        
        elif X.shape[1]==1: # for 1D case

            # plot data points
            plt.scatter(X[y==0][:,0], np.zeros(X[y==0].shape[0]), c='red', label='negative samples')
            plt.scatter(X[y==1][:,0], np.zeros(X[y==1].shape[0]), c='blue', label='positive samples')
            
            # plot calculated line
            x2 = np.linspace(-2,2)
            x1 = -self.w[0]/self.w[1] + 0.0*x2
            plt.plot(x1,x2, '--', c='green', label='decision boundary')
            plt.legend()
            plt.suptitle(msg)
            plt.show()

    
    def fit(self, X, y, make_plots=False, verbose=False):
        """
        Takes X and y as input data and train the perceptron model parameters.
        
        Parameters:
            X : input data features, , shape = (n_samples, n_features)
            y : input data labels ( 0 or 1 ), shape = (n_samples,)
            make_plots : whether to show the plots for intermediate updates
            verbose : set to True for viewing updates printing on the terminal.

        Returns:
            None
        """
        m,n = X.shape

        # insert column of ones
        X_ = np.insert(X,0,np.ones(m),axis=1)
        y_ = np.copy(y)
        y_[y==0] = -1

        # init model parameters
        self.w = np.zeros(n+1)

        updates = 0

        # train loop
        for epoch in range(self.epochs):
            changes_done = False

            # iterate over the data samples
            for i in range(m):

                t = y_[i]*np.dot(self.w,X_[i])
                
                # if misclassified or not exclusively classified
                if t<=0:

                    # make updates to the parameters
                    self.w+=y_[i]*X_[i]
                    updates += 1
                    changes_done = True

                    if verbose:
                        print("Step",updates,"\t","Weights:",self.w)

                    if make_plots:
                        self.plot(X,y, msg="Iteration "+str(epoch+1)+", Sample "+str(i+1)+"/"+str(m))
            
            if not changes_done: # in case of early convergence, break the loop
                break
        
        print("Total Updates:",updates)
        
    def predict(self, X):
        """
        Takes X as input features and returns predicted labels
        
        Parameters:
            X : input data features, shape = (n_samples, n_features)
            
        Returns:
            y : predicted labels, shape = (n_samples,)
        """
        return 1*(np.dot(X,self.w.reshape(-1,1))>=0)
