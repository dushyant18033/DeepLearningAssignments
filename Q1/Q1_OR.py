import numpy as np
from perceptron import MyPerceptron

from os import environ

# suppressing matplotlib warnings
environ["QT_DEVICE_PIXEL_RATIO"] = "0"
environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
environ["QT_SCREEN_SCALE_FACTORS"] = "1"
environ["QT_SCALE_FACTOR"] = "1"


# init a model object
model = MyPerceptron()

# set the train data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])

# train the model
model.fit(X,y, make_plots=True, verbose=True)

# dump model to a file
import pickle
pickle.dump(model, open("Models/OR","wb"))