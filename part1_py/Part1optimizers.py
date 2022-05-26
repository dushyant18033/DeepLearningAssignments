from dl_toolkit import MLPClassifier
import pickle
import numpy as np
from matplotlib import pyplot as plt



TRAIN_MODELS=True
RELU_RESULTS=False



# importing dataset from openml
train = pickle.load(open("Datasets\\train_set.pkl", "rb"))
test = pickle.load(open("Datasets\\val_set.pkl", "rb"))

y_train = train['Labels']
X_train = []

for img in train['Image']:
    X_train.append(np.array(img).reshape(-1))

X_train = np.stack(X_train)


y_test = test['Labels']
X_test = []

for img in test['Image']:
    X_test.append(np.array(img).reshape(-1))

X_test = np.stack(X_test)


# for training models
if TRAIN_MODELS:
    model = MLPClassifier(layers=[784, 256, 128, 64, 10], activation_function='relu', learning_rate=0.01, batch_size=64, num_epochs=20, optimizer='gradient_descent_with_momentum')
    model = model.fit(X_train,y_train, Xtest=X_test, ytest=y_test, save_error=True)
    pickle.dump(model, open("Models/relu_momentum.pkl", "wb"))
    print("Train Accuracy:",model.score(X_train,y_train))
    print("Tests Accuracy:",model.score(X_test,y_test))

    model = MLPClassifier(layers=[784, 256, 128, 64, 10], activation_function='relu', learning_rate=0.01, batch_size=64, num_epochs=20, optimizer='NAG')
    model = model.fit(X_train,y_train, Xtest=X_test, ytest=y_test, save_error=True)
    pickle.dump(model, open("Models/relu_nag.pkl", "wb"))
    print("Train Accuracy:",model.score(X_train,y_train))
    print("Tests Accuracy:",model.score(X_test,y_test))

    model = MLPClassifier(layers=[784, 256, 128, 64, 10], activation_function='relu', learning_rate=0.01, batch_size=64, num_epochs=20, optimizer='AdaGrad')
    model = model.fit(X_train,y_train, Xtest=X_test, ytest=y_test, save_error=True)
    pickle.dump(model, open("Models/relu_adagrad.pkl", "wb"))
    print("Train Accuracy:",model.score(X_train,y_train))
    print("Tests Accuracy:",model.score(X_test,y_test))

    model = MLPClassifier(layers=[784, 256, 128, 64, 10], activation_function='relu', learning_rate=0.01, batch_size=64, num_epochs=18, optimizer='RMSProp')
    model = model.fit(X_train,y_train, Xtest=X_test, ytest=y_test, save_error=True)
    pickle.dump(model, open("Models/relu_rmsprop.pkl", "wb"))
    print("Train Accuracy:",model.score(X_train,y_train))
    print("Tests Accuracy:",model.score(X_test,y_test))

    model = MLPClassifier(layers=[784, 256, 128, 64, 10], activation_function='relu', learning_rate=0.01, batch_size=64, num_epochs=20, optimizer='Adam')
    model = model.fit(X_train,y_train, Xtest=X_test, ytest=y_test, save_error=True)
    pickle.dump(model, open("Models/relu_adam.pkl", "wb"))
    print("Train Accuracy:",model.score(X_train,y_train))
    print("Tests Accuracy:",model.score(X_test,y_test))


# Results were generated using the following
if RELU_RESULTS:
    model = pickle.load(open("Models\\relu.pkl","rb"))

    print("Train Accuracy:",model.score(X_train,y_train))
    print("Tests Accuracy:",model.score(X_test,y_test))

    # LOSS PLOTS
    epochs = list(range(model.num_epochs))
    plt.clf()
    plt.plot(epochs, model.train_CE, label='train loss')
    plt.plot(epochs, model.test_CE, '--', label='test loss')
    plt.legend()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cross Entropy Loss')
    plt.suptitle("Cross Entropy Loss Vs Epochs | ReLU")
    plt.savefig("Plots/relu.png")
    plt.clf()

    # CF MATRIX
    from sklearn.metrics import confusion_matrix
    cf_matrix = confusion_matrix(y_test, model.predict(X_test))
    print(cf_matrix)

    # ROC CURVE
    from sklearn.metrics import roc_curve
    y_pred = model.predict_proba(X_test)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Base Line')
    for i in range(10):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred[:,i], pos_label=i)
        plt.plot(fpr, tpr, linewidth=2, label=f"Class {i}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve | ReLU')
    plt.legend()
    plt.savefig("Plots/roc_relu.png")
    plt.clf()
