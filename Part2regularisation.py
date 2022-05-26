from dl_toolkit import MLPClassifier
import pickle
import numpy as np
from matplotlib import pyplot as plt



TRAIN_MODELS=False
RELU_RESULTS=True



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
    model = MLPClassifier(layers=[784, 256, 128, 64, 10], activation_function='relu', learning_rate=0.01, batch_size=64, num_epochs=20, optimizer='AdaGrad', weight_init='xavier', regularization='l1', l1=0.01)
    model = model.fit(X_train,y_train, Xtest=X_test, ytest=y_test, save_error=True)
    pickle.dump(model, open("Models/relu_adagrad_l1.pkl", "wb"))
    print("Train Accuracy:",model.score(X_train,y_train))
    print("Tests Accuracy:",model.score(X_test,y_test))

    model = MLPClassifier(layers=[784, 256, 128, 64, 10], activation_function='relu', learning_rate=0.01, batch_size=64, num_epochs=20, optimizer='AdaGrad', weight_init='xavier', regularization='l2', l2=3)
    model = model.fit(X_train,y_train, Xtest=X_test, ytest=y_test, save_error=True)
    pickle.dump(model, open("Models/relu_adagrad_l2.pkl", "wb"))
    print("Train Accuracy:",model.score(X_train,y_train))
    print("Tests Accuracy:",model.score(X_test,y_test))

    model = MLPClassifier(layers=[784, 256, 128, 64, 10], activation_function='relu', learning_rate=0.01, batch_size=64, num_epochs=20, optimizer='AdaGrad', weight_init='xavier', dropouts=0.92, regularization=None)
    model = model.fit(X_train,y_train, Xtest=X_test, ytest=y_test, save_error=True)
    pickle.dump(model, open("Models/relu_adagrad_dp.pkl", "wb"))
    print("Train Accuracy:",model.score(X_train,y_train))
    print("Tests Accuracy:",model.score(X_test,y_test))

    model = MLPClassifier(layers=[784, 256, 128, 64, 10], activation_function='relu', learning_rate=0.01, batch_size=64, num_epochs=20, optimizer='AdaGrad', weight_init='random', dropouts=0.8, regularization=None)
    model = model.fit(X_train,y_train, Xtest=X_test, ytest=y_test, save_error=True)
    pickle.dump(model, open("Models/relu_adagrad_dp_using_random.pkl", "wb"))
    print("Train Accuracy:",model.score(X_train,y_train))
    print("Tests Accuracy:",model.score(X_test,y_test))


# Results were generated using the following
if RELU_RESULTS:
    model = pickle.load(open("Models\\relu_adagrad_dp_using_random.pkl","rb"))

    print("Train Accuracy:",model.score(X_train,y_train))
    print("Tests Accuracy:",model.score(X_test,y_test))
    print("Train Loss",model.train_CE[-1])
    print("Val Loss",model.test_CE[-1])

    # LOSS PLOTS
    epochs = list(range(model.num_epochs))
    plt.clf()
    plt.plot(epochs, model.train_CE, label='train loss')
    plt.plot(epochs, model.test_CE, '--', label='test loss')
    plt.legend()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cross Entropy Loss')
    plt.suptitle("Cross Entropy Loss Vs Epochs | ReLU | AdaGrad | Random | Dropout=0.8")
    plt.savefig("Plots/relu_dp_using_random.png")
    plt.clf()

    # CF MATRIX
    from sklearn.metrics import confusion_matrix
    cf_matrix = confusion_matrix(y_test, model.predict(X_test))
    import seaborn as sns
    sns.heatmap(cf_matrix, annot=True)
    plt.savefig("Plots/cf_dp_using_random.png")
    plt.clf()

    # ROC CURVE
    from sklearn.metrics import roc_curve
    y_pred = model.predict_proba(X_test)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Base Line')
    for i in range(10):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred[:,i], pos_label=i)
        plt.plot(fpr, tpr, linewidth=2, label=f"Class {i}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve | ReLU | AdaGrad | Random | Dropout=0.8')
    plt.legend()
    plt.savefig("Plots/roc_relu_dp_using_random.png")
    plt.clf()
