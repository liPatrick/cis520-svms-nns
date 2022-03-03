import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import utils
import time

from support_vector_machines import SVM


# Helper functions to draw decision boundary plot
def plot_contours(clf, X, y, n=100):
    """
    Produce classification decision boundary

    Args:
        clf:
            Any classifier object that predicts {-1, +1} labels
        
        X (numpy.array):
            A 2d feature matrix

        y (numpy.array):
            A {-1, +1} label vector

        n (int)
            Number of points to partition the meshgrids
            Default = 100.

    Returns:
        (fig, ax)
            fig is the figure handle
            ax is the single axis in the figure

        One can use fig to save the figure.
        Or ax to modify the title/axis label etc

    """
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    X0, X1 = X[:, 0], X[:, 1]

    # Set-up grid for plotting.
    xx, yy = np.meshgrid(np.linspace(X0.min()-1, X0.max()+1, n),\
                         np.linspace(X1.min()-1, X1.max()+1, n),\
                        )
    # Do prediction for every single point on the mesh grid
    # This will take a few seconds
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=ListedColormap(["cyan", "pink"]))

    # Scatter the -1 points
    ax.scatter([X0[i] for i,v in enumerate(y) if v == -1],
                        [X1[i] for i,v in enumerate(y) if v == -1], 
                        c="blue", label='- 1',
                        marker='x')
    # Scatter the +1 points
    ax.scatter([X0[i] for i,v in enumerate(y) if v == 1],
                        [X1[i] for i,v in enumerate(y) if v == 1], 
                        edgecolor="red", label='+1', facecolors='none', s=10,
                        marker='o')

    ax.set_ylabel('x_2')
    ax.set_xlabel('x_1')
    ax.legend()
    return fig, ax


# Your code starts here.


#rbf kernels

(x_train, y_train, x_test, y_test) = utils.load_all_train_test_data("./P1/Spam-Dataset")
folds = utils.load_all_cross_validation_data("P1/Spam-Dataset/CrossValidation")

Cs = np.logspace(-3,2,6)
lambdas = np.logspace(-2,2,5)

train_errors = [] 
test_errors = []
totalCVErrors = []
minCs = []

for q in lambdas: 
    kernel = lambda x, y: np.exp(-1*q*(np.linalg.norm(x-y)**2))
    
    #selecting min C via cross validation 
    minC = 0
    minError = 100
    for i in range(len(Cs)):
        C = Cs[i]
        totalError = 0
        for index, fold in enumerate(folds): 
            test_data, train_data = utils.partition_cross_validation_fold(folds, index)
            test_x, test_y = test_data
            train_x, train_y = train_data
            svm = SVM(kernel, C)
            svm.fit(train_x, train_y)
            test_prediction = svm.predict(test_x)
            test_error = utils.classification_error(test_prediction, test_y)
            totalError += test_error

        avgError = totalError/len(folds)
        if avgError < minError: 
            minError = avgError 
            minC = C 
    
    del svm
    minCs.append(minC)
    #calculate train errors and test errors 

    svm = SVM(kernel, minC)
    svm.fit(x_train, y_train)
    train_predictions = svm.predict(x_train)
    test_predictions = svm.predict(x_test)
    train_errors.append(utils.classification_error(train_predictions, y_train))
    test_errors.append(utils.classification_error(test_predictions, y_test))

    #calculate cv errors using C
    totalError = 0
    for index, fold in enumerate(folds): 
        test_data, train_data = utils.partition_cross_validation_fold(folds, index)
        test_x, test_y = test_data
        train_x, train_y = train_data
        svm = SVM(kernel, minC)
        svm.fit(train_x, train_y)
        test_prediction = svm.predict(test_x)
        test_error = utils.classification_error(test_prediction, test_y)
        totalError += test_error

    totalCVErrors.append(totalError/len(folds))  


print("train errors: ", train_errors)
print("test errors: ", test_errors)
print("total cv errors, ", totalCVErrors)
print("min cs per q:", minCs)

plt.plot(lambdas, train_errors, label="full training error")
plt.plot(lambdas, test_errors, label="full test error")
plt.plot(lambdas, totalCVErrors, label="cross validation error")
plt.xlabel("lambda")
plt.ylabel("error")
plt.title("error vs lambdas")
plt.xscale("log")
plt.legend(loc="upper left")
'''
kernel = lambda x, y: np.exp(-1*(0.1)*(np.linalg.norm(x-y)**2))
svm = SVM(kernel, 0.1)
svm.fit(x_train, y_train)
fig, ax = plot_contours(svm, x_train, y_train)
'''

plt.show()



'''


#polynomial kernels
(x_train, y_train, x_test, y_test) = utils.load_all_train_test_data("./P1/Spam-Dataset")
folds = utils.load_all_cross_validation_data("P1/Spam-Dataset/CrossValidation")

Cs = np.logspace(-4,2,7)
qs = [1,2,3,4,5]

train_errors = []
test_errors = []
totalCVErrors = []
minCs = []

for q in qs: 
    kernel = lambda x, y: (np.transpose(x).dot(y)+1)**q
    
    #selecting min C via cross validation 
    minC = 0
    minError = 100
    for i in range(len(Cs)):
        C = Cs[i]
        totalError = 0
        for index, fold in enumerate(folds): 
            test_data, train_data = utils.partition_cross_validation_fold(folds, index)
            test_x, test_y = test_data
            train_x, train_y = train_data
            svm = SVM(kernel, C)
            svm.fit(train_x, train_y)
            test_prediction = svm.predict(test_x)
            test_error = utils.classification_error(test_prediction, test_y)
            totalError += test_error

        avgError = totalError/len(folds)
        if avgError < minError: 
            minError = avgError 
            minC = C 
    
    del svm
    minCs.append(minC)
    #calculate train errors and test errors 

    svm = SVM(kernel, minC)
    svm.fit(x_train, y_train)
    train_predictions = svm.predict(x_train)
    test_predictions = svm.predict(x_test)
    train_errors.append(utils.classification_error(train_predictions, y_train))
    test_errors.append(utils.classification_error(test_predictions, y_test))

    #calculate cv errors using C
    totalError = 0
    for index, fold in enumerate(folds): 
        test_data, train_data = utils.partition_cross_validation_fold(folds, index)
        test_x, test_y = test_data
        train_x, train_y = train_data
        svm = SVM(kernel, minC)
        svm.fit(train_x, train_y)
        test_prediction = svm.predict(test_x)
        test_error = utils.classification_error(test_prediction, test_y)
        totalError += test_error

    totalCVErrors.append(totalError/len(folds))  


print("train errors: ", train_errors)
print("test errors: ", test_errors)
print("total cv errors, ", totalCVErrors)
print("min cs per q:", minCs)

plt.plot(qs, train_errors, label="full training error")
plt.plot(qs, test_errors, label="full test error")
plt.plot(qs, totalCVErrors, label="cross validation error")
plt.xlabel("q")
plt.ylabel("error")
plt.title("error vs q")
#plt.xscale("log")
plt.legend(loc="upper left")

plt.show()
'''

'''


#linear svm 
Cs = np.logspace(-4,2,7)
kernel = lambda x, y: np.sum(x*y)
(x_train, y_train, x_test, y_test) = utils.load_all_train_test_data("./P1/Spam-Dataset")
folds = utils.load_all_cross_validation_data("P1/Spam-Dataset/CrossValidation")

train_errors = [] 
test_errors = [] 

for i in range(len(Cs)):
    C = Cs[i]
    svm = SVM(kernel, C)
    svm.fit(x_train, y_train)
    train_predictions = svm.predict(x_train)
    test_predictions = svm.predict(x_test)
    train_errors.append(utils.classification_error(train_predictions, y_train))
    test_errors.append(utils.classification_error(test_predictions, y_test))

    del svm

totalCVErrors = []

for i in range(len(Cs)):
    C = Cs[i]

    totalError = 0
    for index, fold in enumerate(folds): 
        test_data, train_data = utils.partition_cross_validation_fold(folds, index)
        test_x, test_y = test_data
        train_x, train_y = train_data
        svm = SVM(kernel, C)
        svm.fit(train_x, train_y)
        test_prediction = svm.predict(test_x)
        test_error = utils.classification_error(test_prediction, test_y)
        totalError += test_error

    totalCVErrors.append(totalError/len(folds))
        
    del svm

print("train errors: ", train_errors)
print("test errors: ", test_errors)
print("total cv errors, ", totalCVErrors)

plt.plot(Cs, train_errors, label="full training error")
plt.plot(Cs, test_errors, label="full test error")
plt.plot(Cs, totalCVErrors, label="cross validation error")
plt.xlabel("C")
plt.ylabel("error")
plt.title("error vs C")
plt.xscale("log")
plt.legend(loc="upper left")

plt.show()

'''