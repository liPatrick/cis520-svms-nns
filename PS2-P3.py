import numpy as np 
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import utils
import time

from neural_network import NeuralNetworkClassification


#relu on spam dataset
# Your code starts here.
start = time.time()

(x_train, y_train, x_test, y_test) = utils.load_all_train_test_data("./P3/Spam-Dataset")
folds = utils.load_all_cross_validation_data("P3/Spam-Dataset/CrossValidation")

hiddenUnits = [1,5,10,15,25,50]

train_errors = [] 
test_errors = [] 
cv_errors = []

for i in range(len(hiddenUnits)):
    print("hidden unit: ", i)
    N,d = x_train.shape

    d1 = hiddenUnits[i]
    initParams = utils.load_initial_weights("P3/Spam-Dataset/InitParams/relu/"+str(d1))
    
    nn = NeuralNetworkClassification(d, d1, "relu", initParams["W1"], initParams["b1"], initParams["W2"], initParams["b2"])
    nn.fit(x_train, y_train, step_size=0.01)
    train_predictions = nn.predict(x_train)
    test_predictions = nn.predict(x_test)

    print("train_predictions shape", train_predictions.shape)
    print("y_train shape", y_train.shape)
    train_errors.append(utils.classification_error(train_predictions, y_train))
    test_errors.append(utils.classification_error(test_predictions, y_test))



    #kfolds
    totalError = 0
    for index, fold in enumerate(folds): 
        test_data, train_data = utils.partition_cross_validation_fold(folds, index)
        test_x, test_y = test_data
        train_x, train_y = train_data


        nn = NeuralNetworkClassification(d, d1, "relu", initParams["W1"], initParams["b1"], initParams["W2"], initParams["b2"])
        nn.fit(train_x, train_y, step_size=0.01)
        test_prediction = nn.predict(test_x)
        test_error = utils.classification_error(test_prediction, test_y)
        totalError += test_error

    cv_errors.append(totalError/len(folds))
    del nn



        
end = time.time()


print("train errors: ", train_errors)
print("test errors: ", test_errors)
print("total cv errors, ", cv_errors)
print("total time: ", end - start)

plt.plot(hiddenUnits, train_errors, label="full training error")
plt.plot(hiddenUnits, test_errors, label="full test error")
plt.plot(hiddenUnits, cv_errors, label="cross validation error")
plt.xlabel("hidden units")
plt.ylabel("error")
plt.title("error vs hidden units")
#plt.xscale("log")
plt.legend(loc="upper left")

plt.show()


'''
#sigmoid on spam dataset
# Your code starts here.
start = time.time()

(x_train, y_train, x_test, y_test) = utils.load_all_train_test_data("./P3/Spam-Dataset")
folds = utils.load_all_cross_validation_data("P3/Spam-Dataset/CrossValidation")

hiddenUnits = [1,5,10,15,25,50]

train_errors = [] 
test_errors = [] 
cv_errors = []

for i in range(len(hiddenUnits)):
    print("hidden unit: ", i)
    N,d = x_train.shape

    d1 = hiddenUnits[i]
    initParams = utils.load_initial_weights("P3/Spam-Dataset/InitParams/sigmoid/"+str(d1))
    
    nn = NeuralNetworkClassification(d, d1, "sigmoid", initParams["W1"], initParams["b1"], initParams["W2"], initParams["b2"])
    nn.fit(x_train, y_train)
    train_predictions = nn.predict(x_train)
    test_predictions = nn.predict(x_test)

    print("train_predictions shape", train_predictions.shape)
    print("y_train shape", y_train.shape)
    train_errors.append(utils.classification_error(train_predictions, y_train))
    test_errors.append(utils.classification_error(test_predictions, y_test))



    #kfolds
    totalError = 0
    for index, fold in enumerate(folds): 
        test_data, train_data = utils.partition_cross_validation_fold(folds, index)
        test_x, test_y = test_data
        train_x, train_y = train_data


        nn = NeuralNetworkClassification(d, d1, "sigmoid", initParams["W1"], initParams["b1"], initParams["W2"], initParams["b2"])
        nn.fit(train_x, train_y)
        test_prediction = nn.predict(test_x)
        test_error = utils.classification_error(test_prediction, test_y)
        totalError += test_error

    cv_errors.append(totalError/len(folds))
    del nn



        
end = time.time()


print("train errors: ", train_errors)
print("test errors: ", test_errors)
print("total cv errors, ", cv_errors)
print("total time: ", end - start)

plt.plot(hiddenUnits, train_errors, label="full training error")
plt.plot(hiddenUnits, test_errors, label="full test error")
plt.plot(hiddenUnits, cv_errors, label="cross validation error")
plt.xlabel("hidden units")
plt.ylabel("error")
plt.title("error vs hidden units")
#plt.xscale("log")
plt.legend(loc="upper left")

plt.show()
'''


'''
#relu
start = time.time()

(x_train, y_train, x_test, y_test) = utils.load_all_train_test_data("./P3/Synthetic-Dataset")
folds = utils.load_all_cross_validation_data("P3/Synthetic-Dataset/CrossValidation")

hiddenUnits = [1,5,10,15,25,50]

train_errors = [] 
test_errors = [] 
cv_errors = []

for i in range(len(hiddenUnits)):
    print("hidden unit: ", i)
    N,d = x_train.shape

    d1 = hiddenUnits[i]
    initParams = utils.load_initial_weights("P3/Synthetic-Dataset/InitParams/relu/"+str(d1))
    
    nn = NeuralNetworkClassification(d, d1, "relu", initParams["W1"], initParams["b1"], initParams["W2"], initParams["b2"])
    nn.fit(x_train, y_train, step_size=0.01)
    train_predictions = nn.predict(x_train)
    test_predictions = nn.predict(x_test)

    print("train_predictions shape", train_predictions.shape)
    print("y_train shape", y_train.shape)
    train_errors.append(utils.classification_error(train_predictions, y_train))
    test_errors.append(utils.classification_error(test_predictions, y_test))



    #kfolds
    totalError = 0
    for index, fold in enumerate(folds): 
        test_data, train_data = utils.partition_cross_validation_fold(folds, index)
        test_x, test_y = test_data
        train_x, train_y = train_data


        nn = NeuralNetworkClassification(d, d1, "relu", initParams["W1"], initParams["b1"], initParams["W2"], initParams["b2"])
        nn.fit(train_x, train_y, step_size=0.01)
        test_prediction = nn.predict(test_x)
        test_error = utils.classification_error(test_prediction, test_y)
        totalError += test_error

    cv_errors.append(totalError/len(folds))
    del nn



end = time.time()

print("train errors: ", train_errors)
print("test errors: ", test_errors)
print("total cv errors, ", cv_errors)
print("total time: ", end - start)

plt.plot(hiddenUnits, train_errors, label="full training error")
plt.plot(hiddenUnits, test_errors, label="full test error")
plt.plot(hiddenUnits, cv_errors, label="cross validation error")
plt.xlabel("hidden units")
plt.ylabel("error")
plt.title("error vs hidden units")
#plt.xscale("log")
plt.legend(loc="upper left")

plt.show()



#sigmoid
start = time.time()

(x_train, y_train, x_test, y_test) = utils.load_all_train_test_data("./P3/Synthetic-Dataset")
folds = utils.load_all_cross_validation_data("P3/Synthetic-Dataset/CrossValidation")

hiddenUnits = [1,5,10,15,25,50]

train_errors = [] 
test_errors = [] 
cv_errors = []

for i in range(len(hiddenUnits)):
    print("hidden unit: ", i)
    N,d = x_train.shape

    d1 = hiddenUnits[i]
    initParams = utils.load_initial_weights("P3/Synthetic-Dataset/InitParams/sigmoid/"+str(d1))
    
    nn = NeuralNetworkClassification(d, d1, "sigmoid", initParams["W1"], initParams["b1"], initParams["W2"], initParams["b2"])
    nn.fit(x_train, y_train)
    train_predictions = nn.predict(x_train)
    test_predictions = nn.predict(x_test)

    print("train_predictions shape", train_predictions.shape)
    print("y_train shape", y_train.shape)
    train_errors.append(utils.classification_error(train_predictions, y_train))
    test_errors.append(utils.classification_error(test_predictions, y_test))



    #kfolds
    totalError = 0
    for index, fold in enumerate(folds): 
        test_data, train_data = utils.partition_cross_validation_fold(folds, index)
        test_x, test_y = test_data
        train_x, train_y = train_data


        nn = NeuralNetworkClassification(d, d1, "sigmoid", initParams["W1"], initParams["b1"], initParams["W2"], initParams["b2"])
        nn.fit(train_x, train_y)
        test_prediction = nn.predict(test_x)
        test_error = utils.classification_error(test_prediction, test_y)
        totalError += test_error

    cv_errors.append(totalError/len(folds))
    del nn



end = time.time()

print("train errors: ", train_errors)
print("test errors: ", test_errors)
print("total cv errors, ", cv_errors)
print("total time: ", end - start)

plt.plot(hiddenUnits, train_errors, label="full training error")
plt.plot(hiddenUnits, test_errors, label="full test error")
plt.plot(hiddenUnits, cv_errors, label="cross validation error")
plt.xlabel("hidden units")
plt.ylabel("error")
plt.title("error vs hidden units")
#plt.xscale("log")
plt.legend(loc="upper left")

plt.show()

'''