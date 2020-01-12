from numpy import count_nonzero, logical_and, logical_or, concatenate, mean, array_split, poly1d, polyfit, array, linspace
from numpy.random import permutation
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt


SVM_DEFAULT_DEGREE = 3
SVM_DEFAULT_GAMMA = 'auto'
SVM_DEFAULT_C = 1.0
ALPHA = 1.5


def prepare_data(data, labels, max_count=None, train_ratio=0.8):
    """
    :param data: a numpy array with the features dataset
    :param labels:  a numpy array with the labels
    :param max_count: max amount of samples to work on. can be used for testing
    :param train_ratio: ratio of samples used for train
    :return: train_data: a numpy array with the features dataset - for train
             train_labels: a numpy array with the labels - for train
             test_data: a numpy array with the features dataset - for test
             test_labels: a numpy array with the features dataset - for test
    """
    if max_count:
        data = data[:max_count]
        labels = labels[:max_count]

    train_data = array([])
    train_labels = array([])
    test_data = array([])
    test_labels = array([])

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    data = concatenate((data, labels[:, None]), axis=1)
    data = permutation(data)
    length = len(data)
    train_length = int (train_ratio * length)
    train = data[:train_length]
    test = data[train_length:]

    #split labels and data
    train_data = train[:, :-1]
    train_labels = train[:, -1]
    test_data = test[:, :-1]
    test_labels = test[:, -1]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return train_data, train_labels, test_data, test_labels


def get_stats(prediction, labels):
    """
    :param prediction: a numpy array with the prediction of the model
    :param labels: a numpy array with the target values (labels)
    :return: tpr: true positive rate
             fpr: false positive rate
             accuracy: accuracy of the model given the predictions
    """

    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    leny = len(labels)

    #tpr
    p = count_nonzero(labels)
    if p:
        tp = count_nonzero(logical_and(prediction, labels))
        tpr = tp / p

    #fpr
    positive = count_nonzero(prediction)
    fp = positive - tp
    n = leny - p
    if n:
        fpr = fp / n

    #accuracy
    tn = n - fp
    #accuracy = tn + tp / leny
    counter = leny - count_nonzero(prediction - labels)
    accuracy = counter / leny
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return tpr, fpr, accuracy


def get_k_fold_stats(folds_array, labels_array, clf):
    """
    :param folds_array: a k-folds arrays based on a dataset with M features and N samples
    :param labels_array: a k-folds labels array based on the same dataset
    :param clf: the configured SVC learner
    :return: mean(tpr), mean(fpr), mean(accuracy) - means across all folds
    """
    tpr = []
    fpr = []
    accuracy = []

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    k = len(folds_array)
    for i in range(k):
        #test data
        test_data = folds_array[i]
        test_label = labels_array[i]

        #concat train folds
        train_data = concatenate(folds_array[:i] + folds_array[i + 1:])
        train_labels = concatenate(labels_array[:i] + labels_array[i + 1:])

        #svm
        clf.fit(train_data, train_labels)
        prediction = clf.predict(test_data)

        #append parameters
        t, f, a = get_stats(prediction, test_label)
        tpr.append(t)
        fpr.append(f)
        accuracy.append(a)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return mean(tpr), mean(fpr), mean(accuracy)


def compare_svms(data_array,
                 labels_array,
                 folds_count,
                 kernels_list=('poly', 'poly', 'poly', 'rbf', 'rbf', 'rbf',),
                 kernel_params=({'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05}, {'gamma': 0.5},)):


    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernels_list: a list of strings defining the SVM kernels
    :param kernel_params: a dictionary with kernel parameters - degree, gamma, c
    :return: svm_df: a dataframe containing the results as described below
    """
    svm_df = pd.DataFrame()
    svm_df['kernel'] = kernels_list
    svm_df['kernel_params'] = kernel_params
    svm_df['tpr'] = None
    svm_df['fpr'] = None
    svm_df['accuracy'] = None

    tpr = []
    fpr = []
    acc = []

    # SVM_DEFAULT_DEGREE = 3
    # SVM_DEFAULT_GAMMA = 'auto'
    # SVM_DEFAULT_C = 1.0
    # ALPHA = 1.5
    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    arrays = array_split(data_array, folds_count)
    labels = array_split(labels_array, folds_count)

    for kernel, params in zip(kernels_list, kernel_params):
        if 'degree' in params.keys():
            degree = params["degree"]
        else:
            degree = SVM_DEFAULT_DEGREE

        if 'gamma' in params.keys():
            gamma = params["gamma"]
        else:
            gamma = SVM_DEFAULT_GAMMA

        if 'C' in params.keys():
            C = params["C"]
        else:
            C = SVM_DEFAULT_C

        clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
        t, f, a = get_k_fold_stats(arrays, labels, clf)
        tpr.append(t)
        fpr.append(f)
        acc.append(a)

    svm_df['tpr'] = tpr
    svm_df['fpr'] = fpr
    svm_df['accuracy'] = acc

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return svm_df


def get_most_accurate_kernel(accuracy=None):
    """
    :return: integer representing the row number of the most accurate kernel
    """
    if accuracy:
        accuracy = accuracy.tolist()
        return accuracy.index(max(accuracy))
    else:
        return 5


def get_kernel_with_highest_score(score=None):
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    if score:
        score = score.tolist()
        return score.index(max(score))
    else:
        return 5


def plot_roc_curve_with_score(df, alpha_slope=1.5):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    x = df.fpr.tolist()
    y = df.tpr.tolist()

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    best_kernel = get_kernel_with_highest_score(df["score"])
    line_x = [0]
    line_y = [y[best_kernel] - alpha_slope * x[best_kernel]]
    line_x.append(x[best_kernel])
    line_y.append(y[best_kernel])
    line_x.append(x[best_kernel] + 1)
    line_y.append(y[best_kernel] + alpha_slope)

    plt.plot(x, y, '.', ms=4, mec='r')
    plt.plot(line_x, line_y)
    plt.ylim(0.9, 1.01)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def evaluate_c_param(data_array, labels_array, folds_count, best_kernel):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """

    res = pd.DataFrame()
    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    kernel_list = ()
    kernel_params = ()

    i_list = [1, 0, -1, -2, -3, -4]
    j_list = [3, 2 ,1]
    C_list = []

    #create C list
    for j in j_list:
        for i in i_list:
            C_list.append((10 ** i) * (j / 3))

    #append best kernel type len(C_list) times
    kernel_list = [best_kernel[0] for i in range(len(C_list))]

    #append parameters dictionary to kernel_params
    for C in C_list:
        params_dict = best_kernel[1].copy()
        params_dict['C'] = C
        kernel_params = kernel_params + (params_dict,)

    res = compare_svms(data_array, labels_array, folds_count, kernel_list, kernel_params)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return res


def get_test_set_performance(train_data, train_labels, test_data, test_labels, best_kernel):
    """
    :param train_data: a numpy array with the features dataset - train
    :param train_labels: a numpy array with the labels - train

    :param test_data: a numpy array with the features dataset - test
    :param test_labels: a numpy array with the labels - test
    :return: kernel_type: the chosen kernel type (either 'poly' or 'rbf')
             kernel_params: a dictionary with the chosen kernel's parameters - c value, gamma or degree
             clf: the SVM leaner that was built from the parameters
             tpr: tpr on the test dataset
             fpr: fpr on the test dataset
             accuracy: accuracy of the model on the test dataset
    """

    kernel_type = best_kernel[0]
    kernel_params = best_kernel[1].copy()

    clf = SVC(class_weight='balanced', kernel=kernel_type, gamma=kernel_params['gamma'], C=kernel_params['C'])  # TODO: set the right kernel
    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    clf.fit(train_data, train_labels)
    tpr, fpr, accuracy = get_stats(clf.predict(test_data), test_labels)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy
