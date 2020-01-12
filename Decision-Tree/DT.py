import numpy as np

np.random.seed(42)

chi_table = {0.01: 6.635,
             0.005: 7.879,
             0.001: 10.828,
             0.0005: 12.116,
             0.0001: 15.140,
             0.00001: 19.511}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if len(data.shape) == 1:
        classes, count = np.unique(data, return_counts=True)
        gini = 1 - np.sum([(count[i] / np.sum(count)) ** 2 for i in range(len(classes))])
    else:
        classes, count = np.unique(data.T[-1], return_counts=True)
        gini = 1 - np.sum([(count[i] / np.sum(count)) ** 2 for i in range(len(classes))])
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if len(data.shape) == 1:
        classes, count = np.unique(data, return_counts=True)
        entropy = np.sum([(-count[i] / np.sum(count)) * np.log2(count[i] / np.sum(count)) for i in range(len(classes))])
    else:
        classes, count = np.unique(data.T[-1], return_counts=True)
        entropy = np.sum([(-count[i] / np.sum(count)) * np.log2(count[i] / np.sum(count)) for i in range(len(classes))])
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy


class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.

    def __init__(self, feature, value):
        self.feature = feature  # column index of criteria being tested
        self.value = value  # value necessary to get a true result
        self.labels = {}  # dictionary to hold the labels and their amount at each node
        self.children = []  # list to hold the children for each node
        self.parent = None  # the parent of a node (if root none)

    def add_child(self, node):
        self.children.append(node)

    def build(self, data, impurity, p_value):
        """
        Build a tree using the given impurity measure, training dataset and p-value.
        if p_value = 1: full grow of the tree until all leaves are pure.
        else grow the tree according to chi-square testing
        Input:
        - data: the training dataset.
        - impurity: the chosen impurity measure.
        - p-value: the Probability value for the chi square test
        This function has no return value
        """
        # Labels count for current node
        label, count = np.unique(data.T[-1], return_counts=True)
        self.labels = dict(zip(label, count))

        # Impurity check
        if impurity(data) == 0:
            return
        else:
            # Find best feature and value
            self.feature, self.value = find_best_feature(data, impurity)
            # split the data according to the threshold value
            dataR, dataL = partition(data, self.feature, self.value)

            # Chi square test
            if p_value != 1:
                if chi_square(data, dataR, dataL, self) < chi_table.get(p_value):
                    return

            # Initiate children and assign to parent
            left = DecisionNode(None, None)
            right = DecisionNode(None, None)
            left.parent = self
            right.parent = self
            self.add_child(left)
            self.add_child(right)

            # build the tree recursively
            left.build(dataL, impurity, p_value)
            right.build(dataR, impurity, p_value)

    def __str__(self):
        if not self.children:
            return f"leaf: [{self.labels}]"
        else:
            return f"[X{self.feature} <= {self.value}]"


def build_tree(data, impurity, p_value=1):
    """
    Build a tree using the given impurity measure and training dataset.
    You are required to fully grow the tree until all leaves are pure.

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - p-value: the Probability value for the chi square test
    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    root = DecisionNode(None, None)
    root.build(data, impurity, p_value)  # p_value=1 corresponding to no pruning
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root


def find_best_feature(data, impurity):
    """
        find the best feature that will split the data that will get us closer to perfect classification
        according to the best threshold value and base on information gain decisions.
        Input:
        - data: the training dataset.
        - impurity: the chosen impurity measure.
        Output:
        - best feature
        - best threshold value
    """
    best_feature = 0
    best_threshold = 0
    max_gain = 0
    for i in range(data.shape[1] - 1):
        threshold, gain = find_best_threshold(data, i, impurity)
        if gain > max_gain:
            max_gain = gain
            best_feature = i
            best_threshold = threshold

    return best_feature, best_threshold


def find_best_threshold(data, index, impurity):
    """
        find the best threshold value that will split the data that will get us the maximum information gain.
        Input:
        - data: the training dataset.
        - index: the index of the feature in the dataset
        - impurity: the chosen impurity measure.
        Output:
        - best threshold value
        - the max infoGain of the best threshold
    """
    thresholds = get_thresholds(data, index)
    max_gain = 0
    best_value = 0
    for val in thresholds:
        gain = infoGain(data, index, val, impurity)
        if gain > max_gain:
            max_gain = gain
            best_value = val

    return best_value, max_gain


def get_thresholds(data, index):
    """
    get a list of thresholds, each threshold is the average of each consecutive pair of values.
    Input:
    - data: the training dataset.
    - index: the index of the feature in the dataset
    Output: list of thresholds
    """
    thresholds = []
    attribute = np.sort(data[:, index], kind='mergesort')

    for i in range(len(attribute) - 1):
        thresholds.append((attribute[i] + attribute[i + 1]) / 2)

    return thresholds


def infoGain(data, index, split, impurity):
    """
        Calculate the information gain of a dataset.
        Input:
        - data: The dataset for whose feature the IG should be calculated
        - index: the index of the feature.
        - split: the threshold value which the information gain should be calculated
        - impurity: the chosen impurity measure
        Output: information gain calculation
    """
    # Calculate the impurity of the total dataset
    current_impurity = impurity(data)

    rows = data.shape[0]
    d1, d2 = partition(data, index, split)

    # Calculate the weighted impurity
    weighted_impurity = (len(d1) / rows) * impurity(d1) + (len(d2) / rows) * impurity(d2)
    # Calculate the information gain
    information_gain = current_impurity - weighted_impurity

    return information_gain


def partition(data, index, split):
    """
        split the data into 2 list according to the threshold value (split value)
        Input:
        - data: the training dataset.
        - index: the index of the feature in the dataset
        - split: the threshold value which the data need to split
        Output:
        - d1 - list of instances that are greater than the split value
        - d2 - list of instances that are less than the split value
    """
    d1 = []
    d2 = []

    for line in data:
        if line[index] > split:
            d1.append(line)
        else:
            d2.append(line)

    return np.array(d1), np.array(d2)


def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    while len(node.children) > 0:
        if instance[node.feature] > node.value:
            node = node.children[1]
        else:
            node = node.children[0]
    pred = max(node.labels, key=node.labels.get)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred


def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    count = 0
    for row in dataset:
        pred = predict(node, row)
        if pred == row[-1]:
            count += 1
    accuracy = 100 * count / dataset.shape[0]
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy


def chi_square(data, dataR, dataL, node):
    """
        Calculate the chi square, according to the formula as we learned in class
        Input:
        - data: The dataset that the node holding
        - dataR: the split data that supposed to move to the right child of the node
        - dataL: the split data that supposed to move to the left child of the node
        - node: the node that need to split
        Output: chi square value
    """
    x = 0.0
    num_of_instance = len(data)
    py0 = node.labels.get(0) / num_of_instance
    py1 = node.labels.get(1) / num_of_instance
    if (py0 * py1) == 0.0:
        return 0

    # Greater than threshold
    df0 = len(dataR)
    label0, count0 = np.unique(dataR.T[-1], return_counts=True)
    pf0 = 0.0
    nf0 = 0.0
    for i in range(len(label0)):
        # pf
        if label0[i] == 0.0:
            pf0 = count0[i]
        # nf
        else:
            nf0 = count0[i]
    e0r = df0 * py0
    e1r = df0 * py1
    x += (((pf0 - e0r) ** 2) / e0r) + (((nf0 - e1r) ** 2) / e1r)

    # Less than threshold
    df1 = len(dataL)
    label1, count1 = np.unique(dataL.T[-1], return_counts=True)
    pf1 = 0.0
    nf1 = 0.0
    for i in range(len(label1)):
        # pf
        if label1[i] == 0.0:
            pf1 = count1[i]
        # nf
        else:
            nf1 = count1[i]
    e0l = df1 * py0
    e1l = df1 * py1
    x += (((pf1 - e0l) ** 2) / e0l) + (((nf1 - e1l) ** 2) / e1l)

    return x


def print_tree(node):
    """
        prints the tree according to the example in the notebook

        Input:
        - node: a node in the decision tree

        This function has no return value
    """

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    print_preorder(node, 0)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def print_preorder(node, level):
    """
        prints the tree recursively according to the example in the notebook
        Input:
        - node: a node in the decision tree
        This function has no return value
    """
    print('  ' * level + str(node))
    if node.children:
        print_preorder(node.children[1], level + 1)
        print_preorder(node.children[0], level + 1)
    return


def post_pruning(root, train_data, test_data):
    """
        unlink a node from is children, calculate the accuracy of the tree, link the children
        Input:
        - root: the root of the decision tree
        - train data: the training data of the decision tree
        - test data: the testing data of the decision tree
        Output:
        - tree_size: list of the amount of nodes in the tree while pruning
        - train_accuracy: list of the accuracy on the train data as the tree pruning
        - test_accuracy:  list of the accuracy on the test data as the tree pruning
    """
    tree_size = []
    train_accuracy = []
    test_accuracy = []

    while root.children:
        tree_size.append(count_intenal_nodes(root))
        train_accuracy.append(calc_accuracy(root, train_data))
        test_accuracy.append(calc_accuracy(root, test_data))

        parents = list_parents(root)
        max_acc = 0.0
        node_to_prune = None
        for parent in parents:
            acc = calc_accuracy_parent(root, parent, test_data)
            if acc > max_acc:
                max_acc = acc
                node_to_prune = parent
        node_to_prune.children.clear()

    tree_size.append(count_intenal_nodes(root))
    train_accuracy.append(calc_accuracy(root, train_data))
    test_accuracy.append(calc_accuracy(root, test_data))

    return tree_size, train_accuracy, test_accuracy


def calc_accuracy_parent(root, parent, data):
    """
        unlink a node from is children, calculate the accuracy of the tree, link the children
        Input:
        - root: the root of the decision tree
        - parent: node of the decision tree
        - data: data to calculate the accuracy
        Output:
        - accuracy: the accuracy of the decision tree on the data
    """
    templ = parent.children[0]
    tempr = parent.children[1]
    parent.children.clear()
    accuracy = calc_accuracy(root, data)
    parent.children.append(templ)
    parent.children.append(tempr)
    return accuracy


def count_intenal_nodes(root):
    """
        counts recursively the number of nodes of a tree
        Input:
        - root: the root of the decision tree
        Output: the amount of nodes
    """
    if not root.children:  # if leaf, returns 0
        return 0
    return count_intenal_nodes(root.children[0]) + count_intenal_nodes(root.children[1]) + 1


def list_parents(root):
    """
         return
         Input:
         - root: the root of the decision tree
         Output: parents - list of parents of leaves
    """
    parents = []
    q = [root]
    while q:
        node = q.pop(0)
        if node.children:
            q.append(node.children[0])
            q.append(node.children[1])
        else:
            parents.append(node.parent)

    return parents
