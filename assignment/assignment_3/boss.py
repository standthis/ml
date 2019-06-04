#!/usr/bin/env python
import sys
import argparse 
import numpy as np
import operator
#sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from difflib import get_close_matches

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

base = ['load_iris']
#parser = MyParser(description='Binary Classifier')
parser = argparse.ArgumentParser(description='Binary Classifier')
parser.add_argument('-c', '--classifier', nargs='*',
        choices=['k-Nearest Neighbours', 'Naive Bayes', 'Logistic Regression',
            'Linear SVM', 'Decision Tree', 'Random Forest', 'Multilayer Perceptron', 'all'],
        help="Specifiy a binary classifier")
parser.add_argument('-m', '--metric', nargs='*',
        choices=['accuracy', 'precision', 'recall'],
        help="Specifiy a metric for comparision", default=['accuracy'])
parser.add_argument('-g', '--grid', help="Apply grid search for hyper-parameter tuning", action='store_true')
parser.add_argument('-i', '--info',nargs='?', help="List information about datasets", default=42)
parser.add_argument('dataset', nargs='*', help="Provide binary-class dataset. Defaults to load_iris", default=base)
args = parser.parse_args()

if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

if args.info is None:
    help(datasets)
    sys.exit()

classifiers = {'k-Nearest Neighbours':-1, 'Naive Bayes':-1, 'Logistic Regression':-1,
        'Linear SVM':-1, 'Decision Tree':-1, 'Random Forest':-1, 'Multilayer Perceptron':-1}
classdiff = ['k-Nearest Neighbours', 'Naive Bayes', 'Logistic Regression',
        'Linear SVM', 'Decision Tree', 'Random Forest', 'Multilayer Perceptron']

def main():
    for data in args.dataset:
        if data.rfind('.') == -1:
            data = 'datasets.' + data  + '()'
        else:
            data = data + '()'
        # 2 is here to be generous. Since we will remove one data point to be lenient toward the user
        if not all(n==0 or n==1 or n==2 for n in eval(data).target.tolist()):
            print(data, "is not binary. Please provide a binary dataset")
            return None
        for clss in vars(args)['classifier']:
            #clss = get_close_matches(clss, classdiff, n=1, cutoff=0.1)
            if clss == 'Multilayer Perceptron':
                perceptron(data, clss)
            elif clss == 'Logistic Regression':
                logReg(data, clss)
            elif clss == 'Random Forest':
                randomForest(data, clss)
            elif clss == 'k-Nearest Neighbours':
                nearest(data, clss)
            elif clss == 'Naive Bayes':
                bayes(data, clss)
            elif clss == 'Linear SVM':
                linSVM(data, clss)
            elif clss == 'Decision Tree':
                decisionTree(data, clss)
            elif clss == 'all':
                decisionTree(data, 'Decision Tree')
                linSVM(data, 'Linear SVM')
                bayes(data, 'Naive Bayes')
                nearest(data, 'k-Nearest Neighbours')
                randomForest(data, 'Random Forest')
                logReg(data, 'Logistic Regression')
                perceptron(data, 'Multilayer Perceptron')
            else: 
                print("FALLEN THROUGH")
        makeplt(data) 
        if args.grid:
            gridSearch(data)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++') 
    print('BEST CLASSIFIERS') 
    #classi = sorted(classifiers.items(), key=operator.itemgetter(1))
    #classifiers = classi

    # CLASSIFERS BAR GRAPH -> 
    classif = {k: v for k, v in classifiers.items() if v >= 0}
    if len(classif) <= 1:
        print("Provide more than one classifier to see bar graph")
        sys.exit()
    plt.title(args.metric)
    plt.bar(range(len(classif)), list(classif.values()), align='center')
    plt.xticks(range(len(classif)), list(classif.keys()))

    plt.show()

# Kudos to Dane
def gridSearch(dataset):

    # Loading the Digits dataset
    digits = eval(dataset)

    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.data)
    X = digits.data.reshape((n_samples, -1))
    y = digits.target

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)

    tune_param = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        grid = GridSearchCV(SVC(), tune_param, cv=5,
                           scoring='%s_macro' % score)
        grid.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(grid.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        y_pred = grid.predict(X_test)
        print(metrics.classification_report(y_test, y_pred))
        print("Accuracy",metrics.accuracy_score(y_test, y_pred))

def makeplt(dataset):
    dataset = eval(dataset) 
    X = dataset.data
    plt.scatter(X[:, 0], X[:, 1])
    plt.axis('equal')
    plt.show()

def decisionTree(dataset, clss):
    dataset = eval(dataset) 
    X = dataset.data
    y = dataset.target
    idx = dataset.target != 2
    data = dataset.data[idx].astype(np.float32)
    target = dataset.target[idx].astype(np.float32)

    X, y = data.reshape(len(data),-1), target
    # Split to 30% test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)

    y_pred=clf.predict(X_test)
    accuracy=metrics.accuracy_score(y_test, y_pred) *100
    precision=metrics.precision_score(y_test, y_pred) *100
    recall=metrics.recall_score(y_test, y_pred) *100
    print("DecisionTreeClassifier")
    print("Generic accuracy_score: {:.2f}%".format(accuracy))
    print("Generic precision_score: {:.2f}%".format(precision))
    print("Generic recall_score: {:.2f}%".format(recall))
    print("---------------------------------------------------------------------------")

    classifiers[clss] = eval(args.metric[0]) 
    #plot_decision_boundary(clf, data, target, y_pred)

def linSVM(dataset, clss):
    dataset = eval(dataset) 
    X = dataset.data
    y = dataset.target
    idx = dataset.target != 2
    data = dataset.data[idx].astype(np.float32)
    target = dataset.target[idx].astype(np.float32)

    X, y = data.reshape(len(data),-1), target
    # Split to 30% test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(X, y)  
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
            intercept_scaling=1, loss='squared_hinge', max_iter=1000,
            multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)
    y_pred=clf.predict(X_test)
    #print("MLP \n",y_pred)
    accuracy=metrics.accuracy_score(y_test, y_pred) *100
    precision=metrics.precision_score(y_test, y_pred) *100
    recall=metrics.recall_score(y_test, y_pred) *100
    print("LinearSVM")
    print("Generic accuracy_score: {:.2f}%".format(accuracy))
    print("Generic precision_score: {:.2f}%".format(precision))
    print("Generic recall_score: {:.2f}%".format(recall))
    print("---------------------------------------------------------------------------")
    classifiers[clss] = eval(args.metric[0]) 

def bayes(dataset, clss):
    dataset = eval(dataset) 
    X = dataset.data
    y = dataset.target

    idx = dataset.target != 2
    data = dataset.data[idx].astype(np.float32)
    target = dataset.target[idx].astype(np.float32)
    X, y = data.reshape(len(data),-1), target
    # Split to 30% test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    #print("MLP \n",y_pred)
    accuracy=metrics.accuracy_score(y_test, y_pred) *100
    precision=metrics.precision_score(y_test, y_pred) *100
    recall=metrics.recall_score(y_test, y_pred) *100
    print("Naive Bayes")
    print("Generic accuracy_score: {:.2f}%".format(accuracy))
    print("Generic precision_score: {:.2f}%".format(precision))
    print("Generic recall_score: {:.2f}%".format(recall))
    print("---------------------------------------------------------------------------")
    classifiers[clss] = eval(args.metric[0]) 


def nearest(dataset, clss):
    dataset = eval(dataset) 
    X = dataset.data
    y = dataset.target
    idx = dataset.target != 2
    data = dataset.data[idx].astype(np.float32)
    target = dataset.target[idx].astype(np.float32)
    X, y = data.reshape(len(data),-1), target
    # Split to 30% test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    accuracy=metrics.accuracy_score(y_test, y_pred) *100
    precision=metrics.precision_score(y_test, y_pred) *100
    recall=metrics.recall_score(y_test, y_pred) *100
    print("k-Nearest Neighbours")
    print("Generic accuracy_score: {:.2f}%".format(accuracy))
    print("Generic precision_score: {:.2f}%".format(precision))
    print("Generic recall_score: {:.2f}%".format(recall))
    print("---------------------------------------------------------------------------")
    classifiers[clss] = eval(args.metric[0]) 


    # -------------------------------------------------------------------------------------------------------------------

def randomForest(dataset, clss):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

    dataset = eval(dataset) 
    X = dataset.data
    y = dataset.target
    idx = dataset.target != 2
    data = dataset.data[idx].astype(np.float32)
    target = dataset.target[idx].astype(np.float32)
    X, y = data.reshape(len(data),-1), target
    # Split to 30% test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    accuracy=metrics.accuracy_score(y_test, y_pred) *100
    precision=metrics.precision_score(y_test, y_pred) *100
    recall=metrics.recall_score(y_test, y_pred) *100
    print("Random Forest")
    print("Generic accuracy_score: {:.2f}%".format(accuracy))
    print("Generic precision_score: {:.2f}%".format(precision))
    print("Generic recall_score: {:.2f}%".format(recall))
    print("---------------------------------------------------------------------------")
    classifiers[clss] = eval(args.metric[0]) 


def logReg(dataset, clss):
    clf = linear_model.LogisticRegression()
    dataset = eval(dataset) 
    X = dataset.data
    y = dataset.target
    idx = dataset.target != 2
    data = dataset.data[idx].astype(np.float32)
    target = dataset.target[idx].astype(np.float32)
    X, y = data.reshape(len(data),-1), target
    # Split to 30% test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    accuracy=metrics.accuracy_score(y_test, y_pred) *100
    precision=metrics.precision_score(y_test, y_pred) *100
    recall=metrics.recall_score(y_test, y_pred) *100
    print("Logistic Regression")
    print("Generic accuracy_score: {:.2f}%".format(accuracy))
    print("Generic precision_score: {:.2f}%".format(precision))
    print("Generic recall_score: {:.2f}%".format(recall))
    print("---------------------------------------------------------------------------")

    classifiers[clss] = eval(args.metric[0]) 

# arg for num of layers
def perceptron(dataset, clss):
    n_hidden = 2
    n_hidden_layers = 1
    clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(n_hidden,n_hidden_layers), activation='tanh')
    dataset = eval(dataset) 
    X = dataset.data
    y = dataset.target
    idx = dataset.target != 2
    data = dataset.data[idx].astype(np.float32)
    target = dataset.target[idx].astype(np.float32)
    X, y = data.reshape(len(data),-1), target
    # Split to 30% test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    accuracy=metrics.accuracy_score(y_test, y_pred) *100
    precision=metrics.precision_score(y_test, y_pred) *100
    recall=metrics.recall_score(y_test, y_pred) *100
    print("Multilayer Perceptron")
    print("Generic accuracy_score: {:.2f}%".format(accuracy))
    print("Generic precision_score: {:.2f}%".format(precision))
    print("Generic recall_score: {:.2f}%".format(recall))
    print("---------------------------------------------------------------------------")

    classifiers[clss] = eval(args.metric[0]) 

main()
