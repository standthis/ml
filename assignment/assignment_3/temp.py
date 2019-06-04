print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
                clf.fit(X, y)

                    # Plot the decision boundary. For that, we will assign a color to each
                        # point in the mesh [x_min, x_max]x[y_min, y_max].
                            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                                                     np.arange(y_min, y_max, h))
                                        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

                                            # Put the result into a color plot
                                                Z = Z.reshape(xx.shape)
                                                    plt.figure()
                                                        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

                                                            # Plot also the training points
                                                                plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                                                                                        edgecolor='k', s=20)
                                                                    plt.xlim(xx.min(), xx.max())
                                                                        plt.ylim(yy.min(), yy.max())
                                                                            plt.title("3-Class classification (k = %i, weights = '%s')"
                                                                                                  % (n_neighbors, weights))

                                                                            plt.show()
 classifiers = {u'Label1':26, u'Label2': 17, u'Label3':30}


#parser.add_argument('-d', '--dataset', nargs='?', help="Provide binary-class dataset", default=base)
#parser.add_argument('-k', '--k-nearest', nargs='?', help="Please provide a valid png file")
#parser.add_argument('-k', '--k-nearest', nargs='+', help="Please provide a valid png file",)
#parser.add_argument('-k', nargs='?', help="Please supply a image to hide", default=base)
#parser.add_argument('-c', '--convert', nargs='?', help="Please provide a valid png file", const=base)
#parser.add_argument('-s', '--show', nargs='?', help="Please provide a valid ppm file to uncover", const=output)
    #plt.style.use('ggplot')

    #iris = eval(data)

    ## -------------------------------------------------------------------------------------------------------------------
    ## Using Logistic Regression as a binary classifier
    ## classifiersiscard all data points belonging to for e.g. class label 2, by selecting all the rows that do not belong to class 2:
    #idx = iris.target != 2
    #data = iris.data[idx].astype(np.float32)
    #target = iris.target[idx].astype(np.float32)
    ## Notice the first 100 rows are class 0 and 1
    ## print(target.size)
    ## This is now a binary problem

    ## Visualizing the data using a scatter plot of the first two features
    #plt.figure(figsize=(10, 6))
    #plt.scatter(data[:, 0], data[:, 1], c=target, cmap=plt.cm.Paired, s=100)
    #plt.xlabel(iris.feature_names[0])
    #plt.ylabel(iris.feature_names[1])
    ## plt.show()

    ## Split the data into training and test sets
    #X_train, X_test, y_train, y_test = train_test_split(
    #    data, target, test_size=0.4, random_state=42
    #)

    ## End up with 90 training data points and 10 test data points
    ## print(X_train.shape, y_train.shape)
    ## print(X_test.shape, y_test.shape)

    ## -------------------------------------------------------------------------------------------------------------------
    ## Training the LR model
    ## CV method
    ##lr = cv2.ml.LogisticRegression_create()

    ## sklearn method
    ## Choice of a training method:`cv2.ml.LogisticRegression_BATCH` or `cv2.ml.LogisticRegression_MINI_BATCH`
    ## Choose to update the model after every data point using cv2.ml.LogisticRegression_MINI_BATCH:
    ##lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
    ##lr.setMiniBatchSize(1)

    ## Specify the number of iterations for the batch
    ##lr.setIterations(100)
    ## OpenCV's `train` method returns True upon success:
    ##lr.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    ## sklearn
    #clf.fit(X_train, y_train)

    ## Get the learned weights:
    ##lr.get_learnt_thetas()

    ## -------------------------------------------------------------------------------------------------------------------
    ## ### Testing the classifier
    ## Using the learnt weights calculate the accuracy score on the training set (seen data).
    ## This test will show how well the model was able to memorize the training dataset
    ##ret, y_pred = lr.predict(X_train)

    #y_pred = clf.predict(X_train)
    ##print(y_train)
    ##print(y_pred[0])
    ##print(len(y_train))
    ##print(len(y_pred))
    ##sys.exit()
    #print('Seen Accuracy:', metrics.accuracy_score(y_train, y_pred))

    #
    ##X = X_train
    ##y = y_train
    ##pca = decomposition.PCA(n_components=2)
    ##X = pca.fit_transform(X)
    ##plt.figure(figsize=(10, 6))
    ##print(X.shape)
    ##plot_decision_boundary(clf, X, y)
    ##plt.xlabel('x1')
    ##plt.ylabel('x2')
    ##plt.show()

    ##plt.style.use('ggplot')
    ##plt.figure(figsize=(10, 6))
    ##plt.scatter(X[:, 0], X[:, 1], s=100, c=y);
    ##plt.xlabel('x1')
    ##plt.ylabel('x2')
    ##plt.figure(figsize=(10, 6))
    ##print(X.shape)
    ##print(y.shape)
    ##plot_decision_boundary(clf, X, y)
    ##plt.xlabel('x1')
    ##plt.ylabel('x2')
    ##plt.show()

    ## But, how well can it classify unseen data points:
    #y_pred = clf.predict(X_test)
    ##ret, y_pred = lr.predict(X_test)
    #print('Unseen Accuracy:', metrics.accuracy_score(y_test, y_pred))
    #return None

#######################################################


    #plt.style.use('ggplot')

    #iris = eval(data)

    ## Loading the iris dataset
    ##iris = datasets.load_iris()

    ## The structure of the boston object:
    ## - `classifiersESCR`: Get a description of the data
    ## - `data`: The actual data, <`num_samples` x `num_features`>
    ## - `feature_names`: The names of the features
    ## - `target`: The class labels, <`num_samples` x 1>
    ## - `target_names`: The names of the class labels

    ## print(dir(iris))


    ## There are 150 data points; each have four feature values:
    ## print(iris.data.shape)

    ## Four features: sepal and petal in two-dimensions as shown in slides:
    ## print(iris.feature_names)

    ## Inspecting the class labels reveals a total of three classes:
    ##print(np.unique(iris.target))

    ## Let's see what the feature values are:

    ##print(iris.data)

    ## -------------------------------------------------------------------------------------------------------------------
    ## Using Logistic Regression as a binary classifier
    ## classifiersiscard all data points belonging to for e.g. class label 2, by selecting all the rows that do not belong to class 2:
    #idx = iris.target != 2
    #data = iris.data[idx].astype(np.float32)
    #target = iris.target[idx].astype(np.float32)
    ## Notice the first 100 rows are class 0 and 1
    ## print(target.size)
    ## This is now a binary problem

    ## Visualizing the data using a scatter plot of the first two features
    #plt.figure(figsize=(10, 6))
    #plt.scatter(data[:, 0], data[:, 1], c=target, cmap=plt.cm.Paired, s=100)
    #plt.xlabel(iris.feature_names[0])
    #plt.ylabel(iris.feature_names[1])
    ## plt.show()

    ## Split the data into training and test sets
    #X_train, X_test, y_train, y_test = train_test_split(
    #    data, target, test_size=0.4, random_state=42
    #)

    ## End up with 90 training data points and 10 test data points
    ## print(X_train.shape, y_train.shape)
    ## print(X_test.shape, y_test.shape)

    ## -------------------------------------------------------------------------------------------------------------------
    ## Training the LR model
    ## CV method
    ##lr = cv2.ml.LogisticRegression_create()

    ## sklearn method
    ## Choice of a training method:`cv2.ml.LogisticRegression_BATCH` or `cv2.ml.LogisticRegression_MINI_BATCH`
    ## Choose to update the model after every data point using cv2.ml.LogisticRegression_MINI_BATCH:
    ##lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
    ##lr.setMiniBatchSize(1)

    ## Specify the number of iterations for the batch
    ##lr.setIterations(100)
    ## OpenCV's `train` method returns True upon success:
    ##lr.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    ## sklearn
    #logreg.fit(X_train, y_train)

    ## Get the learned weights:
    ##lr.get_learnt_thetas()

    ## -------------------------------------------------------------------------------------------------------------------
    ## ### Testing the classifier
    ## Using the learnt weights calculate the accuracy score on the training set (seen data).
    ## This test will show how well the model was able to memorize the training dataset
    ##ret, y_pred = lr.predict(X_train)

    #y_pred = logreg.predict(X_train)
    ##print(y_train)
    ##print(y_pred[0])
    ##print(len(y_train))
    ##print(len(y_pred))
    ##sys.exit()
    #print('Seen Accuracy:', metrics.accuracy_score(y_train, y_pred))



    ## But, how well can it classify unseen data points:
    #y_pred = logreg.predict(X_test)
    ##ret, y_pred = lr.predict(X_test)
    #print('Unseen Accuracy:', metrics.accuracy_score(y_test, y_pred))

##################################################################3
    #plt.style.use('ggplot')

    # -------------------------------------------------------------------------------------------------------------------
    ##Simple example for generating training data
    #np.random.seed(42) #again for reproducibilty

    ## Generate a coordinate such that `0 <= x <= 100` and `0 <= y <= 100`:
    #single_data_point = np.random.randint(0, 100, 2)
    #print(single_data_point)

    ## Assign a label to the data point:
    #single_label = np.random.randint(0, 2)
    #print(single_label)

    ##End of simple example
    ## -------------------------------------------------------------------------------------------------------------------

    ## -------------------------------------------------------------------------------------------------------------------
    ## Create a function instead that allows us to specify the amount of data and features
    ## num_samples is the number of data points
    ## num_features is the number of features every data point has
    ## note that class labels are randomly assigned to random data points
    ## note every row is a data point and one label per row
    #def generate_data(num_samples, num_features):
    ##    """Randomly generates a number of data points"""
    #    data_size = (num_samples, num_features)
    #    data = np.random.randint(0, 100, size=data_size) # N X 2 array of random integers [0, 100]
    #    labels_size = (num_samples, 1)
    #    labels = np.random.randint(0, 2, size=labels_size)
    #    
    #    return data.astype(np.float32), labels
    ## -------------------------------------------------------------------------------------------------------------------

    ## Test the generate_data function with an 11 x 2 array. Note the dual tuple return:
    ##data = generate_data(11,2)
    ##print(data.shape)
    ##train_data, train_labels = generate_data(11,2)
    ##train_data, train_labels = data

    #idx = dataset.target != 2
    #data = dataset.data[idx].astype(np.float32)
    #target = dataset.target[idx].astype(np.float32)
    #train_data, train_labels = data.reshape(len(data),-1), target
    #print('-!-')
    ##print(data.shape)
    #print(train_data.shape)
    #print('-!-')
    ##print(target.shape)
    ##train_labels.ravel()
    #print(train_labels.shape)
    ##print(train_data)
    ##print(train_labels)
    ##sys.exit()
    ## print(train_data)
    ## print(train_labels)

    ## Inspect the first data point and corresponding label:
    #print(train_data[0], train_labels[0])

    ## The first data point (71, 60) is a blue square ('sb') and belongs to class 0

    #test = plt.plot(train_data[0, 0], train_data[0, 1], 'sb')
    #plt.xlabel('x coordinate')
    #plt.ylabel('y coordinate')
    ## plt.savefig("First class")
    ## plt.show()

    ## -------------------------------------------------------------------------------------------------------------------
    ## Better yet, create a function that visualizes the whole training set.
    ## Input a list of all the data points that are blue squares and a second list for red triangles
    #def plot_data(all_blue, all_red):
    #    plt.figure(figsize=(10, 6))
    #    plt.scatter(all_blue[:, 0], all_blue[:, 1], c='b', marker='s', s=180)
    #    plt.scatter(all_red[:, 0], all_red[:, 1], c='r', marker='^', s=180)
    #    plt.xlabel('x coordinate (feature 1)')
    #    plt.ylabel('y coordinate (feature 2)')
    ## -------------------------------------------------------------------------------------------------------------------


    ## Next, split all the data points into red and blue sets.
    ## ravel flattens the array to easily select all the elements of the `train_labels` array created earlier that are equal to 0
    ## print("Original array",train_labels)
    ## print("Flattened array (1classifiers)",train_labels.ravel())
    ## print("True when label = 1",train_labels.ravel() == 1)

    ## Blue data points are rows in `train_data` and set to label 0:
    #blue = train_data[train_labels.ravel() == 0]

    ## Red data points are rows in `train_data` and set to label 1:
    #red = train_data[train_labels.ravel() == 1]

    ## Finally, call the custom plot function:
    #plot_data(blue, red)
    ## plt.show()

    ## -------------------------------------------------------------------------------------------------------------------
    ## Train the classifier by following the "cvML Methodology" outline in the slides
    ## Use OpenCV's `ml` module:
    ##knnModel = cv2.ml.KNearest_create()

    ## Pass training data to the train method:
    ##knnModel.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    ## cv2.ml.ROW_SAMPLE tells k-NN the data is an N X 2 array i.e. every row is a data point
    ## train function returns True if data is the correct format
    ## -------------------------------------------------------------------------------------------------------------------

    ## -------------------------------------------------------------------------------------------------------------------
    ## Predict the label of a new data point
    ## `findNearest` predicts the label of a new data point based on its k-nearest neighbours.
    ## Use our `generate_data` function to generate a new (unseen) data point!
    ## Think of this new data point as a dataset of size 1:
    #newcomer, _ = generate_data(1,2) #Test data does not have a label thus the "ignore" underscore variable
    ## The idea of the previous line is for the classifier to predict the (unseen) label
    ## print(newcomer)

    ## Plot the training set once again, but add the new data point (from test set) as a green circle (since it is unseen
    ## data):
    #plot_data(blue, red)
    #plt.plot(newcomer[0, 0], newcomer[0, 1], 'go', markersize=14)
    ## plt.show()
    ## Forget the classifier, visually what would you guess the label is based on its neighbours, blue or red?
    ## classifiersepends, doesn't it?
    ## NN is(86, 73)

    ## so what our classifier would predict for k=1:
    ##ret, results, neighbour, dist = knnModel.findNearest(newcomer, 1)
    ## print("Predicted label:\t", results)
    ## print("neighbour's label:\t", neighbour)
    ## print("classifiersistance to neighbour:\t", dist)

    ## The same would be true if we looked at the k=2 nearest neighbours (and k=3)
    ## classifierso not to pick even numbers for k. Why? Refer to the slides or stay tuned

    ## What would happen if k=7?
    ##ret, results, neighbour, dist = knnModel.findNearest(newcomer, 7)
    ## print("Predicted label:\t", results)
    ## print("neighbour's label:\t", neighbour)
    ## print("classifiersistance to neighbour:\t", dist)

    ## Clearly, k-NN uses majority voting since four blue squares (label 0), and only three red triangles (label 1)

    ## What would happen if k=6?
    ##ret, results, neighbours, dist = knnModel.findNearest(newcomer, 6)
    ## print("Predicted label:\t", results)
    ## print("neighbours' labels:\t", neighbours)
    ## print("classifiersistance to neighbours:\t", dist)


    ## Following the cvML Methodology the `predict` method can instead be used. But first set `k`:
    ##knnModel.setclassifiersefaultK(7)
    ##ret, pred = knnModel.predict(newcomer)
    ### print(pred)
    ##knnModel.setclassifiersefaultK(6)
    ##ret, pred = knnModel.predict(newcomer)
    ### print(int(pred))
    ##knnModel.setclassifiersefaultK(1)
    ##ret, pred = knnModel.predict(newcomer)
    ## print(int(pred))

    ## Which value for $k$ is the most suitable? A naive solution is just to try a bunch of values for k
    ## Better solutions will be covered later

    ## generate test data. This time we have 10 data points to predict:
    ##data.reshape(len(data),-1), target
    ##test_data, test_labels = generate_data(11,2)
    #test_data, test_labels = data, target

    ## print(test_data.ravel())
    ## print(test_labels.ravel())

    ##knnModel.setclassifiersefaultK(3)
    ##ret, pred = knnModel.predict(test_data)
    ##opencv_pred = pred.flatten().astype(int)
    ## print(opencv_pred)

    ##According the cvML Methodology:
    ## print("Generic accuracy_score: {:.2f}%".format(accuracy_score(test_labels.ravel(), opencv_pred) *100))
    ## print("Generic precision_score: {:.2f}%".format(precision_score(test_labels.ravel(), opencv_pred) *100))
    ## print("Generic recall_score: {:.2f}%".format(recall_score(test_labels.ravel(), opencv_pred) *100))
    ## Note that these are scikit learn metric functions that work for OpenCV with the help of ravel()
    ## # -------------------------------------------------------------------------------------------------------------------

    ## The Scikit way
    ## -------------------------------------------------------------------------------------------------------------------
    #acc = model.score(test_data, test_labels)
    ## print("k-NN's version of accuracy_score: {:.2f}%".format(acc * 100))
    #scikit_pred = model.predict(test_data)
    ## notice how Scikit stores predictions as a single item separated by space unlike OpenCV
    ## print(scikit_pred)
    ##According the cvML Methodology:
    #print("Generic accuracy_score: {:.2f}%".format(metrics.accuracy_score(test_labels.ravel(), scikit_pred) *100))
    #print("Generic precision_score: {:.2f}%".format(metrics.precision_score(test_labels.ravel(), scikit_pred) *100))
    #print("Generic recall_score: {:.2f}%".format(metrics.recall_score(test_labels.ravel(), scikit_pred) *100))


    #print("MLP \n",y_pred)
