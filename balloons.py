import numpy as np
import tensorflow.keras as tf
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def read_file(file_name):
    print("FILE: " + file_name)
    balloon_list = []
    categories_list = []
    # READING FILE
    balloon_file = open(file_name, "r")
    read_list = balloon_file.readlines()
    # PREPROCESSING
    for line in read_list:
        split_line = line.split(sep=',')
        category = split_line.pop()
        if category == "F\n":
            categories_list.append(0.)
        else:
            categories_list.append(1.)
        balloon_list.append(split_line)
    # ENCODING DATA
    enc = OneHotEncoder()
    numbers_from_categories_table = enc.fit_transform(balloon_list).toarray()
    result_for_classifiers(numbers_from_categories_table, categories_list)
    neural_network(numbers_from_categories_table, categories_list)
    print("_________________________________________________________________________________________________")


def classifiers(clf, clf_name, data, categories, is_neural_network):
    clf_new = clf
    clf_new.fit(data, categories)
    predictions = clf_new.predict(data)
    result = np.array([1 if predictions[i] == categories[i] else 0 for i in range(len(predictions))]).sum() / float(
        len(predictions))
    if is_neural_network:
        print("Result for neural network " + clf_name + " " + str(result))
    else:
        print("Result for classifier " + clf_name + " " + str(result))


def result_for_classifiers(data, categories_list):
    # NAIVE BAYES
    classifiers(MultinomialNB(alpha=0.05), "Naive Bayes", data, categories_list, False)
    # COMPLEMENT NAIVE BAYES
    classifiers(ComplementNB(alpha=0.05), "Complement Naive Bayes", data, categories_list, False)
    # GAUSSIAN NAIVE BAYES
    classifiers(GaussianNB(), "Gaussian Naive Bayes", data, categories_list, False)
    # RANDOM FOREST
    classifiers(RandomForestClassifier(), "Random Forest", data, categories_list, False)
    # ADABOOST
    classifiers(AdaBoostClassifier(), "AdaBoost", data, categories_list, False)
    # KNN
    classifiers(KNeighborsClassifier(), "KNN", data, categories_list, False)
    # SVM
    classifiers(SVC(), "SVM", data, categories_list, False)
    # DECISION TREES
    classifiers(DecisionTreeClassifier(), "Decision Trees", data, categories_list, False)
    # NEURAL NETWORK
    classifiers(MLPClassifier(hidden_layer_sizes=2, random_state=0), "Multilayer Perceptron",
                data, categories_list, True)


def neural_network(data, categories_list):
    # NEURAL NETWORK - TENSORFLOW
    model = tf.Sequential([
        tf.layers.Dense(32, input_shape=(8,)),
        tf.layers.Activation('relu'),
        tf.layers.Dense(10),
        tf.layers.Activation('softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(np.asarray(data), np.asarray(categories_list), epochs=50)
    test_loss, test_acc = model.evaluate(np.asarray(data), np.asarray(categories_list), verbose=2)
    print('\nTest accuracy:', test_acc)


read_file("data/yellow-small.data")
read_file("data/adult+stretch.data")
read_file("data/adult-stretch.data")
read_file("data/yellow-small+adult-stretch.data")
