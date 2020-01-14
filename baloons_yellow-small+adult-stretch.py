import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ballonList = []
categoriesList = []
# READING FILE
ballonFile = open("data/yellow-small+adult-stretch.data", "r")
readlist = ballonFile.readlines()
# PREPROCESSING
for line in readlist:
    splitLine = line.split(sep=',')
    category = splitLine.pop()
    if(category == "F\n"):
        categoriesList.append(0.)
    else:
        categoriesList.append(1.)
    ballonList.append(splitLine)
# ENDODING DATA
enc = OneHotEncoder()
numbersFromCategoriesTable = enc.fit_transform(ballonList).toarray()
# NAIVE BAYES
clfNB = MultinomialNB(alpha=0.05)
clfNB.fit(numbersFromCategoriesTable, categoriesList)
predictionsNB = clfNB.predict(numbersFromCategoriesTable)
resultNB = np.array([1 if predictionsNB[i] == categoriesList[i] else 0 for i in range(len(predictionsNB))]).sum()/float(len(predictionsNB))
print("Result for classifier Naive Bayes " + str(resultNB))
# COMPLEMENT NAIVE BAYES
clfCNB = ComplementNB(alpha=0.05)
clfCNB.fit(numbersFromCategoriesTable, categoriesList)
predictionsCNB = clfCNB.predict(numbersFromCategoriesTable)
resultCNB = np.array([1 if predictionsCNB[i] == categoriesList[i] else 0 for i in range(len(predictionsCNB))]).sum()/float(len(predictionsCNB))
print("Result for classifier Complement Naive Bayes " + str(resultCNB))
# GAUSSIAN NAIVE BAYES
clfGNB = GaussianNB()
clfGNB.fit(numbersFromCategoriesTable, categoriesList)
predictionsGNB = clfGNB.predict(numbersFromCategoriesTable)
resultGNB = np.array([1 if predictionsGNB[i] == categoriesList[i] else 0 for i in range(len(predictionsGNB))]).sum()/float(len(predictionsGNB))
print("Result for classifier Gaussian Naive Bayes " + str(resultGNB))
# RANDOM FOREST
clfRF = RandomForestClassifier()
clfRF.fit(numbersFromCategoriesTable, categoriesList)
predictionsRF = clfRF.predict(numbersFromCategoriesTable)
resultRF = np.array([1 if predictionsRF[i] == categoriesList[i] else 0 for i in range(len(predictionsRF))]).sum()/float(len(predictionsRF))
print("Result for classifier Random Forest " + str(resultRF))
# ADABOOST
clfAB = AdaBoostClassifier()
clfAB.fit(numbersFromCategoriesTable, categoriesList)
predictionsAB = clfAB.predict(numbersFromCategoriesTable)
resultAB = np.array([1 if predictionsAB[i] == categoriesList[i] else 0 for i in range(len(predictionsAB))]).sum()/float(len(predictionsAB))
print("Result for classifier AdaBoost " + str(resultAB))
# KNN
clfKNN = KNeighborsClassifier()
clfKNN.fit(numbersFromCategoriesTable, categoriesList)
predictionsKNN = clfKNN.predict(numbersFromCategoriesTable)
resultKNN = np.array([1 if predictionsKNN[i] == categoriesList[i] else 0 for i in range(len(predictionsKNN))]).sum()/float(len(predictionsKNN))
print("Result for classifier KNN " + str(resultKNN))
# SVM
clfSVM = SVC()
clfSVM.fit(numbersFromCategoriesTable, categoriesList)
predictionsSVM = clfSVM.predict(numbersFromCategoriesTable)
resultSVM = np.array([1 if predictionsSVM[i] == categoriesList[i] else 0 for i in range(len(predictionsSVM))]).sum()/float(len(predictionsSVM))
print("Result for classifier SVM " + str(resultSVM))
# DECISION TREES
clfDT = DecisionTreeClassifier()
clfDT.fit(numbersFromCategoriesTable, categoriesList)
predictionsDT = clfDT.predict(numbersFromCategoriesTable)
resultDT = np.array([1 if predictionsDT[i] == categoriesList[i] else 0 for i in range(len(predictionsDT))]).sum()/float(len(predictionsDT))
print("Result for classifier Decision Trees " + str(resultDT))
# SIEC NEURONOWA
clfMP = MLPClassifier(hidden_layer_sizes=2, random_state=0)
clfMP.fit(numbersFromCategoriesTable, categoriesList)
predictionsMP = clfMP.predict(numbersFromCategoriesTable)
resultMP = np.array([1 if predictionsMP[i] == categoriesList[i] else 0 for i in range(len(predictionsMP))]).sum()/float(len(predictionsMP))
print("Result for neural network Multilayer Perceptron " + str(resultMP))