import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

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
print("Wynik klasyfikatora Naive Bayes " + str(resultNB))
# COMPLEMENT NAIVE BAYES
clfCNB = ComplementNB(alpha=0.05)
clfCNB.fit(numbersFromCategoriesTable, categoriesList)
predictionsCNB = clfCNB.predict(numbersFromCategoriesTable)
resultCNB = np.array([1 if predictionsCNB[i] == categoriesList[i] else 0 for i in range(len(predictionsCNB))]).sum()/float(len(predictionsCNB))
print("Wynik klasyfikatora Complement Naive Bayes " + str(resultCNB))
# GAUSSIAN NAIVE BAYES
clfGNB = GaussianNB()
clfGNB.fit(numbersFromCategoriesTable, categoriesList)
predictionsGNB = clfGNB.predict(numbersFromCategoriesTable)
resultGNB = np.array([1 if predictionsGNB[i] == categoriesList[i] else 0 for i in range(len(predictionsGNB))]).sum()/float(len(predictionsGNB))
print("Wynik klasyfikatora Gaussian Naive Bayes " + str(resultGNB))
# RANDOM FOREST
clfRF = RandomForestClassifier()
clfRF.fit(numbersFromCategoriesTable, categoriesList)
predictionsRF = clfRF.predict(numbersFromCategoriesTable)
resultRF = np.array([1 if predictionsRF[i] == categoriesList[i] else 0 for i in range(len(predictionsRF))]).sum()/float(len(predictionsRF))
print("Wynik klasyfikatora Random Forest " + str(resultRF))
# SIEC NEURONOWA
clfMP = MLPClassifier(hidden_layer_sizes=2, random_state=0)
clfMP.fit(numbersFromCategoriesTable, categoriesList)
predictionsMP = clfMP.predict(numbersFromCategoriesTable)
resultMP = np.array([1 if predictionsMP[i] == categoriesList[i] else 0 for i in range(len(predictionsMP))]).sum()/float(len(predictionsMP))
print("Wynik sieci neuronowej Multilayer Perceptron " + str(resultMP))