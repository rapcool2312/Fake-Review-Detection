import numpy as np
import pandas as pd
import joblib
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('reviews1.tsv',sep="\t")
# print(df.head())

not_fakeDf=df[df['label'] == "not_fake"]
fakeDf=df[df['label'] == "fake"]

# print(not_fakeDf)
# print(fakeDf)

not_fakeDf = not_fakeDf.sample(fakeDf.shape[0])

print(not_fakeDf.shape)
print(fakeDf.shape)

finalDf = not_fakeDf.append(fakeDf, ignore_index = True)

print(finalDf.shape)

X_train, X_test,Y_train,Y_test = train_test_split(finalDf['message'], finalDf['label'], test_size = 0.2, random_state = 0, shuffle = True, stratify = finalDf['label'])

# Pipeline

# model = Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier(n_estimators=100, n_jobs = -1))])

model = Pipeline([('tfidf', TfidfVectorizer()), ('model',SVC(C = 1000, gamma= 'auto'))])

model.fit(X_train, Y_train)

Y_predict = model.predict(X_test)

print(accuracy_score(Y_test,Y_predict))

# print(confusion_matrix(Y_test, Y_predict))
# print(classification_report(Y_test, Y_predict))
# print(accuracy_score(Y_test, Y_predict))

print(model.predict(["very bad product"]))

# joblib.dump(model, "myModel2.pkl")

joblib.dump(model, "mySVCModel1.pkl")



