import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

data = pd.read_excel("Dry_Bean_Dataset.xlsx")

labelEncoder = preprocessing.LabelEncoder()

x = (data[["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4", "Class"]])

cls = labelEncoder.fit_transform(list(data["Class"]))

y = list(cls)

x = x.drop("Class", axis=1)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(x, y)

predictions = knn_model.predict(x)

beanTypes = ["Seker", "Barbunya", "Bombay", "Cali", "Dermosan", "Horoz", " Sira"]

timesWrong = 0

for i in range(len(x)):
    actual = beanTypes[cls[i]]
    predicted = beanTypes[predictions[i]]\

    print("Actual: ", actual, ". Predicted: ", predicted)

    if(actual != predicted):
        timesWrong += 1


print("Times Computer was wrong: ", timesWrong)