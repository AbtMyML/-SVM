from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import pandas as pd

# def loadIris():
#     iris = datasets.load_iris()
#     Data = iris.data
#     Label = iris.target
#     np.random.seed(0)
#     indices = np.random.permutation(len(Data))
#     DataTrain = Data[indices[:-10]]
#     LabelTrain = Label[indices[:-10]]
#     DataTest = Data[indices[:-10]]
#     LabelTest = Label[indices[:-10]]
#     return DataTrain, LabelTrain, DataTest, LabelTest



def import_data():
    """
    :return: 数据导入
    """
    #data_frame = np.array(pd.read_csv("./data/NASA/MW1.csv"))
    data_frame = np.array(pd.read_csv("./data/ar1.csv"))
    data = data_frame[:, :-1]
    # 标准化数据
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)
    #indices = np.random.permutation(len(data))
    label = data_frame[:, -1]
    data_train, label_train, data_test, label_test = train_test_split(data, label, test_size=0.2, random_state=1)
    return data_train, label_train, data_test, label_test

def calPrecision(prediction, truth):
    numSamples = len(prediction)
    #print(prediction.shape)
    numCorrect = 0
    for k in range(0, numSamples):
        if prediction[k] == truth[k]:
            numCorrect += 1
    p = float(numCorrect) / float(numSamples)
    return p


def main():
    data_train, label_train, data_test, label_test = import_data()
    clf = SVC(kernel='rbf')
    clf.fit(data_train, label_train)
    prediction_label = clf.predict(data_test)
    p = calPrecision(prediction_label, label_test)
    #print(p)
    print(data_test.shape)

    print(data_train)

main()
