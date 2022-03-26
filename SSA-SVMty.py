import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import SSA as spa
import ISSA as spa1


def import_data():
    """
    :return: 数据导入
    """
    #data_frame = np.array(pd.read_csv("./data/NASA/ar1.csv"))
    data_frame = np.array(pd.read_csv("./data/ar1.csv"))
    data = data_frame[:, :-1]
    # 标准化数据
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)
    target = data_frame[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)

    return x_train, x_test, y_train, y_test


def fitness_spaction(parameter):
    """
    :param parameter: SVM参数
    :return: 最小化错误率
    """
    data_train, data_test, label_train, label_test = import_data()
    # SVM参数
    c = parameter[0]
    g = parameter[1]
    # 训练测试
    clf = SVC(gamma=g, C=c)
    clf.fit(data_train, label_train)
    y_predict = clf.predict(data_test)
    acc = accuracy_score(label_test, y_predict)
    p = precision_score(label_test, y_predict)
    r = recall_score(label_test, y_predict)
    f = f1_score(label_test, y_predict)

    return 1-acc


if __name__ == "__main__":

    # SVM预测
    data_train, data_test, label_train, label_test = import_data()
    # 训练测试
    clf = SVC()
    clf.fit(data_train, label_train)
    y_predict = clf.predict(data_test)

    acc = accuracy_score(label_test, y_predict)
    p = precision_score(label_test, y_predict)
    r = recall_score(label_test, y_predict)
    f = f1_score(label_test, y_predict)

    print('SVM acc:{0}'.format(acc))
    print("SVM precision:{0}".format(p))
    print("SVM recall:{0}".format(r))
    print("SVM F-measure:{0}".format(f))
    print("."*20)


    # SSA初始化参数
    SearchAgents_no = 30  # 种群数量
    Max_iteration = 100  # 迭代次数
    dim = 2  # 优化参数的个数
    lb = [10 ** (-1), 2 ** (-5)]
    ub = [10 ** 1, 2 ** 4]

    fMin, bestX, SSA_curve = spa.SSA(SearchAgents_no, Max_iteration, lb, ub, dim, fitness_spaction)
    print('SSA acc:{0}'.format(1-fMin))
    print("c: {0}, g: {1}".format(bestX[0],bestX[1]))
    clf = SVC(gamma=bestX[1], C=bestX[0])
    clf.fit(data_train, label_train)
    y_predict = clf.predict(data_test)
    acc = accuracy_score(label_test, y_predict)
    p = precision_score(label_test, y_predict)
    r = recall_score(label_test, y_predict)
    f = f1_score(label_test, y_predict)
    print("SSA acc:{0}".format(acc))
    print("SSA precision:{0}".format(p))
    print("SSA recall:{0}".format(r))
    print("SSA F-measure:{0}".format(f))
    print("."*20)

    fMin, bestX, ISSA_curve = spa1.ISSA(SearchAgents_no, Max_iteration, lb, ub, dim, fitness_spaction)
    print('ISSA acc:{0}'.format(1-fMin))
    print("c: {0}, g: {1}".format(bestX[0], bestX[1]))
    clf = SVC(gamma=bestX[1], C=bestX[0])
    clf.fit(data_train, label_train)
    y_predict = clf.predict(data_test)
    acc = accuracy_score(label_test, y_predict)
    p = precision_score(label_test, y_predict)
    r = recall_score(label_test, y_predict)
    f = f1_score(label_test, y_predict)
    print("ISSA acc:{0}".format(acc))
    print("ISSA precision:{0}".format(p))
    print("ISSA recall:{0}".format(r))
    print("ISSA F-measure:{0}".format(f))
    # thr1 = np.arange(len(SSA_curve[0, :]))
    # plt.plot(thr1, SSA_curve[0, :])
    # plt.xlabel('num')
    # plt.ylabel('object value')
    # plt.title('line')
    # plt.show()
