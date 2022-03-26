import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import sparrow as spa
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt


def import_data():
    """
    :return: 数据导入
    """
    data_frame = np.array(pd.read_csv("./data/KC1.csv"))
    data = data_frame[:, :-1]
    # 标准化数据
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)
    target = data_frame[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=1)

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
    return 1 - acc


if __name__ == "__main__":
    SearchAgents_no = 20  # 种群数量
    Max_iteration = 30  # 迭代次数
    dim = 2  # 优化参数的个数
    lb = [10**(-1), 2**(-5)]
    ub = [10**1, 2**4]
    fMin, bestX, SSA_curve = spa.SSA(SearchAgents_no, Max_iteration, lb, ub, dim, fitness_spaction)
    print('c和g为最优值时准确率为：', 1 - fMin)
    print('最优变量c:{0}, g:{1}'.format(bestX[0], bestX[1]))

    thr1 = np.arange(len(SSA_curve[0, :]))
    plt.plot(thr1, SSA_curve[0, :])
    plt.xlabel('num')
    plt.ylabel('object value')
    plt.title('line')
    plt.show()
