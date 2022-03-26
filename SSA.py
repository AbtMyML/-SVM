import numpy as np
import math
import random


def Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[0, i]:
            temp[i] = Lb[0, i]
        elif temp[i] > Ub[0, i]:
            temp[i] = Ub[0, i]

    return temp


def SSA(pop, M, c, d, dim, fun):
    """
    :param fun: 适应度函数
    :param pop: 种群数量
    :param M: 迭代次数
    :param c: 迭代范围下界
    :param d: 迭代范围上界
    :param dim: 优化参数的个数
    :return: 适应度值最小的值 对应得位置
    """
    P_percent = 0.2
    pNum = round(pop*P_percent)
    lb = c*np.ones((1, dim))
    ub = d*np.ones((1, dim))
    X = np.zeros((pop, dim))
    fit = np.zeros((pop, 1))

    for i in range(pop):
        X[i, :] = lb+(ub-lb)*np.random.rand(1, dim)
        fit[i, 0] = fun(X[i, :])
    pFit = fit
    pX = X
    fMin = np.min(fit[:, 0])
    bestI = np.argmin(fit[:, 0])
    bestX = X[bestI, :]
    Convergence_curve = np.zeros((1, M))
    for t in range(M):
        sortIndex = np.argsort(pFit.T)
        fmax = np.max(pFit[:, 0])
        B = np.argmax(pFit[:, 0])
        worse = X[B, :]
        # 发现者位置更新
        r2 = np.random.rand(1)
        if r2 < 0.8:
            for i in range(pNum):
                r1=np.random.rand(1)
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :]*np.exp(-(i)/(r1*M))
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
                fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :])
        elif r2 >= 0.8:
            for i in range(pNum):
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :]+np.random.rand(1)*np.ones((1, dim))
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
                fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :])
        bestII = np.argmin(fit[:, 0])
        bestXX = X[bestII, :]
        # 加入者位置更新
        for ii in range(pop-pNum):
            i = ii+pNum
            A = np.floor(np.random.rand(1, dim)*2)*2-1
            if i > pop/2:
                X[sortIndex[0, i], :] = np.random.rand(1)*np.exp(worse-pX[sortIndex[0, i], :]/np.square(i))
            else:
                X[sortIndex[0, i], :] = bestXX+np.dot(np.abs(pX[sortIndex[0, i], :]-bestXX), 1/(A.T*np.dot(A, A.T)))*np.ones((1, dim))
            X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
            fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :])
        arrc = np.arange(len(sortIndex[0, :]))
        # c=np.random.shuffle(arrc)
        # 意识到危险得麻雀位置更新
        c = np.random.permutation(arrc)
        b = sortIndex[0, c[0:20]]
        for j in range(len(b)):
            if pFit[sortIndex[0, b[j]], 0] > fMin:
                X[sortIndex[0, b[j]], :] = bestX+np.random.rand(1, dim)*np.abs(pX[sortIndex[0, b[j]], :]-bestX)
            else:
                X[sortIndex[0, b[j]], :] = pX[sortIndex[0, b[j]], :]+(2*np.random.rand(1)-1)*np.abs(pX[sortIndex[0, b[j]], :]-worse)/(pFit[sortIndex[0, b[j]]]-fmax+10**(-50))
            X[sortIndex[0, b[j]], :] = Bounds(X[sortIndex[0, b[j]], :], lb, ub)
            fit[sortIndex[0, b[j]], 0] = fun(X[sortIndex[0, b[j]]])
        for i in range(pop):
            if fit[i, 0] < pFit[i, 0]:
                pFit[i, 0] = fit[i, 0]
                pX[i, :] = X[i, :]
            if pFit[i, 0] < fMin:
                fMin = pFit[i, 0]
                bestX = pX[i, :]
        Convergence_curve[0, t] = fMin
    return fMin, bestX, Convergence_curve

