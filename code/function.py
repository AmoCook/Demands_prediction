"""
该部分包含一些基回归器的设定
"""

def TVPP(history_data,interval_in_one_hours,season_with_hours,alpha):
    iioh = interval_in_one_hours
    swh = season_with_hours
    history_len = len(history_data)
    back_steps = swh*iioh
    theta_list = []
    for i in range(history_len-1,-1,-int(back_steps)):
        theta_list.append(history_data[i])
    weight = []
    for i in range(len(theta_list)):
        weight.append(1*(1-alpha)**i)
    sum_alpha = sum(weight)

    theta = 0
    for i in range(len(theta_list)):
        theta += theta_list[i]*weight[i]/sum_alpha
    return theta


from sklearn import tree, linear_model, svm
import numpy as np
import math


#决策树

class DTR(object):
    def __init__(self, lag_num):
        self.lag_num = lag_num
        self.y_hat = np.NAN
        self.y = np.NAN
        self.model_DecisionTreeRegressor = tree.DecisionTreeRegressor()

    def train(self, train_reg):
        self.model_DecisionTreeRegressor.fit(train_reg[:, 1:self.lag_num+1], train_reg[:, 0])

    def predict(self, test_reg):
        if(test_reg.shape[1] < self.lag_num):
            print("test 数据提供的lag量，小于模型需要的lag量")

        self.y_hat = self.model_DecisionTreeRegressor.predict(test_reg[:, 1:self.lag_num+1])
        self.y = test_reg[:, 0]
        return self.y_hat

    def evaluate_rmse(self):
        if(math.isnan(self.y_hat)):
            print("Error: 该模型还没有预测值, 无法对无结果的预测值进行评估！！！")
        elif(math.isnan(self.y)):
            print("Error: 该模型还没有targets数据, 无法对无结果的预测值进行评估！！！")
        else:
            #计算标准
            pass


#线性模型

class LR(object):
    def __init__(self, lag_num):
        self.lag_num = lag_num
        self.y_hat = np.NAN
        self.y = np.NAN
        self.model_LinearRegression = linear_model.LinearRegression()

    def train(self, train_reg):
        self.model_LinearRegression.fit(train_reg[:, 1:self.lag_num+1], train_reg[:, 0])

    def predict(self, test_reg):
        if (test_reg.shape[1] < self.lag_num):
            print("test 数据提供的lag量，小于模型需要的lag量")

        self.y_hat = self.model_LinearRegression.predict(test_reg[:, 1:self.lag_num + 1])
        self.y = test_reg[:, 0]
        return self.y_hat

    def evaluate_rmse(self):
        if (math.isnan(self.y_hat)):
            print("Error: 该模型还没有预测值, 无法对无结果的预测值进行评估！！！")
        elif (math.isnan(self.y)):
            print("Error: 该模型还没有targets数据, 无法对无结果的预测值进行评估！！！")
        else:
            # 计算标准
            pass

#svr
class SVR(object):
    def __init__(self, lag_num, kernel_index, degree=3):
        self.lag_num = lag_num
        self.y_hat = np.NAN
        self.y = np.NAN
        self.kernel_index = kernel_index
        self.degree = degree
        self.kernel_list = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        if(kernel_index == 1):
            self.model_SVR = svm.SVR(kernel=self.kernel_list[kernel_index], degree=self.degree)
        else:
            self.model_SVR = svm.SVR(kernel=self.kernel_list[kernel_index])

    def train(self, train_reg):
        self.model_SVR.fit(train_reg[:, 1:self.lag_num+1], train_reg[:, 0])

    def predict(self, test_reg):
        if (test_reg.shape[1] < self.lag_num):
            print("test 数据提供的lag量，小于模型需要的lag量")

        self.y_hat = self.model_SVR.predict(test_reg[:, 1:self.lag_num + 1])
        self.y = test_reg[:, 0]
        return self.y_hat

    def evaluate_rmse(self):
        if (math.isnan(self.y_hat)):
            print("Error: 该模型还没有预测值, 无法对无结果的预测值进行评估！！！")
        elif (math.isnan(self.y)):
            print("Error: 该模型还没有targets数据, 无法对无结果的预测值进行评估！！！")
        else:
            # 计算标准
            pass

#随机森林
from sklearn import ensemble

class RF(object):
    def __init__(self, lag_num, tree_num):
        self.lag_num = lag_num
        self.y_hat = np.NAN
        self.y = np.NAN
        self.tree_num = tree_num
        self.model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=tree_num)

    def train(self, train_reg):
        self.model_RandomForestRegressor.fit(train_reg[:, 1:self.lag_num+1], train_reg[:, 0])

    def predict(self, test_reg):
        if (test_reg.shape[1] < self.lag_num):
            print("test 数据提供的lag量，小于模型需要的lag量")

        self.y_hat = self.model_RandomForestRegressor.predict(test_reg[:, 1:self.lag_num + 1])
        self.y = test_reg[:, 0]
        return self.y_hat

    def evaluate_rmse(self):
        if (math.isnan(self.y_hat)):
            print("Error: 该模型还没有预测值, 无法对无结果的预测值进行评估！！！")
        elif (math.isnan(self.y)):
            print("Error: 该模型还没有targets数据, 无法对无结果的预测值进行评估！！！")
        else:
            # 计算标准
            pass

#adboost
class ADB(object):
    def __init__(self, lag_num, tree_num):
        self.lag_num = lag_num
        self.y_hat = np.NAN
        self.y = np.NAN
        self.tree_num = tree_num
        self.model_AdaBoostRegressor = ensemble.model_AdaBoostRegressor(n_estimators=tree_num)

    def train(self, train_reg):
        self.model_AdaBoostRegressor.fit(train_reg[:, 1:self.lag_num+1], train_reg[:, 0])

    def predict(self, test_reg):
        if (test_reg.shape[1] < self.lag_num):
            print("test 数据提供的lag量，小于模型需要的lag量")

        self.y_hat = self.model_AdaBoostRegressor.predict(test_reg[:, 1:self.lag_num + 1])
        self.y = test_reg[:, 0]
        return self.y_hat

    def evaluate_rmse(self):
        if (math.isnan(self.y_hat)):
            print("Error: 该模型还没有预测值, 无法对无结果的预测值进行评估！！！")
        elif (math.isnan(self.y)):
            print("Error: 该模型还没有targets数据, 无法对无结果的预测值进行评估！！！")
        else:
            # 计算标准
            pass










