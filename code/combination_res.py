'''
进行最终的组合，得出预测值
输入：
    1. winner_list 数据结构为list 共有n个elem ， 每个elem是一个npa的vector，代表t时刻的winner下标
    2. test_res_npa 是各个地区各个时间步的结果
    3. output_val_res_list 数据结构为list 共有n个elem ， 每个elem是一个npa，代表t时刻上各个分类器在t时刻对应的val中的输出结果

'''

import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from math import isnan



def combination_res(winner_list, test_res_npa, output_val_res_list):
    """

    :param winner_list: 数据结构为list 共有n个elem ， 每个elem是一个npa的vector，代表t时刻的winner下标
    :param test_res_npa: 是一个各个时间步各个回归器的预测结果
    :param output_val_res_list: 数据结构为list 共有n个elem ， 每个elem是一个npa，代表t时刻上各个分类器在t时刻对应的val中的输出结果
    :return: predict_res list
    """
    predict_res_list = []
    for t_time_step  in range(test_res_npa.shape[0]):
        #在t时刻的操作

        #循环计算loss 采用函数是 rmse
        t_winner_res_npa = output_val_res_list[t_time_step][:,(winner_list[t_time_step]+1)]
        t_rmse_list = []
        for t_winner_res_col_index in range(t_winner_res_npa.shape[1]):
            rmse = sqrt(mean_squared_error(t_winner_res_npa[:, t_winner_res_col_index], output_val_res_list[t_time_step][:, 0]))
            t_rmse_list.append(rmse)

        #对rmse进行归一化

        t_rmse_npa = np.array(t_rmse_list)
        if(t_rmse_npa.shape[0]==1):
            t_com_res = test_res_npa[t_time_step, winner_list[t_time_step][0]]

        elif(max(t_rmse_npa)==min(t_rmse_npa)):

            t_rmse_npa = (t_rmse_npa-min(t_rmse_npa))/(max(t_rmse_npa)-min(t_rmse_npa)+0.00001)

            #组合预测结果
            t_com_res = 0
            for i in range(winner_list[t_time_step].shape[0]):
                t_com_res = t_com_res + (test_res_npa[t_time_step,winner_list[t_time_step][i]]*(1-t_rmse_npa[i]))/sum(1-t_rmse_npa)
        else:
            t_rmse_npa = (t_rmse_npa - min(t_rmse_npa)) / (max(t_rmse_npa) - min(t_rmse_npa))

            # 组合预测结果
            t_com_res = 0
            for i in range(winner_list[t_time_step].shape[0]):
                t_com_res = t_com_res + (
                            test_res_npa[t_time_step, winner_list[t_time_step][i]] * (1 - t_rmse_npa[i])) / sum(
                    1 - t_rmse_npa)

        if(t_com_res < 0):
                t_com_res = 0
        if(isnan(t_com_res)):
            print('sum(1-t_rmse_npa):',sum(1-t_rmse_npa))
            print('fenzi',(test_res_npa[t_time_step,winner_list[t_time_step][i]]*(1-t_rmse_npa[i])))
            print(np.array(t_rmse_list))
            print('max:',max(t_rmse_npa))
            print('min:',min(t_rmse_npa))
            print('winner_list[t_time_step]:',winner_list[t_time_step])
        predict_res_list.append(t_com_res)

    return  predict_res_list