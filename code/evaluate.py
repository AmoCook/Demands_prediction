import pandas as pd
import numpy as np
from math import sqrt
from datetime import datetime


def current_time():
    year = str(datetime.now().year)
    month = str(datetime.now().month)
    day = str(datetime.now().day)

    hour = str(datetime.now().hour)
    minu = str(datetime.now().minute)
    second = str(datetime.now().second)

    return year+'/'+ month+'/'+day+'  '+hour+':'+minu+':'+second


def nrmse(y_hat, y):
    return sqrt(sum((y_hat-y)**2)/y.shape[0])/(y.mean())


def rmse(y_hat, y):
    return sqrt(sum((y_hat-y)**2)/y.shape[0])


def evaluate(parameter_set_dic, instance_file_path, test_target_file_name, test_res_file_name):

    """进行test与res数据的读取"""
    y_hat_set = pd.read_csv(instance_file_path + test_res_file_name)

    y_set = pd.read_csv(instance_file_path + test_target_file_name)

    """计算相关的结果:如 nrmse, rmse"""

    '''计算nrmse'''
    nrmse_res = []
    for i in range(y_set.shape[0]):
        nrmse_res.append(nrmse(y_hat_set.values[i, :], y_set.values[i, :]))

    '''计算rmse'''
    rmse_res = []

    for i in range(y_set.shape[0]):
        rmse_res.append(rmse(y_hat_set.values[i, :], y_set.values[i, :]))


    """统计相关数据： 1 平均nmres_mean 2 低于设定nrmse的个数 nrmse_low 3 设定值来自para..dic['nrmse_stander_float'] """
    nrmse_res = np.array(nrmse_res)
    nrmse_mean = nrmse_res.mean()

    rmse_res = np.array(rmse_res)
    rmse_mean = rmse_res.mean()

    nrmse_low = 0
    for i in nrmse_res:
        if i < parameter_set_dic['nrmse_stander_float']:
            nrmse_low = nrmse_low + 1

    import matplotlib.pyplot as plt

    plt.figure()

    '''画出nrmse线'''
    plt.plot(nrmse_res, color='b', label='nrmse')

    '''画出nrmse_stander_line'''
    plt.plot([i for i in range(len(nrmse_res))], [parameter_set_dic['nrmse_stander_float'] for i in range(len(nrmse_res))], alpha=0.4, color='green')

    '''标出超越阈值stander的部分'''
    # print('max ', int(max(nrmse_res)))
    # for x, values in enumerate(np.array(nrmse_res)):
    #     if values > parameter_set_dic['nrmse_stander_float']:
    #         plt.plot([x for i in range(int(max(nrmse_res)))], [i for i in range(int(max(nrmse_res)))], alpha=0.3, color='orange')

    '''画出target对应的值'''
    plt.plot([y_set.values[i, :].mean() for i in range(y_set.shape[0])], color='r', label='test_target', alpha=0.4)
    plt.legend()
    plt.xlabel('time interval 30 mins')
    plt.ylabel('num')
    plt.title(parameter_set_dic['date']+'_instance_'+parameter_set_dic['instance_id']+'  mean_nrmse: '+str(nrmse_mean)[0:4]+' low_num_rate: '+str(nrmse_low/nrmse_res.shape[0])[0:4])
    plt.savefig(instance_file_path+parameter_set_dic['nrmse_fig_name'])
    # plt.show()

    """
    编写log文件，编写两个log文件:
        1.是在para..['output_file_path']目录下的log_[date].txt: 记录这一天跑的实验实例的目的信息
        2.是在instance_id_[instance_id]目录下的log_instance_[instance].txt: 记录本次实验的详细信息
    """

    '''写入log_[date].txt文件'''
    log_date_file = open(parameter_set_dic['output_file_path']+'log_'+parameter_set_dic['date']+'.txt', 'a')
    log_date_file.writelines('==============='+'instance_'+parameter_set_dic['instance_id']+'================\n')
    log_date_file.writelines(current_time()+'\n')
    log_date_file.writelines(parameter_set_dic['info']+'\n')
    log_date_file.writelines('mean_nrmse: ' + str(nrmse_mean)[0:4] + ' and rate of lower 0.8 : ' + str(nrmse_low / nrmse_res.shape[0])[0:4]+'\n')
    log_date_file.writelines('\n')
    log_date_file.close()

    '''写入log_[date].txt文件'''
    log_instance_file = open(instance_file_path+'log_instance_'+parameter_set_dic['instance_id']+'.txt', 'a')
    log_instance_file.writelines(current_time() + '\n')
    for key in parameter_set_dic:
        log_instance_file.writelines(str(key)+': '+str(parameter_set_dic[key])+'\n')

    log_instance_file.writelines('mean_nrmse: ' + str(nrmse_mean)[0:4] + ' and rate of lower 0.8 : ' + str(nrmse_low / nrmse_res.shape[0])[0:4] + 'mean_rmse: ' + str(rmse_mean) + '\n')
    log_instance_file.writelines('\n')

    log_instance_file.close()

    """生成分析文件analysis_file_name"""
    analysis_res_pd = pd.DataFrame({
        'nrmse_res': nrmse_res,
        'rmse': rmse_res
    })

    analysis_res_pd.to_csv(instance_file_path+parameter_set_dic['analysis_file_name'], index=False)



