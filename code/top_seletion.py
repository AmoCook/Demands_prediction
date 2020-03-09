from scipy.stats import pearsonr
import math
import numpy as np
#利用SRC进行top selection

from model_val_predict import model_val_predict
#进行模型预测的模块
'''
输入：
    1.在移动窗口获取的各回归器的结果，res{'base_regressor_0':[x1,x2...xw],...}字典 npa
    2.移动窗口的真实数据，Xw:[x1,x2...xw]
返回：
    1.包含SRC排序的list对象res_sorted[
    [top0_SRC_value,top0_name],[top1_SRC_value,top1_name],...[]
    ]
'''

def takekey(elem):
    return elem[0]


# def caculate_SRC_and_sort(res,Xw):
#     res_sorted_list = []
#     '''
#     step1: caculate SRC
#     '''
#
#     for regressor_name,regressor_res in res.items():
#
#         #对每一个回归器的结果进行SRC计算
#         tao = sum([ regressor_res[i]*Xw[i] for i in range(regressor_res.shape[0])])
#
#         corr = (tao-((sum(Xw)*sum(regressor_res))/(Xw.shape[0])))/(((sum(regressor_res*regressor_res)-(sum(regressor_res)**2)/(Xw.shape[0]))**0.5)*(((sum(Xw**2))-((sum(Xw))**2/Xw.shape[0]))**0.5))
#
#         SRC = ((1-corr)/2)**0.5
#
#         res_sorted_list.append([SRC,regressor_name])
#
#     #利用存储的SRC的值对所有回归器的结果进行排序 ； 当预测越精准时，SRC的值越接近于0，所以排序选择升序
#     res_sorted_list.sort(key=takekey, reverse=False)
#
#     return res_sorted_list
def caculate_SRC(x,y):
    c = pearsonr(x,y)[0]
    rn = math.sqrt((1-c)/2)
    return rn


def top_k_selection(model_list, data_train_npa, data_test_npa, val_length_int, lim_float, tp1_int,parameter_set_dic, v_train_npa, v_test_npa):
    """

    :param model_list: 存储model的数据结构
    :param data_train_npa: 训练集，[[target1,lagn,lag(n-1)...lag1],[target2,lag...]...[targetn]]
    :param data_test_npa: 测试集 形式同训练集一样 都是np.array的数据结构
    :param val_length_int: 验证集窗口的长度
    :param lim_float: 针对第一轮选择出来的模型，利用lim对比各个模型的SRC值进行挑选
    :param tp1_int: 对验证集的结果与真实数据的皮尔逊系数高低，进行个数的挑选
    :return: model_select_index_list: 挑选出来的模型的下标 [[time_step_t_index_list]....]
            alarm_record_list: drift 发生改变的记录 按照每个t时间步进行记录
    """
    time_step_int = 0

    alarm_record_list = []

    model_select_index_list = []

    output_val_res_list = [] #记录time_step时刻窗口所有回归器的输出值

    output_m_npa = model_val_predict(model_list, data_train_npa, data_test_npa, time_step_int, val_length_int,v_train_npa, v_test_npa,parameter_set_dic)


    output_val_res_list.append(output_m_npa)
    '''
    output_m_npa 是一个array,size为 w_size * (1+Num_models) 第一列为target
    '''
    pearsonr_record_list = []
    # 计算target与每个输出的皮尔逊系数
    for each_res_index in range(output_m_npa.shape[1]):
        if(each_res_index == 0):
            continue
        else:
            pearsonr_record_list.append([abs(pearsonr(output_m_npa[:, 0], output_m_npa[:, each_res_index])[0]), each_res_index-1])#！！！这里的减1，使得后边得到的下标就是model的全局下标

    #排序
    pearsonr_record_list.sort(key=takekey, reverse=True)

    # model_sel 是一个list 里面的元素也是一个list ，elem = [pearsonr_value,model_index] ，取了先tp1个模型

    model_sel_value_and_index = pearsonr_record_list[0:tp1_int]

    model_sel_pearson_list = [elem[1] for elem in model_sel_value_and_index]

    model_sel_index = []

    #通过lim_float参数对选出来的数据进行二次筛选
    for each_res in model_sel_value_and_index:
        if(each_res[0] > lim_float):
            model_sel_index.append(each_res[1])

    if(len(model_sel_index)==0):
        model_sel_index.append(model_sel_value_and_index[0][1])

    #计算挑选出来的模型的结果与target的SRC值，然后计算最小值

    min_d = min([caculate_SRC(output_m_npa[:, 0], output_m_npa[:, index+1]) for index in model_sel_pearson_list])

    updated_sel_list = []
    updated_sel_list.append(model_sel_index)

    alarm_record_list.append(0)
    time_step_int = time_step_int + 1

    while(time_step_int < data_test_npa.shape[0]):

        output_m1_npa = model_val_predict(model_list, data_train_npa, data_test_npa, time_step_int, val_length_int,v_train_npa, v_test_npa,parameter_set_dic)

        output_val_res_list.append(output_m1_npa)
        pearsonr_record1_list = []
        # 计算target与每个输出的皮尔逊系数
        for each_res_index in range(output_m1_npa.shape[1]):
            if (each_res_index == 0):
                continue
            else:
                pearsonr_record1_list.append([abs(pearsonr(output_m1_npa[:, 0], output_m1_npa[:, each_res_index])[0]), each_res_index - 1])

        pearsonr_record1_list.sort(key=takekey, reverse=True)

        model_sel_value_and_index1 = pearsonr_record1_list[0:tp1_int]

        model_sel_pearson_list1 = [elem[1] for elem in model_sel_value_and_index1]


        l_cor = min([caculate_SRC(output_m1_npa[:, 0], output_m1_npa[:, index + 1]) for index in model_sel_pearson_list1])

        dd = min_d - l_cor

        if(math.isnan(dd)):
            dd = 0

        if(abs(dd) > math.sqrt(math.log(1/0.95)/(2*output_m1_npa.shape[0]))):


            min_d = l_cor
            #统计新的selection 下标
            model_sel_index1 = []

            # 通过lim_float参数对选出来的数据进行二次筛选
            for each_res in model_sel_value_and_index1:
                if (each_res[0] > lim_float):
                    model_sel_index1.append(each_res[1])

            if (len(model_sel_index1) == 0):
                model_sel_index1.append(model_sel_value_and_index1[0][1])

            model_sel_index = model_sel_index1

            updated_sel_list.append(model_sel_index)
            alarm_record_list.append(1)

        else:
            updated_sel_list.append(model_sel_index)
            alarm_record_list.append(0)

        time_step_int = time_step_int + 1
    model_select_index_list = updated_sel_list
    return model_select_index_list, alarm_record_list, output_val_res_list