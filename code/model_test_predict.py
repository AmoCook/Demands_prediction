import numpy as np
"""
根据time_step_int 进行数据reg与ts的分配，同时根据对预测的结果进行整理

"""
def model_test_predict(model_list, data_test_npa, v_data_test_npa,parameter_set_dic):
    """

    :param model_list: 存储model的数据结构
    :param data_train_npa: 训练集，[[target1,lagn,lag(n-1)...lag1],[target2,lag...]...[targetn]]
    :param data_test_npa: 测试集 形式同训练集一样 都是np.array的数据结构
    :param time_step_int: 时间步，根据此来调整val_reg
    :param val_length_int: 验证集窗口的长度
    :return: output_m_npa: 第一列为ts，其他列为各回归器预测结果的矩阵
    """


    """
    接下来是各个回归器的输出结果
    """
    res_regressor_list = []
    # res_regressor_list.append()....
    for m_index,each_regressor in enumerate(model_list):
        if(m_index in parameter_set_dic['v_model_index']):
            res_regressor_list.append(each_regressor.predict(v_data_test_npa))
        else:
            res_regressor_list.append(each_regressor.predict(data_test_npa))

    #res_regressor_list有M个元素，这个元素是npa，代表每个回归器在val上的预测序列

    # print("检测v数据是否正确!!!!!!!")
    # print('test:', data_test_npa[:, 0])
    # print('v_test:', v_data_test_npa[:, 0])
    output_m_npa = np.array([elem for elem in data_test_npa[:, 0]]).reshape(data_test_npa.shape[0], 1)
    for each_res_npa in res_regressor_list:
        output_m_npa = np.hstack((output_m_npa, each_res_npa.reshape(each_res_npa.shape[0], 1)))



    return output_m_npa
