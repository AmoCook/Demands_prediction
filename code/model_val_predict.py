import numpy as np
"""
根据time_step_int 进行数据reg与ts的分配，同时根据对预测的结果进行整理

"""
def model_val_predict(model_list, data_train_npa, data_test_npa, time_step_int, val_length_int, v_train_npa, v_test_npa,parameter_set_dic):
    """

    :param model_list: 存储model的数据结构
    :param data_train_npa: 训练集，[[target1,lagn,lag(n-1)...lag1],[target2,lag...]...[targetn]]
    :param data_test_npa: 测试集 形式同训练集一样 都是np.array的数据结构
    :param time_step_int: 时间步，根据此来调整val_reg
    :param val_length_int: 验证集窗口的长度
    :return: output_m_npa: 第一列为ts，其他列为各回归器预测结果的矩阵
    """

    num_train_int = data_train_npa.shape[0]
    data_val_npa = data_train_npa[num_train_int - val_length_int:num_train_int, ]
    data_train_true_npa = data_train_npa[0:(num_train_int-val_length_int), ]

    #根据time_step_int 计算真实的验证集 ts_npa , data_val_true_npa
    if(time_step_int==0):
        ts_npa = data_val_npa[:,0]
        data_val_true_npa = data_val_npa
    elif(time_step_int >= val_length_int):
        ts_npa = data_test_npa[time_step_int-val_length_int:time_step_int, 0]
        data_val_true_npa = data_test_npa[time_step_int-val_length_int:time_step_int, ]
    else:
        ts_npa = np.hstack((data_val_npa[time_step_int:val_length_int, 0], data_test_npa[0:time_step_int, 0]))
        data_val_true_npa = np.vstack((data_val_npa[time_step_int:val_length_int, ], data_test_npa[0:time_step_int, ]))


    """===================v_data===================="""

    v_num_train_int = v_train_npa.shape[0]
    v_data_val_npa = v_train_npa[v_num_train_int - val_length_int:v_num_train_int, ]
    v_data_train_true_npa = v_train_npa[0:(v_num_train_int - val_length_int), ]

    # 根据time_step_int 计算真实的验证集 ts_npa , data_val_true_npa
    if(parameter_set_dic['related_region_num_int']!=0):
        if (time_step_int == 0):
            # print(v_data_val_npa.shape)
            # print(v_data_val_npa)
            v_ts_npa = v_data_val_npa[:, 0]
            v_data_val_true_npa = v_data_val_npa
        elif (time_step_int >= val_length_int):
            v_ts_npa = v_test_npa[time_step_int - val_length_int:time_step_int, 0]
            v_data_val_true_npa = v_test_npa[time_step_int - val_length_int:time_step_int, ]
        else:
            v_ts_npa = np.hstack((v_data_val_npa[time_step_int:val_length_int, 0], v_test_npa[0:time_step_int, 0]))
            v_data_val_true_npa = np.vstack((v_data_val_npa[time_step_int:val_length_int, ], v_test_npa[0:time_step_int, ]))





    """===================v_data===================="""

    """
    接下来是各个回归器的输出结果
    """
    res_regressor_list = []
    # res_regressor_list.append()....
    for m_index,each_regressor in enumerate(model_list):
        if(m_index in parameter_set_dic['v_model_index']):
            res_regressor_list.append(each_regressor.predict(v_data_val_true_npa))
        else:
            res_regressor_list.append(each_regressor.predict(data_val_true_npa))

    #res_regressor_list有M个元素，这个元素是npa，代表每个回归器在val上的预测序列


    output_m_npa = np.array([elem for elem in ts_npa]).reshape(val_length_int, 1)
    for each_res_npa in res_regressor_list:
        output_m_npa = np.hstack((output_m_npa, each_res_npa.reshape(val_length_int, 1)))



    return output_m_npa
