import numpy as np

def takekey(elem):
    return elem[2]

def find_neighborhood(region_index,region_data_pd,k):

    """

    :param index: 目标区域的下标
    :param region_data_pd: 完整的区域数据
    :param k: 选择距离进的前k个区域
    :return: 1.前k个邻近区域的下标 2.前k区域的信息[[x,y,dis,index]...], 3.完整所有区域的信息排名
    """
    #获取目标区域的坐标值， target_location = [x,y]  x与y 是string类型
    target_location = region_data_pd.columns[region_index].split(',')
    target_location_x = int(target_location[0])
    target_location_y = int(target_location[1])




    #获取所有区域的坐标数值
    location_list = []
    for index,location in enumerate(region_data_pd.columns):
        x = int(location.split(',')[0])
        y = int(location.split(',')[1])

        location_list.append([x, y, (target_location_x-x)**2+(target_location_y-y)**2, index])

    # 设置变量记录邻域的下标
    location_list.sort(key=takekey)


    related_k_region_index_npa = location_list[0:k]

    related_k_index = []

    for each_info in related_k_region_index_npa:
        related_k_index.append(each_info[-1])
    related_k_index_npa = np.array(related_k_index)


    return related_k_index_npa,related_k_region_index_npa, location_list


def V_data_preparation(parameter_set_dic, region_index, region_data_pd):

    if parameter_set_dic['related_region_num_int'] == 0:
        return np.array([]), np.array([])

    #找出所有的邻近区域下标 related_k_index_npa

    related_k_index_npa, related_k_region_index_npa, location_list = find_neighborhood(region_index, region_data_pd, parameter_set_dic['related_region_num_int'])

    v_data = region_data_pd.values[:, related_k_index_npa]

    v_data_target = v_data[:,0]

    t_index = parameter_set_dic['lag_int']
    # t_index-1 是把上 1 个时刻作为输入
    v_data_1 = np.hstack((v_data_target[t_index],v_data[t_index-1, :]))

    t_index = t_index + 1

    while(t_index < region_data_pd.shape[0]):
        v_data_1 = np.vstack((v_data_1, np.hstack((v_data_target[t_index],v_data[t_index-1, :]))))

        t_index = t_index + 1

    train_data_length = int(region_data_pd.shape[0] * parameter_set_dic['train_data_rate_float'])

    v_train_reg = v_data_1[:train_data_length, ]

    v_test_reg = v_data_1[train_data_length:, ]
    if (parameter_set_dic['test_data_num_int'] == -1):
        v_test_reg = v_test_reg
    else:
        v_test_reg = v_test_reg[0:parameter_set_dic['test_data_num_int'], ]

    return v_train_reg, v_test_reg
