import numpy as np
def create_reg_data(region_data_pd, lag_num):
    """

    :param data_region_pd: 从cvs文件导入的数据，整理成可以进行预测的形式。
    :return data_regions_list: [[target,lagn,lagn-1,...lag1]:npa...]
    """

    data_regions_list = []
    for region_index in range(region_data_pd.shape[1]):
        index = lag_num
        region_reg_npa = np.array([region_data_pd.values[index-i, region_index] for i in range(lag_num+1)])

        index = index + 1
        while(index < region_data_pd.shape[0]):
            print(region_index, index)
            region_reg_npa = np.vstack((region_reg_npa, np.array([region_data_pd.values[index-i, region_index] for i in range(lag_num+1)])))
            index = index + 1
        data_regions_list.append(region_reg_npa)
    return data_regions_list