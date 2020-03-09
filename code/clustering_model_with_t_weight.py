"""
该部分主要是进行对基回归器的聚类工作，得到最终的需要组合的回归器，主要的步骤分两步：
1。对model.sel选出来的回归器进行聚类操作，方法是对目标回归器们在验证集上的预测值进行聚类操作，假设窗口大小为W，回归器个数为M，
则得到M个W维的向量，对这些向量进行GMM-EM的聚类操作。假设生成N个类别即N个family
2。在这N个family中，选择最靠近本身family的回归器，每个family得到一个称之为winner，利用N个winner对test数据进行组合预测。

聚类部分的输入与输出工作：
input：
    model.sel：一组含有top k 回归器的下标 数据类型是np.array
    ...相关参数：包含窗口数据长度，等根据具体算法确定

output：
    winners: 一组含有各个winner回归器的下标 数据类型是np.array
    ...如果可以
    cluster_familys:[[family_1.members]...]  一个含有聚类情况的实时list，list[1]代表t=1时刻的family情况 list[1]也是一个list
        包含N个family<因为每次聚类的个数不同，这里的N可能是个变量> list[2][3]代表t=2时刻，famliy下标是3的类内包含哪些回归器
"""
from sklearn import mixture
import numpy as np
import math

def winner(cluster_family_index, output_val_res):
    """

    :param cluster_family_index: 数据结构为list [0,11,12] 说明 0,11,12下标的model是一个family
    :param output_val_res: 这个时刻，所有model在val上的输出
    :return: 在总的model上的下标
    """

    center_vector = sum(output_val_res[cluster_family_index])/(len(cluster_family_index))

    dis = sum(abs(output_val_res[cluster_family_index[0]]-center_vector))
    winner_id = cluster_family_index[0]
    for model_index in cluster_family_index:
        #计算距离的绝对值
        dis_1 = sum(abs(output_val_res[model_index] - center_vector))
        if(dis_1 < dis):
            dis = dis_1
            winner_id = model_index

    return winner_id


def cluster_models_with_t_weight(model_select_index_list, alarm_record_list, output_val_res_list, n_components_int, parameter_set_dic):
    """

    :param model_select_index_list: 挑选出来的模型的下标 [[time_step_t_index_list]....]
    :param alarm_record_list: drift 发生改变的记录 按照每个t时间步进行记录
    :param output_val_res_list: [[time_step_t_val_res]...]
    :return:


    """
    winner_list = []
    for time_step_t, each_index_combination_list in enumerate(model_select_index_list):
        # 获取model_select_index_list
        model_num = len(each_index_combination_list)
        # print('model_num:',model_num)
        if (model_num < 2):
            cluster_num = 1
            # print('cluster_num:1')
        else:
            cluster_num = int(model_num / 2)
            # print('cluster_num:', cluster_num)

        n_components_int = cluster_num

        gmm = mixture.GaussianMixture(n_components_int)
        cluster_data = (output_val_res_list[time_step_t][:, np.array(model_select_index_list[time_step_t]) + 1].T).copy()

        theta = parameter_set_dic['time_weight_theta']
        for i in range(cluster_data.shape[1]):
            cluster_data[:, i] = cluster_data[:,i]*math.exp(theta*-1*i)

        if (cluster_data.shape[0] == 1):
            labels = np.array([0])
        else:
            gmm.fit(cluster_data)
            # gmm.fit(output_val_res_list[time_step_t][:, np.array(model_select_index_list[time_step_t]) + 1].T)
            labels = gmm.predict(cluster_data)

        #cluster_time_t_index 记录是针对某个时间t中，labels
        cluster_time_t_index = []
        for cluster_id in range(labels.shape[0]):
            if (cluster_id in labels):
                cluster_time_t_index.append([model_select_index_list[time_step_t][i] for i, elem in enumerate(labels) if elem == cluster_id])

        #进行winner的挑选 循环执行每个family
        t_winner_id_list = []
        for cluster_family_index in cluster_time_t_index:
            winner_id = winner(cluster_family_index, output_val_res_list[time_step_t])
            t_winner_id_list.append(winner_id)

        winner_list.append(np.array(t_winner_id_list))

    return winner_list