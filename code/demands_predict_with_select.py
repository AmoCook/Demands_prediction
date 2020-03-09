import pandas as pd
import os
import time
import numpy as np
from clustering_model_with_t_weight import cluster_models_with_t_weight
from evaluate import evaluate
from V_data_preparation import V_data_preparation
from parameter_set_dic_file import experiment_instance_set


def predict_one_region(region_index, region_data_pd, parameter_set_dic):
    """
    进行单个区域的实验实例的运行
    :param region_index: 目标实验区域
    :param region_data_pd: 所有区域的实验数据
    :param parameter_set_dic: 实验实例的相关信息
    :return:
    """

    """读取对应地区的数据 """
    og_data = region_data_pd.values[:,region_index]

    """创建自回归所需要的输入表格数据，data_reg 的第一列是target 后面的多列是input, input有多少列是空过para...['lag_int']进行控制的"""

    '''根据 lag_int 确定初始数据的下标， 例如采用lag_int =4 ,则进行train的数据不能包含前4项，需要从第5个时间步开始，第5个时间步对应的数据是第一个target数据，而对应的这个target输入的数据为前四个时间步的值'''
    index = parameter_set_dic['lag_int']

    '''先对第一行输入表格数据生成， 第一个数据的下标为lag_int, 也就是原始数据的第lag_int+1个数据项'''
    region_reg_npa = np.array([og_data[index-i] for i in range(parameter_set_dic['lag_int']+1)])

    index = index + 1

    '''生成后续的输入表格数据项'''
    while(index < og_data.shape[0]):
        # print(index)
        region_reg_npa = np.vstack((region_reg_npa, np.array([og_data[index-i] for i in range(parameter_set_dic['lag_int']+1)])))
        index = index + 1

    """根据para...[train_data_rate_float]来确定训练集,根据test_data_num_int来生成测试集"""

    '''确定训练集'''
    train_data_length = int(og_data.shape[0]*parameter_set_dic['train_data_rate_float'])
    train_reg = region_reg_npa[:train_data_length, ]

    '''确定测试集, 若para...['test_data_num_int']==-1则表明剩下全部的行(项)都为测试集，否则按照其值进行截取得出测试集'''
    test_reg = region_reg_npa[train_data_length:, ]
    if(parameter_set_dic['test_data_num_int']==-1):
        test_reg = test_reg
    else:
        test_reg = test_reg[0:parameter_set_dic['test_data_num_int'], ]


    """ 添加v_data  给空间模型建立合适且与上面的time_step对应的数据集"""
    v_train_reg, v_test_reg = V_data_preparation(parameter_set_dic, region_index, region_data_pd)

    # print("检测正确性")
    # print('no_v_train:', train_reg[0:4, 0])
    # print('v_train:', v_train_reg[0:4,0])
    # print('no_v_test:', test_reg[0:4, 0])
    # print('v_test:', v_test_reg[0:4,0])

    from generate_model import generate_model_set_0
    """利用train训练各个回归器，并生成model_list"""
    model_train_start_time = time.time()

    '''检车文件是否存在，若不存在边创建文件夹'''
    instance_file_path = parameter_set_dic['output_file_path'] + parameter_set_dic['instance_file_path']
    isExists = os.path.exists(instance_file_path)

    if (isExists == False):
        print('Instacne_file dictionary is creating! ')

        os.makedirs(instance_file_path)
        log_program_file = open(instance_file_path + 'log_program_' + parameter_set_dic['instance_id'] + '.txt', 'a')
        print('Instacne_file dictionary has been created! ')
        log_program_file.writelines('Day ' + parameter_set_dic['date'] + ' instance ' + parameter_set_dic[
            'instance_id'] + '_file dictionary has been created! \n')
        log_program_file.close()

    '''打开program记录文件'''
    log_program_file = open(parameter_set_dic['output_file_path'] + parameter_set_dic['instance_file_path']+'log_program_'+parameter_set_dic['instance_id']+'.txt', 'a')
    print('Day '+parameter_set_dic['date']+' instance '+parameter_set_dic['instance_id']+' region '+str(region_index)+' traiming models is starting !')
    log_program_file.writelines('Day '+parameter_set_dic['date']+' instance '+parameter_set_dic['instance_id']+' region '+str(region_index)+' traiming models is starting ! \n')

    model_list = generate_model_set_0(train_reg, v_train_reg, parameter_set_dic)

    model_train_end_time = time.time()

    print('Day '+parameter_set_dic['date']+' instance '+parameter_set_dic['instance_id']+' region '+str(region_index)+' training models has finished, and cost time of training models: ',model_train_end_time - model_train_start_time)
    log_program_file.writelines('Day '+parameter_set_dic['date']+' instance '+parameter_set_dic['instance_id']+' region '+str(region_index)+' training models has finished, and cost time of training models: '+str(model_train_end_time - model_train_start_time)[0:4]+'s \n')
    from top_seletion import top_k_selection
    """在val上进行滑动实验，获得 1.model_select信息， 2.获得drift检测信息。 3.获得所有回归器在滑动val上的预测结果"""
    print('start top selection!')
    log_program_file.writelines('start top selection! \n')
    model_select_index_list, alarm_record_list, output_val_res_list = top_k_selection(model_list, train_reg, test_reg, parameter_set_dic['val_length_int'], parameter_set_dic['lim_float'], parameter_set_dic['tp1_int'],parameter_set_dic, v_train_reg, v_test_reg)

    from clustering_model import cluster_models
    """对各个时刻的val进行聚类，得到各个时刻的winners信息"""
    print('start cluster!')
    log_program_file.writelines('start cluster! \n')
    if(parameter_set_dic['time_weight_theta']==-1):
        winner_list = cluster_models(model_select_index_list, alarm_record_list, output_val_res_list, parameter_set_dic['n_components_int'],parameter_set_dic)
    else:
        winner_list = cluster_models_with_t_weight(model_select_index_list, alarm_record_list, output_val_res_list,
                                     parameter_set_dic['n_components_int'], parameter_set_dic)

    from model_test_predict import model_test_predict
    """利用model进行test的预测"""
    print('start test predict')
    log_program_file.writelines('start test predict \n')
    test_res_npa = model_test_predict(model_list, test_reg, v_test_reg,parameter_set_dic)

    from combination_res import combination_res
    """进行对预测结果的集成"""
    print('start combination!')
    log_program_file.writelines('start combination! \n')
    predict_res_list = combination_res(winner_list, test_res_npa, output_val_res_list)

    print('Day ' + parameter_set_dic['date'] + ' instance ' + parameter_set_dic['instance_id'] + ' region ' + str(region_index)+' has finished!')
    log_program_file.writelines('Day ' + parameter_set_dic['date'] + ' instance ' + parameter_set_dic['instance_id'] + ' region ' + str(region_index)+' has finished! \n')
    log_program_file.close()
    return np.array(predict_res_list),test_reg[:, 0]

def run_one_experiment_instance(parameter_set_dic):
    """
    该函数是进行对一个experiment instance进行实验  parameter_set_dic 是一个实验实例 数据结构为dic
    :param parameter_set_dic: 这是一个字典数据类型，是一个experiment_instance_info_dic 包含各种参数
    :return: 返回的是3个信息，第一个是instance的路径，一个是target的文件名，最后一个是res的文件名，这三个信息提供给evaluate函数进行输入文件的确定
    """

    """读取目标文件的数据"""
    ex90df = pd.read_csv(parameter_set_dic['input_file_path']+parameter_set_dic['input_file_name'])
    ex90_df_dic = {}
    test_target_dic = {}
    # whole_start_time = time.time()

    """针对instance中设置的start region到 end region进行实验"""
    for region_index in range(parameter_set_dic['region_index_start_int'], parameter_set_dic['region_index_end_int']):
        # each_start_time = time.time()

        '''进行一个区域的实验运行，返回的是预测结果res与对应的真实值target'''
        res, target = predict_one_region(region_index, ex90df, parameter_set_dic)

        '''进行结果的记录，ex90_df_dic是记录预测结果的dic数据结构, test_target_dic是记录真实值的dic'''
        ex90_df_dic[ex90df.columns[region_index]] = res
        test_target_dic[ex90df.columns[region_index]] = target
        # each_end_time = time.time()
        # dtime = each_end_time - each_start_time
        # whole_dtime = each_end_time - whole_start_time
        # print(str(region_index)+':  '+ex90df.columns[region_index])
        # print("计算region"+str(region_index)+" 耗费时间：%.8s s" % dtime)
        # print("累计耗时：%.8s s" % whole_dtime)
        # print('=====================================================')

    ex90_df_new = pd.DataFrame(ex90_df_dic)
    test_target_df = pd.DataFrame(test_target_dic)

    """将test_target数据进行存储  文件在instance_id_* 目录下，文件名为格式为test_target_[date]_id_[instance_id] """

    '''检车文件是否存在，若不存在边创建文件夹'''
    instance_file_path = parameter_set_dic['output_file_path'] + parameter_set_dic['instance_file_path']
    isExists = os.path.exists(instance_file_path)

    if(isExists==False):
        print('Instacne_file dictionary is creating! ')

        os.makedirs(instance_file_path)
        log_program_file = open(instance_file_path+'log_program_'+parameter_set_dic['instance_id']+'.txt', 'a')
        print('Instacne_file dictionary has been created! ')
        log_program_file.writelines('Day ' + parameter_set_dic['date'] + ' instance ' + parameter_set_dic['instance_id'] + '_file dictionary has been created! \n')
        log_program_file.close()

    '''设置target和res的文件名字'''
    test_target_file_name = 'test_target_'+parameter_set_dic['date']+'_id_'+parameter_set_dic['instance_id']+'.csv'
    test_res_file_name = 'test_res_'+parameter_set_dic['date']+'_id_'+parameter_set_dic['instance_id']+'.csv'

    '''存储test_target_[date]_id_[instance_id]文件'''
    test_target_df.to_csv(instance_file_path+test_target_file_name, index=False)

    '''存储test_res_[date]_id_[instance_id].csv'''
    ex90_df_new.to_csv(instance_file_path+test_res_file_name, index=False)



    return instance_file_path, test_target_file_name, test_res_file_name


def run_one_day_instance_set_with_selection(experiment_instance_set, date, instance_index=-1):
    """
    该函数可以批量运行设置为同一天的experiment_instance_set

    :param experiment_instance_set: 从parameter_set_dic_file.py 文件获取到的需要进行运行的实验实例
    :param date: 具体运行设置运行哪一天的实验实例集合
    :param instance_index: 若不进行设置会运行date里面所有的实验实例，否则可以通过[2,3,6]来设置只运行instance_id为2，3，6的实验实例
    :return: 0
    """

    '''instance_index==[] return 0'''
    if instance_index == []:
        return 0

    '''instance_set 获取设置时间对应的所有的实验实例'''
    instance_set = experiment_instance_set[date]

    '''判断需要运行该天全部的实验还是部分实验'''
    if(instance_index==-1):
        print('Run all experiment instances on day '+date+'.')
        for key in instance_set:
            print('Day' + date + ' instance ' + str(key) + ' is running!')
            one_instance_start_time = time.time()
            instance_file_path, test_target_file_name, test_res_file_name = run_one_experiment_instance(
                instance_set[str(key)])
            evaluate(instance_set[str(key)], instance_file_path, test_target_file_name, test_res_file_name)

            log_program_file = open(
                instance_set[key]['output_file_path'] + instance_set[key]['instance_file_path'] + 'log_program_' +
                instance_set[key]['instance_id'] + '.txt', 'a')
            print('Day' + date + ' instance ' + str(key) + ' has finished!')
            log_program_file.writelines('Day' + date + ' instance ' + str(key) + ' has finished!\n')
            one_instance_end_time = time.time()
            one_instance_cost_time = one_instance_end_time - one_instance_start_time
            print('Day' + date + ' instance ' + str(key) + " time cost：%.8s s" % one_instance_cost_time)
            log_program_file.writelines(
                'Day' + date + ' instance ' + str(key) + " time cost：%.8s s" % one_instance_cost_time)
            print('\n\n')
            log_program_file.close()

    else:
        print('Run experiment instances '+str(instance_index)+' on day ' + date + '.')
        for key in instance_index:
            print('Day' + date + ' instance ' + str(key) + ' is running!')
            one_instance_start_time = time.time()
            instance_file_path, test_target_file_name, test_res_file_name = run_one_experiment_instance(instance_set[str(key)])
            evaluate(instance_set[str(key)], instance_file_path, test_target_file_name, test_res_file_name)

            log_program_file = open(instance_set[str(key)]['output_file_path'] + instance_set[str(key)]['instance_file_path']+'log_program_'+instance_set[str(key)]['instance_id']+'.txt', 'a')
            print('Day' + date + ' instance ' + str(key) + ' has finished!')
            log_program_file.writelines('Day' + date + ' instance ' + str(key) + ' has finished!\n')
            one_instance_end_time = time.time()
            one_instance_cost_time = one_instance_end_time - one_instance_start_time
            print('Day' + date + ' instance ' + str(key) + " time cost：%.8s s" % one_instance_cost_time)
            log_program_file.writelines('Day' + date + ' instance ' + str(key) + " time cost：%.8s s" % one_instance_cost_time)
            print('\n\n')
            log_program_file.close()

    return 0


# run_one_day_instance_set_with_selection(experiment_instance_set, '20200304', -1)
