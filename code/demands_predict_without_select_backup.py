import pandas as pd
import time
import numpy as np
from clustering_model_with_t_weight import cluster_models_with_t_weight
from evaluate import evaluate

def predict_one_region(region_index,region_data_pd, parameter_set_dic):



    """由于数据过大，这里采用下标为33号的地区进行单区域的实验    lag_num = 96 """
    og_data = region_data_pd.values[:,region_index]

    index = parameter_set_dic['lag_int']
    region_reg_npa = np.array([og_data[index-i] for i in range(parameter_set_dic['lag_int']+1)])

    index = index + 1
    while(index < og_data.shape[0]):
        # print(index)
        region_reg_npa = np.vstack((region_reg_npa, np.array([og_data[index-i] for i in range(parameter_set_dic['lag_int']+1)])))
        index = index + 1

    """设定好train与test 前1500个时间片为train ， 后48*7个为test 即后7天"""

    train_data_length = int(og_data.shape[0]*parameter_set_dic['train_data_rate_float'])

    train_reg = region_reg_npa[:train_data_length, ]

    test_reg = region_reg_npa[train_data_length:, ]
    if(parameter_set_dic['test_data_num_int']==-1):
        test_reg = test_reg
    else:
        test_reg = test_reg[0:parameter_set_dic['test_data_num_int'], ]

    from function import DTR, LR, SVR, RF
    """利用train训练各个回归器，并生成model_list"""

    res_list = []
    res_name_list = []
    res_list.append(test_reg[:,0])
    res_name_list.append('targets')

    #dtr_48
    dtr_48 = DTR(48)
    dtr_48.train(train_reg)
    # res_list.append(dtr_48.predict(test_reg))
    # res_name_list.append('dtr_48')

    #dtr_96
    dtr_96 = DTR(96)
    dtr_96.train(train_reg)
    # res_list.append(dtr_96.predict(test_reg))
    # res_name_list.append('dtr_96')

    #LR_48
    lr_48 = LR(48)
    lr_48.train(train_reg)
    # res_list.append(lr_48.predict(test_reg))
    # res_name_list.append('lr_48')

    #LR_96
    lr_96 = LR(96)
    lr_96.train(train_reg)
    # res_list.append(lr_96.predict(test_reg))
    # res_name_list.append('lr_96')

    #svr_48_0
    svr_48_0 = SVR(48, 0)
    svr_48_0.train(train_reg)
    # res_list.append(svr_48_0.predict(test_reg))
    # res_name_list.append('svr_48_0')

    #svr_96_0
    svr_96_0 = SVR(96, 0)
    svr_96_0.train(train_reg)
    # res_list.append(svr_96_0.predict(test_reg))
    # res_name_list.append('svr_96_0')

    #svr_48_1
    svr_48_1 = SVR(48, 1)
    svr_48_1.train(train_reg)
    # res_list.append(svr_48_1.predict(test_reg))
    # res_name_list.append('svr_48_1')

    #svr_96_1
    svr_96_1 = SVR(96, 1)
    svr_96_1.train(train_reg)
    # res_list.append(svr_96_1.predict(test_reg))
    # res_name_list.append('svr_96_1')

    #svr_48_2
    svr_48_2 = SVR(48, 2)
    svr_48_2.train(train_reg)
    # res_list.append(svr_48_2.predict(test_reg))
    # res_name_list.append('svr_48_2')

    #svr_96_2
    svr_96_2 = SVR(96, 2)
    svr_96_2.train(train_reg)
    # res_list.append(svr_96_2.predict(test_reg))
    # res_name_list.append('svr_96_2')

    #RF_48_5
    rf_48_5 = RF(48, 5)
    rf_48_5.train(train_reg)
    # res_list.append(rf_48_5.predict(test_reg))
    # res_name_list.append('rf_48_5')

    #RF_48_10
    rf_48_10 = RF(48, 10)
    rf_48_10.train(train_reg)
    # res_list.append(rf_48_10.predict(test_reg))
    # res_name_list.append('rf_48_10')

    #RF_48_15
    rf_48_15 = RF(48, 15)
    rf_48_15.train(train_reg)
    # res_list.append(rf_48_15.predict(test_reg))
    # res_name_list.append('rf_48_15')


    #RF_96_5
    rf_96_5 = RF(96, 5)
    rf_96_5.train(train_reg)
    # res_list.append(rf_96_5.predict(test_reg))
    # res_name_list.append('rf_96_5')

    #RF_96_10
    rf_96_10 = RF(96, 10)
    rf_96_10.train(train_reg)
    # res_list.append(rf_96_10.predict(test_reg))
    # res_name_list.append('rf_96_10')

    #RF_96_15
    rf_96_15 = RF(96, 15)
    rf_96_15.train(train_reg)
    # res_list.append(rf_96_15.predict(test_reg))
    # res_name_list.append('rf_96_15')


    model_list = []

    model_list.append(dtr_48)
    model_list.append(dtr_96)
    model_list.append(lr_48)
    model_list.append(lr_96)
    model_list.append(svr_48_0)
    model_list.append(svr_96_0)
    model_list.append(svr_48_1)
    model_list.append(svr_96_1)
    model_list.append(svr_48_2)
    model_list.append(svr_96_2)
    model_list.append(rf_48_5)
    model_list.append(rf_48_10)
    model_list.append(rf_48_15)
    model_list.append(rf_96_5)
    model_list.append(rf_96_10)
    model_list.append(rf_96_15)

    from top_seletion import top_k_selection
    """在val上进行滑动实验，获得 1.model_select信息， 2.获得drift检测信息。 3.获得所有回归器在滑动val上的预测结果"""

    model_select_index_list, alarm_record_list, output_val_res_list = top_k_selection(model_list, train_reg, test_reg, parameter_set_dic['val_length_int'], parameter_set_dic['lim_float'], parameter_set_dic['tp1_int'])

    # from clustering_model import cluster_models
    # """对各个时刻的val进行聚类，得到各个时刻的winners信息"""
    # if(parameter_set_dic['time_weight_theta']==-1):
    #     winner_list = cluster_models(model_select_index_list, alarm_record_list, output_val_res_list, parameter_set_dic['n_components_int'],parameter_set_dic)
    # else:
    #     winner_list = cluster_models_with_t_weight(model_select_index_list, alarm_record_list, output_val_res_list,
    #                                  parameter_set_dic['n_components_int'], parameter_set_dic)
    #
    from model_test_predict import model_test_predict
    """利用model进行test的预测"""

    test_res_npa = model_test_predict(model_list, test_reg)

    from combination_res import combination_res
    """进行对预测结果的集成"""

    predict_res_list = combination_res([np.array([index for index in range(len(model_list))]) for i in range(len(output_val_res_list))], test_res_npa, output_val_res_list)

    return np.array(predict_res_list),test_reg[:, 0]

def demand_predict(parameter_set_dic):

    print()
    ex90df = pd.read_csv(parameter_set_dic['file_path']+parameter_set_dic['file_name'])
    ex90_df_dic = {}
    test_target_dic = {}
    whole_start_time = time.time()
    for region_index in range(parameter_set_dic['region_index_start_int'], parameter_set_dic['region_index_end_int']):
        each_start_time = time.time()
        res,target = predict_one_region(region_index, ex90df, parameter_set_dic)
        ex90_df_dic[ex90df.columns[region_index]] = res
        test_target_dic[ex90df.columns[region_index]] = target
        each_end_time = time.time()
        dtime = each_end_time - each_start_time
        whole_dtime = each_end_time - whole_start_time
        print(str(region_index)+':  '+ex90df.columns[region_index])
        print("计算region"+str(region_index)+" 耗费时间：%.8s s" % dtime)
        print("累计耗时：%.8s s" % whole_dtime)
        print('=====================================================')

    ex90_df_new = pd.DataFrame(ex90_df_dic)
    test_target_df = pd.DataFrame(test_target_dic)
    test_target_df.to_csv(parameter_set_dic['res_file_path']+parameter_set_dic['test_target_file_name'], index=False)
    ex90_df_new.to_csv(parameter_set_dic['res_file_path']+parameter_set_dic['res_file_name'], index=False)

def main():
    parameter_set_dic = {
        'lag_int': 96,
        'train_data_rate_float': 0.5,
        'test_data_num_int': 48,
        'val_length_int': 48,
        'lim_float': 0.1,
        'tp1_int': 10,
        'n_components_int': -1,
        'time_weight_theta': -1,
        'region_index_start_int': 0,
        'region_index_end_int': 1,
        'file_path': '../data_set/',
        'file_name': 'ex90_demands.csv',
        'res_file_path': '../res_data/20200226/',
        'res_file_name': 'res_region_0_40_20200226.csv',
        'test_target_file_name': 'target_region_0_40_20200226_ttt.csv',
        'log_file_name': 'log_20200226.txt',
        'nrmse_file_name': 'nrmse_20200226ttt.csv',
        'info': 'Without selection_这是个测试'
    }

    time_start = time.time()
    demand_predict(parameter_set_dic)
    time_end = time.time()
    evaluate(parameter_set_dic)

    return 0

main()