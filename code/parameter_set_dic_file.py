experiment_instance_set = {
    '20200305': {
        '0': {
            "date": '20200305',
            'instance_id': '0',
            'lag_int': 96,
            'train_data_rate_float': 0.5,

            #test_data_num_int 为-1时，test为剩下全部
            'test_data_num_int': -1,
            'val_length_int': 96,
            'lim_float': 0.3,
            'tp1_int': 10,
            'n_components_int': -1,

            #time_weight_theta = -1 为不添加时间权重，该值越大，衰减越厉害
            'time_weight_theta': -1,

            #设置跑的区域
            'region_index_start_int': 0,
            'region_index_end_int': 187,

            #输入数据的文件
            'input_file_path': './../data_set/',
            'input_file_name': 'ex90_demands.csv',

            #输出文件的路径和文件名设置
            'output_file_path': '../res_data/20200305/',
            'instance_file_path': 'instance_id_0/',
            'res_file_name': 'res_id_0.csv',
            'test_target_file_name': 'target_id_0.csv',
            # 'log_file_name': 'log_20.txt',
            'analysis_file_name': 'analysis_id_0.csv',
            'nrmse_fig_name': 'nrmse_id_0.png',

            #nrmse的画图参数设置
            'nrmse_stander_float': 0.8,

            #空间关系修改修改的参数
            'related_region_num_int': 0,
            'v_model_index': [],
            'model_set_id_int': 1,

            'info': '全部187个区域，剩下近一个月数据，无空间关系，无时间权重 只有selection'
        },
        '1': {
            "date": '20200305',
            'instance_id': '1',
            'lag_int': 96,
            'train_data_rate_float': 0.5,
            'test_data_num_int': -1,
            'val_length_int': 96,
            'lim_float': 0.3,
            'tp1_int': 10,
            'n_components_int': -1,
            'time_weight_theta': -1,
            'region_index_start_int': 0,
            'region_index_end_int': 187,
            'input_file_path': './../data_set/',
            'input_file_name': 'ex90_demands.csv',
            'output_file_path': '../res_data/20200305/',
            'instance_file_path': 'instance_id_1/',
            'res_file_name': 'res_id_1.csv',
            'test_target_file_name': 'target_id_1.csv',
            # 'log_file_name': 'log_20.txt',
            'analysis_file_name': 'analysis_id_1.csv',
            'nrmse_fig_name': 'nrmse_id_1.png',
            'nrmse_stander_float': 0.8,
            'related_region_num_int': 16,
            'v_model_index': [0, 1, 2, 3],
            'model_set_id_int': 0,
            'info': '全部187个区域，剩下近一个月数据，有selection，有空间关系，无时间权重'
        },
        '2': {
            "date": '20200305',
            'instance_id': '2',
            'lag_int': 96,
            'train_data_rate_float': 0.5,
            'test_data_num_int': -1,
            'val_length_int': 96,
            'lim_float': 0.3,
            'tp1_int': 10,
            'n_components_int': -1,
            'time_weight_theta': 0.5,
            'region_index_start_int': 0,
            'region_index_end_int': 187,
            'input_file_path': './../data_set/',
            'input_file_name': 'ex90_demands.csv',
            'output_file_path': '../res_data/20200305/',
            'instance_file_path': 'instance_id_2/',
            'res_file_name': 'res_id_2.csv',
            'test_target_file_name': 'target_id_2.csv',
            # 'log_file_name': 'log_20.txt',
            'analysis_file_name': 'analysis_id_2.csv',
            'nrmse_fig_name': 'nrmse_id_2.png',
            'nrmse_stander_float': 0.8,
            'related_region_num_int': 0,
            'v_model_index': [],
            'model_set_id_int': 1,
            'info': '全部187个区域，剩下近一个月数据，有selection，无空间关系，有时间权重'
        },
        '3': {
            "date": '20200305',
            'instance_id': '3',
            'lag_int': 96,
            'train_data_rate_float': 0.5,
            'test_data_num_int': -1,
            'val_length_int': 96,
            'lim_float': 0.3,
            'tp1_int': 10,
            'n_components_int': -1,
            'time_weight_theta': -1,
            'region_index_start_int': 0,
            'region_index_end_int': 187,
            'input_file_path': './../data_set/',
            'input_file_name': 'ex90_demands.csv',
            'output_file_path': '../res_data/20200305/',
            'instance_file_path': 'instance_id_3/',
            'res_file_name': 'res_id_3.csv',
            'test_target_file_name': 'target_id_3.csv',
            # 'log_file_name': 'log_20.txt',
            'analysis_file_name': 'analysis_id_3.csv',
            'nrmse_fig_name': 'nrmse_id_3.png',
            'nrmse_stander_float': 0.8,
            'related_region_num_int': 0,
            'v_model_index': [],
            'model_set_id_int': 1,
            'info': '全部187个区域，剩下近一个月数据，不带有selection，不带权重，不带空间关系'
        }
    },
    '20200309':{
        '0': {
            "date": '20200309',
            'instance_id': '0',
            'lag_int': 96,
            'train_data_rate_float': 0.5,

            #test_data_num_int 为-1时，test为剩下全部
            'test_data_num_int': -1,
            'val_length_int': 96,
            'lim_float': 0.3,
            'tp1_int': 10,
            'n_components_int': -1,

            #time_weight_theta = -1 为不添加时间权重，该值越大，衰减越厉害
            'time_weight_theta': -1,

            #设置跑的区域
            'region_index_start_int': 0,
            'region_index_end_int': 187,

            #输入数据的文件
            'input_file_path': './../data_set/',
            'input_file_name': 'ex90_demands.csv',

            #输出文件的路径和文件名设置
            'output_file_path': '../res_data/20200309/',
            'instance_file_path': 'instance_id_0/',
            'res_file_name': 'res_id_0.csv',
            'test_target_file_name': 'target_id_0.csv',
            # 'log_file_name': 'log_20.txt',
            'analysis_file_name': 'analysis_id_0.csv',
            'nrmse_fig_name': 'nrmse_id_0.png',

            #nrmse的画图参数设置
            'nrmse_stander_float': 0.8,

            #空间关系修改修改的参数
            'related_region_num_int': 16,
            'v_model_index': [0,1,2,3],
            'model_set_id_int': 0,

            'info': '全部187个区域，剩下近一个月数据，无selection 有空间关系的对比实验，查看selection的效果'
        },
        '1':{
            "date": '20200309',
            'instance_id': '1',
            'lag_int': 96,
            'train_data_rate_float': 0.5,

            #test_data_num_int 为-1时，test为剩下全部
            'test_data_num_int': -1,
            'val_length_int': 48,
            'lim_float': 0.3,
            'tp1_int': 10,
            'n_components_int': -1,

            #time_weight_theta = -1 为不添加时间权重，该值越大，衰减越厉害
            'time_weight_theta': -1,

            #设置跑的区域
            'region_index_start_int': 0,
            'region_index_end_int': 187,

            #输入数据的文件
            'input_file_path': './../data_set/',
            'input_file_name': 'ex90_demands.csv',

            #输出文件的路径和文件名设置
            'output_file_path': '../res_data/20200309/',
            'instance_file_path': 'instance_id_1/',
            'res_file_name': 'res_id_1.csv',
            'test_target_file_name': 'target_id_1.csv',
            # 'log_file_name': 'log_20.txt',
            'analysis_file_name': 'analysis_id_1.csv',
            'nrmse_fig_name': 'nrmse_id_1.png',

            #nrmse的画图参数设置
            'nrmse_stander_float': 0.8,

            #空间关系修改修改的参数
            'related_region_num_int': 16,
            'v_model_index': [0,1,2,3],
            'model_set_id_int': 0,

            'info': '全部187个区域，剩下近一个月数据，有空间关系，无时间权重 val_length_int 降低为1天'
        },
        '2':{
            "date": '20200309',
            'instance_id': '2',
            'lag_int': 96,
            'train_data_rate_float': 0.5,

            #test_data_num_int 为-1时，test为剩下全部
            'test_data_num_int': -1,
            'val_length_int': 96,
            'lim_float': 0.1,
            'tp1_int': 10,
            'n_components_int': -1,

            #time_weight_theta = -1 为不添加时间权重，该值越大，衰减越厉害
            'time_weight_theta': -1,

            #设置跑的区域
            'region_index_start_int': 0,
            'region_index_end_int': 187,

            #输入数据的文件
            'input_file_path': './../data_set/',
            'input_file_name': 'ex90_demands.csv',

            #输出文件的路径和文件名设置
            'output_file_path': '../res_data/20200309/',
            'instance_file_path': 'instance_id_2/',
            'res_file_name': 'res_id_2.csv',
            'test_target_file_name': 'target_id_2.csv',
            # 'log_file_name': 'log_20.txt',
            'analysis_file_name': 'analysis_id_2.csv',
            'nrmse_fig_name': 'nrmse_id_2.png',

            #nrmse的画图参数设置
            'nrmse_stander_float': 0.8,

            #空间关系修改修改的参数
            'related_region_num_int': 16,
            'v_model_index': [0,1,2,3],
            'model_set_id_int': 0,

            'info': '全部187个区域，剩下近一个月数据，有空间关系，无时间权重 lim_float 降低为0.1'
        },
        '3':{
            "date": '20200309',
            'instance_id': '3',
            'lag_int': 96,
            'train_data_rate_float': 0.5,

            #test_data_num_int 为-1时，test为剩下全部
            'test_data_num_int': -1,
            'val_length_int': 96,
            'lim_float': 0.6,
            'tp1_int': 10,
            'n_components_int': -1,

            #time_weight_theta = -1 为不添加时间权重，该值越大，衰减越厉害
            'time_weight_theta': -1,

            #设置跑的区域
            'region_index_start_int': 0,
            'region_index_end_int': 187,

            #输入数据的文件
            'input_file_path': './../data_set/',
            'input_file_name': 'ex90_demands.csv',

            #输出文件的路径和文件名设置
            'output_file_path': '../res_data/20200309/',
            'instance_file_path': 'instance_id_3/',
            'res_file_name': 'res_id_3.csv',
            'test_target_file_name': 'target_id_3.csv',
            # 'log_file_name': 'log_20.txt',
            'analysis_file_name': 'analysis_id_3.csv',
            'nrmse_fig_name': 'nrmse_id_3.png',

            #nrmse的画图参数设置
            'nrmse_stander_float': 0.8,

            #空间关系修改修改的参数
            'related_region_num_int': 16,
            'v_model_index': [0,1,2,3],
            'model_set_id_int': 0,

            'info': '全部187个区域，剩下近一个月数据，有空间关系，无时间权重 lim_float 升高为0.6'
        }
    }
}



print(type(experiment_instance_set))

