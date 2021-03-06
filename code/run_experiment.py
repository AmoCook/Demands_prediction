import pandas as pd
import os
import time
import numpy as np
from clustering_model_with_t_weight import cluster_models_with_t_weight
from evaluate import evaluate
from V_data_preparation import V_data_preparation
from parameter_set_dic_file import experiment_instance_set
from demands_predict_with_select import run_one_day_instance_set_with_selection
from demands_predict_without_select import run_one_day_instance_set_with_out_selection

#设置带有selection的instance下标，不带有selection的instance下标

selection_instance_index_list = [1,2,3]
with_out_selection_instance_index_list = [0]

experiment_time = '20200309'

run_one_day_instance_set_with_out_selection(experiment_instance_set, experiment_time, with_out_selection_instance_index_list)
run_one_day_instance_set_with_selection(experiment_instance_set, experiment_time, selection_instance_index_list)


