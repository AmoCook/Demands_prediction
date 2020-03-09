from function import DTR, LR, SVR, RF

def generate_model_set_0(train_reg, v_train_reg, parameter_set_dic):
    """利用train训练各个回归器，并生成model_list"""


    # res_list = []
    # res_name_list = []
    # res_list.append(test_reg[:, 0])
    # res_name_list.append('targets')
    if parameter_set_dic['model_set_id_int'] == 0:
        # LR_v
        lr_v = LR(parameter_set_dic['related_region_num_int'] + 1)
        lr_v.train(v_train_reg)
        # res_list.append(lr_96.predict(test_reg))
        # res_name_list.append('lr_96')

        # svr_v_0
        svr_v_0 = SVR(parameter_set_dic['related_region_num_int'] + 1, 0)
        svr_v_0.train(v_train_reg)

        # RF_96_5
        rf_v_5 = RF(parameter_set_dic['related_region_num_int'] + 1, 5)
        rf_v_5.train(v_train_reg)

        # #svr_v_2
        svr_v_2 = SVR(parameter_set_dic['related_region_num_int'] + 1, 2)
        svr_v_2.train(v_train_reg)

        # dtr_48
        dtr_48 = DTR(48)
        dtr_48.train(train_reg)
        # res_list.append(dtr_48.predict(test_reg))
        # res_name_list.append('dtr_48')

        # dtr_96
        dtr_96 = DTR(96)
        dtr_96.train(train_reg)
        # res_list.append(dtr_96.predict(test_reg))
        # res_name_list.append('dtr_96')

        # LR_48
        lr_48 = LR(48)
        lr_48.train(train_reg)
        # res_list.append(lr_48.predict(test_reg))
        # res_name_list.append('lr_48')

        # LR_96
        lr_96 = LR(96)
        lr_96.train(train_reg)
        # res_list.append(lr_96.predict(test_reg))
        # res_name_list.append('lr_96')

        # svr_48_0
        svr_48_0 = SVR(48, 0)
        svr_48_0.train(train_reg)
        # res_list.append(svr_48_0.predict(test_reg))
        # res_name_list.append('svr_48_0')

        # svr_96_0
        svr_96_0 = SVR(96, 0)
        svr_96_0.train(train_reg)
        # res_list.append(svr_96_0.predict(test_reg))
        # res_name_list.append('svr_96_0')

        # svr_48_1
        svr_48_1 = SVR(48, 1)
        svr_48_1.train(train_reg)
        # res_list.append(svr_48_1.predict(test_reg))
        # res_name_list.append('svr_48_1')

        # svr_96_1
        svr_96_1 = SVR(96, 1)
        svr_96_1.train(train_reg)
        # res_list.append(svr_96_1.predict(test_reg))
        # res_name_list.append('svr_96_1')

        # svr_48_2
        svr_48_2 = SVR(48, 2)
        svr_48_2.train(train_reg)
        # res_list.append(svr_48_2.predict(test_reg))
        # res_name_list.append('svr_48_2')

        # svr_96_2
        svr_96_2 = SVR(96, 2)
        svr_96_2.train(train_reg)
        # res_list.append(svr_96_2.predict(test_reg))
        # res_name_list.append('svr_96_2')

        # RF_48_5
        rf_48_5 = RF(48, 5)
        rf_48_5.train(train_reg)
        # res_list.append(rf_48_5.predict(test_reg))
        # res_name_list.append('rf_48_5')

        # RF_48_10
        rf_48_10 = RF(48, 10)
        rf_48_10.train(train_reg)
        # res_list.append(rf_48_10.predict(test_reg))
        # res_name_list.append('rf_48_10')

        # RF_48_15
        rf_48_15 = RF(48, 15)
        rf_48_15.train(train_reg)
        # res_list.append(rf_48_15.predict(test_reg))
        # res_name_list.append('rf_48_15')


        # RF_96_5
        rf_96_5 = RF(96, 5)
        rf_96_5.train(train_reg)
        # res_list.append(rf_96_5.predict(test_reg))
        # res_name_list.append('rf_96_5')


        # RF_96_10
        rf_96_10 = RF(96, 10)
        rf_96_10.train(train_reg)
        # res_list.append(rf_96_10.predict(test_reg))
        # res_name_list.append('rf_96_10')

        # RF_96_15
        rf_96_15 = RF(96, 15)
        rf_96_15.train(train_reg)
        # res_list.append(rf_96_15.predict(test_reg))
        # res_name_list.append('rf_96_15')


        model_list = []
        model_list.append(lr_v)
        model_list.append(svr_v_0)
        model_list.append(rf_v_5)
        model_list.append(svr_v_2)
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
    if parameter_set_dic['model_set_id_int'] == 1:
        # dtr_48
        dtr_48 = DTR(48)
        dtr_48.train(train_reg)
        # res_list.append(dtr_48.predict(test_reg))
        # res_name_list.append('dtr_48')

        # dtr_96
        dtr_96 = DTR(96)
        dtr_96.train(train_reg)
        # res_list.append(dtr_96.predict(test_reg))
        # res_name_list.append('dtr_96')

        # LR_48
        lr_48 = LR(48)
        lr_48.train(train_reg)
        # res_list.append(lr_48.predict(test_reg))
        # res_name_list.append('lr_48')

        # LR_96
        lr_96 = LR(96)
        lr_96.train(train_reg)
        # res_list.append(lr_96.predict(test_reg))
        # res_name_list.append('lr_96')

        # svr_48_0
        svr_48_0 = SVR(48, 0)
        svr_48_0.train(train_reg)
        # res_list.append(svr_48_0.predict(test_reg))
        # res_name_list.append('svr_48_0')

        # svr_96_0
        svr_96_0 = SVR(96, 0)
        svr_96_0.train(train_reg)
        # res_list.append(svr_96_0.predict(test_reg))
        # res_name_list.append('svr_96_0')

        # svr_48_1
        svr_48_1 = SVR(48, 1)
        svr_48_1.train(train_reg)
        # res_list.append(svr_48_1.predict(test_reg))
        # res_name_list.append('svr_48_1')

        # svr_96_1
        svr_96_1 = SVR(96, 1)
        svr_96_1.train(train_reg)
        # res_list.append(svr_96_1.predict(test_reg))
        # res_name_list.append('svr_96_1')

        # svr_48_2
        svr_48_2 = SVR(48, 2)
        svr_48_2.train(train_reg)
        # res_list.append(svr_48_2.predict(test_reg))
        # res_name_list.append('svr_48_2')

        # svr_96_2
        svr_96_2 = SVR(96, 2)
        svr_96_2.train(train_reg)
        # res_list.append(svr_96_2.predict(test_reg))
        # res_name_list.append('svr_96_2')

        # RF_48_5
        rf_48_5 = RF(48, 5)
        rf_48_5.train(train_reg)
        # res_list.append(rf_48_5.predict(test_reg))
        # res_name_list.append('rf_48_5')

        # RF_48_10
        rf_48_10 = RF(48, 10)
        rf_48_10.train(train_reg)
        # res_list.append(rf_48_10.predict(test_reg))
        # res_name_list.append('rf_48_10')

        # RF_48_15
        rf_48_15 = RF(48, 15)
        rf_48_15.train(train_reg)
        # res_list.append(rf_48_15.predict(test_reg))
        # res_name_list.append('rf_48_15')

        # RF_96_5
        rf_96_5 = RF(96, 5)
        rf_96_5.train(train_reg)
        # res_list.append(rf_96_5.predict(test_reg))
        # res_name_list.append('rf_96_5')

        # RF_96_10
        rf_96_10 = RF(96, 10)
        rf_96_10.train(train_reg)
        # res_list.append(rf_96_10.predict(test_reg))
        # res_name_list.append('rf_96_10')

        # RF_96_15
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

    return model_list
