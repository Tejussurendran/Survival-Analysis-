def cph_model(dataframe):
    from sklearn.model_selection import KFold
    from sksurv.metrics import integrated_brier_score
    from scipy import interpolate
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines import CoxPHFitter
    from statistics import mean
    import lifelines
    import math
    from io import BytesIO
    
    
    oai_right_temp_SBL_Merged_zeros = dataframe.loc[dataframe['right_tkr'] == 0]
    oai_right_temp_SBL_Merged_ones = dataframe.loc[dataframe['right_tkr'] == 1]

    # oai_right_temp = pd.concat([oai_side_1_merge.loc[ :, 'F0':'T199', ],oai_side_1_merge[["time","right_tkr",]]], axis = 1)
    kf = KFold(n_splits = 5, shuffle = True, random_state = 120)
    kf.get_n_splits(oai_right_temp_SBL_Merged_zeros)
    print(kf)


    zb = KFold(n_splits = 5, shuffle = True, random_state = 120)
    zb.get_n_splits(oai_right_temp_SBL_Merged_ones)
    print(zb)
    iteration = 1

    for i,j in zip(list(kf.split(oai_right_temp_SBL_Merged_zeros)),list(zb.split(oai_right_temp_SBL_Merged_ones)) ):
            train_concat = pd.concat([oai_right_temp_SBL_Merged_zeros.iloc[i[0]],oai_right_temp_SBL_Merged_ones.iloc[j[0]]], axis = 0  )
            test_concat = pd.concat([oai_right_temp_SBL_Merged_zeros.iloc[i[1]],oai_right_temp_SBL_Merged_ones.iloc[j[1]]], axis = 0 )
            # print(test_concat)
            train = train_concat
            test =  test_concat
            
            print('test BML: ', test)
            
            t_max = (min(test['time'].max(),train['time'].max()))
            t_min = (max(test['time'].min(),train['time'].min()))
            # print('t_min: ', t_min)
            # print('t_max: ', t_max)
            
            test = test.loc[(test['time'] >= train['time'].min()) & (test['time'] < train['time'].max())]
            # print(test)
            print('iteration: ', iteration, '\n') 
            # print(test.loc[test['right_tkr'] == 1])
            
            oai_right_temp_kl_BML_2_brier_train = train[['right_tkr','time']].copy()
            oai_right_temp_kl_BML_2_brier_train['right_tkr'] =  oai_right_temp_kl_BML_2_brier_train['right_tkr'].astype(bool)
            oai_right_temp_kl_BML_2_brier_train = oai_right_temp_kl_BML_2_brier_train.to_numpy()
            aux_train = [(e1,e2) for e1,e2 in oai_right_temp_kl_BML_2_brier_train]
            train_data_y = np.array(aux_train, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])


            cph = CoxPHFitter(penalizer=0.0015)
            #brier score should be on test data
            oai_right_temp_kl_BML_2_brier_test = test[['right_tkr','time']].copy()
            oai_right_temp_kl_BML_2_brier_test['right_tkr'] =  oai_right_temp_kl_BML_2_brier_test['right_tkr'].astype(bool)
            oai_right_temp_kl_BML_2_brier_test = oai_right_temp_kl_BML_2_brier_test.to_numpy()
            aux_test = [(e1,e2) for e1,e2 in oai_right_temp_kl_BML_2_brier_test]
            test_data_y = np.array(aux_test, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
            # # oai_right_temp = oai_right_temp.dropna(axis=0, subset=['V99ERKDAYS'])
            cph.fit(train,"time",event_col = "right_tkr", show_progress = True, robust = True )
            # cph.print_summary()

            d_data = test
            
            # print('d_data', d_data)
            
            
            # print('len d_Data:',len( d_data))
            # print(d_data['time'].min())
            cph_data = cph.predict_survival_function(d_data)
            
            return cph_data, train,test, train_data_y,test_data_y
        