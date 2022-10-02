def cph_creator_SBL(dataframe, title):
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
    from graph_gen import survival_curve, average_curve
    import numpy.ma as ma
    from itertools import zip_longest
    from sklearn.model_selection import train_test_split
    from lifelines.utils.sklearn_adapter import sklearn_adapter
    averages = []
    vals = []
    bad_cols = []
    
    list_bad = ['time', 'right_tkr']
    ############### SBL cox model
    # print(dataframe["right_tkr"])
    events = dataframe["right_tkr"].astype(bool)
    for i in dataframe.columns:
        if i not in list_bad:
            # print(dataframe.loc[events, 'T99'].var())
            # print(dataframe.loc[~events, 'T99'].var())
            #np isclose, we could say the column we generated is outside/very close to the edge of the knee
            # print(i,np.allclose(zeros, temp , rtol=1, atol=1), dataframe[i].mean())
            # print(oai_right_temp_kl_2.loc[events, i].var())
            if dataframe.loc[events, i].var() == 0.0:
                # print(i)
                dataframe = dataframe.drop(labels = i, axis=1)
                bad_cols.append(i)    
                # print(oai_right_temp_kl_2.loc[~events, i].var())

    print('bad cols: ', bad_cols)

    
    # print('all cols', dataframe.columns)


    # oai_right_temp = pd.concat([oai_side_1_merge.loc[ :, 'F0':'T199', ],oai_side_1_merge[["time","right_tkr",]]], axis = 1)

    # print(kf)


    brier_scores_SBL = []






    # print(test_concat)

    train, test = train_test_split(dataframe, test_size=0.3, random_state = 120)
    # print('tkr occurs 1',len(test.loc[test['right_tkr'] == 1]))
    # print(len(train.loc[train['right_tkr'] == 0]))
    # print('tester ones',len(oai_right_temp_SBL_Merged_ones.iloc[j[1]]))

    # print('test BML: ', test)
    
            
    t_max = (min(test['time'].max(),train['time'].max()))
    t_min = (max(test['time'].min(),train['time'].min()))
    # print('t_min: ', t_min)
    # print('t_max: ', t_max)
    # print(test.time.unique())
    # print(train.time.unique())
    # print('len test data before time range', len(test))
    test = test.loc[(test['time'] >= train['time'].min()) & (test['time'] < train['time'].max())].copy()
    # print('len test data after time range', len(test))
    # print('tkr occurs 2',len(test.loc[test['right_tkr'] == 1]))
    # print(test)
    
    # print(test.loc[test['right_tkr'] == 1])
    
    oai_right_temp_kl_BML_2_brier_train = train[['right_tkr','time']].copy()
    oai_right_temp_kl_BML_2_brier_train['right_tkr'] =  oai_right_temp_kl_BML_2_brier_train['right_tkr'].astype(bool)
    oai_right_temp_kl_BML_2_brier_train = oai_right_temp_kl_BML_2_brier_train.to_numpy()
    aux_train = [(e1,e2) for e1,e2 in oai_right_temp_kl_BML_2_brier_train]
    train_data_y = np.array(aux_train, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    #principal component analysis
    
    #brier score should be on test data
    oai_right_temp_kl_BML_2_brier_test = test[['right_tkr','time']].copy()
    oai_right_temp_kl_BML_2_brier_test['right_tkr'] =  oai_right_temp_kl_BML_2_brier_test['right_tkr'].astype(bool)
    oai_right_temp_kl_BML_2_brier_test = oai_right_temp_kl_BML_2_brier_test.to_numpy()
    aux_test = [(e1,e2) for e1,e2 in oai_right_temp_kl_BML_2_brier_test]
    test_data_y = np.array(aux_test, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    # # oai_right_temp = oai_right_temp.dropna(axis=0, subset=['V99ERKDAYS'])
    
    # print('train_data_y tkrs', train_data_y)
    # print('test_data_y tkrs', test_data_y)
    
    X = train.copy().drop('time', axis=1) # keep as a dataframe
    Y = train.copy().pop('time')
    
    base_class = sklearn_adapter(CoxPHFitter, event_col='right_tkr')
    wf = base_class()

    # scores = cross_val_score(wf, X, Y, cv=5)
    # print(scores)
    
    from sklearn.model_selection import GridSearchCV

    clf = GridSearchCV(wf, {
    "penalizer": 10.0 ** np.arange(-2, 3),
    }, cv=4)
    clf.fit(X,Y)


    # print(clf.lifelines_model)
    penalizer_val = clf.best_params_.get('penalizer')   
    print('penalizer_val: ',penalizer_val) 
    # col_list = list(train.columns.values)
    # col_list.remove('right_tkr')
    # col_list.remove('time')
    # cph.fit(train,"time",event_col = "right_tkr", show_progress = True, strata = col_list , robust = True )
    cph = CoxPHFitter(penalizer=penalizer_val)
    cph.fit(train,"time",event_col = "right_tkr", show_progress = True, robust = True )
    
    print('cph standard error', cph.standard_errors_)

    print('Concordance: ', cph.concordance_index_)
    # cph.check_assumptions(train, p_value_threshold=0.05, show_plots=True)
    cph.print_summary(decimals=3)
    # print('training done iteration',  iteration)
    # print('tester ones',len(oai_right_temp_SBL_Merged_ones.iloc[j[1]]))
    # print('tkr occurs',len(train.loc[train['right_tkr'] == 1]))
    # print('tkr occurs 3',len(test.loc[test['right_tkr'] == 1]))
    # print()
    d_data = test.copy()

    # print('d_data', d_data)
    
    
    # print('len d_Data:',len( d_data))
    # print(d_data['time'].min())
    cph_data = cph.predict_survival_function(d_data)
    # print(cph_data)
    time_points = cph_data.index.tolist()
    
    # print(d_data)
    # print('time_points', time_points)
    # cph_data['time_loc'] = cph_data.index
    # cph_data = cph_data[cph_data['time_loc'] >=360.0]
    

    
    # print(cph_data)
    

    
    numpy_array = cph_data.to_numpy()
    new_nump =  np.transpose(numpy_array)
    # print('after transpose\n')
    # print(new_nump)
    # print('len new_nump', len(new_nump))
    cph_data_avg = cph_data.assign(avg=cph_data.mean(axis=1))
    std_dev_DF = cph_data.std(axis=1)
    avgs = cph_data_avg['avg'].tolist()
    std_dev = std_dev_DF.tolist()
    
    # print('testing min', test['time'].min())
    
    # cph.predict_survival_function(d_data).plot(legend = None)
    # plt.ylabel('Survival Probability SBL')
    # plt.title("CPH plot")
    # plt.xlabel('timeline')
    # print(type(cph_data))
    # print(cph_data)
    # time_locs = []
    # score,interp = retrieve_brier_scores(time_points,new_nump, train, test)

    # print(list(timmy))
    # for x in time_points:
    #     # print(x)
    #     if x >= 360.0:
    #         time_locs.append(x)
            
    # print(type(time_locs))
    # print(type(time_points))
    # print(time_locs)
    t_max = (min(test['time'].max(),train['time'].max()))
    t_min = (max(test['time'].min(),train['time'].min()))
    # print('t_min: ', t_min)
    # print('t_max: ', t_max)
    # print('timepoints: ', time_points)
    # print ('tmax and tmin', t_max, t_min)
    time_new = [x for x in time_points if x >= t_min and x < t_max]

    # print('time_points: ', time_points)
    # # time_new = [t_min] + time_points[1:-1] + [t_max-1]
    # print('time_new: ', time_new)
    # print()
    interpolator = interpolate.PchipInterpolator(time_points, new_nump, axis=1)
    new_nump_trunc = interpolator(time_new)
    # print(new_nump_trunc)
    # # print(np.range(new_data_y))
    # print(len(new_nump_trunc))
    # # print(len(new_data_y))
    # print(len(time_new))
    # print('times: ', time_new)
    
    score = integrated_brier_score(train_data_y, test_data_y, new_nump_trunc, time_new)
    brier_scores_SBL.append(score)
    averages.append( time_points)
    vals.append(avgs)
    # iteration += 1
    print('Values where the surivival plus std dev is greater than or equal to 1:')
    sum_list = list(np.add(avgs, std_dev))
    for i in sum_list:
        if i >= 1.00:
            print(i)
    print('standard deviation: ', len(std_dev), len(cph_data_avg))
    average_curve(time_points,avgs, std_dev,title)
    return bad_cols, brier_scores_SBL