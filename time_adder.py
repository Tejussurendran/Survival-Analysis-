
def add_time(oai_SBL_KL_WOMAC_merge, knee):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines import CoxPHFitter
    num_days_in_month = 30
    if 'time' in oai_SBL_KL_WOMAC_merge.columns:
        
        oai_SBL_KL_WOMAC_merge = oai_SBL_KL_WOMAC_merge.drop('time', 1)
    
    time_val = pd.Series([])
    for i in range(len(oai_SBL_KL_WOMAC_merge)):
        if oai_SBL_KL_WOMAC_merge["right_tkr"][i] == 0:
            if oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 0.0:
                time_val[i] = 0*num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 1.0:
                time_val[i] = 12*num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 2.0:
                time_val[i] = 24*num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 3.0:
                time_val[i] = 36*num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 4.0:
                time_val[i] = 48*num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 5.0:
                time_val[i] = 60*num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 6.0:
                time_val[i] = 72*num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 7.0:
                time_val[i] = 84*num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 8.0:
                time_val[i] = 96*num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 9.0:
                time_val[i] = 108*num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 10.0:
                time_val[i] = 120*num_days_in_month
            elif oai_SBL_KL_WOMAC_merge["V99RNTCNT"][i] == 11.0:
                time_val[i] = 132*num_days_in_month
    
        elif oai_SBL_KL_WOMAC_merge["right_tkr"][i] == 1:
            if knee == 'left':
                
                time_val[i]= oai_SBL_KL_WOMAC_merge["V99ELKDAYS"][i]
            elif knee == 'right':
                time_val[i]= oai_SBL_KL_WOMAC_merge["V99ERKDAYS"][i]

    oai_SBL_KL_WOMAC_merge.insert(2, "time", time_val) 

    print(oai_SBL_KL_WOMAC_merge)
    # print('nan values:',df['V99RNTCNT'].isnull().sum())
    # print(oai_side_1_merge)
    print(oai_SBL_KL_WOMAC_merge.loc[oai_SBL_KL_WOMAC_merge['right_tkr'] == 0])
    print(oai_SBL_KL_WOMAC_merge.loc[oai_SBL_KL_WOMAC_merge['right_tkr'] == 1])

    kmf = KaplanMeierFitter() 
    #censored values match at 4309 between the subtraction and KMF
    kmf.fit(durations = oai_SBL_KL_WOMAC_merge["time"], event_observed = oai_SBL_KL_WOMAC_merge["right_tkr"])
    print (kmf.event_table)
    kmf.plot_survival_function(show_censors = True, censor_styles={'ms': 6, 'marker': 's'})
    plt.ylabel('Survival Probability')
    plt.title("kmf plot")
    # plt.savefig('hit.png')
    return oai_SBL_KL_WOMAC_merge