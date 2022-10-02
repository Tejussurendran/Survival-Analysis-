from scipy.fftpack import diff


def survival_curve(time_points,avgs, iteration, var_name, sex):
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
   
    plt.ylabel('Survival Probability ')
    plt.title("CPH plot " + var_name)
    plt.xlabel('timeline')

    if iteration == 1:
        
        plt.plot(time_points,avgs,'r--.',  label = 'itr_1')
        # print('iteration 1 WOMAC Brier score:', score)
    elif iteration == 2:
        
        plt.plot(time_points,avgs,'g-',  label = 'itr_2')
        # print('iteration 2 WOMAC Brier score:', score)   
    elif iteration == 3:

        plt.plot(time_points,avgs,'b-.',  label = 'itr_3')
        # print('iteration 3 WOMAC Brier score:', score)
    elif iteration == 4:
        
        plt.plot(time_points,avgs,'m:',  label = 'itr_4')
        # print('iteration 4 WOMAC Brier score:', score)
    elif iteration == 5:

        plt.plot(time_points,avgs,'k-',  label = 'itr_5')
        # print('iteration 5 WOMAC Brier score:', score)
        
    plt.legend(loc = 'upper right')   

def average_curve(time_points,avgs, std_dev, title):
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
    parameters = {'axes.labelsize': 15,
          'axes.titlesize': 25,'font.family':'sans-serif'}

    sum_list = list(np.add(avgs, std_dev))
    diff_list = list(np.subtract(avgs,std_dev))
    plt.rcParams.update(parameters)
    csfont = {'fontname':'Comic Sans MS'}
    # print(time_points)
    # print(avgs)
    # print(std_dev)
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams.update({})
    plt.ylabel('Average Survival Probability ')
    plt.title(title)
    plt.xlabel('Time in Days')
    plt.plot(time_points,avgs, 'k-') 
    plt.fill_between(time_points, sum_list, diff_list,facecolor = 'grey')
    plt.yticks(np.arange(0,1.2,0.2))
    # plt.legend(loc = 'upper right')   
