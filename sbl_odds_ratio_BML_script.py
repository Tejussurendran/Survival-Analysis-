import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn
import matplotlib
import seaborn as sns
import matplotlib.colors as mc
import colorsys
from scipy.stats import ttest_ind


###############
# Preparation
###############

cli = pd.read_csv('/data_1/OAI_Backup/merge1.csv')
sbl = pd.read_csv('/data_1/OAI_Backup/SBL_0904.csv')  # Lateral 0-200 Medial
sbl['id'] = sbl['id'].astype(str)
side_SBL_temp = sbl.groupby("SIDE")
side_1_SBL_Right = side_SBL_temp.get_group(1) 
side_2_SBL_Left = side_SBL_temp.get_group(2) 

data_BML_right = pd.read_csv('/data_1/OAI_Backup/rightFilteredbmlMoaks.csv')
data_BML_right['id'] = data_BML_right['id'].astype(str)
data_BML_right = data_BML_right.drop(["Unnamed: 0"],axis=1)
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMNFMA'])
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMNFLA'])
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMNFMC'])
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMNFLC'])
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMNFMP'])
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMNFLP'])
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMNSS'])
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMNTMA'])
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMNTLA'])
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMNTMC'])
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMNTLC'])
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMNTMP'])
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMNTLP'])
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMNPM'])
data_BML_right = data_BML_right.dropna(axis=0, subset=['V00MBMNPL'])
# print(data_BML_right)
column_list = ['V00MBMNFMA','V00MBMNFLA','V00MBMNFMC','V00MBMNFLC','V00MBMNFMP','V00MBMNFLP','V00MBMNSS','V00MBMNTMA','V00MBMNTLA','V00MBMNTMC','V00MBMNTMP','V00MBMNTLP','V00MBMNPM','V00MBMNPL']
data_BML_right['bml_total'] = data_BML_right[column_list].sum(axis=1)
data_BML_left = pd.read_csv('/data_1/OAI_Backup/leftFilteredbmlMoaks.csv')
data_BML_left['id'] = data_BML_left['id'].astype(str)
data_BML_left = data_BML_left.drop(["Unnamed: 0"],axis=1)
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMNFMA'])
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMNFLA'])
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMNFMC'])
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMNFLC'])
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMNFMP'])
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMNFLP'])
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMNSS'])
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMNTMA'])
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMNTLA'])
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMNTMC'])
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMNTLC'])
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMNTMP'])
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMNTLP'])
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMNPM'])
data_BML_left = data_BML_left.dropna(axis=0, subset=['V00MBMNPL'])
# print(data_BML_left)
column_list = ['V00MBMNFMA','V00MBMNFLA','V00MBMNFMC','V00MBMNFLC','V00MBMNFMP','V00MBMNFLP','V00MBMNSS','V00MBMNTMA','V00MBMNTLA','V00MBMNTMC','V00MBMNTMP','V00MBMNTLP','V00MBMNPM','V00MBMNPL']
data_BML_left['bml_total'] = data_BML_left[column_list].sum(axis=1)


cli['id'] = cli['id'].astype(str)
merge1_temp = cli.groupby("SIDE")
merge1_right = merge1_temp.get_group(1) 
merge1_left = merge1_temp.get_group(2) 
# bml_all = pd.concat([data_BML_right,data_BML_left], ignore_index=True)

bml_merge1_right = pd.merge(merge1_right,data_BML_right, how = 'inner', on = ['id'])
bml_merge1_left = pd.merge(merge1_left,data_BML_left, how = 'inner', on = ['id'])
# print(len(bml_merge1_right))
# print(len(bml_merge1_left))

bml_merge1_SBL_right = pd.merge(bml_merge1_right,side_1_SBL_Right, how = 'inner', on = ['id'])
bml_merge1_SBL_left = pd.merge(bml_merge1_left,side_2_SBL_Left, how = 'inner', on = ['id'])

# print((bml_merge1_SBL_right))
# print((bml_merge1_SBL_left))

bml_all = pd.concat([bml_merge1_SBL_right,bml_merge1_SBL_left], ignore_index=True)

# print((bml_all))

sbl_col_names = ['F' + str(i) for i in range(200)] + ['T' + str(i) for i in range(200)]

###############
# normalize SBL
###############
sbl_values = bml_all.loc[:, sbl_col_names].values
for row in range(sbl_values.shape[0]):
    sbl_values[row, :] = sbl_values[row, :] / sbl_values[row, :].mean()  # normalize by the averaged val. of SBL
# flip left to right so left is medial side and right is lateral side
sbl_values = np.concatenate([np.fliplr(sbl_values[:, :200]), np.fliplr(sbl_values[:, 200:])], 1) # Medial 0-200 Lateral
bml_all.loc[:, sbl_col_names] = sbl_values
print(sbl_values)


#  define baseline
sbl_jsn_0_mean = bml_all.loc[(bml_all['V00XRJSM'] == 0) & (bml_all['V00XRJSL'] == 0), sbl_col_names].values.mean(0)
#sbl_KL_0_mean = sbl.loc[cli['V00XRKL'] == 0, sbl_col_names].values.mean(0)
#sbl_pain_0_mean = sbl.loc[cli['V00WOMKP#'] == 0, sbl_col_names].values.mean(0)
baseline = sbl_jsn_0_mean

#####################
# Calculate Odd Ratio
#####################
def get_OR(x, outcome, condition):
    #subjects w/ condition. SBLs and outcomes.
    x = x.loc[condition]
    outcome = outcome.loc[condition]
    print("total number of subject w/ condition:" + str(outcome.shape[0]))

    #find SBL quartile cuttoffs of subjects w/ condition
    quantile = [np.quantile(x, i) for i in [0, 0.25, 0.5, 0.75, 1]]
    print("_______QUARTILE CUTOFFS_______")
    print(quantile)
    OR = np.zeros((2, 2))
    i = 0

    #find outcome data for subjects in 1st quartile
    found = outcome[(x >= quantile[i]) & (x <= quantile[i + 1])]
    print("total number of subjects in quantile: " + str(found.shape[0]))
    OR[1, 0] = (found == 1).sum()
    OR[1, 1] = (found == 0).sum()

    to_print=''
    #find outcome data for subjects in 2nd/3rd/4th quartile then compare w/ 1st quartile.
    for i in range(1, 4):
        found = outcome[(x > quantile[i]) & (x <= quantile[i + 1])]
        OR[0, 0] = (found == 1).sum()
        OR[0, 1] = (found == 0).sum()

        #calculate oddsratio
        oddsratio, pvalue = stats.fisher_exact(OR)
        LOR = np.log(OR[0,0]) + np.log(OR[1,1]) - np.log(OR[0,1]) - np.log(OR[1,0])
        SE = np.sqrt(np.sum(1/OR.astype(np.float64)))
        LCL = np.exp(LOR - 1.96*SE)
        UCL = np.exp(LOR + 1.96*SE)
        if pvalue <= 0.05:
            significance = '*'
        else:
            significance = ' '
        #print("Quantile: ", i + 1, "OR: ", oddsratio, "p-Value: %.4f"+significance % pvalue)
        to_print = to_print+("Q%d, OR: %.2f, p: %.4f"+significance+" CI: %.2f, %.2f") % (i + 1, oddsratio, pvalue, LCL, UCL) + '  '
    print("_______OUTCOMES TABLE for CALC ODDS RATIO________")
    print("1st column = outcome true. 1st row = 1st quartile. 2nd row = 2nd/3rd/4th quartile")
    print(OR)
    print("_______ODDS RATIO________")
    print(to_print)
    
# sum of all the absolute value of sbl difference.
baseline = sbl_jsn_0_mean
sbl_difference = (bml_all.loc[:, sbl_col_names].sub(baseline, axis=1))
sbl_difference_absolute = sbl_difference.abs().sum(1)
print(sbl_difference_absolute)

# get OR
conditions = {
              'BML_totals ≥1': bml_all['bml_total'] >= 1,
              }

# 'JSN Both ≥1': bml_all['V00XRJSL'] >= 1,
# 'Womac Score ≥ 1': bml_all['V00XRKL'] >= 1,
outcomes = {'Future TKR': (bml_all['V99E#KRPSN'] >= 1)}

for outcome in outcomes.keys():
    for condition in conditions.keys():
        print('******* ' + outcome + ' | ' + condition + ' *******')
        get_OR(x=sbl_difference_absolute, outcome=outcomes[outcome],
               condition=conditions[condition])
        print('')
    print('')


# #########################
# #QUARTILE MEAN/SD/MEDIAN
# #########################

def get_stats(x, outcome, condition):
    x = x.loc[condition]
    outcome = outcome.loc[condition]

    #create df w/ sbl_diff, outcomes, and quartile number.
    df = pd.DataFrame({'sbl_difference': x, 'outcome': outcome})
    quartile_label = pd.qcut(df['sbl_difference'], 4, labels=('Q1', 'Q2', 'Q3', 'Q4'))
    df.insert(2, 'Q', quartile_label)

    quartiles = ['Q1', 'Q2', 'Q3', 'Q4']
    #calculate mean/std/medial of each quartile
    for quartile in quartiles:
        q = df[df['Q'] == quartile]
        q_stat = q['sbl_difference']

        #stats for quartile
        print("__________________________")
        print("total subjects in " + quartile + ": " + str(q.shape[0]))
        qmn = q_stat.mean()
        qsd = q_stat.std()
        qmd = q_stat.median()
        print(quartile +' Mean: ', qmn, ' SD: ', qsd, ' Median: ', qmd)


        q_outcome_true = q[q['outcome'] == 1]
        q_stat = q_outcome_true['sbl_difference']

        #stats for quartile with outcome
        print("total subjects w/ outcome in " + quartile + ": " + str(q_outcome_true.shape[0]))
        qmn = q_stat.mean()
        qsd = q_stat.std()
        qmd = q_stat.median()
        print(quartile +' Mean: ', qmn, ' SD: ', qsd, ' Median: ', qmd)

#get OR stats
for outcome in outcomes.keys():
    for condition in conditions.keys():
        print('******* ' + outcome + ' | ' + condition + ' *******')
        get_stats(x=sbl_difference_absolute, outcome=outcomes[outcome],
               condition=conditions[condition])
        print('')
    print('')