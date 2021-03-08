import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

disease = 'MetaSyn'
year = '2019'
df = df_list[year]

def metasyn(x):
    sex = x['sex']
    sbp = x['HE_sbp']
    dbp = x['HE_dbp']
    hpdr =  x['DI1_pt']
    glu = x['HE_glu']
    dmdr = x['DE1_pt']
    tg = x['HE_TG']
    hldr = x['DI2_pt']
    hdl = x['HE_HDL_st2']
    wc = x['HE_wc']
    return (sum([
        ((sbp >= 130) | (dbp >= 85) | (hpdr == 1)),
        ((glu >= 100) | (dmdr == 1)),
        ((tg >= 150) | (hldr == 1)),
        ((sex == 1) & (hdl < 40)) | ((sex == 2) & (hdl < 50)),
        ((sex == 1) & (wc >= 90)) | ((sex == 2) & (wc >= 85)),
    ]) >= 3).astype(int)

df[disease] = metasyn(df)

df = df.query('age >= 20')

df_X = df.iloc[:, :-1]
df_Y = df.iloc[:, -1]

X_train, X_val, Y_train, Y_val = train_test_split(df_X, df_Y, test_size=0.1, random_state=1)
print(X_train.columns)
print(X_train.shape)
print(X_val.shape)




















# def Diab(he_glu, he_hba1c, de1_dg):
#     return ((he_glu >= 126) or (he_hba1c >= 6.5) or (de1_dg == 1))

# def metasyn(x):
#     sex = x['sex']
#     sbp = x['HE_sbp']
#     dbp = x['HE_dbp']
#     hpdr =  x['DI1_pt']
#     glu = x['HE_glu']
#     dmdr = x['DE1_pt']
#     tg = x['HE_TG']
#     hldr = x['DI2_pt']
#     hdl = x['HE_HDL_st2']
#     wc = x['HE_wc']
#     return (sum([
#         ((sbp >= 130) | (dbp >= 85) | (hpdr == 1)),
#         ((glu >= 100) | (dmdr == 1)),
#         ((tg >= 150) | (hldr == 1)),
#         ((sex == 1) & (hdl < 40)) | ((sex == 2) & (hdl < 50)),
#         ((sex == 1) & (wc >= 90)) | ((sex == 2) & (wc >= 85)),
#     ]) >= 3).astype(int)

# for year in df_list:
#     df_list[year]['Diab'] = [1 if Diab(he_glu, he_hba1c, de1_dg) else 0 for (he_glu, he_hba1c, de1_dg) in zip(df_list[year]['HE_glu'], df_list[year]['HE_HbA1c'], df_list[year]['DE1_dg'])]
#     df_list[year]['MetaSyn'] = metasyn(df_list[year])


# for year in df_list:
#     df_list[year] = df_list[year].query('age >= 20')

# disease = 'MetaSyn'

# def Diab(he_glu, he_hba1c, de1_dg):
#     return ((he_glu >= 126) or (he_hba1c >= 6.5) or (de1_dg == 1))

# def metasyn(x):
#     sex = x['sex']
#     sbp = x['HE_sbp']
#     dbp = x['HE_dbp']
#     hpdr =  x['DI1_pt']
#     glu = x['HE_glu']
#     dmdr = x['DE1_pt']
#     tg = x['HE_TG']
#     hldr = x['DI2_pt']
#     hdl = x['HE_HDL_st2']
#     wc = x['HE_wc']
#     return (sum([
#         ((sbp >= 130) | (dbp >= 85) | (hpdr == 1)),
#         ((glu >= 100) | (dmdr == 1)),
#         ((tg >= 150) | (hldr == 1)),
#         ((sex == 1) & (hdl < 40)) | ((sex == 2) & (hdl < 50)),
#         ((sex == 1) & (wc >= 90)) | ((sex == 2) & (wc >= 85)),
#     ]) >= 3).astype(int)

# for year in df_list:
#     df_list[year]['Diab'] = [1 if Diab(he_glu, he_hba1c, de1_dg) else 0 for (he_glu, he_hba1c, de1_dg) in zip(df_list[year]['HE_glu'], df_list[year]['HE_HbA1c'], df_list[year]['DE1_dg'])]
#     df_list[year]['MetaSyn'] = metasyn(df_list[year])

# conti_factor = ['age', 'HE_BMI', 'HE_PLS']
# cate_factor = ['sex', 'BD1_11', 'BD2_1', 'BS3_1', 'BE3_31', 'BE5_1', 'marri_1', 'house', 'edu', 'HE_HPfh1', 'HE_HPfh2', 'HE_HPfh3', 'HE_HLfh1', 'HE_HLfh2', 'HE_HLfh3', 'HE_DMfh1', 'HE_DMfh2', 'HE_DMfh3']

# tmp = {}
# for year in df_list:
#     tmp[year] = df_list[year][conti_factor + cate_factor + [disease]].dropna()
# df = pd.concat(tmp.values())

# cat_df = []
# cat_df.append(df[conti_factor])
# for idx in cate_factor:
#     cat_df.append(pd.get_dummies(df[idx], prefix=idx))
# cat_df.append(df[disease])
# df = pd.concat(cat_df, axis=1)
# df_X = df.iloc[:, :-1]
# df_Y = df.iloc[:, -1]

# conti_count = sum([1 for x in df.columns if x in conti_factor])
# scalar = StandardScaler()
# scalar.fit(df_X.iloc[:, :conti_count])
# df_X.iloc[:, :conti_count] = scalar.transform(df_X.iloc[:, :conti_count])
# X_train, X_val, Y_train, Y_val = train_test_split(df_X, df_Y, test_size=0.1, random_state=1)
# print(X_train.columns)
# print(X_train.shape)
# print(X_val.shape)