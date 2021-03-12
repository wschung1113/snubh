# libraries
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler

# from refine_knhanes_sex import conti_factor, cate_factor, disease, df_list, find_n

#################################################### 성별 바꿀때마다 바꿔주기
# 트레이닝 할 성별 지정 1 남자 2 여자
sex = 2

# 각 연도 n수
find_n(df_list)

# train/val split
train_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']
val_years = ['2019']

cate_factor_tmp = copy.copy(cate_factor)

## training data
df_train = {}
for year in train_years:
    df_train[year] = df_list[year][conti_factor + cate_factor + [disease]].dropna()
df = pd.concat(df_train.values())

#################################################### 성별 바꿀때마다 바꿔주기
df = df.query('sex == 2')
# sex drop
df = df.drop('sex', axis = 1)
cate_factor_tmp.remove('sex')
# 생리여부 drop
if sex == 1:
    df = df.drop('HE_mens', axis = 1)
    cate_factor_tmp.remove('HE_mens')

# 카테고리 변수의 카테고리화
cat_df = []
cat_df.append(df[conti_factor])
for idx in cate_factor_tmp:
    cat_df.append(pd.get_dummies(df[idx], prefix=idx))
cat_df.append(df[disease])
df = pd.concat(cat_df, axis=1)
# 마지막 column이 질병 label이므로 이를 y로 지정
X_train = df.iloc[:, :-1]
Y_train = df.iloc[:, -1]

cate_factor_tmp = copy.copy(cate_factor)

## validation data
df_val = {}
for year in val_years:
    df_val[year] = df_list[year][conti_factor + cate_factor + [disease]].dropna()
df = pd.concat(df_val.values())

#################################################### 성별 바꿀때마다 바꿔주기
df = df.query('sex == 2')
# sex drop
df = df.drop('sex', axis = 1)
cate_factor_tmp.remove('sex')
# 생리여부 drop
if sex == 1:
    df = df.drop('HE_mens', axis = 1)
    cate_factor_tmp.remove('HE_mens')

cat_df = []
cat_df.append(df[conti_factor])
for idx in cate_factor_tmp:
    cat_df.append(pd.get_dummies(df[idx], prefix=idx))
cat_df.append(df[disease])
df = pd.concat(cat_df, axis=1)
X_val = df.iloc[:, :-1]
Y_val = df.iloc[:, -1]

# 연속 변수들을 normalization 하고
# 이에 맞게 validation set을 같이 transform 시킨다
conti_count = sum([1 for x in df.columns if x in conti_factor])
scalar = StandardScaler()
scalar.fit(X_train.iloc[:, :conti_count])
X_train.iloc[:, :conti_count] = scalar.transform(X_train.iloc[:, :conti_count])
X_val.iloc[:, :conti_count] = scalar.transform(X_val.iloc[:, :conti_count])

print(X_train.columns)
print(X_train.shape)
print(X_val.shape)

# add prevalence of disease
print(np.sum(Y_train)/len(Y_train))
print(np.sum(Y_val)/len(Y_val))