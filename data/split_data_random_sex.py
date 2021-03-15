# libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# from refine_knhanes import conti_factor, cate_factor, disease, df_list

######################################################################## 성별 바꿀때마다 바꿔주기
sex = 2
cate_factor_tmp = copy.copy(cate_factor)

# concatenate dictionary into one df
tmp = {}
for year in df_list:
    tmp[year] = df_list[year][conti_factor + cate_factor + [disease]].dropna()
df = pd.concat(tmp.values())

######################################################################## 성별 바꿀때마다 바꿔주기
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
for idx in cate_factor_tmp: # drop_first=True for logistic regression
    cat_df.append(pd.get_dummies(df[idx], prefix=idx, drop_first=False))
cat_df.append(df[disease])
df = pd.concat(cat_df, axis=1)

df_X = df.iloc[:, :-1]
df_Y = df.iloc[:, -1]

# 연속 변수들을 normalization
conti_count = sum([1 for x in df.columns if x in conti_factor])
scalar = StandardScaler()
scalar.fit(df_X.iloc[:, :conti_count])
df_X.iloc[:, :conti_count] = scalar.transform(df_X.iloc[:, :conti_count])
# parameters : random_state = seed
X_train, X_val, Y_train, Y_val = train_test_split(df_X, df_Y, test_size=0.25)

print(X_train.columns)
print(X_train.shape)
print(X_val.shape)

# add prevalence of disease
print(np.sum(Y_train)/len(Y_train))
print(np.sum(Y_val)/len(Y_val))