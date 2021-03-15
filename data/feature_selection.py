import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# from refine_knhanes import conti_factor, cate_factor, disease, df_list

tmp = {}
for year in df_list:
    tmp[year] = df_list[year][conti_factor + cate_factor + [disease]].dropna()
df_tmp = pd.concat(tmp.values())

# 카테고리 변수의 카테고리화
tmp = []
tmp.append(df_tmp[conti_factor])
for idx in cate_factor:
    tmp.append(pd.get_dummies(df_tmp[idx], prefix=idx, drop_first=True))
tmp.append(df_tmp[disease])
df_tmp = pd.concat(tmp, axis=1)

# 연속 변수들을 normalization
scalar = StandardScaler()
scalar.fit(df_tmp.iloc[:, :len(conti_factor)])
df_tmp.iloc[:, :len(conti_factor)] = scalar.transform(df_tmp.iloc[:, :len(conti_factor)])

# for f in conti_factor:
#     regressor = sm.Logit(df_tmp['MetaSyn'], df_tmp[f]).fit()
#     print(regressor.summary())

# for f in cate_factor:
#     tmp = [s for s in df_tmp.columns.values if f in s]
#     x_tmp = df_tmp[tmp]
#     regressor = sm.Logit(df_tmp['MetaSyn'], x_tmp).fit()
#     print(regressor.summary())
    
# None of the variables are insignificant via simple logistic regression run-through

# Backward elimination
regressor = sm.Logit(df_tmp['MetaSyn'], df_tmp.iloc[:, :-1]).fit()
# regressor = sm.Logit(df_tmp['MetaSyn'], X_train.iloc[:, X_opt]).fit()
while (np.max(regressor.pvalues) > 0.05):
    # p-value가 0.05보다 큰 항목이 있으면 가장 큰 항목부터 backward elimination 시행
    print(df_tmp.columns.values[np.argmax(regressor.pvalues)])
    df_tmp.drop(df_tmp.columns.values[np.argmax(regressor.pvalues)], inplace=True, axis=1)
    regressor = sm.Logit(df_tmp['MetaSyn'], df_tmp.iloc[:, :-1]).fit()

print(df_tmp.columns)