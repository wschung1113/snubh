# libraries
import copy
import pandas as pd
import numpy as np

# n수 함수
def find_n(dict):
    n = 0
    for year in dict:
        print(year + ' : ' + str(dict[year].shape[0]))
        n += dict[year].shape[0]
    print(n)

# n수
find_n(df_list)

# target disease
disease = 'MetaSyn'

# 결측치 확인
na_feat = ['HE_HPfh1', 'HE_HLfh1', 'HE_DMfh1']
for year in df_list:
    print(year)
    print(df_list[year].loc[:, na_feat].isna().sum())

for i in range(len(na_feat)):
    na_feat[i] not in df_list['2010'].columns.values

############# 다시 돌릴 때 여기서부터 돌리면 됨
# 복사본 생성
df_list_copy = copy.deepcopy(df_list)

# 20세 이상 대상만 선별
for year in df_list_copy:
    df_list_copy[year] = df_list_copy[year].query('age >= 20')

# n수
find_n(df_list_copy)

# 대사증후군 Label 생성 함수
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

# 각 연도별 dataset에 대해 대사증후군 Label 생성
for year in df_list_copy:
    df_list_copy[year]['MetaSyn'] = metasyn(df_list_copy[year])

# 모름/무응답 drop
for year in df_list_copy:
    df_list_copy[year] = df_list_copy[year].query('BD1_11 <= 8')
    df_list_copy[year] = df_list_copy[year].query('BD2_1 <= 8')
    df_list_copy[year] = df_list_copy[year].query('BS3_1 <= 8')
    df_list_copy[year] = df_list_copy[year].query('BE3_31 <= 8')
    df_list_copy[year] = df_list_copy[year].query('BE5_1 <= 8')
    df_list_copy[year] = df_list_copy[year].query('marri_1 <= 2')
    # df_list_copy[year] = df_list_copy[year].query('marri_2 <= 8')
    df_list_copy[year] = df_list_copy[year].query('house <= 3')
    # df_list_copy[year] = df_list_copy[year].query('Total_slp_wk <= 1440') # 24시간 이상인 사람 배제
    # df_list_copy[year] = df_list_copy[year].query('BE8_1 <= 24') # 24시간 이상인 사람 배제
    # df_list_copy[year] = df_list_copy[year].query('BE8_2 <= 60') # 60분 이상인 사람 배제
    # df_list_copy[year] = df_list_copy[year].query('HE_fh != 9')
    # df_list_copy[year] = df_list_copy[year].query('HE_HPfh1 != 9')

# for year in df_list_copy:
#     print(pd.unique(df_list_copy[year]['marri_1']))


# n수
find_n(df_list_copy)

# 연속 변수 리스트
# Total_slp_wk : 주중 하루 평균 수면 시간; 5, 6기에 없음
# HE_PLS : 15초 맥박수
# HE_mPLS : 60초 맥박수; 결측치가 너~무 많음 그냥 HE_PLS 쓰자;
# BE8_1 : 평소 하루 앉아서 보내는 시간(시간); 5기에 없음
# BE8_2 : 평소 하루 앉아서 보내는 시간(분); 5기에 없음
conti_factor = ['age', 'HE_BMI', 'HE_wc', 'HE_PLS']

# 카테고리 변수 리스트
# BD1_11 : 1년간 음주빈도
# BD2_1 : 한번에 마시는 음주량
# BS3_1 : 현재 흡연여부
# HE_rPLS : 맥박 규칙성 여부; 불규칙적인 사람이 없음;
# HE_fh : 만성질환 의사진단 가족력 여부
# HE_HPfh1, 2, 3 : 고혈압 의사진단 여부 (부, 모, 형제)
# HE_HLfh1, 2, 3 : 고지혈증 의사진단 여부 (부, 모, 형제)
# HE_DMfh1, 2, 3 : 당뇨병 의사진단 여부 (부, 모, 형제)
# 부모형제자매 가족력 여부 변수 생성
def isHPfh(he_hpfh1, he_hpfh2, he_hpfh3):
    return ((he_hpfh1 == 1) | (he_hpfh2 == 1) | (he_hpfh3 == 1))

def isHLfh(he_hlfh1, he_hlfh2, he_hlfh3):
    return ((he_hlfh1 == 1) | (he_hlfh2 == 1) | (he_hlfh3 == 1))

def isDMfh(he_dmfh1, he_dmfh2, he_dmfh3):
    return ((he_dmfh1 == 1) | (he_dmfh2 == 1) | (he_dmfh3 == 1))

for year in df_list_copy:
    print(year)
    df_list_copy[year]['HE_HPfh'] = [1 if isHPfh(he_hpfh1, he_hpfh2, he_hpfh3) else 0 for (he_hpfh1, he_hpfh2, he_hpfh3) in zip(df_list_copy[year]['HE_HPfh1'], df_list_copy[year]['HE_HPfh2'], df_list_copy[year]['HE_HPfh3'])]
    df_list_copy[year]['HE_HLfh'] = [1 if isHLfh(he_hlfh1, he_hlfh2, he_hlfh3) else 0 for (he_hlfh1, he_hlfh2, he_hlfh3) in zip(df_list_copy[year]['HE_HLfh1'], df_list_copy[year]['HE_HLfh2'], df_list_copy[year]['HE_HLfh3'])]
    df_list_copy[year]['HE_DMfh'] = [1 if isDMfh(he_dmfh1, he_dmfh2, he_dmfh3) else 0 for (he_dmfh1, he_dmfh2, he_dmfh3) in zip(df_list_copy[year]['HE_DMfh1'], df_list_copy[year]['HE_DMfh2'], df_list_copy[year]['HE_DMfh3'])]

# n수
find_n(df_list_copy)

# 이 신체활동 변수들 쓰면 안되겠음; 결측치가 너무 많음;
# BE3_11 : 1주일간 격렬한 신체활동 일수; 7기에 없음; 2014, 2015 (6기)에도 없음
# BE3_21 : 1주일간 중등도 신체활동 일수; 7기에 없음; 2014, 2015 (6기)에도 없음
# BE3_31 : 1주일간 걷기 일수
# for year in ['2014', '2015', '2016', '2017', '2018']:
#     print(year)
#     print(df_list_copy[year].shape[0])
#     df_list_copy[year] = df_list_copy[year].query('BE3_72 < 8')
#     print(df_list_copy[year].shape[0])
#     df_list_copy[year] = df_list_copy[year].query('BE3_76 < 8')
#     print(df_list_copy[year].shape[0])
#     df_list_copy[year] = df_list_copy[year].query('BE3_82 < 8')
#     print(df_list_copy[year].shape[0])
#     df_list_copy[year] = df_list_copy[year].query('BE3_86 < 8')
#     print(df_list_copy[year].shape[0])
#     df_list_copy[year]['BE3_11'] = [sum([be3_72, be3_76]) if (sum([be3_72, be3_76]) <= 7) else 7 for (be3_72, be3_76) in zip(df_list_copy[year]['BE3_72'], df_list_copy[year]['BE3_76'])]
#     df_list_copy[year]['BE3_21'] = [sum([be3_82, be3_86]) if (sum([be3_82, be3_86]) <= 7) else 7 for (be3_82, be3_86) in zip(df_list_copy[year]['BE3_82'], df_list_copy[year]['BE3_86'])]
# for year in ['2010', '2011', '2012', '2013']:
#     df_list_copy[year] = df_list_copy[year].query('BE3_11 <= 8 & BE3_21 <= 8 & BE3_31 <= 8')

# marri_1 : 결혼여부
# marri_2 : 결혼상태
# house : 주택소유여부
# edu : 교육수준 재분류 코드
# ho_incm : 소득 4분위수 (가구)
# BE3_31 : 1주일간 걷기 일수
# BE5_1 : 1주일간 근력운동 일수
cate_factor = ['sex', 'BD1_11', 'BD2_1', 'BS3_1', 'HE_HPfh', 'HE_HLfh', 'HE_DMfh', 'BE3_31', 'BE5_1', 'marri_1', 'house', 'edu', 'ho_incm']

# df_tmp = {}
# for year in ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']:
#     print('#################### ' + year)
#     df_tmp[year] = df_list_copy[year][conti_factor + cate_factor + [disease]]
#     print(df_list[year].shape)
#     print(df_tmp[year].shape[0])
#     print(df_tmp[year].isna().sum())

# Training을 위한 데이터셋 생성
df_train = {}
for year in ['2013', '2014', '2015', '2016', '2017', '2018']:
    # print(year)
    df_train[year] = df_list_copy[year][conti_factor + cate_factor + [disease]].dropna()
df = pd.concat(df_train.values())
# 카테고리 변수의 카테고리화
cat_df = []
cat_df.append(df[conti_factor])
for idx in cate_factor:
    cat_df.append(pd.get_dummies(df[idx], prefix=idx))
cat_df.append(df[disease])
drop_list = []
# 추가적으로 제거할 변수 이름을 지정하는 곳
## XGB나 LightGBM의 경우는 drop 하지 않아도 가능
# drop_list = ['BD1_11_1.0', 'BD1_11_2.0', 'BD1_11_9.0', 'BS3_1_9.0', 'BS3_1_2.0', 'BS3_1_3.0', 'BS3_1_8.0', 'BD1_11_8.0', 'sex_1.0', 'age_50_0']
df = pd.concat(cat_df, axis=1).drop(drop_list, axis=1)
# 마지막 column이 질병 label이므로 이를 y로 지정
X_train = df.iloc[:, :-1]
Y_train = df.iloc[:, -1]

df_val = {}
#Validation set에 대해서도 같은 작업
for year in ['2010', '2011', '2012']:
    df_val[year] = df_list_copy[year][conti_factor + cate_factor + [disease]].dropna()
df = pd.concat(df_val.values())
cat_df = []
cat_df.append(df[conti_factor])
for idx in cate_factor:
    cat_df.append(pd.get_dummies(df[idx], prefix=idx))
cat_df.append(df[disease])
df = pd.concat(cat_df, axis=1).drop(drop_list, axis=1)
X_val = df.iloc[:, :-1]
Y_val = df.iloc[:, -1]
conti_count = sum([1 for x in df.columns if x in conti_factor])
scalar = StandardScaler()
scalar.fit(X_train.iloc[:, :conti_count])
# 연속 변수들을 normalization 하고
# 이에 맞게 validation set을 같이 transform 시킨다
X_train.iloc[:, :conti_count] = scalar.transform(X_train.iloc[:, :conti_count])
X_val.iloc[:, :conti_count] = scalar.transform(X_val.iloc[:, :conti_count])

X_train.columns
print(X_train.shape)
print(X_val.shape)

# pd.unique(df_train['2017']['marri_1'])

# # for year in df_list:
# #     print(pd.unique(df_list[year]['marri_1']))