# libraries
import copy
import pandas as pd
import numpy as np

# n수 보는 함수
def find_n(dict):
    n = 0
    for year in dict:
        print(year + ' : ' + str(dict[year].shape[0]))
        n += dict[year].shape[0]
    print(n)

# 오리지널 데이터 n수
find_n(df_list)

# target disease
disease = 'MetaSyn'

# 복사본 생성
df_list_copy = copy.deepcopy(df_list)

# 만20세 이상 대상만 선별
for year in df_list:
    df_list[year] = df_list[year].query('age >= 20')

# exclusion criteria; 보류; 이유를 찾기 전까지는
# 뇌졸중
# for year in df_list:
#     df_list[year] = df_list[year].query('DI3_dg != 1')


# 당뇨병 Label 생성 함수
def Diab(he_glu, he_hba1c, de1_dg):
    return ((he_glu >= 126) or (he_hba1c >= 6.5) or (de1_dg == 1))

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

# 각 연도별 dataset에 대해 'disease' Label 생성
for year in df_list:
    df_list[year]['Diab'] = [1 if Diab(he_glu, he_hba1c, de1_dg) else 0 for (he_glu, he_hba1c, de1_dg) in zip(df_list[year]['HE_glu'], df_list[year]['HE_HbA1c'], df_list[year]['DE1_dg'])]
    df_list[year]['MetaSyn'] = metasyn(df_list[year])

# 연속 변수 리스트
# BP8 : 하루 평균 수면 시간; 5, 6기
# Total_slp_wk : 주중 하루 평균 수면 시간 (분);
# 8기에는 BP16_1 : 주중 하루 평균 수면 시간
# 7기
for year in ['2016', '2017', '2018']: 
    df_list[year]['BP8'] = (df_list[year]['Total_slp_wk']/60*5 + df_list[year]['Total_slp_wd']/60*2)/7

# 8기
df_list['2019']['BP8'] = (df_list['2019']['BP16_1']*5 + df_list['2019']['BP16_2']*2)/7


# HE_PLS : 15초 맥박수
# HE_mPLS : 60초 맥박수; 결측치가 너~무 많음 그냥 HE_PLS 쓰자;
# BE8_1 : 평소 하루 앉아서 보내는 시간(시간); 5기에 없음
# BE8_2 : 평소 하루 앉아서 보내는 시간(분); 5기에 없음
# 대사증후군 돌릴때는 HE_wc;허리둘레 빼기; 다시 넣기로 함
conti_factor = ['age', 'HE_BMI', 'HE_PLS', 'HE_wc', 'BP8']
# conti_factor = ['age', 'HE_BMI', 'HE_PLS', 'HE_wc', 'HE_sbp', 'HE_dbp', 'BP8']

# 모름/무응답 drop
for year in df_list:
    df_list[year] = df_list[year].query('BD1_11 <= 8')
    df_list[year] = df_list[year].query('BD2_1 <= 8')
    df_list[year] = df_list[year].query('BS3_1 <= 8')
    df_list[year] = df_list[year].query('BE3_31 <= 8')
    df_list[year] = df_list[year].query('BE5_1 <= 8')
    df_list[year] = df_list[year].query('marri_1 <= 2')
    df_list[year] = df_list[year].query('house <= 3')
    df_list[year] = df_list[year].query('BP8 <= 24')

    # df_list[year] = df_list[year].query('marri_2 <= 8')
    # df_list[year] = df_list[year].query('Total_slp_wk <= 1440') # 24시간 이상인 사람 배제
    # df_list[year] = df_list[year].query('BE8_1 <= 24') # 24시간 이상인 사람 배제
    # df_list[year] = df_list[year].query('BE8_2 <= 60') # 60분 이상인 사람 배제
    # df_list[year] = df_list[year].query('HE_fh != 9')
    # df_list[year] = df_list[year].query('HE_HPfh1 != 9')

# 카테고리 변수 리스트
# BD1_11 : 1년간 음주빈도
# BD2_1 : 한번에 마시는 음주량
# BS3_1 : 현재 흡연여부
# HE_rPLS : 맥박 규칙성 여부; 불규칙적인 사람이 없음;
# HE_fh : 만성질환 의사진단 가족력 여부
# HE_HPfh1, 2, 3 : 고혈압 의사진단 여부 (부, 모, 형제)
# HE_HLfh1, 2, 3 : 고지혈증 의사진단 여부 (부, 모, 형제)
# HE_DMfh1, 2, 3 : 당뇨병 의사진단 여부 (부, 모, 형제)
# 뇌졸중, 허혈성심질환
# 부모형제자매 가족력 여부 변수 생성
def isHPfh(he_hpfh1, he_hpfh2, he_hpfh3):
    return ((he_hpfh1 == 1) | (he_hpfh2 == 1) | (he_hpfh3 == 1))

def isHLfh(he_hlfh1, he_hlfh2, he_hlfh3):
    return ((he_hlfh1 == 1) | (he_hlfh2 == 1) | (he_hlfh3 == 1))

def isDMfh(he_dmfh1, he_dmfh2, he_dmfh3):
    return ((he_dmfh1 == 1) | (he_dmfh2 == 1) | (he_dmfh3 == 1))

def isIHDfh(he_ihdfh1, he_ihdfh2, he_ihdfh3):
    return ((he_ihdfh1 == 1) | (he_ihdfh2 == 1) | (he_ihdfh3 == 1))

def isSTRfh(he_strfh1, he_strfh2, he_strfh3):
    return ((he_strfh1 == 1) | (he_strfh2 == 1) | (he_strfh3 == 1))

for year in df_list:
    df_list[year]['HE_HPfh'] = [1 if isHPfh(he_hpfh1, he_hpfh2, he_hpfh3) else 0 for (he_hpfh1, he_hpfh2, he_hpfh3) in zip(df_list[year]['HE_HPfh1'], df_list[year]['HE_HPfh2'], df_list[year]['HE_HPfh3'])]
    df_list[year]['HE_HLfh'] = [1 if isHLfh(he_hlfh1, he_hlfh2, he_hlfh3) else 0 for (he_hlfh1, he_hlfh2, he_hlfh3) in zip(df_list[year]['HE_HLfh1'], df_list[year]['HE_HLfh2'], df_list[year]['HE_HLfh3'])]
    df_list[year]['HE_DMfh'] = [1 if isDMfh(he_dmfh1, he_dmfh2, he_dmfh3) else 0 for (he_dmfh1, he_dmfh2, he_dmfh3) in zip(df_list[year]['HE_DMfh1'], df_list[year]['HE_DMfh2'], df_list[year]['HE_DMfh3'])]
    df_list[year]['HE_IHDfh'] = [1 if isIHDfh(he_ihdfh1, he_ihdfh2, he_ihdfh3) else 0 for (he_ihdfh1, he_ihdfh2, he_ihdfh3) in zip(df_list[year]['HE_IHDfh1'], df_list[year]['HE_IHDfh2'], df_list[year]['HE_IHDfh3'])]
    df_list[year]['HE_STRfh'] = [1 if isSTRfh(he_strfh1, he_strfh2, he_strfh3) else 0 for (he_strfh1, he_strfh2, he_strfh3) in zip(df_list[year]['HE_STRfh1'], df_list[year]['HE_STRfh2'], df_list[year]['HE_STRfh3'])]

# marri_1 : 결혼여부
# marri_2 : 결혼상태
# house : 주택소유여부
# edu : 교육수준 재분류 코드
# ho_incm : 소득 4분위수 (가구); 유의미하지 않다고 나와서 일단 뺌
# BE3_31 : 1주일간 걷기 일수
# BE5_1 : 1주일간 근력운동 일수
cate_factor = ['sex', 'BD1_11', 'BD2_1', 'BS3_1', 'BE3_31', 'BE5_1', 'marri_1', 'house', 'edu', 'HE_HPfh', 'HE_HLfh', 'HE_DMfh', 'HE_IHDfh', 'HE_STRfh', 'HE_mens']
# cate_factor = ['sex', 'HE_DMfh']



























# BE3_11 : 1주일간 격렬한 신체활동 일수; 7기에 없음; 2014, 2015 (6기)에도 없음
# BE3_21 : 1주일간 중등도 신체활동 일수; 7기에 없음; 2014, 2015 (6기)에도 없음
# BE3_31 : 1주일간 걷기 일수
# 이 신체활동 변수들 쓰면 안되겠음; 결측치가 너무 많음;
# for year in ['2014', '2015', '2016', '2017', '2018']:
#     print(year)
#     print(df_list[year].shape[0])
#     df_list[year] = df_list[year].query('BE3_72 < 8')
#     print(df_list[year].shape[0])
#     df_list[year] = df_list[year].query('BE3_76 < 8')
#     print(df_list[year].shape[0])
#     df_list[year] = df_list[year].query('BE3_82 < 8')
#     print(df_list[year].shape[0])
#     df_list[year] = df_list[year].query('BE3_86 < 8')
#     print(df_list[year].shape[0])
#     df_list[year]['BE3_11'] = [sum([be3_72, be3_76]) if (sum([be3_72, be3_76]) <= 7) else 7 for (be3_72, be3_76) in zip(df_list[year]['BE3_72'], df_list[year]['BE3_76'])]
#     df_list[year]['BE3_21'] = [sum([be3_82, be3_86]) if (sum([be3_82, be3_86]) <= 7) else 7 for (be3_82, be3_86) in zip(df_list[year]['BE3_82'], df_list[year]['BE3_86'])]
# for year in ['2010', '2011', '2012', '2013']:
#     df_list[year] = df_list[year].query('BE3_11 <= 8 & BE3_21 <= 8 & BE3_31 <= 8')