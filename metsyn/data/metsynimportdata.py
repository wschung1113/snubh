import os
import sys

import pandas as pd

# 각 KNHANES 데이터 셋을 불러오기
df_list= {}
# 5기
df_list['2010'] = pd.read_sas('C:/Users/wschu/OneDrive/Desktop/snubh/교보생명/data/국건영/hn10_all.sas7bdat')
df_list['2011'] = pd.read_sas('C:/Users/wschu/OneDrive/Desktop/snubh/교보생명/data/국건영/hn11_all.sas7bdat')
df_list['2012'] = pd.read_sas('C:/Users/wschu/OneDrive/Desktop/snubh/교보생명/data/국건영/hn12_all.sas7bdat')
# 6기
df_list['2013'] = pd.read_sas('C:/Users/wschu/OneDrive/Desktop/snubh/교보생명/data/국건영/hn13_all.sas7bdat')
df_list['2014'] = pd.read_sas('C:/Users/wschu/OneDrive/Desktop/snubh/교보생명/data/국건영/hn14_all.sas7bdat')
df_list['2015'] = pd.read_sas('C:/Users/wschu/OneDrive/Desktop/snubh/교보생명/data/국건영/hn15_all.sas7bdat')
# 7기
df_list['2016'] = pd.read_sas('C:/Users/wschu/OneDrive/Desktop/snubh/교보생명/data/국건영/hn16_all.sas7bdat')
df_list['2017'] = pd.read_sas('C:/Users/wschu/OneDrive/Desktop/snubh/교보생명/data/국건영/hn17_all.sas7bdat')
df_list['2018'] = pd.read_sas('C:/Users/wschu/OneDrive/Desktop/snubh/교보생명/data/국건영/hn18_all.sas7bdat')