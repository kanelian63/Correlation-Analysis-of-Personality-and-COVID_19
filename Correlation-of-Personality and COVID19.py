import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from datetime import datetime, timedelta

plt.style.use('dark_background')
mpl.rcParams['axes.unicode_minus'] = False
"""
mpl.matplotlib_fname()
fm.get_fontconfig_fonts()
font_location = 'D:\\AI\\Char_COVID_Analysis\\NanumBarunGothic.ttf' # For Windows
font_name = fm.FontProperties(fname=font_location).get_name()
mpl.rc('font', family=font_name)
"""

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['font.size'] = 8

#%%

pos_questions = [
    'OPN1','OPN3','OPN5','OPN7','OPN8','OPN9','OPN10',        # 7 Openness 개방성
    'CSN1','CSN3','CSN5','CSN7','CSN9','CSN10',               # 6 Conscientiousness 성실성
    'EXT1','EXT3','EXT5','EXT7','EXT9',                       # 5 Extroversion 외향성
    'AGR2','AGR4','AGR6','AGR8','AGR9','AGR10',               # 6 Agreeableness 친화성
    'EST1','EST3','EST5','EST6','EST7','EST8','EST9','EST10', # 8 Emotional Stability 안정성(신경성)
]

neg_questions = [
    'OPN2','OPN4','OPN6',                # 3 Openness
    'CSN2','CSN4','CSN6','CSN8',         # 4 Conscientiousness
    'EXT2','EXT4','EXT6','EXT8','EXT10', # 5 Extroversion
    'AGR1','AGR3','AGR5','AGR7',         # 4 Agreeableness
    'EST2','EST4',                       # 2 Emotional Stability
]

usecols = pos_questions + neg_questions + ['country']

df = pd.read_csv('D:\\AI\\Char_COVID_Analysis\\IPIP-FFM-data-8Nov2018\\data-final.csv', sep='\t', usecols=usecols) # tsv

df = df.replace(0, np.nan).dropna(axis=0).reset_index(drop=True)
"""
# groupby + agg('function')
# grouby한 데이터에 agg 안의 함수를 적용
a = df.groupby('country').agg('count')
counts_values = a['EXT1'].sort_values(ascending=False)
counts_values.describe()

count       113.000000
mean       7717.646018
std       44938.922475
min         106.000000
25%         296.000000
50%         865.000000
75%        2997.000000
max      472320.000000
Name: EXT1, dtype: float64
"""

df_2 = (df.groupby('country').agg('count')['EXT1'] > 1000).reset_index()
fc = df_2[df_2['EXT1'] == True]['country']
df = df[df['country'].isin(fc)].reset_index(drop=True)
len(df)

#%%
# graph visualization
pos_questions = [
    'OPN1','OPN3','OPN5','OPN7','OPN8','OPN9','OPN10',        # 7 Openness 개방성
    'CSN1','CSN3','CSN5','CSN7','CSN9','CSN10',               # 6 Conscientiousness 성실성
    'EXT1','EXT3','EXT5','EXT7','EXT9',                       # 5 Extroversion 외향성
    'AGR2','AGR4','AGR6','AGR8','AGR9','AGR10',               # 6 Agreeableness 친화성
    'EST1','EST3','EST5','EST6','EST7','EST8','EST9','EST10', # 8 Emotional Stability 안정성(신경성)
]

neg_questions = [
    'OPN2','OPN4','OPN6',                # 3 Openness
    'CSN2','CSN4','CSN6','CSN8',         # 4 Conscientiousness
    'EXT2','EXT4','EXT6','EXT8','EXT10', # 5 Extroversion
    'AGR1','AGR3','AGR5','AGR7',         # 4 Agreeableness
    'EST2','EST4',                       # 2 Emotional Stability
]

fig = plt.figure(figsize=(20, 6))
for question in pos_questions:
    sns.distplot(df[question], kde=True, bins=20, axlabel=False)
fig.legend(pos_questions)
fig.savefig("D:\\AI\\Char_COVID_Analysis\\graph\\pos_questions.png")
    
fig = plt.figure(figsize=(20, 6))
for question in neg_questions:
    sns.distplot(df[question], kde=True, bins=20, axlabel=False)
fig.legend(neg_questions)
fig.savefig("D:\\AI\\Char_COVID_Analysis\\graph\\neg_questions.png")
#%%
"""
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# 개인별 응답 편향 제거
df_pre = df.copy()
df_mean = pd.DataFrame()
df_mean = df_pre.iloc[:,:-1].sum(axis=1) / 50

for i in df_pre.columns[:-1]:
    df_pre[i] = df_pre[i] - df_mean

df_pre
# 나라별 점수 합계
sum_values_country = df_pre.groupby('country').agg('mean').drop('mean', axis=1)
# 나라별 사람수 합계
sum_people_country = df_pre.groupby('country').agg('count')['EXT1']
# 나라별 항목의 평균 점수
mean_values_country = pd.DataFrame()

for i in range(len(sum_values_country)):
    mean_values_country = mean_values_country.append(sum_values_country.iloc[i] / sum_people_country.iloc[i])

# 컬럼 재정렬
mean_values_country = mean_values_country[['EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9',
       'EXT10', 'EST1', 'EST2', 'EST3', 'EST4', 'EST5', 'EST6', 'EST7', 'EST8',
       'EST9', 'EST10', 'AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7',
       'AGR8', 'AGR9', 'AGR10', 'CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6',
       'CSN7', 'CSN8', 'CSN9', 'CSN10', 'OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5',
       'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10']]

"""
#%%
df_pre = df.copy()
# 긍정 부정 질문 숫자와 점수에 가중치 보정
pos_questions_OPN = ['OPN1','OPN3','OPN5','OPN7','OPN8','OPN9','OPN10'] # 7 Openness 개방성
pos_questions_CSN = ['CSN1','CSN3','CSN5','CSN7','CSN9','CSN10'] # 6 Conscientiousness 성실성
pos_questions_EXT = ['EXT1','EXT3','EXT5','EXT7','EXT9'] # 5 Extroversion 외향성
pos_questions_AGR = ['AGR2','AGR4','AGR6','AGR8','AGR9','AGR10'] # 6 Agreeableness 친화성
pos_questions_EST = ['EST1','EST3','EST5','EST6','EST7','EST8','EST9','EST10'] # 8 Emotional Stability 안정성(신경성)

neg_questions_OPN = ['OPN2','OPN4','OPN6'] # 3 Openness
neg_questions_CSN = ['CSN2','CSN4','CSN6','CSN8'] # 4 Conscientiousness
neg_questions_EXT = ['EXT2','EXT4','EXT6','EXT8','EXT10'] # 5 Extroversion
neg_questions_AGR = ['AGR1','AGR3','AGR5','AGR7'] # 4 Agreeableness
neg_questions_EST = ['EST2','EST4'] # 2 Emotional Stability

df_pre[pos_questions_OPN] = df_pre[pos_questions_OPN] / len(pos_questions_OPN) * 0.5
df_pre[pos_questions_CSN] = df_pre[pos_questions_CSN] / len(pos_questions_CSN) * 0.5
df_pre[pos_questions_EXT] = df_pre[pos_questions_EXT] / len(pos_questions_EXT) * 0.5
df_pre[pos_questions_AGR] = df_pre[pos_questions_AGR] / len(pos_questions_AGR) * 0.5
df_pre[pos_questions_EST] = df_pre[pos_questions_EST] / len(pos_questions_EST) * 0.5

df_pre[neg_questions_OPN] = df_pre[neg_questions_OPN] / len(neg_questions_OPN) * 0.5
df_pre[neg_questions_CSN] = df_pre[neg_questions_CSN] / len(neg_questions_CSN) * 0.5
df_pre[neg_questions_EXT] = df_pre[neg_questions_EXT] / len(neg_questions_EXT) * 0.5
df_pre[neg_questions_AGR] = df_pre[neg_questions_AGR] / len(neg_questions_AGR) * 0.5
df_pre[neg_questions_EST] = df_pre[neg_questions_EST] / len(neg_questions_EST) * 0.5

#%%
"""
>>> df = pd.read_csv('NormalizeColumns.csv')
>>> x = df.values.astype(float)
>>> min_max_scaler = preprocessing.MinMaxScaler()
>>> x_scaled = min_max_scaler.fit_transform(x)
>>> df = pd.DataFrame(x_scaled, columns=df.columns)
>>> df

01. 표준화 (Standardization)
  - 수식: (요소값 - 평균) / 표준편차
  - 평균을 기준으로 얼마나 떨어져 있는지를 나타내는 값으로, 이 방법을 적용하려는 때는 2개 이상의 대상이 단위가 다를 때 대상 데이터를 같은 기준으로 볼 수 있게 합니다.
  - 또한 이 방법은 데이터를 다소 평평하게 하는(로그보다는 덜하지만 데이터의 진폭을 줄이는) 특성을 가집니다. 이 방법을 적용하면 간극이 줄어드는 효과가 발생하여 고객별 매출금액과 같이 간극이 큰 데이터의 간극을 줄이는 결과를 얻게 됩니다. 그 결과 분석 대상 고객군을 정하는 데 (약간의) 편의성을 제공하게 됩니다.
02. 정규화 (Normalization)
  - 수식: (요소값 - 최소값) / (최대값 - 최소값)
  - 정규화는 전체 구간을 0~100으로 설정하여 데이터를 관찰하는 방법입니다.

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df['EXT1'])
df = df.T
df.shape
df = df.drop('country')
df
import numpy as np
s = df[:3].values
type(s)
s.shape
xc = min_max_scaler.fit_transform(df[:3])
"""
#%%
#'OPN': '개방성','CSN': '성실성','EXT': '외향성','AGR': '친화성','EST': '안정성'
traits = ['OPN','CSN','EXT','AGR','EST']
df_pre_traits = pd.DataFrame()
df_pre = df.copy()
df_pre[pos_questions] = df_pre[pos_questions].replace({1:-2, 2:-1, 3:0, 4:1, 5:2})
df_pre[neg_questions] = df_pre[neg_questions].replace({1:2, 2:1, 3:0, 4:-1, 5:-2})


for i in traits:
    trait_cols = [col for col in df_pre.columns if i in col]
    df_pre_traits[i] = df_pre[trait_cols].sum(axis=1)

df_pre_traits['country'] = df['country']
df_pre_traits.head()


# traits 별 점수 분포
fig = plt.figure(figsize=(20, 6))

for trait in traits:
    sns.distplot(df_pre_traits[trait], kde=True, bins=60, axlabel=False)
fig.legend(traits)
fig.savefig("D:\\AI\\Char_COVID_Analysis\\graph\\traits.png")
#%%
fig = plt.figure(figsize=(12, 6))

sns.distplot(df_pre_traits[df_pre_traits['country'] == 'KR']['OPN'], bins=40, axlabel=False)
sns.distplot(df_pre_traits[df_pre_traits['country'] == 'DE']['OPN'], bins=40, axlabel=False)
sns.distplot(df_pre_traits[df_pre_traits['country'] == 'US']['OPN'], bins=40, axlabel=False)

fig.legend(['KR', 'DE', 'US'])
fig.savefig("D:\\AI\\Char_COVID_Analysis\\graph\\compare.png")
#%%
df_traits_mean = df_pre_traits.groupby('country').mean().rename_axis('country').reset_index()

df_traits_mean
#%%
df_covid = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv', parse_dates=['Date'])

df_covid.head()
#%%
cc = pd.read_csv('D:\\AI\\Char_COVID_Analysis\\COVID-19-personality-correlation-master\\dataset\\country_code.csv')

cc.head()
#%%
df_covid = df_covid[df_covid['Confirmed'] > 50].reset_index(drop=True)

df_covid = df_covid.groupby(['Country/Region', 'Date']).sum().reset_index()

df_covid[df_covid['Country/Region'] == 'US']
#%%
n_days = 14

filtered = (
    datetime.now() - df_covid.groupby('Country/Region')['Date'].min() > timedelta(days=n_days)
).reset_index().rename(columns={'Date': 'Filtered'})

filtered_countries = filtered[filtered['Filtered'] == True]['Country/Region']

df_covid = df_covid[df_covid['Country/Region'].isin(filtered_countries)]

df_covid_14days = df_covid.groupby('Country/Region').head(n_days).groupby('Country/Region').tail(1)

df_covid_14days
#%%
df_covid_14days = df_covid_14days.merge(cc, left_on='Country/Region', right_on='Name')

df_covid_14days = df_covid_14days.merge(df_traits_mean, left_on='Code', right_on='country')

df_covid_14days.sort_values('Confirmed', ascending=False)
#%%
new_df = df_covid_14days[
    ~df_covid_14days['country'].isin(['CN', 'TR'])
]

for trait in traits:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, n in enumerate(['Confirmed', 'Recovered', 'Deaths']):
        corr = pearsonr(
            new_df[trait], 
            new_df[n]
        )

        sns.regplot(x=trait, y=n, data=new_df, ax=axes[i])
        axes[i].set_title('%s, %s :: r=%.2f, p=%.2f' % (trait, n, corr[0], corr[1]))
        fig.savefig("D:\\AI\\Char_COVID_Analysis\\graph\\{}.png".format(trait))

#%%
# 리커트 척도는 연속형 변수이므로 Pearson 척도로 상관관계 분석을 해야 적절
new_df.sort_values('OPN', ascending=False)
#%%
for trait in traits:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, n in enumerate(['Confirmed', 'Recovered', 'Deaths']):
        corr = spearmanr(
            new_df[trait], 
            new_df[n]
        )

        sns.regplot(x=trait, y=n, data=new_df, ax=axes[i])
        axes[i].set_title('%s, %s :: r=%.2f, p=%.2f' % (trait, n, corr[0], corr[1]))





