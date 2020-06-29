# Correlation-of-Personality and COVID19
Correlation Analysis of Big Five Personality &amp; Coronavirus disease 2019

# Likert scale
A Likert scale is basically a scale used to represent people's opinions and attitudes to a topic or subject matter. The Likert scale ranges from one extreme to another, for example “extremely likely” to “not at all likely. It uses psychometric testing to measure beliefs, attitudes, and opinions of subjects.

상관관계 분석을 위해서 모든 변수는 연속형 변수이거나, 서열척도여야 한다. 연속형변수 일때는 Pearson상관분석을, 서열척도일때는 Spearman상관분석을 사용한다. 리커트 척도이기 때문에 Pearson의 상관분석 기법으로 진행할거랍니다. 

# 응답 편향

# Distribution of Traits
![traits](https://user-images.githubusercontent.com/59387983/86057766-bc978d00-ba9a-11ea-83b5-a6a98915da48.png)

'OPN': '개방성','CSN': '성실성','EXT': '외향성','AGR': '친화성','EST': '안정성'

개방성과 친화성의 경우, 상대적으로 높은 점수 분포를 보인다. 반면 외향성과 안정성의 경우, 상대적으로 낮은 분포를 보인다.
성실성의 경우, 고르고 분포하고 있다.

# Answer Distribution of Positive Questions
![pos_questions](https://user-images.githubusercontent.com/59387983/86057749-b6a1ac00-ba9a-11ea-8000-fca084b082c2.png)

# Answer Distribution of Negative Questions
![neg_questions](https://user-images.githubusercontent.com/59387983/86057755-b7d2d900-ba9a-11ea-8b60-3291979193fc.png)

# Answer Distribution of KR, DE, US
![compare](https://user-images.githubusercontent.com/59387983/86057762-bacdc980-ba9a-11ea-94b6-39381719421e.png)

# Correlation of Openness & COVID-19
![OPN](https://user-images.githubusercontent.com/59387983/86057804-cc16d600-ba9a-11ea-92b8-b547ddb90854.png)

# Correlation of Conscientiousness & COVID-19
![CSN](https://user-images.githubusercontent.com/59387983/86057811-cd480300-ba9a-11ea-90c1-8ffe3d7f2d01.png)

# Correlation of Extroversion & COVID-19
![EXT](https://user-images.githubusercontent.com/59387983/86057816-cf11c680-ba9a-11ea-82d6-be7455ebe0df.png)

# Correlation of Agreeableness & COVID-19
![AGR](https://user-images.githubusercontent.com/59387983/86057818-d042f380-ba9a-11ea-924a-7afe54b3bee2.png)

# Correlation of Emotional Stability & COVID-19
![EST](https://user-images.githubusercontent.com/59387983/86057825-d1742080-ba9a-11ea-8cc5-3698494385ff.png)

# Dataset
1. Big Five Personality Test (1M Answers to 50 personality items, and technical information)

    The Big Five personality traits, also known as the five-factor model (FFM) and the OCEAN model, is a taxonomy, or grouping, for personality traits. When factor analysis (a statistical technique) is applied to personality survey data, some words used to describe aspects of personality are often applied to the same person. For example, someone described as conscientious is more likely to be described as "always prepared" rather than "messy". This theory is based therefore on the association between words but not on neuropsychological experiments. This theory uses descriptors of common language and therefore suggests five broad dimensions commonly used to describe the human personality and psyche.

    source : https://www.kaggle.com/tunguz/big-five-personality-test


2. Coronavirus disease 2019 (COVID-19)

    Coronavirus disease 2019 (COVID-19) time series listing confirmed cases, reported deaths and reported recoveries. Data is disaggregated by country (and sometimes subregion). Coronavirus disease (COVID-19) is caused by the Severe acute respiratory syndrome Coronavirus 2 (SARS-CoV-2) and has had a worldwide effect. On March 11 2020, the World Health Organization (WHO) declared it a pandemic, pointing to the over 118,000 cases of the coronavirus illness in over 110 countries and territories around the world at the time.

    This dataset includes time series data tracking the number of people affected by COVID-19 worldwide, including:

    confirmed tested cases of Coronavirus infection
    the number of people who have reportedly died while sick with Coronavirus
    the number of people who have reportedly recovered from it

    source : https://github.com/datasets/covid-19
