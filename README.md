# Correlation-of-Personality and COVID19
Correlation Analysis of Big Five Personality &amp; Coronavirus disease 2019

# Likert scale
A Likert scale is basically a scale used to represent people's opinions and attitudes to a topic or subject matter. The Likert scale ranges from one extreme to another, for example “extremely likely” to “not at all likely. It uses psychometric testing to measure beliefs, attitudes, and opinions of subjects.

리커트 척도
집단과 집단간의 상관관계가 있냐, 없냐를 확인하기 위한 분석방법 이라고 생각하시면 되는데요 
상관분석을 하기 위해서는 
모든 변수는 연속형 변수 이거나, 서열척도여야 합니다. 
​
단, 연속형변수 일때는 Pearson상관분석을, 서열척도일때는 Spearman상관분석을 합니다. 
그럼 한번 예를 들어볼께요 
서비스 만족도와 제품만족도, 요금만족도, 재구매 의향은 서로 상관관계가 있다 
이런 가설이 있을때 검증을 방식이 상관분석입니다. 
리커트 척도이기 때문에 Pearson의 상관분석 기법으로 진행할거랍니다. 
체크해주시고, 확인 버튼을 눌러주세요 

# Linear Mixed Model (LMM)

# Dataset
1. Big Five Personality Test (1M Answers to 50 personality items, and technical information)

    source : https://www.kaggle.com/tunguz/big-five-personality-test


2. Coronavirus disease 2019 (COVID-19)

    Coronavirus disease 2019 (COVID-19) time series listing confirmed cases, reported deaths and reported recoveries. Data is disaggregated by country (and sometimes subregion). Coronavirus disease (COVID-19) is caused by the Severe acute respiratory syndrome Coronavirus 2 (SARS-CoV-2) and has had a worldwide effect. On March 11 2020, the World Health Organization (WHO) declared it a pandemic, pointing to the over 118,000 cases of the coronavirus illness in over 110 countries and territories around the world at the time.

    This dataset includes time series data tracking the number of people affected by COVID-19 worldwide, including:

    confirmed tested cases of Coronavirus infection
    the number of people who have reportedly died while sick with Coronavirus
    the number of people who have reportedly recovered from it

    source : https://github.com/datasets/covid-19
