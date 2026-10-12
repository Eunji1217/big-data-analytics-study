# %% [markdown]
# # 08. 작업형 3유형 — 회귀분석 · 로지스틱 회귀 (statsmodels)
# 개념서: 02_개념서/06_통계검정_3유형.md (6·7장)
# 3유형에서 "회귀계수 / p값 / R² / 신뢰구간 / 오즈비 / 예측값" 을 묻는 문제가 자주 나옴

# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data') if '__file__' in globals() else 'data'
adv = pd.read_csv(os.path.join(DATA, 'stats_adv.csv'))
dis = pd.read_csv(os.path.join(DATA, 'stats_disease.csv'))
print(adv.head(), dis.head(), sep='\n\n')

# %% 1. 단순 선형회귀 — Sales ~ TV
m1 = ols('Sales ~ TV', data=adv).fit()
print(m1.summary())                        # 연습용. 시험에선 아래처럼 개별 값만 print
print('기울기 :', round(m1.params['TV'], 4))
print('절편   :', round(m1.params['Intercept'], 4))
print('R2     :', round(m1.rsquared, 4))
print('p(TV)  :', round(m1.pvalues['TV'], 4))

# %% 2. 다중 선형회귀 — Sales ~ TV + Radio + Newspaper
m2 = ols('Sales ~ TV + Radio + Newspaper', data=adv).fit()
print(m2.params.round(4))                  # 모든 계수
print(m2.pvalues.round(4))                 # 계수별 p값
print('R2:', round(m2.rsquared, 4), ' adj R2:', round(m2.rsquared_adj, 4))
print('F :', round(m2.fvalue, 4), ' F p:', round(m2.f_pvalue, 6))
print(m2.conf_int().round(4))              # 95% 신뢰구간
print(m2.conf_int(alpha=0.01).loc['Radio'].round(4))   # Radio 계수의 99% CI
print(m2.bse.round(4))                     # 표준오차
print(m2.tvalues.round(4))                 # t값
print('AIC:', round(m2.aic, 2), ' BIC:', round(m2.bic, 2))

# "유의수준 0.05 에서 유의한 변수 개수"
print((m2.pvalues.drop('Intercept') < 0.05).sum())
# "계수가 가장 큰 변수"
print(m2.params.drop('Intercept').idxmax())

# %% 3. 예측 & 잔차
new = pd.DataFrame({'TV': [200], 'Radio': [30], 'Newspaper': [50]})
print(round(m2.predict(new)[0], 4))
resid = m2.resid
print('잔차 평균:', round(resid.mean(), 6), ' 잔차 표준편차:', round(resid.std(), 4))
print('RMSE:', round(np.sqrt((resid ** 2).mean()), 4))
# 잔차 정규성
from scipy import stats
print('Shapiro p:', round(stats.shapiro(resid)[1], 4))
# Durbin-Watson (자기상관, 2 근처면 OK)
from statsmodels.stats.stattools import durbin_watson
print('DW:', round(durbin_watson(resid), 4))

# %% 4. 범주형 변수 포함 / 상호작용 / 절편 제거
score = pd.read_csv(os.path.join(DATA, 'stats_score.csv')).rename(columns={'class': 'cls'})
m3 = ols('score ~ C(cls) + C(method)', data=score).fit()
print(m3.params.round(4))                  # C(cls)[T.B] = A 대비 B 의 효과
m4 = ols('Sales ~ TV * Radio', data=adv).fit()           # TV + Radio + TV:Radio
print(m4.params.round(6))
m5 = ols('Sales ~ TV - 1', data=adv).fit()               # 절편 없음
print(m5.params.round(4))

# %% 5. sm.OLS 방식 (formula 없이) — 상수항 직접 추가
X = sm.add_constant(adv[['TV', 'Radio']])
m6 = sm.OLS(adv['Sales'], X).fit()
print(m6.params.round(4))

# %% 6. 로지스틱 회귀 — disease ~ smoke + age + exercise
lg = logit('disease ~ smoke + age + exercise', data=dis).fit()
print(lg.summary())
print(lg.params.round(4))                  # 로그오즈 계수
print(np.exp(lg.params).round(4))          # 오즈비 ★
print('smoke 오즈비:', round(np.exp(lg.params['smoke']), 4))
print(lg.pvalues.round(4))
print('로그우도:', round(lg.llf, 4), ' 잔차이탈도:', round(-2 * lg.llf, 4))
print('AIC:', round(lg.aic, 4))
print('유사 R2:', round(lg.prsquared, 4))

# 예측 확률
new = pd.DataFrame({'smoke': [1], 'age': [50], 'exercise': [0]})
print(round(lg.predict(new)[0], 4))
# "age 가 10 증가할 때 오즈 몇 배?"
print(round(np.exp(lg.params['age'] * 10), 4))

# 예측 정확도
pred_label = (lg.predict(dis) >= 0.5).astype(int)
print('accuracy:', round((pred_label == dis['disease']).mean(), 4))

# %% 7. sklearn 선형회귀와 비교 (계수만 필요할 때)
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(adv[['TV', 'Radio', 'Newspaper']], adv['Sales'])
print(lr.coef_.round(4), round(lr.intercept_, 4))
print(round(lr.score(adv[['TV', 'Radio', 'Newspaper']], adv['Sales']), 4))   # R2

# %% 8. 다중공선성 VIF (가끔 출제)
from statsmodels.stats.outliers_influence import variance_inflation_factor
Xv = sm.add_constant(adv[['TV', 'Radio', 'Newspaper']])
vif = pd.Series([variance_inflation_factor(Xv.values, i) for i in range(Xv.shape[1])], index=Xv.columns)
print(vif.round(4))

# %% ★ 연습문제
# Q1. Sales ~ Radio + Newspaper 모형에서 Newspaper 계수의 p값 (소수 넷째), 유의한가?
# Q2. 위 모형의 수정 결정계수
# Q3. TV=150, Radio=20, Newspaper=10 일 때 Sales ~ TV+Radio+Newspaper 모형의 예측값 (소수 둘째)
# Q4. disease ~ smoke + age 로지스틱에서 smoke 의 오즈비 (소수 셋째)
# Q5. 위 모형에서 smoke=0, age=40 인 사람의 질병 확률 (소수 셋째)

# %% 정답
q = ols('Sales ~ Radio + Newspaper', data=adv).fit()
print('Q1', round(q.pvalues['Newspaper'], 4), '유의' if q.pvalues['Newspaper'] < 0.05 else '유의하지 않음')
print('Q2', round(q.rsquared_adj, 4))
print('Q3', round(m2.predict(pd.DataFrame({'TV': [150], 'Radio': [20], 'Newspaper': [10]}))[0], 2))
q4 = logit('disease ~ smoke + age', data=dis).fit(disp=0)
print('Q4', round(np.exp(q4.params['smoke']), 3))
print('Q5', round(q4.predict(pd.DataFrame({'smoke': [0], 'age': [40]}))[0], 3))
