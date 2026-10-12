# 작업형 3유형 연습문제 풀이
# %%
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols, logit
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_실습코드', 'data') if '__file__' in globals() else '../03_실습코드/data'
bp = pd.read_csv(os.path.join(DATA, 'stats_bp.csv'))
chi = pd.read_csv(os.path.join(DATA, 'stats_chi.csv'))
score = pd.read_csv(os.path.join(DATA, 'stats_score.csv'))
adv = pd.read_csv(os.path.join(DATA, 'stats_adv.csv'))
dis = pd.read_csv(os.path.join(DATA, 'stats_disease.csv'))
V = lambda p: '기각' if p < 0.05 else '채택'

# %% 문제 1. 대응표본 (감소 → before > after → alternative='greater')
diff = bp['before'] - bp['after']
print(round(diff.mean(), 2))
s, p = stats.ttest_rel(bp['before'], bp['after'], alternative='greater')
print(round(s, 4)); print(round(p, 4)); print(V(p))

# %% 문제 2. 독립표본
m = bp[bp['sex'] == 'M']['before']; f = bp[bp['sex'] == 'F']['before']
lp = stats.levene(m, f)[1]
print(round(lp, 4))
s, p = stats.ttest_ind(m, f, equal_var=lp >= 0.05)
print(round(s, 4)); print(round(p, 4), V(p))

# %% 문제 3. 카이제곱 독립성
t = pd.crosstab(chi['sex'], chi['buy'])
c2, p, dof, exp = stats.chi2_contingency(t)
print(t.loc['M', 'Yes'])
exp_df = pd.DataFrame(exp, index=t.index, columns=t.columns)
print(round(exp_df.loc['M', 'Yes'], 2))
print(round(c2, 4)); print(round(p, 4), V(p))

# %% 문제 4. ANOVA
print(round(score.groupby('class')['score'].mean().max(), 2))
g = [x['score'] for _, x in score.groupby('class')]
fs, p = stats.f_oneway(*g)
print(round(fs, 4)); print(round(p, 4), V(p))

# %% 문제 5. 다중회귀
mdl = ols('Sales ~ TV + Radio + Newspaper', data=adv).fit()
print(round(mdl.params['Radio'], 4))
print(round(mdl.rsquared, 4))
pv = mdl.pvalues.drop('Intercept')
print(pv[pv >= 0.05].index.tolist())
print(round(mdl.conf_int().loc['Radio', 1], 4))
print(round(mdl.predict(pd.DataFrame({'TV': [100], 'Radio': [25], 'Newspaper': [30]}))[0], 2))

# %% 문제 6. 로지스틱
lg = logit('disease ~ smoke + age + exercise', data=dis).fit(disp=0)
print(round(np.exp(lg.params['smoke']), 3))
print(round(np.exp(lg.params['age'] * 5), 3))
print(round(lg.pvalues['exercise'], 4), '유의' if lg.pvalues['exercise'] < 0.05 else '유의하지 않음')
print(round(-2 * lg.llf, 2))
print(round(lg.predict(pd.DataFrame({'smoke': [1], 'age': [45], 'exercise': [1]}))[0], 3))

# %% 문제 7. 단일표본 + 정규성 + 신뢰구간
a = score[score['class'] == 'A']['score']
print(round(stats.shapiro(a)[1], 4))
s, p = stats.ttest_1samp(a, 70, alternative='greater')
print(round(s, 4)); print(round(p, 4), V(p))
print(round(stats.t.interval(0.95, len(a) - 1, loc=a.mean(), scale=stats.sem(a))[0], 3))

# %% 문제 8. 적합도 (알파벳순: SNS, Search, TV → 비율 4:3:3 순서도 맞춰야 함!)
obs = chi['channel'].value_counts().sort_index()      # SNS, Search, TV
print(obs.tolist())
ratio = pd.Series({'SNS': 0.4, 'Search': 0.3, 'TV': 0.3})
exp = ratio[obs.index] * len(chi)
c2, p = stats.chisquare(obs, f_exp=exp)
print(round(c2, 4)); print(round(p, 4), V(p))
