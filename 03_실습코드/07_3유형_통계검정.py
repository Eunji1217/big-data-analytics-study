# %% [markdown]
# # 07. 작업형 3유형 — scipy.stats 통계 검정 실습
# 개념서: 02_개념서/06_통계검정_3유형.md (2장 매핑표를 옆에 두고 진행)
# 모든 검정은 (통계량, p값) 을 돌려줌 → round 해서 print, α=0.05 로 채택/기각

# %%
import pandas as pd
import numpy as np
from scipy import stats
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data') if '__file__' in globals() else 'data'
bp = pd.read_csv(os.path.join(DATA, 'stats_bp.csv'))
score = pd.read_csv(os.path.join(DATA, 'stats_score.csv'))
chi = pd.read_csv(os.path.join(DATA, 'stats_chi.csv'))
print(bp.head(), score.head(), chi.head(), sep='\n\n')

def verdict(p, alpha=0.05):
    return '기각' if p < alpha else '채택'

# %% 1. 단일표본 t검정 — "복용 전 혈압 평균이 140 인가?"
stat, p = stats.ttest_1samp(bp['before'], popmean=140)
print(round(stat, 4), round(p, 4), verdict(p))
# 단측: "140 보다 큰가?"
stat, p = stats.ttest_1samp(bp['before'], 140, alternative='greater')
print(round(stat, 4), round(p, 4), verdict(p))

# %% 2. 독립표본 t검정 — "남/녀 복용 전 혈압 평균이 다른가?"
m = bp[bp['sex'] == 'M']['before']
f = bp[bp['sex'] == 'F']['before']
# 2-1 등분산 검정 먼저 (문제에서 요구하는 경우)
lstat, lp = stats.levene(m, f)
print('Levene', round(lstat, 4), round(lp, 4), '등분산' if lp >= 0.05 else '이분산')
# 2-2 t검정
stat, p = stats.ttest_ind(m, f, equal_var=(lp >= 0.05))
print(round(stat, 4), round(p, 4), verdict(p))
# 순서(m, f)를 바꾸면 stat 부호가 바뀜 — 문제 순서대로!

# %% 3. 대응표본 t검정 — "약 복용 후 혈압이 감소했는가?"
stat, p = stats.ttest_rel(bp['before'], bp['after'])                 # 양측
print(round(stat, 4), round(p, 4))
stat, p = stats.ttest_rel(bp['before'], bp['after'], alternative='greater')   # before > after (감소)
print(round(stat, 4), round(p, 4), verdict(p))
print('평균 차이:', round((bp['before'] - bp['after']).mean(), 4))

# %% 4. 정규성 검정 — Shapiro
stat, p = stats.shapiro(bp['before'])
print(round(stat, 4), round(p, 4), '정규성 만족' if p >= 0.05 else '정규성 불만족')
# 차이값의 정규성 (대응표본 전제 확인)
stat, p = stats.shapiro(bp['before'] - bp['after'])
print(round(p, 4))

# %% 5. 비모수 검정 (정규성 불만족 시 대체)
print(stats.mannwhitneyu(m, f))                       # 독립표본 대체
print(stats.wilcoxon(bp['before'], bp['after']))      # 대응표본 대체

# %% 6. 일원분산분석 ANOVA — "세 반 평균 점수가 같은가?"
groups = [g['score'].values for _, g in score.groupby('class')]
fstat, p = stats.f_oneway(*groups)
print(round(fstat, 4), round(p, 4), verdict(p))
# 비모수 대체
print(stats.kruskal(*groups))

# statsmodels 로 ANOVA 표 (자유도·제곱합까지)
from statsmodels.formula.api import ols
import statsmodels.api as sm
model = ols('score ~ C(class_)', data=score.rename(columns={'class': 'class_'})).fit()   # class 는 예약어라 이름 변경
table = sm.stats.anova_lm(model)
print(table)
print(round(table.loc['C(class_)', 'F'], 4), round(table.loc['C(class_)', 'PR(>F)'], 4))

# 이원분산분석: 반 × 수업방식
model2 = ols('score ~ C(class_) + C(method) + C(class_):C(method)', data=score.rename(columns={'class': 'class_'})).fit()
print(sm.stats.anova_lm(model2, typ=2).round(4))

# 사후검정 (Tukey) — 어느 집단끼리 다른지
from statsmodels.stats.multicomp import pairwise_tukeyhsd
print(pairwise_tukeyhsd(score['score'], score['class']))

# %% 7. 카이제곱 독립성 검정 — "성별과 구매여부가 관련 있는가?"
table = pd.crosstab(chi['sex'], chi['buy'])
print(table)
chi2, p, dof, expected = stats.chi2_contingency(table)
print(round(chi2, 4), round(p, 4), dof, verdict(p))
print(np.round(expected, 2))          # 기대빈도
# 보정 없이 (2x2 에서 Yates 보정 끄기) — 문제에 언급 있을 때만
print(stats.chi2_contingency(table, correction=False)[:2])

# %% 8. 카이제곱 적합도 검정 — "채널 비율이 1:1:1 인가?"
obs = chi['channel'].value_counts().sort_index()
exp = [len(chi) / 3] * 3
chi2, p = stats.chisquare(obs, f_exp=exp)
print(obs.tolist(), round(chi2, 4), round(p, 4), verdict(p))

# %% 9. 상관계수 검정
r, p = stats.pearsonr(bp['age'], bp['before'])
print(round(r, 4), round(p, 4))
rho, p = stats.spearmanr(bp['age'], bp['before'])
print(round(rho, 4), round(p, 4))
print(round(bp['age'].corr(bp['before']), 4))    # pandas (p값 없음)

# %% 10. 신뢰구간
x = bp['before']
n = len(x)
ci = stats.t.interval(0.95, df=n - 1, loc=x.mean(), scale=stats.sem(x))
print(round(ci[0], 4), round(ci[1], 4))
# 직접 계산
t = stats.t.ppf(0.975, n - 1)
print(round(x.mean() - t * x.std() / np.sqrt(n), 4), round(x.mean() + t * x.std() / np.sqrt(n), 4))

# %% 11. 확률분포 계산 (가끔 출제)
print(round(stats.norm.cdf(1.96), 4))                    # 0.975
print(round(stats.norm.ppf(0.95), 4))                    # 1.6449
print(round(1 - stats.norm.cdf(170, loc=165, scale=5), 4))   # 키 170 초과 확률
print(round(stats.binom.pmf(3, n=10, p=0.5), 4))         # 10번 중 3번 성공
print(round(stats.binom.cdf(3, n=10, p=0.5), 4))         # 3번 이하
print(round(stats.poisson.pmf(2, mu=3), 4))
print(round(stats.t.ppf(0.975, df=29), 4))
print(round(stats.chi2.ppf(0.95, df=3), 4))

# %% 12. 검정통계량 직접 계산 (공식 이해용)
# 단일표본 t = (x̄ - μ0) / (s/√n)
t_manual = (x.mean() - 140) / (x.std(ddof=1) / np.sqrt(n))
print(round(t_manual, 4), round(stats.ttest_1samp(x, 140)[0], 4))
# 양측 p = 2 * (1 - cdf(|t|))
print(round(2 * (1 - stats.t.cdf(abs(t_manual), n - 1)), 4))

# %% ★ 연습문제 (stats_* 데이터)
# Q1. 남성의 복용 후(after) 혈압 평균이 130 보다 작다고 할 수 있는가? (t, p 소수 넷째, 채택/기각)
# Q2. 반 A 와 반 C 의 점수 평균 차이 검정 (등분산 가정). t 통계량 절댓값, p
# Q3. channel 과 buy 의 독립성 검정. chi2 통계량, p, 자유도
# Q4. before 와 after 의 차이(before-after)에 대한 95% 신뢰구간 하한

# %% 정답
s, p = stats.ttest_1samp(bp[bp['sex'] == 'M']['after'], 130, alternative='less')
print('Q1', round(s, 4), round(p, 4), verdict(p))
s, p = stats.ttest_ind(score[score['class'] == 'A']['score'], score[score['class'] == 'C']['score'])
print('Q2', round(abs(s), 4), round(p, 4))
c2, p, d, _ = stats.chi2_contingency(pd.crosstab(chi['channel'], chi['buy']))
print('Q3', round(c2, 4), round(p, 4), d)
diff = bp['before'] - bp['after']
print('Q4', round(stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff))[0], 4))
