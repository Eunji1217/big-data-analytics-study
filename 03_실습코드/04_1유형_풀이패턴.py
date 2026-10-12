# %% [markdown]
# # 04. 작업형 1유형 풀이 패턴 모음
# 1유형은 "문제 문장을 pandas 한 줄씩으로 번역" 하는 연습입니다.
# 각 패턴을 **문제 → 코드** 로 익힌 뒤, 04_기출유형_연습문제/1유형 으로 넘어가세요.
#
# 풀이 습관:
#   1) 문장을 단계로 쪼갠다   2) 한 단계마다 print 로 확인   3) 마지막에 요구 형식으로 print 하나만

# %%
import pandas as pd
import numpy as np
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data') if '__file__' in globals() else 'data'
cust = pd.read_csv(os.path.join(DATA, 'customers.csv'))
sales = pd.read_csv(os.path.join(DATA, 'sales.csv'))

# %% 패턴 1. 정렬 → 상위 N → 집계
# "purchase_amt 상위 20명의 visit_cnt 합계"
print(cust.sort_values('purchase_amt', ascending=False).head(20)['visit_cnt'].sum())

# %% 패턴 2. 조건 → 개수/비율
# "income 4000 이상이면서 grade 가 VIP 가 아닌 고객 비율 (소수 셋째 자리)"
cond = (cust['income'] >= 4000) & (cust['grade'] != 'VIP')
print(round(cond.mean(), 3))          # bool Series 의 mean = 비율
print(round(cond.sum() / len(cust), 3))

# %% 패턴 3. 결측 처리 후 통계
# "age 결측을 중앙값으로 채운 후, 60세 이상 고객 수"
age = cust['age'].fillna(cust['age'].median())
print((age >= 60).sum())

# %% 패턴 4. 이상치 개수 (IQR)
q1, q3 = cust['income'].quantile([0.25, 0.75])
iqr = q3 - q1
print(((cust['income'] < q1 - 1.5 * iqr) | (cust['income'] > q3 + 1.5 * iqr)).sum())

# %% 패턴 5. 그룹 집계 → 최댓값 그룹
# "city 별 평균 purchase_amt 가 가장 큰 도시와 그 값(정수)"
g = cust.groupby('city')['purchase_amt'].mean()
print(g.idxmax(), int(g.max()))

# %% 패턴 6. 그룹별 상위 1개 → 다시 집계
# "grade 별로 income 이 가장 높은 고객의 age 평균"
top = cust.dropna(subset=['income']).sort_values('income', ascending=False).groupby('grade').head(1)
print(round(top['age'].mean(), 2))

# %% 패턴 7. 날짜 → 특정 기간/요일/월
sales['date'] = pd.to_datetime(sales['date'])
# "2024년 상반기(1~6월) 강남점 매출 합"
m = (sales['date'].dt.year == 2024) & (sales['date'].dt.month <= 6) & (sales['store'] == '강남점')
print(int(sales[m]['amount'].sum()))
# "월별 매출이 가장 높은 달 (YYYY-MM)"
print(sales.groupby(sales['date'].dt.to_period('M'))['amount'].sum().idxmax())
# "요일별 주문 건수 중 가장 적은 요일"
print(sales['date'].dt.day_name().value_counts().idxmin())

# %% 패턴 8. 표준화/정규화 후 조건
# "purchase_amt 를 표준화한 값이 2 이상인 고객 수"
z = (cust['purchase_amt'] - cust['purchase_amt'].mean()) / cust['purchase_amt'].std()
print((z >= 2).sum())

# %% 패턴 9. 문자열 처리 후 집계
# "phone 가운데 4자리가 5000 이상인 고객 수"
mid = cust['phone'].str.split('-').str[1].astype(int)
print((mid >= 5000).sum())

# %% 패턴 10. 상관계수
# "수치형 컬럼 중 purchase_amt 와 상관이 가장 높은 컬럼 (자기 자신 제외)"
c = cust.select_dtypes('number').corr()['purchase_amt'].drop('purchase_amt')
print(c.abs().idxmax(), round(c.abs().max(), 4))

# %% 패턴 11. 누적합 / 변화율
# "날짜순 누적 매출이 처음으로 1억을 넘는 날짜"
s = sales.sort_values('date')
s['cum'] = s['amount'].cumsum()
print(s[s['cum'] >= 1e8]['date'].iloc[0].strftime('%Y-%m-%d'))
# "월별 매출의 전월 대비 증가율이 가장 큰 달"
mon = sales.groupby(sales['date'].dt.to_period('M'))['amount'].sum()
print(mon.pct_change().idxmax())

# %% 패턴 12. 구간화 후 집계
# "age 를 10대/20대/... 로 나눴을 때 인원이 가장 많은 연령대"
a = cust['age'].dropna()
band = (a // 10 * 10).astype(int)
print(band.value_counts().idxmax())

# %% 패턴 13. 왜도/첨도, 로그변환
print(round(cust['purchase_amt'].skew(), 3), round(np.log1p(cust['purchase_amt']).skew(), 3))

# %% 패턴 14. 결측 비율 / 특정 조건 컬럼 찾기
print(cust.isnull().mean().round(3))
print(cust.columns[cust.isnull().mean() > 0.05].tolist())    # 결측 5% 초과 컬럼

# %% 패턴 15. 두 컬럼 차이·비율 파생 후 정렬
# "purchase_amt / visit_cnt (방문당 구매액) 상위 5명 name"
cust['per_visit'] = cust['purchase_amt'] / cust['visit_cnt'].replace(0, np.nan)
print(cust.nlargest(5, 'per_visit')['name'].tolist())

# %% 패턴 16. 데이터 형식 변환 문제 (wide ↔ long)
pv = pd.pivot_table(sales, index='store', columns='category', values='amount', aggfunc='sum')
print(pv.round(0))
long = pv.reset_index().melt(id_vars='store', var_name='category', value_name='amount')
print(long.head())

# %% 패턴 17. 출력 형식 마무리
val = 1234.56789
print(int(val))          # 1234
print(round(val, 2))     # 1234.57
print(round(val))        # 1235  (반올림 정수)
print(f'{val:.2f}')      # '1234.57' 문자열
# 정수로 나와야 하는데 1234.0 처럼 나오면 int() 로 감싸기
