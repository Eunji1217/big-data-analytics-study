# %% [markdown]
# # 02. pandas 집계 · 날짜 · 문자열 · 결합
# 개념서: 02_개념서/02_pandas_핵심.md  (5장 groupby, 9장 문자열, 10장 날짜, 12장 결합)

# %%
import pandas as pd
import numpy as np
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data') if '__file__' in globals() else 'data'
sales = pd.read_csv(os.path.join(DATA, 'sales.csv'))
cust = pd.read_csv(os.path.join(DATA, 'customers.csv'))
print(sales.head())
print(sales.info())

# %% 1. groupby 기본 ★★★
print(sales.groupby('store')['amount'].sum())                 # 매장별 매출합
print(sales.groupby('store')['amount'].mean().round(0))
print(sales.groupby('category')['qty'].agg(['sum', 'mean', 'count']))
print(sales.groupby(['store', 'category'])['amount'].sum())   # 다중 그룹
print(sales.groupby('store').size())                          # 그룹별 행 수

# %% 2. groupby 결과 다루기
g = sales.groupby('store')['amount'].sum()
print(g.idxmax(), g.max())                             # 매출 최대 매장, 값
print(g.sort_values(ascending=False).index[0])         # 같은 것
print(g.reset_index())                                  # DataFrame 으로
print(sales.groupby('store').agg(총매출=('amount', 'sum'), 건수=('order_id', 'count'), 평균수량=('qty', 'mean')))

# %% 3. pivot_table
pv = pd.pivot_table(sales, index='store', columns='category', values='amount', aggfunc='sum')
print(pv)
print(pv.loc['강남점', '가전'])

# %% 4. 날짜 다루기 ★★
sales['date'] = pd.to_datetime(sales['date'])
sales['year'] = sales['date'].dt.year
sales['month'] = sales['date'].dt.month
sales['dow'] = sales['date'].dt.dayofweek       # 월0 ~ 일6
sales['ym'] = sales['date'].dt.to_period('M')   # 2023-01
print(sales[['date', 'year', 'month', 'dow', 'ym']].head())

print(sales[sales['year'] == 2024]['amount'].sum())               # 2024년 매출
print(sales.groupby('ym')['amount'].sum().idxmax())               # 매출 최대 월
print(sales[sales['dow'] >= 5]['amount'].mean())                  # 주말 평균 매출
print(sales[(sales['date'] >= '2023-06-01') & (sales['date'] <= '2023-06-30')].shape[0])

# resample: 날짜를 인덱스로 두고 기간별 집계
print(sales.set_index('date').resample('QE')['amount'].sum())     # 분기별 (QE: quarter end. 구버전은 'Q')

# 날짜 차이
cust['join_date'] = pd.to_datetime(cust['join_date'])
cust['days_since'] = (pd.Timestamp('2025-01-01') - cust['join_date']).dt.days
print(cust[['join_date', 'days_since']].head())

# %% 5. 문자열 처리 (.str)
print(cust['phone'].str.replace('-', '').head())
print(cust['phone'].str.split('-').str[1].head())        # 가운데 번호
print(cust['phone'].str[-4:].head())                     # 뒷 4자리
print(cust[cust['name'].str.contains('99')].shape[0])
print(cust['name'].str.len().max())
print(cust['name'].str.replace('고객', '').astype(int).head())   # '고객12' → 12

# %% 6. 결합 concat / merge
a = cust[['id', 'name', 'grade']].head(3)
b = cust[['id', 'name', 'grade']].iloc[3:6]
print(pd.concat([a, b]))                          # 위아래
print(pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1))   # 좌우

left = pd.DataFrame({'id': [1, 2, 3], 'x': ['a', 'b', 'c']})
right = pd.DataFrame({'id': [2, 3, 4], 'y': [10, 20, 30]})
print(pd.merge(left, right, on='id', how='inner'))    # 2,3
print(pd.merge(left, right, on='id', how='left'))     # 1,2,3 (4는 NaN)
print(pd.merge(left, right, on='id', how='outer'))    # 1,2,3,4

# %% 7. 구간화 cut / qcut, 순위 rank, 누적 cumsum
cust['age_bin'] = pd.cut(cust['age'], bins=[0, 30, 45, 60, 100], labels=['~30', '31-45', '46-60', '61+'])
print(cust['age_bin'].value_counts())
cust['income_q'] = pd.qcut(cust['income'], q=4, labels=[1, 2, 3, 4])   # 4분위
print(cust.groupby('income_q', observed=True)['purchase_amt'].mean())
cust['amt_rank'] = cust['purchase_amt'].rank(ascending=False, method='min')
print(cust.sort_values('amt_rank')[['name', 'purchase_amt', 'amt_rank']].head())
print(sales.sort_values('date')['amount'].cumsum().tail())

# %% 8. transform — 그룹 통계를 각 행에 붙이기
sales['store_mean'] = sales.groupby('store')['amount'].transform('mean')
sales['diff_from_store_mean'] = sales['amount'] - sales['store_mean']
print(sales[['store', 'amount', 'store_mean', 'diff_from_store_mean']].head())

# %% 9. 상관계수
print(cust[['age', 'income', 'purchase_amt', 'visit_cnt']].corr().round(3))
print(round(cust['income'].corr(cust['purchase_amt']), 4))

# %% ★ 연습문제
# Q1. 2023년 '의류' 카테고리 총 매출 (정수)
# Q2. 매장별 평균 discount 가 가장 높은 매장명
# Q3. 월별(ym) 매출합 중 가장 낮은 달의 매출 (정수)
# Q4. 요일별 평균 qty 가 가장 높은 요일 번호
# Q5. 매장별 매출 상위 카테고리(1개)를 매장:카테고리 형태로 (힌트: groupby 두 번 + idxmax)

# %% 정답
print('Q1', int(sales[(sales['year'] == 2023) & (sales['category'] == '의류')]['amount'].sum()))
print('Q2', sales.groupby('store')['discount'].mean().idxmax())
print('Q3', int(sales.groupby('ym')['amount'].sum().min()))
print('Q4', sales.groupby('dow')['qty'].mean().idxmax())
tmp = sales.groupby(['store', 'category'])['amount'].sum().reset_index()
print('Q5', tmp.loc[tmp.groupby('store')['amount'].idxmax()][['store', 'category']])
