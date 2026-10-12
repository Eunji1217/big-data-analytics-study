# 작업형 1유형 연습문제 풀이
# 각 문제를 셀 단위로 실행. 시험에서는 '답' 부분의 print 하나만 남기면 됨.
# %%
import pandas as pd
import numpy as np
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_실습코드', 'data') if '__file__' in globals() else '../03_실습코드/data'

# %% A-1
df = pd.read_csv(os.path.join(DATA, 'customers.csv'))
df['age'] = df['age'].fillna(df['age'].median())
q3 = df['age'].quantile(0.75)
print('A-1', round(df[df['age'] >= q3]['purchase_amt'].mean(), 2))

# %% A-2
df = pd.read_csv(os.path.join(DATA, 'customers.csv'))
df = df.dropna(subset=['city', 'income'])
q1, q3 = df['income'].quantile([0.25, 0.75])
iqr = q3 - q1
df = df[(df['income'] >= q1 - 1.5 * iqr) & (df['income'] <= q3 + 1.5 * iqr)]
g = df.groupby('city')['income'].mean()
print('A-2', g.idxmax(), int(g.max()))

# %% A-3
df = pd.read_csv(os.path.join(DATA, 'customers.csv'))
df['join_date'] = pd.to_datetime(df['join_date'])
cond = (df['join_date'].dt.year == 2022) & (df['grade'].isin(['VIP', 'Gold']))
print('A-3', cond.sum())

# %% B-1
s = pd.read_csv(os.path.join(DATA, 'sales.csv'))
s['date'] = pd.to_datetime(s['date'])
mon = s.groupby(s['date'].dt.to_period('M'))['amount'].sum()
rate = mon.pct_change()
rate = rate[rate.index.year == 2024]
print('B-1', str(rate.idxmax()))

# %% B-2
s = pd.read_csv(os.path.join(DATA, 'sales.csv'))
g = s.groupby(['store', 'category'])['amount'].sum().reset_index()
top = g.loc[g.groupby('store')['amount'].idxmax()]
print('B-2', (top['category'] == '가전').sum())

# %% B-3
s = pd.read_csv(os.path.join(DATA, 'sales.csv'))
mm = (s['amount'] - s['amount'].min()) / (s['amount'].max() - s['amount'].min())
print('B-3', s[mm >= 0.3]['qty'].sum())

# %% C-1
df = pd.read_csv(os.path.join(DATA, 'customers.csv')).drop_duplicates()
last4 = df['phone'].str[-4:].astype(int)
sub = df[last4 % 2 == 0].sort_values(['visit_cnt', 'id'], ascending=[False, True])
print('C-1', sub.iloc[0]['name'])

# %% C-2
df = pd.read_csv(os.path.join(DATA, 'customers.csv'))
z = (df['purchase_amt'] - df['purchase_amt'].mean()) / df['purchase_amt'].std()
df = df[z.abs() < 2]
print('C-2', df.groupby('grade')['purchase_amt'].std().idxmin())

# %% C-3
df = pd.read_csv(os.path.join(DATA, 'customers.csv'))
before = df['income'].mean()
df['income'] = df.groupby('grade')['income'].transform(lambda x: x.fillna(x.mean()))
print('C-3', round(abs(before - df['income'].mean()), 3))

# %% D-1
s = pd.read_csv(os.path.join(DATA, 'sales.csv'))
s['date'] = pd.to_datetime(s['date'])
print('D-1', s.groupby(s['date'].dt.day_name())['amount'].mean().idxmax())

# %% D-2
s23 = s[s['date'].dt.year == 2023]
q = s23.groupby(s23['date'].dt.quarter)['amount'].sum()
print('D-2', int(q.max() - q.min()))

# %% D-3
s = pd.read_csv(os.path.join(DATA, 'sales.csv'))
d = s[s['discount'] != 0]
p80 = d['unit_price'].quantile(0.8)
print('D-3', round(d[d['unit_price'] >= p80]['amount'].mean(), 1))
