# %% [markdown]
# # 03. 데이터 전처리 실습 — 결측치 · 이상치 · 중복 · 스케일링 · 인코딩
# 개념서: 02_개념서/03_데이터전처리.md

# %%
import pandas as pd
import numpy as np
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data') if '__file__' in globals() else 'data'
df = pd.read_csv(os.path.join(DATA, 'customers.csv'))
print(df.isnull().sum())

# %% 1. 결측치 처리
# 1-1 행 제거
print(df.dropna().shape, df.dropna(subset=['age']).shape)

# 1-2 대치 — 반드시 다시 대입!
df2 = df.copy()
df2['age'] = df2['age'].fillna(df2['age'].median())
df2['income'] = df2['income'].fillna(df2['income'].mean())
df2['city'] = df2['city'].fillna(df2['city'].mode()[0])
print(df2.isnull().sum().sum())     # 0

# 1-3 그룹별 대치 (성별에 따라 다른 평균으로)
df3 = df.copy()
df3['age'] = df3.groupby('gender')['age'].transform(lambda x: x.fillna(x.mean()))
print(df3['age'].isnull().sum())

# 1-4 1유형 스타일: "결측치 채우기 전/후 평균의 차이"
before = df['age'].mean()
after = df['age'].fillna(df['age'].median()).mean()
print(round(abs(before - after), 3))

# 1-5 결측 비율이 가장 높은 컬럼
print((df.isnull().mean()).idxmax(), round(df.isnull().mean().max(), 3))

# %% 2. 이상치 — IQR ★★
col = 'purchase_amt'
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
out = df[(df[col] < lower) | (df[col] > upper)]
print('이상치 개수:', len(out))
clean = df[(df[col] >= lower) & (df[col] <= upper)]
print('제거 후 평균:', round(clean[col].mean(), 2), ' 제거 전:', round(df[col].mean(), 2))

# 경계값으로 자르기(clip)
df['amt_clip'] = df[col].clip(lower, upper)
print(df['amt_clip'].max())

# 표준편차 방식 (평균 ± 3σ)
m, s = df['income'].mean(), df['income'].std()
print('3σ 이상치:', ((df['income'] < m - 3 * s) | (df['income'] > m + 3 * s)).sum())

# 논리적 이상치 (음수 소득)
print('음수 income:', (df['income'] < 0).sum())

# %% 3. 중복
print('중복 행:', df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print(df.shape)
print(df.duplicated(subset=['city', 'grade']).sum())   # 특정 컬럼 기준 중복

# %% 4. 스케일링 — 직접 계산 vs sklearn
df['age'] = df['age'].fillna(df['age'].median())
# 직접
df['age_std'] = (df['age'] - df['age'].mean()) / df['age'].std()
df['age_mm'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
print(df[['age', 'age_std', 'age_mm']].head())
print(round(df['age_std'].std(), 4), df['age_mm'].min(), df['age_mm'].max())

# sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc = StandardScaler()
df['age_std2'] = sc.fit_transform(df[['age']])       # [[ ]] 2차원!
print(np.allclose(df['age_std'] * np.sqrt((len(df) - 1) / len(df)), df['age_std2']))  # sklearn 은 ddof=0
mm = MinMaxScaler()
print(mm.fit_transform(df[['income']].fillna(0))[:3])

# 1유형 스타일: "표준화 후 값이 1 이상인 행 개수"
print((df['age_std'] >= 1).sum())

# %% 5. 인코딩
df['city'] = df['city'].fillna(df['city'].mode()[0])
# 5-1 LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['city_le'] = le.fit_transform(df['city'])
print(dict(zip(le.classes_, range(len(le.classes_)))))
# 5-2 map (순서가 있는 범주는 직접 지정)
df['grade_ord'] = df['grade'].map({'Bronze': 0, 'Silver': 1, 'Gold': 2, 'VIP': 3})
# 5-3 원핫
oh = pd.get_dummies(df[['gender', 'grade']], drop_first=False)
print(oh.head())
print(pd.get_dummies(df, columns=['gender']).columns.tolist()[:5])

# 범주형 컬럼 자동 탐색
print(df.select_dtypes(include='object').columns.tolist())
print(df.select_dtypes(include='number').columns.tolist())

# %% 6. 타입 변환 & 파생변수
df['join_date'] = pd.to_datetime(df['join_date'])
df['join_year'] = df['join_date'].dt.year
df['visit_per_year'] = df['visit_cnt'] / (2025 - df['join_year'] + 1)
df['log_amt'] = np.log1p(df['purchase_amt'])
print(df[['purchase_amt', 'log_amt']].skew().round(3))     # 로그변환으로 왜도 감소 확인
s = pd.Series(['1,200', '3,400', 'abc'])
print(pd.to_numeric(s.str.replace(',', ''), errors='coerce'))   # 숫자 아닌 건 NaN

# %% ★ 연습문제
# Q1. income 결측을 grade 별 중앙값으로 채운 뒤, 전체 income 평균 (정수)
# Q2. visit_cnt 의 IQR 이상치를 제거한 뒤 남은 행 수
# Q3. purchase_amt 를 Min-Max 정규화한 뒤, 0.5 이상인 행 수
# Q4. age 를 표준화한 값의 절댓값이 1.5 초과인 고객의 평균 income (소수 첫째 자리, 결측 제외)

# %% 정답
d = pd.read_csv(os.path.join(DATA, 'customers.csv')).drop_duplicates()
d['income'] = d.groupby('grade')['income'].transform(lambda x: x.fillna(x.median()))
print('Q1', int(d['income'].mean()))

q1, q3 = d['visit_cnt'].quantile([0.25, 0.75])
iqr = q3 - q1
print('Q2', d[(d['visit_cnt'] >= q1 - 1.5 * iqr) & (d['visit_cnt'] <= q3 + 1.5 * iqr)].shape[0])

mmv = (d['purchase_amt'] - d['purchase_amt'].min()) / (d['purchase_amt'].max() - d['purchase_amt'].min())
print('Q3', (mmv >= 0.5).sum())

z = (d['age'] - d['age'].mean()) / d['age'].std()
print('Q4', round(d[z.abs() > 1.5]['income'].mean(), 1))
