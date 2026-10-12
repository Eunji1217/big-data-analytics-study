# %% [markdown]
# # 01. pandas 기초 — 데이터 읽기 · 살펴보기 · 선택 · 정렬
# 개념서: 02_개념서/02_pandas_핵심.md
#
# 사용법: 각 `# %%` 셀 위의 "Run Cell" 또는 Shift+Enter 로 한 셀씩 실행.
# ★ 표시가 있는 셀은 코드를 보지 말고 직접 써보는 연습문제입니다.

# %%
import pandas as pd
import numpy as np
import os

# 이 파일이 있는 폴더 기준으로 data 경로 잡기 (VS Code 에서 어디서 실행해도 동작)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data') if '__file__' in globals() else 'data'
df = pd.read_csv(os.path.join(DATA, 'customers.csv'))
print(df.head())

# %% 1. 데이터 살펴보기 — 문제 풀기 전 무조건 하는 4가지
print(df.shape)            # (행, 열)
print(df.info())           # 타입 + 결측
print(df.describe())       # 수치 요약
print(df.isnull().sum())   # 결측 개수

# %% 2. 컬럼 선택
print(df['age'].head())            # Series
print(df[['name', 'age']].head())  # DataFrame (대괄호 2개)
print(type(df['age']), type(df[['age']]))

# %% 3. 행 선택 — iloc (위치) vs loc (라벨/조건)
print(df.iloc[0])            # 첫 행
print(df.iloc[0:3])          # 0,1,2 행 (3 미포함)
print(df.iloc[0:3, 1:4])     # 행 0~2, 열 1~3
print(df.loc[0:3, ['name', 'age']])   # loc 은 3 포함!

# %% 4. 조건 필터링 ★★ 1유형 핵심
print(df[df['age'] >= 60].shape[0])                          # 60세 이상 몇 명
print(df[(df['gender'] == 'F') & (df['city'] == '서울')].shape[0])   # 괄호 필수
print(df[(df['grade'] == 'VIP') | (df['grade'] == 'Gold')].shape[0])
print(df[df['grade'].isin(['VIP', 'Gold'])].shape[0])       # 위와 같음
print(df[~df['grade'].isin(['VIP', 'Gold'])].shape[0])      # ~ 는 not
print(df[df['city'].isnull()].shape[0])                     # city 결측 행 수

# %% 5. 정렬과 상위 N
print(df.sort_values('purchase_amt', ascending=False).head(5)[['name', 'purchase_amt']])
print(df.nlargest(5, 'purchase_amt')[['name', 'purchase_amt']])   # 같은 결과
print(df.sort_values(['grade', 'age'], ascending=[True, False]).head())

# 1유형 스타일: "구매금액 상위 10명의 평균 나이 (소수 둘째 자리)"
top10 = df.sort_values('purchase_amt', ascending=False).head(10)
print(round(top10['age'].mean(), 2))

# %% 6. 기본 통계
print(df['income'].mean(), df['income'].median(), df['income'].std())
print(df['income'].quantile(0.25), df['income'].quantile(0.75))
print(df['income'].min(), df['income'].max())
print(df['grade'].value_counts())           # 범주 개수
print(df['grade'].value_counts(normalize=True))   # 비율
print(df['city'].nunique(), df['city'].unique())
print(df['grade'].mode()[0])                 # 최빈값 → [0] 필수

# %% 7. 컬럼 추가 / 변경 / 삭제
df['income_man'] = df['income'] * 10000
df['is_senior'] = np.where(df['age'] >= 60, 1, 0)
df['age_group'] = df['age'].apply(lambda x: '60+' if x >= 60 else ('40-59' if x >= 40 else '~39'))
df['grade_num'] = df['grade'].map({'VIP': 4, 'Gold': 3, 'Silver': 2, 'Bronze': 1})
print(df[['age', 'is_senior', 'age_group', 'grade', 'grade_num']].head())
df = df.drop(columns=['income_man'])
df = df.rename(columns={'purchase_amt': 'amount'})
print(df.columns.tolist())

# %% 8. idxmax — "가장 큰 값을 가진 행의 이름"
print(df['amount'].idxmax())                       # 인덱스 번호
print(df.loc[df['amount'].idxmax(), 'name'])       # 그 행의 name

# %% 9. 도움말 찾기 연습 (시험장에서 쓰는 방법)
# help(pd.DataFrame.sort_values)     ← 주석 풀고 실행해 보기
# print(dir(df['age'].str))          ← 문자열 메서드 목록
print([m for m in dir(pd.Series) if m.startswith('n')])   # n으로 시작하는 메서드

# %% ★ 연습문제 (답은 아래 셀. 먼저 직접 풀어보세요)
# Q1. '부산' 고객 중 age 가 40 이상인 사람 수
# Q2. income 이 높은 순으로 정렬했을 때 20번째 사람의 name
# Q3. grade 가 'Silver' 인 고객의 amount 중앙값 (소수 첫째 자리)
# Q4. visit_cnt 가 가장 많은 고객의 city
# Q5. gender 가 'M' 이고 grade 가 'VIP' 또는 'Gold' 인 고객의 평균 age (정수)

# %% 정답
print('Q1', df[(df['city'] == '부산') & (df['age'] >= 40)].shape[0])
print('Q2', df.sort_values('income', ascending=False).iloc[19]['name'])
print('Q3', round(df[df['grade'] == 'Silver']['amount'].median(), 1))
print('Q4', df.loc[df['visit_cnt'].idxmax(), 'city'])
print('Q5', int(df[(df['gender'] == 'M') & (df['grade'].isin(['VIP', 'Gold']))]['age'].mean()))
