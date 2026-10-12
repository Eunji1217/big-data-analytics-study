# 연습용 CSV 데이터 생성기
# 실행하면 이 파일과 같은 폴더(data/)에 csv 파일들이 생성됩니다.
#   python make_data.py
#
# 생성 파일
#   customers.csv   : 1유형/전처리 연습용 (결측치, 이상치, 문자열, 날짜 포함)
#   sales.csv       : 1유형 groupby/날짜 연습용
#   train.csv, test.csv : 2유형 분류 (타겟 churn 0/1)
#   house_train.csv, house_test.csv : 2유형 회귀 (타겟 price)
#   stats_*.csv     : 3유형 통계검정 연습용

import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
HERE = os.path.dirname(os.path.abspath(__file__))

def save(df, name):
    path = os.path.join(HERE, name)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"생성: {name:22s} {df.shape}")

# ---------------------------------------------------------------- customers.csv
n = 500
cities = ['서울', '부산', '대구', '인천', '광주', '대전']
df = pd.DataFrame({
    'id': range(1, n + 1),
    'name': ['고객' + str(i) for i in range(1, n + 1)],
    'gender': rng.choice(['M', 'F'], n),
    'age': rng.integers(18, 70, n).astype(float),
    'city': rng.choice(cities, n, p=[0.35, 0.2, 0.15, 0.12, 0.1, 0.08]),
    'grade': rng.choice(['VIP', 'Gold', 'Silver', 'Bronze'], n, p=[0.1, 0.2, 0.3, 0.4]),
    'income': np.round(rng.normal(4500, 1200, n)).astype(float),
    'purchase_amt': np.round(rng.gamma(2, 150, n), 1),
    'visit_cnt': rng.poisson(6, n),
    'join_date': pd.to_datetime('2021-01-01') + pd.to_timedelta(rng.integers(0, 1200, n), unit='D'),
    'phone': ['010-' + str(rng.integers(1000, 9999)) + '-' + str(rng.integers(1000, 9999)) for _ in range(n)],
})
# 결측치 삽입
df.loc[rng.choice(n, 40, replace=False), 'age'] = np.nan
df.loc[rng.choice(n, 30, replace=False), 'income'] = np.nan
df.loc[rng.choice(n, 15, replace=False), 'city'] = np.nan
# 이상치 삽입
df.loc[rng.choice(n, 5, replace=False), 'purchase_amt'] = rng.uniform(3000, 6000, 5).round(1)
df.loc[rng.choice(n, 3, replace=False), 'income'] = [25000, 30000, -500]
# 중복 행 삽입
df = pd.concat([df, df.iloc[[10, 20, 30]]], ignore_index=True)
df['join_date'] = df['join_date'].dt.strftime('%Y-%m-%d')
save(df, 'customers.csv')

# ---------------------------------------------------------------- sales.csv
n = 2000
dates = pd.to_datetime('2023-01-01') + pd.to_timedelta(rng.integers(0, 730, n), unit='D')
sales = pd.DataFrame({
    'order_id': range(1, n + 1),
    'date': dates,
    'store': rng.choice(['강남점', '홍대점', '잠실점', '판교점'], n),
    'category': rng.choice(['식품', '의류', '가전', '도서', '뷰티'], n),
    'qty': rng.integers(1, 10, n),
    'unit_price': rng.choice([5000, 12000, 25000, 48000, 130000], n),
    'discount': rng.choice([0, 0.1, 0.2, 0.3], n, p=[0.5, 0.3, 0.15, 0.05]),
})
sales['amount'] = (sales['qty'] * sales['unit_price'] * (1 - sales['discount'])).round(0)
sales = sales.sort_values('date').reset_index(drop=True)
sales['date'] = sales['date'].dt.strftime('%Y-%m-%d')
save(sales, 'sales.csv')

# ---------------------------------------------------------------- 2유형 분류: churn
n = 3000
tenure = rng.integers(1, 72, n)
monthly = np.round(rng.uniform(20, 120, n), 2)
contract = rng.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.2])
internet = rng.choice(['DSL', 'Fiber', 'No'], n, p=[0.35, 0.45, 0.2])
support = rng.choice(['Yes', 'No'], n, p=[0.3, 0.7])
senior = rng.choice([0, 1], n, p=[0.85, 0.15])
logit = (-1.5 - 0.03 * tenure + 0.015 * monthly
         + np.where(contract == 'Month-to-month', 1.2, np.where(contract == 'One year', 0.2, -0.8))
         + np.where(internet == 'Fiber', 0.6, 0) + np.where(support == 'No', 0.5, 0) + 0.4 * senior)
p = 1 / (1 + np.exp(-logit))
churn = (rng.uniform(size=n) < p).astype(int)
cust = pd.DataFrame({
    'ID': ['C' + str(i).zfill(5) for i in range(1, n + 1)],
    'gender': rng.choice(['Male', 'Female'], n),
    'SeniorCitizen': senior,
    'tenure': tenure,
    'Contract': contract,
    'InternetService': internet,
    'TechSupport': support,
    'MonthlyCharges': monthly,
    'TotalCharges': np.round(tenure * monthly * rng.uniform(0.9, 1.1, n), 2),
    'churn': churn,
})
cust.loc[rng.choice(n, 60, replace=False), 'TotalCharges'] = np.nan
cust.loc[rng.choice(n, 40, replace=False), 'TechSupport'] = np.nan
idx = rng.permutation(n)
train = cust.iloc[idx[:2400]].reset_index(drop=True)
test = cust.iloc[idx[2400:]].reset_index(drop=True)
save(train, 'train.csv')
test_y = test[['ID', 'churn']]
save(test.drop(columns=['churn']), 'test.csv')
save(test_y, 'test_answer.csv')   # 채점 연습용 정답 (시험에는 없음)

# ---------------------------------------------------------------- 2유형 회귀: house price
n = 2500
area = rng.uniform(40, 200, n).round(1)
rooms = rng.integers(1, 6, n)
age = rng.integers(0, 40, n)
dist = rng.uniform(0.2, 15, n).round(2)
htype = rng.choice(['아파트', '빌라', '오피스텔', '단독'], n, p=[0.5, 0.25, 0.15, 0.1])
region = rng.choice(['A', 'B', 'C', 'D'], n)
region_bonus = pd.Series(region).map({'A': 3000, 'B': 1500, 'C': 500, 'D': 0}).values
price = (3000 + 25 * area + 300 * rooms - 15 * age - 40 * dist
         + np.where(htype == '아파트', 2000, np.where(htype == '단독', 1500, 0))
         + region_bonus + rng.normal(0, 800, n)).round(0)
price = np.clip(price, 500, None)     # 가격은 최소 500 (만원 단위)
house = pd.DataFrame({
    'ID': range(1, n + 1), 'area': area, 'rooms': rooms, 'age': age,
    'dist_subway': dist, 'type': htype, 'region': region, 'price': price,
})
house.loc[rng.choice(n, 50, replace=False), 'age'] = np.nan
idx = rng.permutation(n)
h_train = house.iloc[idx[:2000]].reset_index(drop=True)
h_test = house.iloc[idx[2000:]].reset_index(drop=True)
save(h_train, 'house_train.csv')
save(h_test.drop(columns=['price']), 'house_test.csv')
save(h_test[['ID', 'price']], 'house_test_answer.csv')

# ---------------------------------------------------------------- 3유형용
# (1) 혈압 약 전/후 (대응표본), 성별 (독립표본)
n = 120
before = rng.normal(140, 12, n).round(1)
after = (before - rng.normal(6, 8, n)).round(1)
bp = pd.DataFrame({'id': range(1, n + 1), 'sex': rng.choice(['M', 'F'], n),
                   'before': before, 'after': after, 'age': rng.integers(30, 75, n)})
save(bp, 'stats_bp.csv')

# (2) 세 반 점수 (ANOVA)
score = pd.DataFrame({
    'class': np.repeat(['A', 'B', 'C'], 40),
    'score': np.concatenate([rng.normal(72, 8, 40), rng.normal(76, 8, 40), rng.normal(70, 9, 40)]).round(1),
    'method': rng.choice(['online', 'offline'], 120),
})
save(score, 'stats_score.csv')

# (3) 성별 × 구매 (카이제곱), 광고채널 × 전환
n = 400
sex = rng.choice(['M', 'F'], n)
buy = np.where(sex == 'M', rng.choice(['Yes', 'No'], n, p=[0.35, 0.65]), rng.choice(['Yes', 'No'], n, p=[0.5, 0.5]))
chi = pd.DataFrame({'sex': sex, 'buy': buy, 'channel': rng.choice(['SNS', 'TV', 'Search'], n)})
save(chi, 'stats_chi.csv')

# (4) 회귀분석용: 광고비 → 매출
n = 200
tv = rng.uniform(10, 300, n).round(1)
radio = rng.uniform(0, 50, n).round(1)
news = rng.uniform(0, 100, n).round(1)
sales_y = (5 + 0.045 * tv + 0.18 * radio + 0.002 * news + rng.normal(0, 1.5, n)).round(2)
adv = pd.DataFrame({'TV': tv, 'Radio': radio, 'Newspaper': news, 'Sales': sales_y})
save(adv, 'stats_adv.csv')

# (5) 로지스틱용: 흡연/나이/운동 → 질병
n = 500
smoke = rng.choice([0, 1], n, p=[0.6, 0.4])
age = rng.integers(20, 70, n)
exercise = rng.choice([0, 1], n)
lg = -4 + 1.1 * smoke + 0.06 * age - 0.8 * exercise
disease = (rng.uniform(size=n) < 1 / (1 + np.exp(-lg))).astype(int)
save(pd.DataFrame({'smoke': smoke, 'age': age, 'exercise': exercise, 'disease': disease}), 'stats_disease.csv')

print("\n모든 연습 데이터 생성 완료. 03_실습코드/data/ 폴더를 확인하세요.")
