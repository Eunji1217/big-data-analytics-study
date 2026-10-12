# 02. pandas 핵심 정리 (1유형의 90%)

실습 파일: `03_실습코드/01_pandas_기초.py`, `02_pandas_집계_변환.py`

## 1. 기본 구조

- **Series**: 1차원 (한 컬럼). `df['col']` 은 Series
- **DataFrame**: 2차원 표. `df[['col1','col2']]` 은 DataFrame (대괄호 2개)
- 시험 첫 줄은 거의 항상:
  ```python
  import pandas as pd
  df = pd.read_csv('data/파일.csv')
  ```

## 2. 데이터 살펴보기 (가장 먼저 할 것)

```python
df.head()          # 앞 5행
df.shape           # (행, 열)
df.info()          # 컬럼별 타입, 결측 개수
df.describe()      # 수치형 요약 통계
df.dtypes          # 타입만
df.columns         # 컬럼명
df.isnull().sum()  # 컬럼별 결측 개수
df['col'].unique() / df['col'].nunique() / df['col'].value_counts()
```

## 3. 선택 (indexing)

| 원하는 것 | 코드 |
|---|---|
| 컬럼 하나 | `df['A']` |
| 컬럼 여러 개 | `df[['A','B']]` |
| 행 번호로 (위치) | `df.iloc[0]`, `df.iloc[0:5]`, `df.iloc[0:5, 1:3]` |
| 라벨/조건으로 | `df.loc[0]`, `df.loc[df['A']>10, 'B']` |
| 조건 필터 | `df[df['A'] > 10]` |
| 여러 조건 | `df[(df['A']>10) & (df['B']=='x')]` ← 괄호 필수, `&` `\|` `~` |
| 값 목록에 포함 | `df[df['A'].isin([1,2,3])]` |
| 문자열 포함 | `df[df['name'].str.contains('김')]` |
| 결측 여부 | `df[df['A'].isnull()]`, `df[df['A'].notnull()]` |

`iloc` = 정수 위치(integer location), `loc` = 라벨(label). `loc` 의 슬라이스는 끝 포함, `iloc` 은 끝 미포함.

## 4. 정렬·상위 N개

```python
df.sort_values('A')                          # 오름차순
df.sort_values('A', ascending=False)         # 내림차순
df.sort_values(['A','B'], ascending=[False, True])
df.nlargest(10, 'A')   /  df.nsmallest(10, 'A')
df.sort_values('A', ascending=False).head(10)  # 같은 뜻
df.reset_index(drop=True)                     # 정렬 후 인덱스 0부터 다시
```

> 1유형 단골: "A 기준 내림차순 정렬 후 상위 10개의 B 평균" → `df.sort_values('A', ascending=False).head(10)['B'].mean()`

## 5. 집계 (groupby) ★★★

```python
df.groupby('cat')['val'].mean()                 # 그룹별 평균 (Series)
df.groupby('cat')['val'].agg(['mean','sum','count','max'])
df.groupby(['cat1','cat2'])['val'].sum()        # 다중 그룹
df.groupby('cat').size()                         # 그룹별 행 수
df.groupby('cat')['val'].mean().reset_index()    # DataFrame 으로
df.groupby('cat')['val'].mean().idxmax()         # 평균이 최대인 그룹명
df.groupby('cat')['val'].mean().sort_values(ascending=False).index[0]  # 같은 뜻
df.groupby('cat').agg(평균=('val','mean'), 개수=('id','count'))  # 이름 지정 집계
```

`pivot_table`:
```python
pd.pivot_table(df, index='cat1', columns='cat2', values='val', aggfunc='mean')
```

## 6. 컬럼 추가/변경/삭제

```python
df['new'] = df['A'] * 2
df['grade'] = df['score'].apply(lambda x: 'A' if x >= 90 else 'B')
df['cat_num'] = df['cat'].map({'a':1, 'b':2})
df = df.drop(columns=['A','B'])       # 컬럼 삭제
df = df.drop(index=[0,1])             # 행 삭제
df = df.rename(columns={'old':'new'})
df['A'] = df['A'].astype(int)         # 타입 변경 (float→int, str→int 등)
```

`np.where` (조건에 따라 값):
```python
import numpy as np
df['flag'] = np.where(df['A'] > 10, 'high', 'low')
```

## 7. 결측치

```python
df.isnull().sum()
df.dropna()                              # 결측 있는 행 전부 제거
df.dropna(subset=['A'])                  # A 컬럼 결측인 행만 제거
df['A'].fillna(df['A'].mean())           # 평균 대치
df['A'].fillna(df['A'].median())         # 중앙값 대치
df['A'].fillna(df['A'].mode()[0])        # 최빈값 대치 (mode 는 Series → [0])
df['A'].fillna(0)
df['A'] = df['A'].ffill()  /  .bfill()   # 앞/뒤 값으로 채움
```

> 주의: `fillna` 는 **원본을 바꾸지 않음**. `df['A'] = df['A'].fillna(...)` 로 다시 대입해야 함. (`inplace=True` 도 되지만 최신 pandas에서 경고가 나기도 하니 대입 방식 추천)

## 8. 중복

```python
df.duplicated().sum()
df.drop_duplicates()
df.drop_duplicates(subset=['A'], keep='first')
```

## 9. 문자열 (.str)

```python
df['name'].str.len()
df['name'].str.upper() / .lower() / .strip()
df['name'].str.contains('김')
df['name'].str.startswith('김')
df['name'].str.replace('-', '')
df['name'].str.split('-').str[0]        # 분리 후 첫 조각
df['name'].str[:3]                       # 슬라이싱
df['A'].str.replace(',', '').astype(int) # "1,234" → 1234
```

## 10. 날짜 (.dt) ★

```python
df['date'] = pd.to_datetime(df['date'])     # 문자열 → datetime
df['date'].dt.year / .dt.month / .dt.day
df['date'].dt.dayofweek                     # 월=0 ... 일=6
df['date'].dt.day_name()                    # 'Monday'
df['date'].dt.hour / .dt.minute
df['date'].dt.to_period('M')                # 2024-01 (월 단위 그룹용)
(df['end'] - df['start']).dt.days           # 날짜 차이(일)
df[df['date'].dt.year == 2024]
df[(df['date'] >= '2024-01-01') & (df['date'] < '2024-02-01')]
```

`resample` (시계열 집계, 인덱스가 날짜여야 함):
```python
df.set_index('date').resample('M')['val'].sum()   # 월별 합
```

## 11. 통계 함수

```python
df['A'].mean() / .median() / .std() / .var() / .min() / .max() / .sum()
df['A'].quantile(0.25)          # 1사분위
df['A'].quantile([0.25, 0.75])
df['A'].mode()[0]
df['A'].skew() / .kurt()        # 왜도 / 첨도
df['A'].cumsum()                # 누적합
df['A'].diff()                  # 전 행과 차이
df['A'].pct_change()            # 변화율
df[['A','B']].corr()            # 상관계수 행렬
df['A'].corr(df['B'])
df['A'].abs()
```

> `std()` 는 기본이 **표본표준편차(ddof=1)**. numpy `np.std` 는 기본 ddof=0. 시험은 보통 pandas 기본값으로 채점.

## 12. 결합

```python
pd.concat([df1, df2])                # 위아래로 (행 추가)
pd.concat([df1, df2], axis=1)        # 좌우로
pd.merge(df1, df2, on='id', how='inner')   # inner/left/right/outer
```

## 13. 기타 자주 쓰는 것

```python
df['A'].idxmax()                  # 최댓값의 인덱스
df.loc[df['A'].idxmax(), 'name']  # 최댓값을 가진 행의 name
df['A'].rank(ascending=False)     # 순위
df['A'].clip(lower=0, upper=100)  # 범위 자르기
pd.cut(df['age'], bins=[0,20,40,60,100], labels=['10대','2030','4050','60+'])
pd.qcut(df['A'], q=4, labels=[1,2,3,4])    # 분위수로 4등분
df['A'].round(2)
df.T                              # 전치
df.select_dtypes(include='number') / include='object'
```

## 14. 1유형에서 자주 하는 실수

1. 조건 여러 개 쓸 때 괄호 안 씀 → `(df['A']>1) & (df['B']<2)`
2. `fillna`, `sort_values`, `drop` 결과를 **다시 대입 안 함**
3. `df['A'].mode()` 가 Series 라서 `[0]` 안 붙임
4. `head(10)` 을 정렬 **전에** 함
5. 문제가 "정수로 출력" 인데 float 출력 → `int()` 또는 `round()`
6. 마지막에 `print()` 안 함
