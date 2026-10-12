# %% [markdown]
# # 06. 작업형 2유형 — 회귀 템플릿
# 문제 예시:
#   "house_train.csv 로 학습하여 house_test.csv 의 price 를 예측. ID, pred 컬럼으로 result.csv 저장. 평가: RMSE"
# 분류 템플릿과 차이점: Regressor 사용, predict (확률 없음), 평가지표가 RMSE/MAE/R²

# %% 0~2. 읽기 · 확인 · 분리
import pandas as pd
import numpy as np
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data') if '__file__' in globals() else 'data'
train = pd.read_csv(os.path.join(DATA, 'house_train.csv'))
test = pd.read_csv(os.path.join(DATA, 'house_test.csv'))
print(train.shape, test.shape)
print(train.head(3))
print(train.isnull().sum())
print(train['price'].describe())

target, id_col = 'price', 'ID'
y = train[target]
X = train.drop(columns=[id_col, target])
X_test = test.drop(columns=[id_col])
test_id = test[id_col]

# %% 3~4. 결측 · 인코딩
num_cols = X.select_dtypes(include='number').columns
cat_cols = X.select_dtypes(include='object').columns
for c in num_cols:
    med = X[c].median()
    X[c] = X[c].fillna(med); X_test[c] = X_test[c].fillna(med)
for c in cat_cols:
    mode = X[c].mode()[0]
    X[c] = X[c].fillna(mode); X_test[c] = X_test[c].fillna(mode)

from sklearn.preprocessing import LabelEncoder
for c in cat_cols:
    le = LabelEncoder().fit(pd.concat([X[c], X_test[c]]))
    X[c] = le.transform(X[c]); X_test[c] = le.transform(X_test[c])
print(X.head(3))

# %% 5~7. 분리 · 학습 · 평가
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=0)   # 회귀는 stratify 없음
model = RandomForestRegressor(n_estimators=300, random_state=0)
model.fit(X_tr, y_tr)
val_pred = model.predict(X_val)
print('RMSE:', round(mean_squared_error(y_val, val_pred) ** 0.5, 2))
print('MAE :', round(mean_absolute_error(y_val, val_pred), 2))
print('R2  :', round(r2_score(y_val, val_pred), 4))

# 비교: 선형회귀 (원핫이 더 적합하지만 감만 보기)
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_tr, y_tr)
print('LR RMSE:', round(mean_squared_error(y_val, lr.predict(X_val)) ** 0.5, 2))

# %% 8~10. 재학습 · 예측 · 저장 · 확인
model = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y)
pred = model.predict(X_test)
pd.DataFrame({'ID': test_id, 'pred': pred}).to_csv('result_reg.csv', index=False)
chk = pd.read_csv('result_reg.csv')
print(chk.head(), chk.shape)

# (연습용) 정답 비교
ans = pd.read_csv(os.path.join(DATA, 'house_test_answer.csv'))
print('실제 test RMSE:', round(mean_squared_error(ans['price'], chk['pred']) ** 0.5, 2))
print('실제 test R2  :', round(r2_score(ans['price'], chk['pred']), 4))

# %% [markdown]
# ## 변형 대비 메모
# - 타겟이 심하게 치우쳐 있으면(가격 등): `np.log1p(y)` 로 학습 → `np.expm1(pred)` 로 복원. RMSLE 채점일 때 특히 유효
# - RMSE 는 `mean_squared_error(...) ** 0.5` 로. (`squared=False` 는 최신 sklearn 에서 삭제됨)
# - 예측값이 음수면 안 되는 문제(개수, 가격): `np.clip(pred, 0, None)`
