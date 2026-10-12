# 작업형 2유형 연습문제 풀이 — 재사용 가능한 함수 형태로 정리
# 시험에서는 함수 없이 템플릿을 그대로 쓰면 됨. 여기서는 두 문제를 같은 코드로 푸는 걸 보여줌.
# %%
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, r2_score
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '03_실습코드', 'data') if '__file__' in globals() else '../03_실습코드/data'


def preprocess(train, test, id_col, target):
    """ID/타겟 분리 → 결측 → 라벨인코딩. (X, y, X_test, test_id) 반환"""
    y = train[target]
    X = train.drop(columns=[id_col, target])
    X_test = test.drop(columns=[id_col])
    num_cols = X.select_dtypes(include='number').columns
    cat_cols = X.select_dtypes(include='object').columns
    for c in num_cols:
        med = X[c].median()
        X[c] = X[c].fillna(med); X_test[c] = X_test[c].fillna(med)
    for c in cat_cols:
        mode = X[c].mode()[0]
        X[c] = X[c].fillna(mode); X_test[c] = X_test[c].fillna(mode)
        le = LabelEncoder().fit(pd.concat([X[c], X_test[c]]))
        X[c] = le.transform(X[c]); X_test[c] = le.transform(X_test[c])
    return X, y, X_test, test[id_col]


# %% 문제 1. 분류 (확률 제출, ROC-AUC)
train = pd.read_csv(os.path.join(DATA, 'train.csv'))
test = pd.read_csv(os.path.join(DATA, 'test.csv'))
X, y, X_test, test_id = preprocess(train, test, 'ID', 'churn')

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0).fit(X_tr, y_tr)
print('valid AUC:', round(roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]), 4))

clf.fit(X, y)
pd.DataFrame({'ID': test_id, 'pred': clf.predict_proba(X_test)[:, 1]}).to_csv('result1.csv', index=False)
chk = pd.read_csv('result1.csv')
print(chk.head(3), chk.shape)

# 변형 1-a (라벨 제출, macro F1)
print('valid macro F1:', round(f1_score(y_val, clf.predict(X_val), average='macro'), 4))
# pd.DataFrame({'ID': test_id, 'pred': clf.predict(X_test)}).to_csv('result1a.csv', index=False)
# 변형 1-b (ID 없음)
# pd.DataFrame({'pred': clf.predict_proba(X_test)[:, 1]}).to_csv('result1b.csv', index=False)

# %% 문제 2. 회귀 (RMSE)
train = pd.read_csv(os.path.join(DATA, 'house_train.csv'))
test = pd.read_csv(os.path.join(DATA, 'house_test.csv'))
X, y, X_test, test_id = preprocess(train, test, 'ID', 'price')

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
reg = RandomForestRegressor(n_estimators=300, random_state=0).fit(X_tr, y_tr)
print('valid RMSE:', round(mean_squared_error(y_val, reg.predict(X_val)) ** 0.5, 2))

reg.fit(X, y)
pd.DataFrame({'ID': test_id, 'pred': reg.predict(X_test)}).to_csv('result2.csv', index=False)
print(pd.read_csv('result2.csv').head(3))

# 변형 2-a (RMSLE): 로그 타겟으로 학습 후 복원
reg_log = RandomForestRegressor(n_estimators=300, random_state=0).fit(X_tr, np.log1p(y_tr))
pred_log = np.expm1(reg_log.predict(X_val))
rmsle = np.sqrt(np.mean((np.log1p(y_val) - np.log1p(pred_log)) ** 2))
print('valid RMSLE (log target):', round(rmsle, 4))

# %% 문제 3. 자기 채점
ans1 = pd.read_csv(os.path.join(DATA, 'test_answer.csv'))
r1 = pd.read_csv('result1.csv')
auc = roc_auc_score(ans1['churn'], r1['pred'])
score1 = 40 if auc >= 0.75 else 30 if auc >= 0.70 else 20 if auc >= 0.65 else 10
print(f'문제1 test AUC = {auc:.4f} → {score1}점')

ans2 = pd.read_csv(os.path.join(DATA, 'house_test_answer.csv'))
r2 = pd.read_csv('result2.csv')
print('문제2 test RMSE =', round(mean_squared_error(ans2['price'], r2['pred']) ** 0.5, 2),
      ' R2 =', round(r2_score(ans2['price'], r2['pred']), 4))
