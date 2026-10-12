# %% [markdown]
# # 05. 작업형 2유형 — 분류 템플릿 (★ 이 파일은 통째로 외울 것)
# 문제 예시:
#   "train.csv 로 학습하여 test.csv 고객의 이탈여부(churn)를 예측하시오.
#    예측 확률을 'pred' 컬럼, 'ID' 와 함께 result.csv 로 저장. 평가지표: ROC-AUC"
#
# 흐름: 읽기 → 확인 → ID/타겟 분리 → 결측 → 인코딩 → 검증분리 → 학습 → 평가 → test 예측 → 저장 → 확인

# %% 0. 준비
import pandas as pd
import numpy as np
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data') if '__file__' in globals() else 'data'
train = pd.read_csv(os.path.join(DATA, 'train.csv'))
test = pd.read_csv(os.path.join(DATA, 'test.csv'))

# %% 1. 데이터 확인 (3분 이내)
print(train.shape, test.shape)
print(train.head(3))
print(train.info())
print(train.isnull().sum())
print(test.isnull().sum())
print(train['churn'].value_counts(normalize=True))   # 타겟 분포 (불균형 확인)

# %% 2. ID · 타겟 분리
target = 'churn'
id_col = 'ID'
y = train[target]
X = train.drop(columns=[id_col, target])
X_test = test.drop(columns=[id_col])
test_id = test[id_col]
print(X.shape, X_test.shape)          # 컬럼 수가 같아야 함

# %% 3. 결측치 처리 (수치 → 중앙값, 범주 → 최빈값)
num_cols = X.select_dtypes(include='number').columns
cat_cols = X.select_dtypes(include='object').columns
print('수치:', list(num_cols))
print('범주:', list(cat_cols))

for c in num_cols:
    med = X[c].median()
    X[c] = X[c].fillna(med)
    X_test[c] = X_test[c].fillna(med)
for c in cat_cols:
    mode = X[c].mode()[0]
    X[c] = X[c].fillna(mode)
    X_test[c] = X_test[c].fillna(mode)
print(X.isnull().sum().sum(), X_test.isnull().sum().sum())   # 0 0

# %% 4. 범주형 인코딩 (train + test 합쳐서 fit → 같은 매핑 보장)
from sklearn.preprocessing import LabelEncoder
for c in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([X[c], X_test[c]]))
    X[c] = le.transform(X[c])
    X_test[c] = le.transform(X_test[c])
print(X.head(3))

# (대안) 원핫: X_all = pd.get_dummies(pd.concat([X, X_test])); X = X_all.iloc[:len(X)]; X_test = X_all.iloc[len(X):]

# %% 5. 학습/검증 분리
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
print(X_tr.shape, X_val.shape)

# %% 6. 모델 학습 — RandomForest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0)
model.fit(X_tr, y_tr)

# %% 7. 검증 평가
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
val_proba = model.predict_proba(X_val)[:, 1]     # 1일 확률
val_label = model.predict(X_val)                  # 0/1
print('ROC-AUC :', round(roc_auc_score(y_val, val_proba), 4))
print('Accuracy:', round(accuracy_score(y_val, val_label), 4))
print('F1      :', round(f1_score(y_val, val_label), 4))

# (선택) 다른 모델과 비교
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
print('GB  AUC :', round(roc_auc_score(y_val, gb.predict_proba(X_val)[:, 1]), 4))

# %% 8. 전체 train 으로 재학습 후 test 예측
model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0)
model.fit(X, y)
pred = model.predict_proba(X_test)[:, 1]     # 확률 요구 시
# pred = model.predict(X_test)               # 0/1 라벨 요구 시

# %% 9. 제출 파일 저장 — 문제의 컬럼명·파일명 그대로, index=False 필수
result = pd.DataFrame({'ID': test_id, 'pred': pred})
result.to_csv('result.csv', index=False)

# %% 10. 저장 확인 (반드시!)
chk = pd.read_csv('result.csv')
print(chk.head())
print(chk.shape, test.shape[0])        # 행 수 = test 행 수
print(chk.isnull().sum().sum())        # 결측 0

# %% (연습용) 정답과 비교 — 실제 시험엔 없음
ans = pd.read_csv(os.path.join(DATA, 'test_answer.csv'))
print('실제 test ROC-AUC:', round(roc_auc_score(ans['churn'], chk['pred']), 4))

# %% [markdown]
# ## 변형 대비 메모
# - **다중분류** (타겟이 A/B/C): `predict` 로 라벨 제출. AUC 요구 시 `roc_auc_score(y, proba, multi_class='ovr')`
# - **F1 / accuracy 채점**: `predict` 라벨 제출
# - **불균형 심할 때**: `RandomForestClassifier(class_weight='balanced', ...)`
# - **ID 가 없는 문제**: `pd.DataFrame({'pred': pred}).to_csv(...)`
# - **train 에 test 와 다른 범주가 있을 때**: 4단계처럼 concat 해서 fit 하면 안전
# - **컬럼이 많을 때 특징 중요도**: `pd.Series(model.feature_importances_, index=X.columns).sort_values()`
