# 실습 환경이 제대로 설치됐는지 확인하는 스크립트
# VS Code 에서 ▶ 로 실행하거나, cmd 에서 python 환경확인.py

import sys
print("Python :", sys.version.split()[0])
print("-" * 40)

packages = ["pandas", "numpy", "sklearn", "scipy", "statsmodels", "matplotlib", "xgboost", "lightgbm"]
ok = True
for name in packages:
    try:
        mod = __import__(name)
        print(f"{name:12s} : {mod.__version__}")
    except Exception as e:
        ok = False
        print(f"{name:12s} : 설치 안 됨  ({e})")

print("-" * 40)
if ok:
    print("모든 패키지 정상. 실습 시작 가능!")
else:
    print("일부 패키지가 없습니다. setup.bat 을 다시 실행하거나 pip install <이름> 하세요.")

# 시험장에서 쓰는 '도움말 찾기' 연습 - 아래 주석을 풀고 실행해 보세요
# import pandas as pd
# print(dir(pd.DataFrame))            # DataFrame 이 가진 메서드 목록
# help(pd.DataFrame.fillna)           # 함수 사용법
# print(pd.DataFrame.groupby.__doc__[:500])
