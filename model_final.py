import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from hyperopt import fmin, tpe, hp
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials


# Load your dataframe and define necessary functions and libraries
df = pd.read_csv("final_test_2.csv")
# 피처 이름에 특수 문자가 포함되어 있는 경우 수정
df.columns = df.columns.str.replace(' ', '_')  # 공백을 언더스코어로 변경
df.columns = df.columns.str.replace('-', '_')  # 대시를 언더스코어로 변경
df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True) # 특수문자를 언더스코어로 변경

df.columns

# 필요한 기타 라이브러리도 가져올 수 있습니다.
X = df[['SpO2_AVG', 'Time_under90_Min_', 'Snoring_Size_1_mild_2_morderate_3_severe_']]
y = df["AHI"]


# 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 표준화 (Standard Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# 목적 함수 정의
def objective(params):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=params['learning_rate'],
        max_depth=int(params['max_depth']),
        min_child_weight=int(params['min_child_weight']),
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree']
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_valid_scaled)
    r2 = r2_score(y_valid, y_pred)
    return -r2  # 목적 함수는 최대화를 위해 음수로 반환

# 하이퍼파라미터 탐색 범위 정의
param_space = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
}

# Trials 객체 생성
trials = Trials()

# 베이지안 최적화 수행
best = fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=50, trials=trials, verbose=1)

print("Best hyperparameters:", best)

# 최적 하이퍼파라미터로 모델 훈련
final_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=best['learning_rate'],
    max_depth=int(best['max_depth']),
    min_child_weight=int(best['min_child_weight']),
    subsample=best['subsample'],
    colsample_bytree=best['colsample_bytree']
)

final_model.fit(X_train_scaled, y_train)

# 최종 모델 성능 평가
y_pred_final = final_model.predict(X_valid_scaled)
r2_final = r2_score(y_valid, y_pred_final)
print("Final R2 Score:", r2_final)

# Make pickle file of our model
pickle.dump(final_model, open("xgb_hyperopt.pkl", "wb"))