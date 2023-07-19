import pandas as pd
import numpy as np
import sys

# 데이터 시각화
import seaborn as sb
import matplotlib.pyplot as plt

# 딥러닝 모델링
import keras
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
import tensorflow as tf
import joblib
import warnings
from keras.optimizers import Adam
import silence_tensorflow
warnings.filterwarnings(action='ignore')
silence_tensorflow.silence_tensorflow()

# 데이터 불러오기
temp_data = pd.read_csv('/Users/ghkd1/myenv/eeg_project/열처리 데이터.csv', index_col=0)
print(temp_data)
print(temp_data.describe())

# 데이터 중복 제거
temp_data = temp_data.drop_duplicates(['data_date'])

# 시간 데이터 분석상 데이터를 날짜순으로 정렬
temp_data.sort_values(['data_date'], inplace=True)
temp_data.reset_index(inplace=True, drop=True)

# data_date 열 제거
temp_data = temp_data.drop('data_date', axis=1)

# 데이터 스케일링
scaler = MinMaxScaler()  # 최소값과 최대값을 사용해서 0~1 사이의 범위로 데이터를 표준화 해주는 변환입니다.
temp_data = scaler.fit_transform(temp_data)
joblib.dump(scaler, './scaler.pkl')
scaler = joblib.load('./scaler.pkl')



# 1분을 기준으로 데이터를 1초 단위로 추가하면서 다음 1초를 예측하는 데이터 셋을 만들어줍니다.
# 데이터 전처리 - LSTM 입력 형태로 변환
Look_Back = 600
for k in range(len(temp_data) - Look_Back - 1):
    if k == 0:
        X = temp_data[k:k+Look_Back, :]
        Y = temp_data[k+Look_Back, :]
        print(k)
    else:
        X = np.concatenate((X, temp_data[k:k+Look_Back, :]))
        Y = np.concatenate((Y, temp_data[k+Look_Back, :]))
        print(k)

X = X.reshape(-1, Look_Back, temp_data.shape[1])
Y = Y.reshape(-1, temp_data.shape[1])

# 저장된 데이터 불러오기
X = np.load('./X.npy')
Y = np.load('./Y.npy')

# LSTM 모델 생성
def LSTM_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(Look_Back, temp_data.shape[1]), return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(temp_data.shape[1]))
    model.compile(optimizer=Adam(lr=0.001), loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])
    return model

# LSTM 모델 구조 출력
LSTM_model().summary()

# 데이터 분할
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1004)
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, shuffle=True, random_state=1004)

# 조기 종료 콜백 설정
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

# LSTM 모델 학습
history = LSTM_model().fit(train_x, train_x, batch_size=1, epochs=3, verbose=1, validation_data=(valid_x, valid_y), callbacks=[early_stop])

# 학습된 모델 저장
LSTM_model().save('./lstm_model_1.h5')
model_1 = load_model('./lstm_model_0.h5')
model_2 = load_model('./lstm_model_1.h5')
# model_3 = load_model('./lstm_model_3.h5')
# model_4 = load_model('./lstm_model_4.h5')
# model_5 = load_model('./lstm_model_5.h5')

# 예측
predict = []
num = 3000

model_predict_1 = model_1.predict(test_x[num].reshape(-1, Look_Back, train_x.shape[-1]))
model_predict_2 = model_2.predict(test_x[num].reshape(-1, Look_Back, train_x.shape[-1]))
# model_predict_3 = model_3.predict(test_x[num].reshape(-1, Look_Back, train_x.shape[-1]))
# model_predict_4 = model_4.predict(test_x[num].reshape(-1, Look_Back, train_x.shape[-1]))
# model_predict_5 = model_5.predict(test_x[num].reshape(-1, Look_Back, train_x.shape[-1]))

# 스케일 역변환 후 출력
print("예측1: ", scaler.inverse_transform(model_predict_1)[0].astype(int))
print("예측2: ", scaler.inverse_transform(model_predict_2)[0].astype(int))
# print("예측3: ", scaler.inverse_transform(model_predict_3)[0].astype(int))
# print("예측4: ", scaler.inverse_transform(model_predict_4)[0].astype(int))
# print("예측5: ", scaler.inverse_transform(model_predict_5)[0].astype(int))
print("테스트값: ", scaler.inverse_transform(test_y[num].reshape(-1, train_x.shape[-1]))[0].astype(int))
