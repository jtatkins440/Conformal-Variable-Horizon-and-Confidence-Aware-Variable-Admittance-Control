import os
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# model
# sparse input : interval 24 ms = 6 sample
# lag 20 horizon 20 mean = use 480ms for 480ms prediction


# 데이터 가져오기
folder_path = '/Users/hwangseunghoon/Desktop/Research/Rehab_robot/Preview_CBF_Shoulder/ICRA_Target/Code/AR Model/data1'
file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

pitch_angle_total = []
pitch_vel_total = []
yaw_angle_total = []
yaw_vel_total = []

for file_name in file_list:
    full_file_path = os.path.join(folder_path, file_name)
    print(f'Now reading {full_file_path}')
    
    data = pd.read_csv(full_file_path)
    
    pitch_angle_total.extend(data['ADMI_pitch_pos'])
    pitch_vel_total.extend(data['ADMI_pitch_vel'])
    yaw_angle_total.extend(data['ADMI_yaw_pos'])
    yaw_vel_total.extend(data['ADMI_yaw_vel'])

input_pitch = np.column_stack((pitch_angle_total, pitch_vel_total))
input_yaw = np.column_stack((yaw_angle_total, yaw_vel_total))
inputs = np.column_stack((pitch_angle_total, pitch_vel_total, yaw_angle_total, yaw_vel_total))

split_ratio = 0.8  # 훈련 데이터 비율
train_size = int(split_ratio * inputs.shape[0])  # 훈련 데이터 크기
test_size = inputs.shape[0] - train_size  # 테스트 데이터 크기

train_data = inputs[:train_size]
test_data = inputs[train_size:]

# 일정 간격으로 샘플 선택
interval = 12  # 예시로 48ms 간격 (4ms * 12 = 48ms)
sparse_train_data = train_data[::interval]
sparse_test_data = test_data[::interval]

# Train the VAR model
#num_lags = 20
num_lags = 21 # including current time data too
horizon = 10
model_var = VAR(sparse_train_data)
EstMdl = model_var.fit(num_lags)

# 모델 저장

with open('Var_lag21.pkl', 'wb') as f:
    pickle.dump(EstMdl, f)

print("Model has been saved successfully.")


# Check how many lag model used for training
print("number of lag",EstMdl.k_ar)

