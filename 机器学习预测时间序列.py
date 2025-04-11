import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras import layers
from sklearn import preprocessing
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv(r'C:\Users\yizhuo\Desktop\school_resource\人工智能量化投资\代码\data1.csv',encoding='gbk')
# print(df.columns)
print(df.head())
df = df[df['state'] == 'New York']

# df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d %H:%M')
# df['date'] = df['date'].dt.strftime('%Y-%m-%d')
#print(df['date'])

df1 = pd.concat([df['总计测试人数增长'],df['阳性测试增长'],df['死亡增长']],axis=1)
df1.reset_index(inplace=True,drop=True)
print(df1.head())
# print(df1.iloc[-10:])

min_max_scaler = preprocessing.MinMaxScaler()
df0 = min_max_scaler.fit_transform(df1)
data = pd.DataFrame(df0, columns=df1.columns)
x = data.loc[:, ['总计测试人数增长', '阳性测试增长']].to_numpy()
y = data.loc[:, '死亡增长'].to_numpy()
# cut = 1000
cut = 10
train_x = x[:-cut, :]
train_y = y[:-cut]
test_x = x[-cut:, :]
test_y = y[-cut:]

# SVM
model = svm.SVR()
model.fit(train_x,train_y)
test_predictions = model.predict(test_x)
# 计算均方误差（MSE）
mse = mean_squared_error(test_y, test_predictions)
print("svm mse: %f"% mse)


# 决策树
# 初始化决策树对象，基于信息熵
dtc = DecisionTreeRegressor()
dtc.fit(train_x, train_y)  # 训练模型
test_predictions = dtc.predict(test_x)

# 计算均方误差（MSE）
mse = mean_squared_error(test_y, test_predictions)
print("DecisionTreeRegressor MSE:", mse)


# # 神经网络模型
# # 定义神经网络模型
# model = Sequential()
# model.add(layers.Dense(128, activation='relu', input_shape=[2]))
# model.add(layers.Dense(128, activation='relu', input_shape=[2]))
# model.add(layers.Dense(1))

# # 编译模型
# model.compile(loss='binary_crossentropy',
#               optimizer=tf.keras.optimizers.Adam(0.0001),
#               metrics=['accuracy'])
# # 训练模型
# model.fit(train_x, train_y, epochs=500, verbose=0)

# def inverse_transform_col(scaler,y,n_col):
#     '''scaler是对包含多个feature的X拟合的,y对应其中一个feature,n_col为y在X中对应的列编号.返回y的反归一化结果'''
#     y = y.copy()
#     y -= scaler.min_[n_col]
#     y /= scaler.scale_[n_col]
#     return y

# y_pred = model.predict(test_x)
# y_pred = inverse_transform_col(min_max_scaler, y_pred, n_col=2)  # 对预测值反归一化
# test_y = inverse_transform_col(min_max_scaler, test_y, n_col=2)  # 对实际值反归一化
# train_y = inverse_transform_col(min_max_scaler, train_y, n_col=2)

# # 计算均方差
# mse = mean_squared_error(test_y, y_pred)

# print('LSTM MSE:', mse)

# 修改后的神经网络模型部分
from keras.layers import Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

# 改进后的反归一化函数
def inverse_transform_col(scaler, y, n_col):
    """安全的反归一化方法"""
    dummy = np.zeros((len(y), scaler.n_features_in_))
    dummy[:, n_col] = y.flatten()
    return scaler.inverse_transform(dummy)[:, n_col]

# 增强型神经网络模型
def build_enhanced_model(input_shape):
    model = Sequential([
        layers.Dense(256, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        layers.Dense(128, activation='relu'),
        Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model

# 早停回调
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)

# 重新构建并训练模型
enhanced_model = build_enhanced_model((train_x.shape[1],))
history = enhanced_model.fit(
    train_x, train_y,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 模型预测与评估
enhanced_pred = enhanced_model.predict(test_x).flatten()

# 反归一化处理
enhanced_pred = inverse_transform_col(min_max_scaler, enhanced_pred, 2)
test_y_actual = inverse_transform_col(min_max_scaler, test_y, 2)

# 计算改进后的MSE
enhanced_mse = mean_squared_error(test_y_actual, enhanced_pred)
print(f'Enhanced Model MSE: {enhanced_mse:.4f}')