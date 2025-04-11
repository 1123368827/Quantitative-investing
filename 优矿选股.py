import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC  # 支持向量机分类
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

# 加载数据
df = pd.read_csv(r'C:\Users\yizhuo\Desktop\school_resource\人工智能量化投资\代码\PEtop10.csv')

# 显示数据框的前几行
print(df.head())

# 选择相关特征和目标变量
# 假设我们想预测一个列 'buy_signal'（1表示买入，0表示不买入）
# 此处假设有一个'买入信号'的列。如果没有，需要根据条件生成该列。

# 示例：生成买入信号列（这只是一个假设，需要根据实际情况调整条件）
# 例如，如果CHG为正且CHGPct大于某个阈值，我们定义为买入信号
df['buy_signal'] = ((df['CHG'] > 0) & (df['CHGPct'] > 0.01)).astype(int)

# 选择特征列，这里选择与交易相关的一些基本特征
features = ['preClosePrice', 'openPrice', 'highestPrice', 'lowestPrice', 'closePrice', 'CHG', 'CHGPct', 'turnoverVol']
target = 'buy_signal'

# 分离特征和目标变量
X = df[features]
y = df[target]

# 数据标准化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SVM分类器
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM 准确率: {svm_accuracy:.4f}")

# 决策树分类器
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_predictions)
print(f"决策树准确率: {tree_accuracy:.4f}")

# 神经网络模型
nn_model = Sequential()
nn_model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
nn_model.add(Dense(128, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))

# 编译模型
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# 评估模型
nn_loss, nn_accuracy = nn_model.evaluate(X_test, y_test, verbose=0)
print(f"神经网络准确率: {nn_accuracy:.4f}")

# 预测示例
new_data = pd.DataFrame({
    'preClosePrice': [2.03],
    'openPrice': [2.02],
    'highestPrice': [2.043],
    'lowestPrice': [2.0],
    'closePrice': [2.019],
    'CHG': [-0.011],
    'CHGPct': [-0.0054],
    'turnoverVol': [115672215]
})
new_data_scaled = scaler.transform(new_data)
nn_prediction = nn_model.predict(new_data_scaled)
print(f"神经网络预测结果（买入信号概率）: {nn_prediction[0][0]:.4f}")
