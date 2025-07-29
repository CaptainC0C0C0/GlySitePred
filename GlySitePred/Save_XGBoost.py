import xgboost as xgb
import pandas as pd
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


# 加载CSV数据
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Label']).values  # 特征
    y = df['Label'].values
    return X, y


# 模型训练与验证
def train_and_evaluate(train_file, val_file, num_epochs=100, batch_size=32, lr=0.001, model_save_path='xgboost_model.json'):
    # 加载训练集和验证集
    X_train, y_train = load_data(train_file)
    X_val, y_val = load_data(val_file)

    # XGBoost模型初始化
    model = xgb.XGBClassifier(
        objective='binary:logistic',  # 二分类问题
        eval_metric='logloss',  # 评价指标
        learning_rate=lr,
        max_depth=4,  # 树的最大深度
        n_estimators=num_epochs,  # 树的数量
        use_label_encoder=False,  # 关闭标签编码器
        verbosity=1,  # 控制输出的详细程度
        subsample=0.8,  # 随机采样
        colsample_bytree=0.9,  # 每棵树使用的特征
        #tree_method='hist',  # 使用 CPU 或 GPU 的高效直方图算法
        device='cuda',  # 使用 GPU 进行训练
        #booster='dart'  # 使用 DART（Dropout Additive Regression Trees）
    )

    # 训练模型
    model.fit(X_train, y_train)

    model.save_model(model_save_path)  # 保存XGBoost模型为文件
    print(f'Model saved to {model_save_path}')
    # 预测结果
    y_pred = model.predict(X_val)
    y_probs = model.predict_proba(X_val)[:, 1]  # 获取阳性类的概率

    # 计算评估指标
    acc = accuracy_score(y_val, y_pred)  # 准确率 (ACC)
    mcc = matthews_corrcoef(y_val, y_pred)  # Matthews相关系数 (MCC)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()  # 混淆矩阵
    sn = tp / (tp + fn)  # 灵敏度 (Sensitivity, Recall)
    sp = tn / (tn + fp)  # 特异性 (Specificity)
    auc = roc_auc_score(y_val, y_probs)  # AUC

    # 打印评估结果
    print("\nEvaluation Results:")
    print(f'Accuracy (ACC): {acc * 100:.2f}%')
    print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')
    print(f'Sensitivity (SN): {sn:.4f}')
    print(f'Specificity (SP): {sp:.4f}')
    print(f'AUC: {auc:.4f}')


if __name__ == '__main__':
    # 设置训练集和验证集路径
    train_file = 'D:/毕设/重采样/下采样/esm2+ProstT5_features_train_ncl_cluster.csv'  # 训练集CSV文件路径
    val_file = 'D:/毕设/重采样/esm2+ProstT5_features_val_400_2.csv'  # 验证集CSV文件路径
    model_save_path = 'xgboost_model.json'  # 保存模型的路径

    # 训练和验证模型
    train_and_evaluate(train_file, val_file, num_epochs=2000, batch_size=32, lr=0.0001, model_save_path=model_save_path)
