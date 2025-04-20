import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import pickle
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportion_confint

# 1. 加载数据
data_path = r"C:\Users\zzy\Desktop\Imagine\five\data\Gene\TCGA-model.xlsx"
df = pd.read_excel(data_path, header=0)
print(df.columns)

# 2. 划分训练集和测试集（提前划分，确保后续预处理只基于训练集）
X = df.drop('Status', axis=1)  # 特征
y = df['Status']              # 目标变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3. 数据预处理（仅基于训练集）
# 检查非数值型列
non_numeric_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"非数值型列: {non_numeric_cols}")

# 使用训练集的编码规则对训练集和测试集进行编码
for col in non_numeric_cols:
    if col != 'Status':  # 保留目标列 'Status'
        # 仅使用训练集的唯一值创建编码映射
        train_values = X_train[col].unique()
        value_to_int = {v: i for i, v in enumerate(train_values)}
        
        # 应用编码到训练集
        X_train[col] = X_train[col].map(value_to_int)
        
        # 处理测试集：未见过的值设为-1
        X_test[col] = X_test[col].map(value_to_int)
        X_test[col] = X_test[col].where(pd.notnull(X_test[col]), -1)  # 处理未见过值
        X_test[col] = X_test[col].astype(int)  # 确保类型一致

# 合并 X_train 和 y_train，以便在计算相关性时能够访问 'Status' 列
X_train_with_status = pd.concat([X_train, y_train], axis=1)

# 4. 计算Spearman相关性并绘制散点图（仅使用训练集）
def calculate_spearman_correlation(df, target_col):
    correlations = {}
    for col in df.columns:
        if col != target_col and pd.api.types.is_numeric_dtype(df[col]):
            corr, _ = spearmanr(df[col], df[target_col])
            correlations[col] = corr
    return correlations

# 计算Spearman相关性
spearman_correlations = calculate_spearman_correlation(X_train_with_status, 'Status')

# 将相关性结果转换为DataFrame
corr_df = pd.DataFrame(list(spearman_correlations.items()), columns=['Feature', 'Spearman Correlation'])

# 按相关性排序
corr_df = corr_df.sort_values(by='Spearman Correlation', key=abs, ascending=False)

# 打印相关性结果
print("Spearman Correlation with Target Variable (Status):")
print(corr_df)

# 绘制相关性条形图
plt.figure(figsize=(12, 8))
corr_df.plot(kind='barh', x='Feature', y='Spearman Correlation', legend=False)
plt.title('Spearman Correlation with Target Variable (Status)')
plt.xlabel('Spearman Correlation')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('spearman_correlation.png')  # 保存为图片
plt.show()

# 绘制散点图（选择前5个相关性最高的特征）
top_features = corr_df.head()['Feature'].tolist()

plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_features, 1):
    plt.subplot(2, 3, i)
    plt.scatter(X_train_with_status[feature], X_train_with_status['Status'], alpha=0.5)
    plt.title(f'Scatter Plot of {feature} vs Status')
    plt.xlabel(feature)
    plt.ylabel('Status')
plt.tight_layout()
plt.savefig('scatter_plots.png')  # 保存为图片
plt.show()

# 4.1 绘制数据相关性三角形热图
# 计算所有特征之间的Spearman相关性矩阵
corr_matrix = X_train.corr(method='spearman')

# 创建掩码以隐藏上三角部分（不包括对角线）
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # k=1 表示从对角线的下一个元素开始遮盖

# 绘制三角形热图
plt.figure(figsize=(15, 12))
cmap = 'coolwarm'  # 使用coolwarm调色板

# 确保掩码正确应用，并且热图显示完整
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap=cmap,
    annot=True,
    fmt=".2f",
    square=True,  # 强制正方形显示
    cbar_kws={"shrink": .8}
)

plt.title('Spearman Correlation Triangle Heatmap OF NSCLC-Diabetes Mellitus')
plt.tight_layout()
plt.savefig('correlation_triangle_heatmap.png')  # 保存为图片
plt.show()

# 5. 标准化特征（仅在训练集上进行）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 使用训练集的均值和标准差对测试集进行标准化

# 6. 定义XGBoost模型及其初始参数
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, objective='binary:logistic', random_state=11)

# 7. 定义贝叶斯优化函数
def bayesian_optimization_xgb(model, pbounds, X_train, y_train):
    def optimization_function(**params):
        # 处理XGBoost的max_depth参数
        if 'max_depth' in params:
            params['max_depth'] = int(params['max_depth'])
        
        # 更新模型参数
        model.set_params(**params)
        
        # 使用K折交叉验证评估模型
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # 在每个fold内重新进行标准化
            fold_scaler = StandardScaler()
            X_train_fold_scaled = fold_scaler.fit_transform(X_train_fold)
            X_val_fold_scaled = fold_scaler.transform(X_val_fold)
            
            # 训练模型
            model_fold = model.__class__(**model.get_params())
            model_fold.fit(X_train_fold_scaled, y_train_fold)
            
            # 预测并计算AUC
            y_pred_proba = model_fold.predict_proba(X_val_fold_scaled)[:, 1]
            auc = roc_auc_score(y_val_fold, y_pred_proba)
            auc_scores.append(auc)
        
        return np.mean(auc_scores)
    
    optimizer = BayesianOptimization(
        f=optimization_function,
        pbounds=pbounds,
        random_state=10000
    )
    
    optimizer.maximize(init_points=5, n_iter=25)
    
    best_params = optimizer.max['params']
    
    # 处理XGBoost的max_depth参数
    if 'max_depth' in best_params:
        best_params['max_depth'] = int(best_params['max_depth'])
    
    return best_params

# 8. 执行贝叶斯优化
print("Optimizing XGBoost...")
pbounds = {
    'max_depth': (3, 15),  # 这里可以是浮点数，但最终会被转换为整数
    'gamma': (0, 0.5),
    'min_child_weight': (0.5, 2),
    'reg_alpha': (0, 0.5),
    'reg_lambda': (0.5, 2),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1),
    'scale_pos_weight': (1, 100)  # 修正为合理范围
}

best_params = bayesian_optimization_xgb(xgb_model, pbounds, X_train_scaled, y_train)

# 更新模型参数
xgb_model.set_params(**best_params)

# 训练模型
xgb_model.fit(X_train_scaled, y_train)

# 预测概率和类别
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
y_pred = xgb_model.predict(X_test_scaled)

# --- 新增：绘制混淆矩阵 ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix - XGBoost')
plt.savefig('confusion_matrix_XGBoost.png')
plt.show()

# 计算评估指标
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# 使用K折交叉验证（修正后的标准化流程）
K = 10
cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

# 初始化指标列表
val_auc_scores = []
val_precision_scores = []
val_recall_scores = []
val_f1_scores = []
val_accuracy_scores = []
val_sensitivity_scores = []
val_specificity_scores = []
val_acc_ci_lower = []  # 修正：初始化为列表
val_acc_ci_upper = []  # 修正：初始化为列表

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):  # 使用原始X_train
    # 在每个fold内重新进行标准化
    fold_scaler = StandardScaler()
    X_train_fold_scaled = fold_scaler.fit_transform(X_train.iloc[train_idx])
    X_val_fold_scaled = fold_scaler.transform(X_train.iloc[val_idx])
    y_train_fold = y_train.iloc[train_idx]
    y_val_fold = y_train.iloc[val_idx]
    
    # 训练模型
    model_fold = xgb_model.__class__(**xgb_model.get_params())
    model_fold.fit(X_train_fold_scaled, y_train_fold)
    
    # 验证集评估
    y_val_fold_proba = model_fold.predict_proba(X_val_fold_scaled)[:, 1]
    y_val_fold_pred = model_fold.predict(X_val_fold_scaled)
    
    # 计算指标 - 验证集
    val_auc = roc_auc_score(y_val_fold, y_val_fold_proba)
    val_precision = precision_score(y_val_fold, y_val_fold_pred)
    val_recall = recall_score(y_val_fold, y_val_fold_pred)
    val_f1 = f1_score(y_val_fold, y_val_fold_pred)
    val_cm = confusion_matrix(y_val_fold, y_val_fold_pred)
    val_tn, val_fp, val_fn, val_tp = val_cm.ravel()
    
    val_sensitivity = val_tp / (val_tp + val_fn) if (val_tp + val_fn) != 0 else 0
    val_specificity = val_tn / (val_tn + val_fp) if (val_tn + val_fp) != 0 else 0
    
    val_accuracy = accuracy_score(y_val_fold, y_val_fold_pred)
    # 计算置信区间
    val_ci = proportion_confint(count=val_tp + val_tn, nobs=len(y_val_fold), alpha=0.05, method="normal")
    val_acc_ci_lower_value = val_ci[0]
    val_acc_ci_upper_value = val_ci[1]
    
    # 保存结果
    val_auc_scores.append(val_auc)
    val_precision_scores.append(val_precision)
    val_recall_scores.append(val_recall)
    val_f1_scores.append(val_f1)
    val_accuracy_scores.append(val_accuracy)
    val_sensitivity_scores.append(val_sensitivity)
    val_specificity_scores.append(val_specificity)
    val_acc_ci_lower.append(val_acc_ci_lower_value)  # 修正：存储值
    val_acc_ci_upper.append(val_acc_ci_upper_value)  # 修正：存储值

# 输出分类指标
print("\nXGBoost Validation Results:")
print(f"Validation AUC: {np.mean(val_auc_scores):.4f} ± {np.std(val_auc_scores):.4f}")
print(f"Validation Precision: {np.mean(val_precision_scores):.4f} ± {np.std(val_precision_scores):.4f}")
print(f"Validation Recall: {np.mean(val_recall_scores):.4f} ± {np.std(val_recall_scores):.4f}")
print(f"Validation F1 Score: {np.mean(val_f1_scores):.4f} ± {np.std(val_f1_scores):.4f}")
print(f"Validation Sensitivity: {np.mean(val_sensitivity_scores):.4f} ± {np.std(val_sensitivity_scores):.4f}")
print(f"Validation Specificity: {np.mean(val_specificity_scores):.4f} ± {np.std(val_specificity_scores):.4f}")
print(f"Validation Accuracy: {np.mean(val_accuracy_scores):.4f} ± {np.std(val_accuracy_scores):.4f}")
print(f"Validation Accuracy 95% CI: [{np.mean(val_acc_ci_lower):.4f}, {np.mean(val_acc_ci_upper):.4f}]")

# 12. SHAP解释（适用于XGBoost模型）
# 创建SHAP解释器
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled)

# 创建Explanation对象
shap_explanation = shap.Explanation(
    values=shap_values,
    base_values=explainer.expected_value,
    data=X_test_scaled,
    feature_names=X.columns
)

# 全局特征重要性（条形图）
plt.figure(figsize=(15, 10))
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, plot_type="bar")
plt.title("Global Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig('shap_global_importance.png')
plt.show()

# 局部解释（单个样本）
shap.plots.waterfall(shap_explanation[0], max_display=15)

# 依赖图（特征与SHAP值的关系）
feature_name = X.columns[0]
shap.plots.scatter(shap_explanation[:, feature_name])

# SHAP摘要图（小提琴图）
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)

# Force Plot（样本解释）
plt.figure(figsize=(28, 6))  # 调整为更合理的尺寸（推荐宽度15-20）
ax = plt.gca()

# 生成force plot
shap_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test_scaled[0],
    feature_names=X.columns.tolist(),
    matplotlib=True,
    show=False  # 关闭自动显示
)

# 设置x轴范围（关键新增代码）
ax.set_xlim(8, 12)  # 强制限定x轴显示区间

# 调整字体和标签参数
plt.rcParams.update({
    'font.size': 8,           # 主字体大小
    'xtick.labelsize': 7,     # x轴标签大小
    'ytick.labelsize': 7      # y轴标签大小
})

# 旋转特征标签（关键调整！）
ax.tick_params(axis='x', which='both', rotation=45)  # 45度旋转

# 手动调整边距
plt.subplots_adjust(
    left=0,    # 左边距
    right=0.9,   # 右边距
    bottom=0.4, # 底部增加空间给旋转的标签
    top=0.9      # 顶部边距
)

plt.show()

# 13. 保存XGBoost模型
with open('XGBoost.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)

print("XGBoost模型已保存为 XGBoost.pkl 文件")
X_test.to_csv('X_test.csv', index=False)
