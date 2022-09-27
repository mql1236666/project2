import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scorecardpy as sc
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
# 导入数据
data = pd.read_csv('E:/金融风控模型/评分卡/信用卡评分原始数据/cs-training.csv', index_col=0)
print('数据集维度：', data.shape)


# 1数据清洗和预处理
# 1.1缺失值处理
# 'MonthlyIncome'单独归为一类
data['NumberOfDependents'] = data['NumberOfDependents'].fillna(data['NumberOfDependents'].mode()[0])  # 众数填充

# 1.2异常值处理
# 划分数据集，分层抽样,保证训练集和测试集分布一致
train, test = train_test_split(data, test_size=0.2, stratify=data.loc[:, 'SeriousDlqin2yrs'], random_state=2022)
# 删除年龄大于18的数据
train = train[train['age'] >= 18]
train_processed = train
test_processed = test  # 测试集不处理异常值


# 2特征工程
# 2.1连续特征离散化分箱
bins = sc.woebin(train, y='SeriousDlqin2yrs', stop_limit=0.1, count_distr_limit=0.05,
                 bin_num_limit=8, method='chimerge')

sc.woebin_plot(bins)  # 查看分箱单调性
# 手动调整分箱
breaks_adj = {
    'DebtRatio': [0.3, 0.4, 0.5],
    'MonthlyIncome': [3500, 5000, 6500, 8000, 1000],
    'NumberOfOpenCreditLinesAndLoans': [3, 4, 6, 8],
    'NumberRealEstateLoansOrLines': [3]}
bins_adj = sc.woebin(train, y='SeriousDlqin2yrs', breaks_list=breaks_adj)


# 2.2将测试集和训练集替换为编码
train_woe = sc.woebin_ply(train, bins_adj)
test_woe = sc.woebin_ply(test, bins_adj)

# 2.3特征筛选
# 2.3.1根据iv值剔除小于0.03的特征
iv_dict = {}
for i in bins.keys():
    iv_dict[i] = bins[i]['bin_iv'].sum()
print(iv_dict)
# 特征iv值都大于0.03，不剔除


# 2.1.2共线性筛选
def calculate_vif(df):  # 不能有缺失值和无穷大
    vif = pd.DataFrame()
    vif['index'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif


vif = calculate_vif(train_woe.drop('SeriousDlqin2yrs', axis=1))
# 特征vif值都小于10，无多重共线性，不剔除
print('完成共线性筛选,查看数据')
print(train_woe.head(10))

# 3模型训练与调参，逻辑回归算法
y_train = train_woe.loc[:, 'SeriousDlqin2yrs']
x_train = train_woe.loc[:, train_woe.columns != 'SeriousDlqin2yrs']
y_test = test_woe.loc[:, 'SeriousDlqin2yrs']
x_test = test_woe.loc[:, train_woe.columns != 'SeriousDlqin2yrs']
# # 3.1调参
lr_adj = LogisticRegression(random_state=2022)
params = {'C': [0.5, 0.75, 1, 1.25, 1.5, 2]}
best = GridSearchCV(lr_adj, param_grid=params, refit=True, cv=3, scoring='roc_auc').fit(x_train, y_train)
print('best parameters:', best.best_params_)
print('已完成参数调优')
# # 3.2训练
lr = LogisticRegression(penalty='l1', C=1.5, solver='saga', class_weight='balanced', n_jobs=-1)
lr.fit(x_train, y_train)


# 4.模型评估,ks 和 roc
test_pred = lr.predict_proba(x_test)[:, 1]
test_perf = sc.perf_eva(y_test, test_pred, title="test")

# 5.转化成评分卡形式,假设几率为1/20时，参考评分为600，调整刻度为20
card = sc.scorecard(bins_adj, lr, x_train.columns, points0=600, odds0=1/20, pdo=20)
print(card)

# 计算信用得分
train_score = sc.scorecard_ply(train_processed, card)
test_score = sc.scorecard_ply(test_processed, card)

# 查看稳定性指标psi
sc.perf_psi(
  score={'train': train_score, 'test': test_score},
  label={'train': y_train, 'test': y_test}
)