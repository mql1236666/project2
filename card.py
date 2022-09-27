import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scorecardpy as sc
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
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
# 2.1.1计算卡方值
def cal_Chi2(df):
    """从列联表计算出卡方值"""
    res = []
    # 计算values的和
    num_sum = sum(df.values.flatten())
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            # 计算位置i,j上的期望值
            e = sum(df.iloc[i, :]) * sum(df.iloc[:, j]) / num_sum
            tt = (df.iloc[i, j] - e) ** 2 / e
            res.append(tt)
    return sum(res)


# 2.2.2区间合并
def line_merge(df, i, j):
    """将i,j行合并"""
    df.iloc[i, 1] = df.iloc[i, 1] + df.iloc[j, 1]
    df.iloc[i, 2] = df.iloc[i, 2] + df.iloc[j, 2]
    df.iloc[i, 0] = df.iloc[j, 0]
    df = pd.concat([df.iloc[:j, :], df.iloc[j + 1:, :]])
    return df


# 2.2.3定义一个卡方分箱函数（可设置参数置信度水平与箱的个数）停止条件为大于置信水平且小于bin的数目
def ChiMerge(df, variable, flag, confidenceVal=3.841, bin=10):
    '''
    df:传入一个数据框仅包含一个需要卡方分箱的变量与正负样本标识（正样本为1，负样本为0）
    variable:需要卡方分箱的变量名称（字符串）
    flag：正负样本标识的名称（字符串）
    confidenceVal：置信度水平（默认是不进行抽样95%）
    bin：最多箱的数目
    '''

    # 初始化分箱，每个特征值为一个切分点，升序排列
    regroup = df.groupby([variable])[flag].agg(["size", "sum"])
    regroup.columns = ['total_num', 'positive_class']
    regroup['negative_class'] = regroup['total_num'] - regroup['positive_class']  # 统计需分箱变量每个值负样本数
    regroup = regroup.drop('total_num', axis=1).reset_index()
    col_names = regroup.columns

    print('已完成数据读入,正在计算数据初处理')

    # 处理连续没有正样本或负样本的区间，并进行区间的合并（以免卡方值计算报错）
    i = 0
    while i <= (regroup.shape[0] - 2):
        # 如果正样本(1)列或负样本(2)列的数量求和等于0 (求和等于0,说明i和i+1行的值都等于0)
        if sum(regroup.iloc[[i, i + 1], [1, 2]].sum() == 0) > 0:
            # 合并两个区间
            regroup = line_merge(regroup, i, i + 1)
            i = i - 1
        i = i + 1

        # 对相邻两个区间进行卡方值计算
    chi_ls = []  # 创建一个数组保存相邻两个区间的卡方值
    for i in np.arange(regroup.shape[0] - 1):
        chi = cal_Chi2(regroup.iloc[[i, i + 1], [1, 2]])
        chi_ls.append(chi)

    print('已完成数据初处理，正在进行卡方分箱核心操作')

    # 把卡方值最小的两个区间进行合并（卡方分箱核心）
    while True:
        if (len(chi_ls) <= (bin - 1) and min(chi_ls) >= confidenceVal):
            break

        min_ind = chi_ls.index(min(chi_ls))  # 找出卡方值最小的位置索引
        #       合并两个区间
        regroup = line_merge(regroup, min_ind, min_ind + 1)

        if (min_ind == regroup.shape[0] - 1):  # 最小值是最后两个区间的时候
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_ls[min_ind - 1] = cal_Chi2(regroup.iloc[[min_ind, min_ind - 1], [1, 2]])
            # 删除替换前的卡方值
            del chi_ls[min_ind]

        else:
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_ls[min_ind - 1] = cal_Chi2(regroup.iloc[[min_ind, min_ind - 1], [1, 2]])

            # 计算合并后当前区间与后一个区间的卡方值并替换
            chi_ls[min_ind] = cal_Chi2(regroup.iloc[[min_ind, min_ind + 1], [1, 2]])

            # 删除替换前的卡方值
            del chi_ls[min_ind + 1]

    print('已完成卡方分箱核心操作，正在保存结果')

    # 把结果保存成一个数据框

    regroup['variable'] = [variable] * regroup.shape[0]  # 结果表第一列：变量名
    list_temp = []
    for i in np.arange(regroup.shape[0]):
        if i == 0:
            x = '-inf' + ',' + str(regroup.iloc[i, 0])
        elif i == regroup.shape[0] - 1:
            x = str(regroup.iloc[i - 1, 0]) + ',' + 'inf'
        else:
            x = str(regroup.iloc[i - 1, 0]) + ',' + str(regroup.iloc[i, 0])
        list_temp.append(x)
    regroup['bucket'] = list_temp  # 结果表第二列：区间
    # 计算各个箱的woe值
    regroup['ratio_0'] = regroup['positive_class'] / regroup['positive_class'].sum()
    regroup['ratio_1'] = regroup['negative_class'] / regroup['negative_class'].sum()
    regroup['woe'] = np.log((regroup['ratio_0']+0.01)/(regroup['ratio_1']+0.01))
    regroup['max'] = regroup['bucket'].apply(lambda x: x.split(',')[-1]).astype(float)
    regroup['min'] = regroup['bucket'].apply(lambda x: x.split(',')[0]).astype(float)
    return regroup


bins = {}
for var in train_processed.columns:
    bins[var] = ChiMerge(train_processed, var, 'SeriousDlqin2yrs', confidenceVal=3.841, bin=10)

# 2.2将测试集和训练集替换为编码
cutoff = {}
cutpoint = {}
for var in train_processed.columns:
    cutoff[var] = pd.concat([bins['max'], bins['min']], axis=0)
    cutpoint[var] = list(set(cutoff)).sort()
    train_processed[var+'_woe'] = pd.cut(train_processed[var], bins=cutoff, labes=woe_list)
    test_processed[var + '_woe'] = pd.cut(train_processed[var], bins=cutoff, labes=woe_list)
train_woe = train_processed.loc[:, [x for x in train_processed.columns if x.find('_woe') > 0]+['SeriousDlqin2yrs']]
test_woe = test_processed.loc[:, [x for x in test_processed if x.find('_woe') > 0]+['SeriousDlqin2yrs']]

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

# 3模型训练，逻辑回归算法
y_train = train_woe.loc[:, 'SeriousDlqin2yrs']
X_train = train_woe.loc[:, train_woe.columns != 'SeriousDlqin2yrs']
y_test = test_woe.loc[:, 'SeriousDlqin2yrs']
X_test = test_woe.loc[:, train_woe.columns != 'SeriousDlqin2yrs']
lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', class_weight='balanced', n_jobs=-1)
lr.fit(X_train, y_train)


# 4.模型评估
test_pred = lr.predict_proba(X_test)[:, 1]
# ks 和 roc 的性能表现 -----
test_perf = sc.perf_eva(y_test, test_pred, title="test")

# 5.转化成评分卡形式
card = sc.scorecard(bins_adj, lr, X_train.columns, points0=600, odds0=1/19, pdo=20)
print(card)

# 信用得分
train_score = sc.scorecard_ply(train_processed, card, print_step=0)
test_score = sc.scorecard_ply(test_processed, card, print_step=0)

# 查看稳定性指标psi
sc.perf_psi(
  score={'train': train_score, 'test': test_score},
  label={'train': y_train, 'test': y_test}
)