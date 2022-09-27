import pandas as pd
import numpy as np
import scorecardpy as sc
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import learning_curve  # 查看是否过拟合
import warnings
train = pd.read_csv('E:/金融风控模型/评分卡/信用卡评分原始数据/cs-training.csv', index_col=0)
test = pd.read_csv('E:/金融风控模型/评分卡/信用卡评分原始数据/cs-test.csv', index_col=0)
print(train.columns)
print('训练数据集：', train.shape, '测试数据集：', test.shape)
# 1数据清洗和预处理
# 缺失值处理
for data in [train, test]:
    data['MonthlyIncome'] = data['MonthlyIncome'].fillna(data['MonthlyIncome'].median())
    data['NumberOfDependents'] = data['NumberOfDependents'].fillna(data['NumberOfDependents'].median())


def fill_miss_byRandomForest(data_df, obj_column, missing_other_column):
    """
    data_df: DataFrame类型的数据
    obj_column：待填补缺失值的列名
    missing_other_column：数据中含义空值的其他列
    """
    data_df = data_df.drop(missing_other_column, axis=1)  # 先把有缺失的其他列删除掉missing_other_column
    # 分成已知该特征和未知该特征两部分
    known = data_df[data_df[obj_column].notnull()]
    unknown = data_df[data_df[obj_column].isnull()]
    # y为结果标签值
    y_know = known[obj_column]
    # X为特征属性值
    X_know= known.drop(obj_column, axis=1)
    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(X_know, y_know)
    # 用得到的模型进行未知特征值预测
    # X为特征属性值
    X_unknow = unknown.drop(obj_column, axis=1)
    predicted = rfr.predict(X_unknow).round(0)
    data_df.loc[(data_df[obj_column].isnull()), obj_column] = predicted
    return data_df


# 初步划分连续特征和离散特征
def get_numerical_serial_features(df, feas):
    numerical_serial_feature = []
    numerical_noserial_feature = []
    for fea in feas:
        temp = df[fea].nunique()
        if temp <= 10:
            numerical_noserial_feature.append(fea)
        else:
            numerical_serial_feature.append(fea)
    return numerical_serial_feature, numerical_noserial_feature


feature = train.drop("SeriousDlqin2yrs", axis=1).columns
print(feature)
numerical_serial_feature, numerical_noserial_feature = get_numerical_serial_features(train, feature)
# 异常值处理
train = train[train['age'] >= 18]

#  特征工程
# 特征编码
# 特征衍生
# 特征分箱

# 2.3连续特征离散化（卡方分箱）
# 2.3.1计算卡方值
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


# 2.3.2区间合并
def line_merge(df, i, j):
    """将i,j行合并"""
    df.iloc[i, 1] = df.iloc[i, 1] + df.iloc[j, 1]
    df.iloc[i, 2] = df.iloc[i, 2] + df.iloc[j, 2]
    df.iloc[i, 0] = df.iloc[j, 0]
    df = pd.concat([df.iloc[:j, :], df.iloc[j + 1:, :]])
    return df


# 定义一个卡方分箱（可设置参数置信度水平与箱的个数）停止条件为大于置信水平且小于bin的数目
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
            x = '-inf' + '~' + str(regroup.iloc[i, 0])
        elif i == regroup.shape[0] - 1:
            x = str(regroup.iloc[i - 1, 0]) + '~' + 'inf'
        else:
            x = str(regroup.iloc[i - 1, 0]) + '~' + str(regroup.iloc[i, 0])
        list_temp.append(x)
    regroup['bucket'] = list_temp  # 结果表第二列：区间
    # 计算各个箱的woe值
    regroup['ratio_0'] = regroup['positive_class'] / regroup['positive_class'].sum()
    regroup['ratio_1'] = regroup['negative_class'] / regroup['negative_class'].sum()
    regroup['woe'] = np.log(regroup['ratio_0'] / regroup['ratio_1'])
    regroup.loc[regroup['woe'] == float('inf'), 'woe'] = 9999
    regroup['cutoff'] = regroup['bucket'].apply(lambda x: x.split('~')[-1]).astype(float)
    return regroup


# woe替换函数
# 切分点要和woe值对应，woe值对应的是区间的最小值,区间左开右闭
def woe_replace(df, var, woe, cut):
    '''
    df:dataframe数据
    var：变量名，字符串格式
    woe：woe值列表
    cut：切分点，列表
    '''
    var_woe = var + '_woe'
    for i in range(len(woe)):
        if i == 0:
            df.loc[(df[var] <= cut[i + 1]), var_woe] = woe[i]
        elif (i > 0) and (i <= len(woe) - 2):
            df.loc[(df[var] > cut[i]) & (df[var] <= cut[i + 1]), var_woe] = woe[i]
        else:
            df.loc[(df[var] > cut[len(woe) - 1]), var_woe] = woe[len(woe) - 1]
    return df.drop(var, axis=1)


dict_var = {}
for i in numerical_serial_feature:
    dict_var[i] = ChiMerge(train, i, 'SeriousDlqin2yrs', confidenceVal=3.841, bin=10)
# 计算切分点
    woe = np.array(dict_var[i]['woe'])
    cutoff = np.array(dict_var[i]['min'])
    train_woe = woe_replace(train, i, woe, cutoff)
    test_woe = woe_replace(test, i, woe, cutoff)
print('完成特征分箱', '查看数据前10行')
print(train.head(10))

# 特征筛选
# iv值初筛选
dict_iv = {}
for i in numerical_serial_feature:
    dict_var[i]['iv'] = (dict_var[i]['ratio_0'] - dict_var[i]['ratio_1']) * dict_var[i]['woe']
    dict_iv[i] = dict_var[i]['iv'].sum()

print(dict_iv)
dt = pd.DataFrame(list(dict_iv.items()), columns=['feas', 'iv'])
feature_delect = list(dt[dt['iv'] <= 0.02]['feas'])
train_iv = train.drop(feature_delect, axis=1)
test_iv = train.drop(feature_delect, axis=1)
# 共线性筛选
# 2.4.2共线性筛选
def calculate_vif(df):  # 不能有缺失值和无穷大
    vif = pd.DataFrame()
    vif['index'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif


# 以VIF100为阈值
vif = calculate_vif(train_iv.drop('SeriousDlqin2yrs', axis=1))
feature = vif[vif['VIF'] > 100]
vif_list = list(feature['index'])
train_vif = train_iv.drop(vif_list, axis=1)
test_vif = test_iv.drop(vif_list, axis=1)
print('完成共线性筛选', '查看数据')
print(train_vif.head(10))
# 将测试集集转换为woe值


y_train = train_vif.loc[:, 'SeriousDlqin2yrs']
x_train = train_vif.loc[:, train_vif.columns != 'SeriousDlqin2yrs']
y_test = test_vif.loc[:, 'SeriousDlqin2yrs']
x_test = test_vif.loc[:, test_vif.columns != 'SeriousDlqin2yrs']
print('训练集特征：', train_vif.columns, '测试集特征：', test.columns)
# 3.2调参
lr_adj = LogisticRegression(random_state=2022)
params = {'C': [0.01, 0.05, 0.1, 0.15, 0.2]}
best = GridSearchCV(lr_adj, param_grid=params, refit=True, cv=3, scoring='roc_auc').fit(x_train, y_train)
print('best parameters:', best.best_params_)
print('已完成参数调优')

# 3.2模型训练
model_lr = LogisticRegression(penalty='l1', solver='saga', n_jobs=-1, random_state=2022)
model_lr.fit(x_train, y_train)
score = cross_val_score(model_lr, x_train, y_train, cv=3, scoring='roc_auc')
print(score.mean())
print('完成模型训练')
# 4.保存模型，使用sklearn中的模块joblib
joblib.dump(model_lr, 'D:/个人违约预测/model_lr.pkl')
print('已完成模型保存')
# 4模型评估


# 泛化能力，查看学习曲线
# 构建学习曲线评估器，train_sizes：控制用于生成学习曲线的样本的绝对或相对数量
train_sizes, train_scores, test_scores = learning_curve(estimator=model_lr, X=x_train, y=y_train,
                                                        train_sizes=np.linspace(0.1, 1, 10), cv=3, n_jobs=-1)
# 统计结果
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# 绘制效果
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test accuracy')
plt.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.3, 1.2])  # 纵坐标起始值
plt.show()

# 模型训练，选择逻辑回归算法

model_lr.coef_
model_lr.intercept_

# predicted proability
# 可能性预测
train_pred = model_lr.predict_proba(x_train)[:, 1]
test_pred = model_lr.predict_proba(x_test)[:, 1]

# 模型评估
# ks 和 roc 的性能表现 -----
# 评估指标
y_pre = model_lr.predict(x_test)
# auc值
auc = roc_auc_score(y_test, y_pre)
print('auc', auc)
# roc曲线

yscore = model_lr.predict_proba(x_test)
y_score = yscore[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)  # 假正率，真正率和阈值
plt.plot(fpr, tpr, color='brown')  # 线图
plt.title("ROC curve")  # 标题
plt.xlabel("FPR")  # 横坐标
plt.ylabel("TPR")  # 纵坐标
label = ["Test - AUC:" + str(round(auc, 3))]  # 标签
plt.legend(labels='roc', loc="lower right")
plt.show()


# # 转化成评分卡
# card = sc.scorecard(bins, model_lr, x.columns)
# # credit score
# # 信用得分
# train_score = sc.scorecard_ply(train, card, print_step=0)
# test_score = sc.scorecard_ply(test, card, print_step=0)
#
# # 模型稳定性psi
# #
# sc.perf_psi(
#   score={'train': train_score, 'test': test_score},
#   label={'train': y, 'test': y_test})

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
# 2.3连续特征离散化（卡方分箱）
# 2.3.1计算卡方值
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


# 2.3.2区间合并
def line_merge(df, i, j):
    """将i,j行合并"""
    df.iloc[i, 1] = df.iloc[i, 1] + df.iloc[j, 1]
    df.iloc[i, 2] = df.iloc[i, 2] + df.iloc[j, 2]
    df.iloc[i, 0] = df.iloc[j, 0]
    df = pd.concat([df.iloc[:j, :], df.iloc[j + 1:, :]])
    return df


# 定义一个卡方分箱（可设置参数置信度水平与箱的个数）停止条件为大于置信水平且小于bin的数目
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
bins = sc.woebin(train, y='SeriousDlqin2yrs', stop_limit=0.1, count_distr_limit=0.05,
                 bin_num_limit=8, method='chimerge')
cutoff = pd.concat([bins['max'], bins['min']], axis=0)
cutpoint = list(set(cutoff)).sort()
sc.woebin_plot(bins)  # 查看分箱单调性
# 手动调整分箱
breaks_adj = {
    'DebtRatio': [0.3, 0.4, 0.5, 0.7],
    'MonthlyIncome': [3500, 5000, 6500, 8000, 1000],
    'NumberOfOpenCreditLinesAndLoans': [3, 4, 6, 8, 9],
    'NumberRealEstateLoansOrLines': [3]
}
bins_adj = sc.woebin(train, y='SeriousDlqin2yrs', breaks_list=breaks_adj)


# 2.2将测试集和训练集替换为woe值
def woe_replace(df, var, bins):
    """
    df:dataframe数据
    var：变量名，字符串格式
    woe：woe值列表
    cut：切分点，列表"""
    woe = bins['woe']
    cut = bins['cut']
    var_woe = var + '_woe'
    for i in range(len(woe)):
        if i == 0:
            df.loc[(df[var] <= cut[i + 1]), var_woe] = woe[i]
        elif (i > 0) and (i <= len(woe) - 2):
            df.loc[(df[var] > cut[i]) & (df[var] <= cut[i + 1]), var_woe] = woe[i]
        else:
            df.loc[(df[var] > cut[len(woe) - 1]), var_woe] = woe[len(woe) - 1]
    return df.drop(var, axis=1)
train_woe = sc.woebin_ply(train, bins)
test_woe = sc.woebin_ply(test, bins)

# 2.3特征筛选
# 2.3.1特征初筛选，根据iv值剔除小于0.03的特征
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
# 特征vif值都小于10，不剔除
print('完成共线性筛选', '查看数据')
print(train_woe.head(10))

# 3模型训练，逻辑回归算法
y_train = train_woe.loc[:, 'SeriousDlqin2yrs']
X_train = train_woe.loc[:, train_woe.columns != 'SeriousDlqin2yrs']
y_test = test_woe.loc[:, 'SeriousDlqin2yrs']
X_test = test_woe.loc[:, train_woe.columns != 'SeriousDlqin2yrs']
lr = LogisticRegression(penalty='l1', C=0.9, solver='saga', class_weight='balanced', n_jobs=-1)
lr.fit(X_train, y_train)


# 4.模型评估
train_pred = lr.predict_proba(X_train)[:, 1]
test_pred = lr.predict_proba(X_test)[:, 1]
# ks 和 roc 的性能表现 -----
train_perf = sc.perf_eva(y_train, train_pred, title="train")
test_perf = sc.perf_eva(y_test, test_pred, title="test")

# 5.转化成评分卡形式
def myfunc(x):
    return str(x[0]) + '_' + str(x[1])


##生成评分卡
def create_score(bins, dict_params, dict_cont_bin, dict_disc_bin):
    ##假设几率在1:60时对应的参考分值为600分，分值调整刻度PDO为20。
    odds, PDO = 1 / 60, 20
    B = PDO / np.log(2)
    A = base_point + B * np.log(odds)
    # 计算基础分
    base_points = round(A - B * dict_params['intercept'])
    d1 = pd.DataFrame()
    dict_bin_score = {}
    for k in dict_params.keys():
        if k != 'intercept':
            d2 = pd.DataFrame([dict_woe_map[k.split(sep='_woe')[0]]]).T
            d2.reset_index(inplace=True)
            d2.columns = ['bin', 'woe_val']
            ##计算分值
            d2['score'] = round(-params_B * df_temp.woe_val * dict_params[k])
            dict_bin_score[k.split(sep='_BIN')[0]] = dict(zip(d2['bin'], d2['score']))
            ##连续变量的计算
            if k.split(sep='_BIN')[0] in dict_cont_bin.keys():
                df_1 = dict_cont_bin[k.split(sep='_BIN')[0]]
                df_1['var_name'] = df_1[['bin_low', 'bin_up']].apply(myfunc, axis=1)
                df_1 = df_1[['total', 'var_name']]
                d1 = pd.merge(d1, df_1, on='bin')
                d1['var_name_raw'] = k.split(sep='_BIN')[0]
                d1 = pd.concat([d1, d2], axis=0)
            ##离散变量的计算
            elif k.split(sep='_BIN')[0] in dict_disc_bin.keys():
                d2 = pd.merge(d2, dict_disc_bin[k.split(sep='_BIN')[0]], on='bin')
                d2['var_name_raw'] = k.split(sep='_BIN')[0]
                d1 = pd.concat([d1, d2], axis=0)

    d1['score_base'] = base_points
    return d1


##利用评分卡计算样本分数
def cal_score(df_1, dict_bin_score, dict_cont_bin, dict_disc_bin, base_points):
    ##先对原始数据分箱映射，然后，用分数字典dict_bin_score映射分数，基础分加每项的分数就是最终得分
    df_1.reset_index(drop=True, inplace=True)
    df_all_score = pd.DataFrame()
    ##连续变量
    for i in dict_cont_bin.keys():
        if i in dict_bin_score.keys():
            df_all_score = pd.concat(
                [df_all_score, varbin_meth.cont_var_bin_map(df_1[i], dict_cont_bin[i]).map(dict_bin_score[i])], axis=1)
    ##离散变量
    for i in dict_disc_bin.keys():
        if i in dict_bin_score.keys():
            df_all_score = pd.concat(
                [df_all_score, varbin_meth.disc_var_bin_map(df_1[i], dict_disc_bin[i]).map(dict_bin_score[i])], axis=1)

    df_all_score.columns = [x.split(sep='_BIN')[0] for x in list(df_all_score.columns)]
    df_all_score['base_score'] = base_points
    df_all_score['score'] = df_all_score.apply(sum, axis=1)
    df_all_score['target'] = df_1.target
    return df_all_score


card = sc.scorecard(bins, lr, X_train.columns)
print(card)

# 信用得分
train_score = sc.scorecard_ply(train_woe, card, print_step=0)
test_score = sc.scorecard_ply(test_woe, card, print_step=0)

# psi
def calculate_psi(base_list, test_list, bins=20, min_sample=10):
    try:
        base_df = pd.DataFrame(base_list, columns=['score'])
        test_df = pd.DataFrame(test_list, columns=['score'])

        # 1.去除缺失值后，统计两个分布的样本量
        base_notnull_cnt = len(list(base_df['score'].dropna()))
        test_notnull_cnt = len(list(test_df['score'].dropna()))

        # 空分箱
        base_null_cnt = len(base_df) - base_notnull_cnt
        test_null_cnt = len(test_df) - test_notnull_cnt

        # 2.最小分箱数
        q_list = []
        if type(bins) == int:
            bin_num = min(bins, int(base_notnull_cnt / min_sample))
            q_list = [x / bin_num for x in range(1, bin_num)]
            break_list = []
            for q in q_list:
                bk = base_df['score'].quantile(q)
                break_list.append(bk)
            break_list = sorted(list(set(break_list)))  # 去重复后排序
            score_bin_list = [-np.inf] + break_list + [np.inf]
        else:
            score_bin_list = bins

        # 4.统计各分箱内的样本量
        base_cnt_list = [base_null_cnt]
        test_cnt_list = [test_null_cnt]
        bucket_list = ["MISSING"]
        for i in range(len(score_bin_list) - 1):
            left = round(score_bin_list[i + 0], 4)
            right = round(score_bin_list[i + 1], 4)
            bucket_list.append("(" + str(left) + ',' + str(right) + ']')

            base_cnt = base_df[(base_df.score > left) & (base_df.score <= right)].shape[0]
            base_cnt_list.append(base_cnt)

            test_cnt = test_df[(test_df.score > left) & (test_df.score <= right)].shape[0]
            test_cnt_list.append(test_cnt)

        # 5.汇总统计结果
        stat_df = pd.DataFrame({"bucket": bucket_list, "base_cnt": base_cnt_list, "test_cnt": test_cnt_list})
        stat_df['base_dist'] = stat_df['base_cnt'] / len(base_df)
        stat_df['test_dist'] = stat_df['test_cnt'] / len(test_df)

        def sub_psi(row):
            # 6.计算PSI
            base_list = row['base_dist']
            test_dist = row['test_dist']
            # 处理某分箱内样本量为0的情况
            if base_list == 0 and test_dist == 0:
                return 0
            elif base_list == 0 and test_dist > 0:
                base_list = 1 / base_notnull_cnt
            elif base_list > 0 and test_dist == 0:
                test_dist = 1 / test_notnull_cnt

            return (test_dist - base_list) * np.log(test_dist / base_list)

        stat_df['psi'] = stat_df.apply(lambda row: sub_psi(row), axis=1)
        stat_df = stat_df[['bucket', 'base_cnt', 'base_dist', 'test_cnt', 'test_dist', 'psi']]
        psi = stat_df['psi'].sum()

    except:
        print('error!!!')
        psi = np.nan
        stat_df = None
    return psi, stat_df
sc.perf_psi(
  score={'train': train_score, 'test': test_score},
  label={'train': y_train, 'test': y_test}
)