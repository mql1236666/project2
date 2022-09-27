import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 20)
# 导入数据
data = pd.read_csv('E:/金融风控模型/评分卡/信用卡评分原始数据/cs-training.csv', index_col=0)
# 1.查看数据维度、类型、统计值,了解特征含义
print(data.info())
print(data.describe())
"""
变量名	                                           描述	                         类型	
SeriousDlqin2yrs	                               好坏客户，0表示好客户，1表示坏客户   标签
RevolvingUtilizationOfUnsecuredLines	           贷款以及信用卡可用额度与总额度比例	 数值型	
age	                                               借款人当时的年龄		         数值型
NumberOfTime30-59DaysPastDueNotWorse	           35-59天逾期次数		         数值型
DebtRatio	                                       负债比率	                     数值型	
MonthlyIncome	                                   月收入	                     数值型	
NumberOfOpenCreditLinesAndLoans	                   开放式信贷和贷款数量              数值型	
NumberOfTimes90DaysLate	                           借款者有90天或更高逾期的次数	     数值型
NumberRealEstateLoansOrLines	                   不动产贷款数量              	 数值型	
NumberOfTime60-89DaysPastDueNotWorse               60-89天逾期还次数	             数值型	
NumberOfDependents	                               家属数量                        数值型
"""

# 2.查看缺失值异常值
# 2.1查看缺失值
missing = data.isnull().sum()/len(data)  # 缺失率
miss = missing[missing > 0]
miss.sort_values(inplace=True)  # 将大于0的拿出来并排序
label = miss.index
plt.xticks(range(len(miss)), label)
plt.bar(range(len(miss)), miss)  # 缺失率柱状图可视化
plt.show()
"""
结论：
'MonthlyIncome'缺失值比例接近20%，对模型效果影响较大，考虑单独归为一类, 'NumberOfDependents'缺失值比例不超过5%，对模型效果影响不大，
考虑均值填充，
"""

# 2.2箱型图查看异常值
feature = list(data.columns)
label = 'SeriousDlqin2yrs'
feature.remove(label)


# 初步划分连续变量和离散变量
def get_numerical_serial_features(df, feas):
    numerical_serial_feature=[]
    numerical_noserial_feature=[]
    for fea in feas:
        temp = df[fea].nunique()
        if temp <= 10:
            numerical_noserial_feature.append(fea)
        else:
            numerical_serial_feature.append(fea)
    return numerical_serial_feature, numerical_noserial_feature


numerical_serial_feature, numerical_noserial_feature = get_numerical_serial_features(data, feature)
print('连续变量：', numerical_serial_feature)
print('离散变量：', numerical_noserial_feature)

f1 = pd.melt(data, value_vars=feature)  # 行列转化
g1 = sns.FacetGrid(f1, col="variable", col_wrap=4, sharex=False, sharey=False)  # 生成多个图
g1 = g1.map(sns.boxplot, 'value')
plt.show()
"""
结论：
年龄不可能小于18岁，确定错误的异常值考虑删除
"""


# 3.查看特征分布
# 3.1连续特征分布可视化
f2 = pd.melt(data, value_vars=numerical_serial_feature)  # 行列转化
g2 = sns.FacetGrid(f2, col="variable",  col_wrap=4, sharex=False, sharey=False)  # 生成多个图
g2 = g2.map(sns.distplot, "value", kde_kws={'bw': 1})
plt.show()


# 4.共线性分析
plt.figure(figsize=(16, 16))
data_corr = data.corr()
data_corr[data_corr <= 0.8] = 0.01   # 相关系数小于等于0.8的全部赋值为0.01，便于图片显示
sns.heatmap(data_corr)
plt.show()
"""
结论：
存在相关性大于0.8的特征，线性模型需要考虑去除共线性
"""