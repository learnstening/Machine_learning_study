import os
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


def knn_dna():
    column_knn = []
    for item in range(0, 180):
        s = 'A' + str(item)
        column_knn.append(s)
        # column_name.append('')
    column_knn.append('class')
    data = pd.read_csv("./HomeWork_DNA/dna.data", sep=' ', names=column_knn)
    print(data)

    x = data[column_knn[0:180]]
    y = data[column_knn[-1]]
    print(x)
    print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # 3.特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.KNN算法预估器
    estimator = KNeighborsClassifier()
    estimator.fit(x_train, y_train)  # 对训练集的特征值和目标值进行训练，训练完就有了模型

    # 5.模型评估
    # 法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接对比真实值和预测值：\n", y_test == y_predict)
    # 法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率：\n", score)

    return None


def id3_dna():
    # 随机森林预测的准确率为： 0.95

    # 1.数据预处理：①读取数据集②并给每一列命名③划分特征集和标签④划分训练集和测试集
    column_id3 = []
    for item in range(0, 180):
        s = 'A' + str(item)
        column_id3.append(s)
        # column_id3.append('')
    column_new = column_id3
    column_id3.append('class')
    data = pd.read_csv("./HomeWork_DNA/dna.data", sep=' ', names=column_id3)
    print(data)

    x = data[column_id3[0:180]]
    y = data[column_id3[-1]]
    print(x)
    print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # # 随机森林去进行预测
    # rf = RandomForestClassifier()
    #
    # param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    #
    # 超参数调优
    # gc = GridSearchCV(rf, param_grid=param, cv=2)

    # 进行决策树的建立和预测
    dc = DecisionTreeClassifier(max_depth=7, random_state=22)

    dc.fit(x_train, y_train)

    # print("决策树预测的准确率为：", dc.score(x_test, y_test))

    y_predict = dc.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接对比真实值和预测值：\n", y_test == y_predict)

    print("预测的准确率为：", dc.score(x_test, y_test))

    dot_data = export_graphviz(dc, out_file=None, feature_names=x_train.columns, class_names=['1', '2', '3'],
                               rounded=True, filled=True)  # rounded和字体有关，filled设置颜色填充
    # 将生成的dot_data内容导入到txt文件中
    f = open('dot_data.txt', 'w')
    f.write(dot_data)
    f.close()

    # 修改字体设置，避免中文乱码！
    # f_old = open('dot_data.txt', 'r')
    # f_new = open('dot_data_new.txt', 'w', encoding='utf-8')
    # for line in f_old:
    #     if 'fontname' in line:
    #         font_re = 'fontname=(.*?)]'
    #         old_font = re.findall(font_re, line)[0]
    #         line = line.replace(old_font, 'SimHei')
    #     f_new.write(line)
    # f_old.close()
    # f_new.close()

    # 以PNG的图片形式存储生成的可视化文件
    os.system('dot -Tpng dot_data.txt -o 决策树模型.png')
    print('决策树模型.png已经保存在代码所在文件夹！')
    # 以PDF的形式存储生成的可视化文件
    os.system('dot -Tpdf dot_data.txt -o 决策树模型.pdf')
    print('决策树模型.pdf已经保存在代码所在文件夹！')

    return None


if __name__ == "__main__":
    # knn_dna()
    id3_dna()
