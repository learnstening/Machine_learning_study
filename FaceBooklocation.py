from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler


def knncls():
    """
    K近邻算法预测入住位置类别
    :return:
    """
    # 一、处理数据以及特征工程
    # 1、读取收，缩小数据的范围
    data = pd.read_csv("./FBlocation/train.csv")

    # 数据逻辑筛选操作 df.query()
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")

    # 删除time这一列特征
    data = data.drop(['time'], axis=1)

    print("第一次输出：\n", data)

    # 删除入住次数少于三次位置
    place_count = data.groupby('place_id').count()

    tf = place_count[place_count.row_id > 3].reset_index()

    data = data[data['place_id'].isin(tf.place_id)]

    # 3、取出特征值和目标值
    y = data['place_id']
    # y = data[['place_id']]

    x = data.drop(['place_id', 'row_id'], axis=1)

    # 4、数据分割与特征工程?

    # （1）、数据分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # (2)、标准化
    std = StandardScaler()

    # 对训练集进行标准化操作
    x_train = std.fit_transform(x_train)
    print(x_train)

    # 进行测试集的标准化操作
    x_test = std.fit_transform(x_test)

    # 二、算法的输入训练预测
    # K值：算法传入参数不定的值    理论上：k = 根号(样本数)
    # K值：后面会使用参数调优方法，去轮流试出最好的参数[1,3,5,10,20,100,200]
    knn = KNeighborsClassifier(n_neighbors=1)

    # 调用fit()
    knn.fit(x_train, y_train)

    # 预测测试数据集，得出准确率
    y_predict = knn.predict(x_test)

    print("预测测试集类别：", y_predict)

    print("准确率为：", knn.score(x_test, y_test))

    return None


if __name__ == "__main__":
    # knn_iris()
    knncls()
