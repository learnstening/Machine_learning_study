from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def knn_iris():
    # 1.获取数据
    iris = load_iris()

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=99)
    print("x_train:\n", x_train)
    print("y_train:\n", y_train)

    # 3.特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
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


def knn_iris_gscv():
    # 1.获取数据
    iris = load_iris()

    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=99)
    print("x_train:\n", x_train)
    print("y_train:\n", y_train)

    # 3.特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.KNN算法预估器
    estimator = KNeighborsClassifier()
    # 加入网格搜索与交叉验证
    # 参数准备
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
    estimator.fit(x_train, y_train)  # 对训练集的特征值和目标值进行训练，训练完就有了模型

    # 5.模型评估
    # 法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接对比真实值和预测值：\n", y_test == y_predict)
    # 法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率：\n", score)

    print("最佳参数：\n", estimator.best_params_)
    print("最佳结果：\n", estimator.best_score_)
    print("最佳估计器：\n", estimator.best_estimator_)
    print("交叉验证结果：\n", estimator.cv_results_)

    return None


if __name__ == "__main__":
    # knn_iris()
    knn_iris_gscv()
