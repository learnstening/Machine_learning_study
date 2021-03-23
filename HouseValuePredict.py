from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def linearregression1():
    """
    正规方程预测房子价格
    :return:
    """
    # 1.获取数据
    boston = load_boston()
    print("特征数量：\n", boston.data.shape)

    # 2.对数据集进行划分
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=24)

    # 3.需要做标准化处理,对于特征值处理
    transfer = StandardScaler()

    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.使用预估器
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 5.得出模型
    print("正规方程权重系数为：\n", estimator.coef_)
    print("正规方程偏置为：\n", estimator.intercept_)

    # 6.模型评估
    y_predict = estimator.predict(x_test)
    print("正规方程预测房价为：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程均方误差：\n", error)

    return None


def linearregression2():
    """
    梯度下降预测房子价格
    :return:
    """
    # 1.获取数据
    boston = load_boston()
    print("特征数量：\n", boston.data.shape)

    # 2.对数据集进行划分
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=24)

    # 3.需要做标准化处理,对于特征值处理
    transfer = StandardScaler()

    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.使用预估器
    estimator = SGDRegressor()
    estimator.fit(x_train, y_train)

    # 5.得出模型
    print("梯度下降权重系数为：\n", estimator.coef_)
    print("梯度下降偏置为：\n", estimator.intercept_)

    # 6.模型评估
    y_predict = estimator.predict(x_test)
    print("梯度下降预测房价为：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降均方误差：\n", error)

    return None


def linearregression3():
    """
    岭回归预测房子价格
    :return:
    """
    # 1.获取数据
    boston = load_boston()
    print("特征数量：\n", boston.data.shape)

    # 2.对数据集进行划分
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=24)

    # 3.需要做标准化处理,对于特征值处理
    transfer = StandardScaler()

    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.使用预估器
    estimator = Ridge()
    estimator.fit(x_train, y_train)

    # 5.得出模型
    print("岭回归权重系数为：\n", estimator.coef_)
    print("岭回归偏置为：\n", estimator.intercept_)

    # 6.模型评估
    y_predict = estimator.predict(x_test)
    print("岭回归预测房价为：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("岭回归均方误差：\n", error)

    return None


if __name__ == "__main__":
    linearregression1()
    linearregression2()
    linearregression3()
