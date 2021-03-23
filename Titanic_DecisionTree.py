import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def decisioncls():
    """
    决策树进行乘客生存预测,后面改为随机森林
    :return:
    """
    # 1、获取数据
    titan = pd.read_csv("./titanic/titanic.csv")

    print(titan)

    # 2、数据的处理
    x = titan[['pclass', 'age', 'sex']]

    y = titan['survived']

    # print(x , y)
    # 缺失值需要处理，将特征当中有类别的这些特征进行字典特征抽取
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 对于x转换成字典数据x.to_dict(orient="records")
    # [{"pclass": "1st", "age": 29.00, "sex": "female"}, {}]

    dict = DictVectorizer(sparse=False)

    x = dict.fit_transform(x.to_dict(orient="records"))

    print(dict.get_feature_names())
    print(x)

    # 分割训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 进行决策树的建立和预测
    dc = DecisionTreeClassifier(max_depth=4)

    dc.fit(x_train, y_train)

    # # 随机森林去进行预测
    # rf = RandomForestClassifier()
    #
    # param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}

    # # 超参数调优
    # gc = GridSearchCV(rf, param_grid=param, cv=2)

    dc.fit(x_train, y_train)

    print("随机森林预测的准确率为：", dc.score(x_test, y_test))

    y_predict = dc.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接对比真实值和预测值：\n", y_test == y_predict)

    print("预测的准确率为：", dc.score(x_test, y_test))

    return None


if __name__ == "__main__":
    decisioncls()
