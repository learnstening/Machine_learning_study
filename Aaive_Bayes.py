from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def nbcls():
    """
    朴素贝叶斯对新闻数据集进行预测
    :return:
    """
    # 获取新闻的数据，20个类别
    news = fetch_20newsgroups(subset='all')

    # 进行数据集分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.3)

    # 对于文本数据，进行特征抽取
    tf = TfidfVectorizer()

    x_train = tf.fit_transform(x_train)
    # 这里打印出来的列表是：训练集当中的所有不同词的组成的一个列表
    print(tf.get_feature_names())
    # print(x_train.toarray())

    # 不能调用fit_transform
    x_test = tf.transform(x_test)

    # estimator估计器流程
    mlb = MultinomialNB(alpha=1.0)

    mlb.fit(x_train, y_train)

    # 进行预测
    y_predict = mlb.predict(x_test)

    print("预测每篇文章的类别：", y_predict[:100])
    print("真实类别为：", y_test[:100])

    print("预测准确率为：", mlb.score(x_test, y_test))

    return None


if __name__ == "__main__":
    nbcls()