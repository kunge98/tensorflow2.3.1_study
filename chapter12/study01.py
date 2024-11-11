if __name__ == '__main__':

    # RNN 循环网络
    # 航空公司评论数据实例
    # 本案例包含三种评价（积极、消极、中肯），但是只用到了积极和消极作为一个二分类问题处理
    # 查看到数量发现消极的数量很多，要保证两种评论数量相同，所以取部分的消极评论

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import re

    # 读取文件
    data = pd.read_csv('..//data_set//Tweets.csv')
    # print(data.head(1))
    # print(data.info())
    # 取出数据的两列
    data = data[['airline_sentiment','text']]
    print(data.head(2))

    # 查看评论的种类
    print(data.airline_sentiment.unique())
    # 查看评论对应的数量
    print(data.airline_sentiment.value_counts())

    # 取出积极和消极的数据
    data_positive = data[data.airline_sentiment == 'positive']
    data_negative = data[data.airline_sentiment == 'negative']

    # 取出消极数据，数量和积极的数据相等
    data_negative = data_negative.iloc[:len(data_positive)]

    # 两者的长度相同 2363
    print(len(data_positive),len(data_negative))
    #
    # 将两种数据合并在一起
    data = pd.concat([data_positive,data_negative])
    # 4726
    print(len(data))

    # 混合，输出data数据
    print(data.sample(len(data)))

    # 将positive变为1和negative变为0,作为一个新的列review
    data['review'] = (data.airline_sentiment == 'positive').astype(int)

    # 将原来的airline_sentiment删除，已经没有作用了
    del data['airline_sentiment']

    print(data)

    # 处理文本

    # 正则表达式，转换成一个单词一个单词的样式
    token = re.compile('[A-Za-z]+|[!?,.()]')

    # 转换为小写字母函数
    def regular_text(text):
        # 读取所有的文本内容
        new_text = token.findall(text)
        # 转换为小写字母
        new_text = [i.lower() for i in new_text]
        return new_text

    # 输出得到处理好的单词
    data['text'] = data.text.map(regular_text)
    print(data)


    # 为每个单词赋值一个索引

    # 定义一个字母集合
    word_set = set()

    for text in data.text:
        for word in text:
            # 把遍历到的单词全部加进word_set中
            word_set.add(word)
    # print(len(word_set))
    # print(word_set)

    max_word_counts = len(word_set) + 1
    # 长度7100
    print(max_word_counts)

    # 将set集合转换成list,因为set没有index的属性
    word_list = list(word_set)

    # print(word_list)
    # print(word_list.index('mtff'))

    # index加1因为索引不能从0开始
    word_index = dict((word,word_list.index(word) + 1) for word in word_list)

    print(word_index)

    # 得到处理好的数据
    # 把每一行文本变成了一个整数索引列表
    data_ok = data.text.apply(lambda x: [word_index.get(word,0) for word in x])

    print(data_ok)
    # print(data_ok.values)

    # 打印数据的长度
    # 第一条评论
    # print(len(data_ok.iloc[0]))
    # 第二条评论
    # print(len(data_ok.iloc[1]))

    # # 求出最大的评论长度
    max_comments_len = max(len(x) for x in data_ok)
    # # 40
    # # print(max_comments_len)
    #
    # # 对不同的评论长度进行填充
    # # data_ok.values
    # # maxlen最大的填充长度
    # data_ok = tf.keras.preprocessing.sequence.pad_sequences(data_ok.values,maxlen=max_comments_len)
    # # (4726, 40)
    # print(data_ok.shape)
    #
    model = tf.keras.Sequential()
    # 50为映射成的密集向量的长度
    model.add(tf.keras.layers.Embedding(max_word_counts,50,input_length=max_comments_len))

    model.add(tf.keras.layers.LSTM(64))

    # 二分类激活函数
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),loss='binary_crossentropy',metrics=['acc'])

    his = model.fit(data_ok,
                  # 0和1
                  data.review.values,
                  epochs=30,
                  # 添加的测试数据占总比例的0.2
                  validation_split=0.2)

    #绘出train和test对于acc准确率的折线图
    plt.plot(his.epoch,his.history.get('acc'),label='acc')
    plt.plot(his.epoch,his.history.get('val_acc'),label='val_acc')

    #绘出train和test对于loss函数折线图
    # plt.plot(his.epoch,his.history.get('loss'),label='loss')
    # plt.plot(his.epoch,his.history.get('val_loss'),label='val_loss')

    plt.show()
    #
    #
    #
    #


