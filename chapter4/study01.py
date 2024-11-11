import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':

    #tf.keras解决电影评论序列问题，数据集中把评论的每一个单词转换为一个索引
    #对于电影评价要么是好的，要么是坏的，（不考虑中立的评价）属于二分类问题

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    #读取数据
    data = tf.keras.datasets.imdb

    #对序列的预处理

    #设置单词数目不超过10000，超过则丢弃
    (x_train,y_train),(x_test,y_test) = data.load_data(num_words=10000)
    #输出数据的长度，长度有大有小，所以做一个数据的填充，小的填充，大的截断
    # print([len(i) for i in x_train])

    # print(x_train.shape,x_test.shape)
    # print(y_train.shape,y_test.shape)
    # print(x_train[0])

    #文本处理的最好方法：将文本训练成密集向量
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,300)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,300)
    #填充之后train和test每个长度都变为300
    # print([len(i) for i in x_train])
    # print([len(i) for i in x_test])


    # #建立模型
    #
    # model = tf.keras.models.Sequential()
    #
    # # 文本处理的最好方法：将文本训练成密集向量
    # #输入数据的维度10000，映射成长度为50的向量，输入的长度为300
    # #经过Embedding长度为300的评论会被映射成长度为50的向量
    # #维度变为25000*300*50
    # model.add(tf.keras.layers.Embedding(10000,50,input_length=300))
    # # print(x_train.shape)
    #
    # #将三维数据变为二维的,GlobalMaxPooling1D要比flatten方法更好
    # model.add(tf.keras.layers.GlobalMaxPooling1D())
    #
    # model.add(tf.keras.layers.Dense(64,activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # #二分类问题使用sigmoid激活函数，输出的数量为1
    # model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    model.summary()
    # optimizer=tf.keras.optimizers.Adam 优化函数也可以这么写
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

    his = model.fit(x_train,y_train,epochs=15,batch_size=256,validation_data=(x_test,y_test))
    #查看his的属性
    his.history.keys()
    # plt.plot(his.epoch,his.history.get('loss'),label='loss')
    # plt.plot(his.epoch,his.history.get('val_loss'),label='val-loss')
    plt.plot(his.epoch,his.history.get('acc'),label='acc')
    plt.plot(his.epoch,his.history.get('val_acc'),label='val-acc')
    plt.legend()
    plt.show()



