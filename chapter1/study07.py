if __name__ == '__main__':

    #对数几率回归解决的是二分类的问题，对于多个选项的问题，我们可以使用softmax函数它是对数几率回归在N个可能不同的值上的推广
    #softmax要求每个样本必须属于某个类别，且所有可能的样本均被覆盖
    #softmax个样本分量之和为 1   当只有两个类别时，与对数几率回归完全相同
    #在tf.keras里，对于多分类问题我们使用
    #categorical_crossentropy（label进行独热编码的时候）
    #sparse_categorical_crossentropy（使用数字编码0-9分别代表各类东西）
    #两种loss函数来计算softmax交叉熵
    #softmax解决多分类问题

    #该页面对categorical_crossentropy（label进行独热编码的时候）的loss函数进行应用
    #独热编码 是利用0和1表示一些参数，使用N位状态寄存器来对N个状态进行编码。

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    #数据集存放路径c:/用户/dell/.keras/datasets/的里面，否则回去外国网站下载太慢了
    (train_image,train_label),(test_image,test_label) = tf.keras.datasets.fashion_mnist.load_data()
    #数据集有70000张28*28的图片，训练集有60000张，测试集有10000张

    train_image = train_image / 255
    test_image = test_image / 255

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
    #optimizer的参数可以自己设置学习速率
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['acc'])
    train_label_onehot = tf.keras.utils.to_categorical(train_label)
    test_label_onehot = tf.keras.utils.to_categorical(test_label)
    # print(train_label_onehot[0])
    # print(test_label_onehot)

    #validation_data设置为测试的时候，在后台训练也一并打印出测试的数据集
    model.fit(train_image,train_label_onehot,epochs=10,validation_data=(test_image,test_label_onehot))
    model.evaluate(test_image,test_label_onehot)
    predict = model.predict(test_image)
    print(np.argmax(predict[0]))


