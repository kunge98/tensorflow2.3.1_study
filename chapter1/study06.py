if __name__ == '__main__':

    #对数几率回归解决的是二分类的问题，对于多个选项的问题，我们可以使用softmax函数它是对数几率回归在N个可能不同的值上的推广
    #softmax要求每个样本必须属于某个类别，且所有可能的样本均被覆盖
    #softmax个样本分量之和为 1   当只有两个类别时，与对数几率回归完全相同
    #在tf.keras里，对于多分类问题我们使用
    #categorical_crossentropy（label进行独热编码的时候）
    #sparse_categorical_crossentropy（使用数字编码0-9分别代表各类东西）
    #两种loss函数来计算softmax交叉熵
    #softmax解决多分类问题
    #该页面对sparse_categorical_crossentropy（使用数字编码0-9分别代表各类东西）的loss函数进行应用

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    #数据集存放路径c:/用户/dell/.keras/datasets/的里面，否则回去外国网站下载太慢了
    (train_image,train_label),(test_image,test_label) = tf.keras.datasets.fashion_mnist.load_data()
    #数据集有70000张28*28的图片，训练集有60000张，测试集有10000张

    # print(train_image.shape,train_label.shape)
    # print(test_image.shape,test_label.shape)
    # print(train_image[0])

    #取值范围0-255
    # print(np.max(train_image[0]),np.min(train_image[0]))

    #label的取值范围0-9，用数字表示分类
    # print(np.max(train_label),np.min(train_label))

    train_image = train_image /255
    test_image = test_image / 255

    # plt.imshow(train_image[1])
    #plt.imshow()之后不显示，然后在输入show方法即可
    # plt.show()

    model = tf.keras.Sequential()
    #图片都是二维形式存放的首相用Flatten转化为一维的向量，输入的是一个28*28的二维数组
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    #添加一个隐藏层,上面的flatten展平的时候已经给出了输入
    model.add(tf.keras.layers.Dense(256,activation='relu'))
    model.add(tf.keras.layers.Dense(256,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256,activation='relu'))

    #输出层，传入10个概率分布数加起来为1
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    # model.summary()
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
    #训练模型
    model.fit(train_image,train_label,epochs=50)
    #用测试集来评估模型
    #算算loss和acc
    model.evaluate(test_image,test_label)
