import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':

    #搭建一个卷积神经网络

    import numpy as np

    print(tf.test.is_gpu_available())

    (train_img,train_label),(test_img,test_label) = tf.keras.datasets.fashion_mnist.load_data()
    # print(train_img.shape,train_label.shape)
    # print(test_img.shape,test_label.shape)

    #对train_img的最后一维进行维度的扩充
    train_img = np.expand_dims(train_img,-1)
    #第一维度个数，第二维度高度，第三维度宽度，第四维度厚度
    print(train_img.shape)

    #对test_img的最后一维进行维度的扩充
    test_img = np.expand_dims(test_img,-1)
    #第一维度个数，第二维度高度，第三维度宽度，第四维度厚度
    print(test_img.shape)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(3,3),
                                     input_shape=train_img.shape[1:],
                                     activation='relu'))
    #默认的ksize为2*2的，就是卷积之后图像缩小一半
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
    #GlobalAvgPool2D全局平均池化
    model.add(tf.keras.layers.GlobalAvgPool2D())
    #输出一个维度为10的输出层
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    # model.summary()
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
    his = model.fit(train_img,train_label,epochs=30,validation_data=(test_img,test_label))
    #查看参数 loss、acc、val_loss、val_acc
    # his.history.keys()

    #绘出train和test对于acc准确率的折线图
    plt.plot(his.epoch,his.history.get('acc'),label='acc')
    plt.plot(his.epoch,his.history.get('val_acc'),label='val_acc')

    #绘出train和test对于loss函数折线图
    # plt.plot(his.epoch,his.history.get('loss'),label='loss')
    # plt.plot(his.epoch,his.history.get('val_loss'),label='val_loss')

    plt.show()

    # layers = tf.keras.layers.Conv2D()
    # layers = tf.keras.layers.AvgPool2D()


