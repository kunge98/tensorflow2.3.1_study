if __name__ == '__main__':

    # 自定义训练,非常关键

    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    #1.加载数据集
    (train_img, train_label), (test_img, test_label) = tf.keras.datasets.mnist.load_data()
    # print(train_img.shape, train_label.shape)

    #对数据进行预处理

    # 2.使用卷积神经网络必须要有厚度，所以将维度进行扩充
    # 扩充数据的维度
    train_img = tf.expand_dims(train_img, -1)
    test_img = tf.expand_dims(test_img, -1)
    # 数据的大小 60000,28,28
    # print(train_img.shape)

    # 3.转换数据类型（求微分要用到float类型），归一化
    train_img = tf.cast(train_img / 255, tf.float32)
    train_label = tf.cast(train_label,tf.int64)
    test_img = tf.cast(test_img / 255, tf.float32)
    test_label = tf.cast(test_label,tf.int64)

    # 4.加载训练和测试数据集
    # 里面是一个元组
    dataset_train = tf.data.Dataset.from_tensor_slices((train_img,train_label))
    dataset_test = tf.data.Dataset.from_tensor_slices((test_img,test_label))
    # print(dataset)

    # 5.对训练和测试数据集打乱
    dataset_train = dataset_train.shuffle(buffer_size=10000).repeat().batch(batch_size=32)
    dataset_test = dataset_test.batch(batch_size=32)

    # 6.构建神经网络
    model = tf.keras.Sequential()
    # input_shape=(None,None,1)中的none表示任意形状
    model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),input_shape=(None,None,1),activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    # model.summary()
    # print(model.trainable_variables)



    # 7.设置优化器和损失函数
    optimizer = tf.keras.optimizers.Adam()
    # SparseCategoricalCrossentropy首字母大写的时候是一个可调用的对象
    loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss_fun = tf.keras.losses.sparse_categorical_crossentropy(from_logits=True)

    # 8.iter（）让对象变成一个生成器，next（）取出下一个数据出来

    # features,labels = next(iter(dataset_train))
    # print(features.shape)
    # print(labels.shape)

    # predictions = model(features)
    # print(predictions.shape)
    # print(tf.argmax(predictions,axis=0))
    # print(labels)

    # 9.应用计算汇总模块，具体见study04

    train_loss = tf.keras.metrics.Mean('train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

    # 10.写自定义函数
    # 定义一个loss函数，x是train数据，y是label
    # 当传入一个训练数据和它的label时就会返回一个它的损失值
    # 传入三个参数，model、当前的训练数据。当前数据的label
    # 调用model方法传入当前的训练数据，得到一个预测的label
    # 在调用loss_fun方法，传入当前数据的label和预测的label得到一个交叉熵损失
    def loss(model,data_train,data_label):
        # data_pre_label是预测的label
        pre_data_label = model(data_train)
        # 返回的是一个交叉熵损失
        return loss_fun(data_label,pre_data_label)

    # 建立一个batch的train的优化函数
    def train_one_step(model,img,label):
        # 建立一个上下文管理器，追踪训练过程
        with tf.GradientTape() as t:
            pre = model(img)
            # 跟踪并得到每一步的损失函数
            loss_step = loss_fun(label,pre)
        # model.trainable_variables 就是model的可训练参数
        # 计算损失函数对可训练参数的梯度
        grad = t.gradient(loss_step,model.trainable_variables)
        # 将grad,model.trainable_variables打包并用之前的optimizer优化器应用apply_gradients方法进行优化
        # 该步骤为最核心的地方
        optimizer.apply_gradients(zip(grad,model.trainable_variables))
        train_loss(loss_step)
        train_accuracy(label,pre)


    # 建立一个batch的test的优化
    def test_one_step(model,img,label):
            pre = model(img)
            # 跟踪并得到每一步的损失函数
            loss_step = loss_fun(label,pre)
            test_loss(loss_step)
            test_accuracy(label,pre)

    def train():
        # 训练10个epoch
        for epoch in range(10):
            # batch是由enumerate产生的序号，(img,label)由dataset迭代所得
            for (batch,(img,label)) in enumerate(dataset_train):
                train_one_step(model,img,label)
            print('epoch {} train_loss is {},train_accuracy is {}'.format(epoch,
                                                                          train_loss.result(),
                                                                          train_accuracy.result()))
            #test
            for (batch,(img,label)) in enumerate(dataset_test):
                test_one_step(model,img,label)
            print('epoch {} test_loss is {},test_accuracy is {}'.format(epoch,
                                                                        test_loss.result(),
                                                                        test_accuracy.result()))
            # 重置
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()

    print(1)
    print(train())
    print('确定是在执行了')








