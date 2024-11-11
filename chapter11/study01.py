if __name__ == '__main__':

    # 第十一章 保存完整的模型

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    (train_img,train_label),(test_img,test_label) = tf.keras.datasets.fashion_mnist.load_data()
    # print(train_img.shape,train_label.shape)
    # print(test_img.shape,test_label.shape)

    #对train_img的最后一维进行维度的扩充
    train_img = np.expand_dims(train_img,-1)
    #第一维度个数，第二维度高度，第三维度宽度，第四维度厚度
    # print(train_img.shape)

    #对test_img的最后一维进行维度的扩充
    test_img = np.expand_dims(test_img,-1)
    #第一维度个数，第二维度高度，第三维度宽度，第四维度厚度
    # print(test_img.shape)

    # 归一化
    train_img = train_img / 255
    test_img = test_img / 255

    check_point_path = 'training_save/cp.ckpt'

    # save_weights_only=True只保存权重
    cp_callback = tf.keras.callbacks.ModelCheckpoint(check_point_path,save_weights_only=True)


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    # model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
    # model.add(tf.keras.layers.MaxPooling2D())

    #输出一个维度为10的输出层
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    # model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.000001),loss='sparse_categorical_crossentropy',metrics=['acc'])

    his = model.fit(train_img,
                    train_label,
                    epochs=10,
                    validation_data=(test_img,test_label),
                    callbacks=[cp_callback])

    #查看参数 loss、acc、val_loss、val_acc
    # his.history.keys()

    #绘出train和test对于acc准确率的折线图
    plt.plot(his.epoch,his.history.get('acc'),label='acc')
    plt.plot(his.epoch,his.history.get('val_acc'),label='val_acc')

    #绘出train和test对于loss函数折线图
    # plt.plot(his.epoch,his.history.get('loss'),label='loss')
    # plt.plot(his.epoch,his.history.get('val_loss'),label='val_loss')

    # plt.show()

    model.evaluate(test_img,test_label)

    model.load_weights(check_point_path)



    # 评价预测
    # print(model.evaluate(test_img,test_label))


    # layers = tf.keras.layers.Conv2D()
    # layers = tf.keras.layers.AvgPool2D()


    # 1.保存整个模型
    # model.save('test.h5')
    #
    # 加载新的模型
    # new_model = tf.keras.models.load_model('test.h5')
    #
    # 输出和保存的model一模一样
    # new_model.summary()
    # new_model.evaluate(test_img,test_label)
    # new_model.fit()



    # 2.只保存模型的架构，无需保存权重值或者优化器

    # 返回一个json数据
    # json_config = model.to_json()
    # print(json_config)
    #
    # 重建模型
    # new_model = tf.keras.models.model_from_json(json_config)
    #
    # 查看网络架构
    # new_model.summary()
    # 权重随机的初始化，需要重新进行优化器的配置
    # reinitialized_model.commile()



    # 3.只保存模型的权重值或者优化器

    # 获取权重
    # weight = model.get_weights()
    # print(weight)
    #
    # new_model.setweights(weight)
    #
    # model.save_weights('test_weigths.h5')



    # 4.训练期间保存检查点

    # check_point_path = 'pro8_tensorflow2.3.1/training_save/cp.ckpt'
    #
    # # save_weights_only=True只保存权重
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(check_point_path,save_weights_only=True)



