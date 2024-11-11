if __name__ == '__main__':

    # 第七章

    # 猫狗综合实例 和图片增强
    # 实质为二分类问题

    import tensorflow as tf

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)


    import matplotlib.pyplot as plt
    import numpy as np
    import glob
    import os

    # 1.加载路径
    train_img_path = glob.glob('..//data_set//dc//train//*//*.jpg')
    test_img_path = glob.glob('..//data_set//dc//test//*.jpg')
    print(len(train_img_path))      #250000
    print(len(test_img_path))       #125000
    print(train_img_path)

    # 2.设置train_img_label
    #输出的目录的形式，..// data_set // dc // train\cat\cat.0.jpg
    # 利用split（‘//’）进行分割获取图片的名字
    # 假设 p = '..// data_set // dc // train\cat\cat.0.jpg'
    # print(p.split('\\'))
    # 分割后的形式['..// data_set // dc // train', 'cat', 'cat.0.jpg']，数组里面有三个元素，取出第二个元素就是
    # 设置猫为1，狗为0

    # 写法一：train_img_label = [int(p.split('\\')[1] == 'cat') for p in train_img_path]
    train_img_label = [int(p.split('\\')[1] == 'cat') for p in train_img_path]
    test_img_label = [int(p.split('\\')[1] == 'cat') for p in test_img_path]
    # 写法二：
    # for i in train_img_path:
    #     train_img_label = int(i.split('\\')[1] == 'cat')
        # print(train_img_label)
    # print(train_img_label)

    # 3.加载预处理图片
    def load_process_img(path,label):
        # (1)读取路径
        img = tf.io.read_file(path)
        # （2）解码,默认的channels为0，黑白的。
        img = tf.image.decode_jpeg(img,channels=3)
        # （3）改变图片的大小
        img = tf.image.resize(img,[360,360])

        # 对图片进行随机裁剪，可应用于图片增强
        img = tf.image.random_crop(img,[256,256,3])
        # 对图片进行随机的左右翻转
        img = tf.image.random_flip_left_right(img)
        # 对图片进行随机的左右翻转
        img = tf.image.random_flip_up_down(img)
        # 随机改变图片的亮度
        img = tf.image.random_brightness(img,0.5)
        # 随机改变图片的对比度
        img = tf.image.random_contrast(img,0,1)

        # (4)转换类型
        # 注意：tf.image.convert_image_dtype方法在原数据为非float类型的情况下会顺便把归一化处理
        img = tf.cast(img,tf.float32)
        # (5)图片归一化
        img = img / 255
        # (6)对label做一个处理 [1,2,3] ---->  [[1],[2],[3]]
        label = tf.reshape(label,[1])
        return img,label

    #  4.创建dataset
    train_img_dataset = tf.data.Dataset.from_tensor_slices((train_img_path,train_img_label))
    test_img_dataset = tf.data.Dataset.from_tensor_slices((test_img_path,test_img_label))

    # 5.运用图片处理函数
    # 根据cpu自动处理并行运算
    autotune = tf.data.experimental.AUTOTUNE
    train_img_dataset = train_img_dataset.map(load_process_img,num_parallel_calls=autotune)
    test_img_dataset = test_img_dataset.map(load_process_img,num_parallel_calls=autotune)
    # print(train_img_dataset)

    for img,label in train_img_dataset.take(1):
        plt.imshow(img)
        plt.show()

    # 6.将数据打乱
    batch_size = 32
    train_count = len(train_img_dataset)
    # 长度是25000
    train_img_dataset = train_img_dataset.shuffle(25000).batch(32)
    train_img_dataset = train_img_dataset.prefetch(autotune)

    test_img_dataset = test_img_dataset.batch(32)
    test_img_dataset = test_img_dataset.prefetch(autotune)

    # 7.转换成生成器并取出一个批次的数据
    imgs,labels = next(iter(train_img_dataset))
    # print(imgs.shape)
    # (32, 256, 256, 3)
    # print(labels.shape)
    # （32,1）

    # plt.imshow(imgs[6])
    # plt.show()

    # test_img_path = glob.glob('./data_set/dc/test/*/*.jpg')
    # print(test_img_path)


    # 6.搭建神经网络

    # VGG 网络模型
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64,(3,3),input_shape=(256,256,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(256,(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(256,(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(256,(1,1),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(512,(1,1),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(512,(1,1),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(4096,activation='relu'))
    model.add(tf.keras.layers.Dense(4096,activation='relu'))
    model.add(tf.keras.layers.Dense(4096,activation='relu'))
    model.add(tf.keras.layers.Dense(1000,activation='softmax'))

    #输出的summary中的none代表着此层的batch，代表批次
    # model.summary()

    # 7.进行预测
    # 创建好了模型可以直接进行模型的预测
    pred = model(imgs)

    #输出结果为（32,1）
    # print(pred.shape)
    # 将预测结果转换成和label的类型一样

    #预测结果,将pred中大于0的数据转换类型为int32
    p1 = np.array([p[0].numpy() for p in  tf.cast(pred > 0,tf.int32)])
    #实际的结果
    l1 = np.array([l[0].numpy() for l in labels])
    #两者一对比发现没有训练的前提下，准确率根本无从谈起
    # print(p1,l1)

    # 7.定义loss函数和优化函数
    # 损失函数为二分类问题
    optimizer = tf.keras.optimizers.Adam()
    # BinaryCrossentropy如果是小写的函数，可以直接传入参数进行损失函数的计算
    loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # 8.metrics计算模块
    train_loss = tf.keras.metrics.Mean('train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy('train_accuracy')

    test_loss = tf.keras.metrics.Mean('test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy('test_accuracy')


    train_loss_result = []
    train_accuracy_result = []

    test_loss_result = []
    test_accuracy_result = []

    def loss(model,train_data,label_data):
        pre_data_label = model(train_data)
        return loss_fun(label_data,pre_data_label)


    def train_one_step(model,img,label):
        with tf.GradientTape() as t:
            pred = model(img)
            loss_step = loss_fun(label,pred)
        grad = t.gradient(loss_step,model.trainable_variables)
        optimizer.apply_gradients(zip(grad,model.trainable_variables))
        train_loss(loss_step)
        train_accuracy(label,tf.cast(pred>0,tf.int32))


    def test_one_step(model,img,label):
        # training=False设置为非训练状态
        pred = model(img,training=False)
        loss_step = loss_fun(label,pred)
        test_loss(loss_step)
        test_accuracy(label,tf.cast(pred>0,tf.int32))

    def train():
        for epoch in range(50):
            for img_,label_ in train_img_dataset:
                train_one_step(model,img_,label_)
                print('.',end='')
            print()
            # 将train_loss和train_accuracy的结果分别放入列表中
            train_loss_result.append(train_loss.result())
            train_accuracy_result.append(train_accuracy.result())

            print('epoch: {} ， loss : {:.3f} , accuracy : {:.3f}'.format(epoch+1,
                                                                       train_loss.result(),
                                                                       train_accuracy.result()))

            for img_,label_ in train_img_dataset:
                test_one_step(model,img_,label_)
                print('.',end='')
            print()
            # 将test_loss和test_accuracy的结果分别放入列表中
            test_loss_result.append(test_loss.result())
            test_accuracy_result.append(test_accuracy.result())

            print('epoch: {} ， loss : {:.3f} , accuracy : {:.3f}'.format(epoch+1,
                                                                       test_loss.result(),
                                                                       test_accuracy.result()))
            # 重置
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()

    train()








