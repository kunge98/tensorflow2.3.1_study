if __name__ == '__main__':

    # 第8章
    # 常见的预训练网络
    # 以Xception为例

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import numpy as np
    import glob
    import os

    # 1.加载路径
    train_img_path = glob.glob('..//data_set//dc_2000//train//*//*.jpg')
    test_img_path = glob.glob('..//data_set//dc_2000//test//*//*.jpg')
    print(len(train_img_path))      #2000
    print(len(test_img_path))       #1000
    # print(train_img_path[0])

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

    # 3.加载预处理图片
    def load_process_img(path,label):
        # (1)读取路径
        img = tf.io.read_file(path)
        # （2）解码,默认的channels为0，黑白的。
        img = tf.image.decode_jpeg(img,channels=3)
        # （3）改变图片的大小
        img = tf.image.resize(img,[256,256])
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

    # for img,label in train_img_dataset.take(1):
    #     plt.imshow(img)
    #     plt.show()

    # 6.将数据打乱
    batch_size = 32
    train_count = len(train_img_dataset)
    test_count = len(test_img_dataset)
    # 长度是25000
    train_img_dataset = train_img_dataset.shuffle(train_count).repeat().batch(32)
    train_img_dataset = train_img_dataset.prefetch(autotune)

    test_img_dataset = test_img_dataset.repeat().batch(32)
    # test_img_dataset = test_img_dataset.prefetch(autotune)


    # 7.使用keras内置的经典网络实现

    # weight='imagenet' 用imagenet上预训练好的这些权重，如果设置为none则就仅使用VGG——net，而不使用权重
    # include_top=False是否包含全连接层即输出层，false只是用卷积基
    # input_shape是输入的形状，先后顺序必须是高度、宽度、通道（厚度）
    # pooling='avg'相当于添加了一个GlobalAveragePooling2D
    model = tf.keras.Sequential()
    conv_base = tf.keras.applications.xception.Xception(weights='imagenet',
                                                        include_top=False,
                                                        input_shape=(256,256,3),
                                                        pooling='avg')

    # 将conv_base中的参数设置为不可训练
    conv_base.trainable = False

    # 微调

    # 输出打印conv_base的层数
    # print(len(conv_base.layers))
    fine_tune_at = -33
    # 遍历conv_base的层，让除了后三层之外的层数都不可训练
    for layer in conv_base.layers[:fine_tune_at]:
        layer.trainable = False

    model.add(conv_base)
    # 作用和flatten作用相似，但是更高效
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(512,activation='relu'))
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

    conv_base.summary()

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

    # 设置一个微调的训练次数和原先的训练次数
    initial_epochs = 10
    fine_tune_epochs = 10
    total_epochs = initial_epochs + fine_tune_epochs

    his = model.fit(train_img_dataset,
                    # 设置总的训练次数
                    epochs=total_epochs,
                    # 设置微调之前的训练次数
                    initial_epochs=initial_epochs,
                    steps_per_epoch=train_count//32,
                    validation_data=test_img_dataset,
                    validation_steps=test_count//32)

    plt.plot(his.epoch,his.history.get('acc'),label='acc')
    plt.plot(his.epoch,his.history.get('val_acc'),label='val_acc')
    # plt.plot(his.epoch,his.history.get('loss'),label='loss')
    plt.legend()
    plt.show()









