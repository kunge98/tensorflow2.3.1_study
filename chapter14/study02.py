if __name__ == '__main__':

    # 图像分割

    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import glob
    import os

    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


    # 读取所有的原图片
    images = glob.glob('../data_set/oxford-iiit-pet/images/*.jpg')
    print(len(images))
    print(images[3:8])

    # 读取标记分割图片
    pngs = glob.glob('../data_set/oxford-iiit-pet/annotations/trimaps/*.png')
    print(len(pngs))
    print(pngs[3:8])

    # 将两者进行相同的乱序
    np.random.seed(2021)
    index = np.random.permutation(len(images))
    images = np.array(images)[index]
    pngs = np.array(pngs)[index]

    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((images,pngs))

    # 划分训练集和测试集
    test_count = int(len(images)*0.2)
    train_count = len(images) - test_count
    print(test_count,train_count)

    data_train = dataset.skip(test_count)
    data_test = dataset.take(test_count)

    # 读取原图片
    def read_jpg(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img,channels=3)
        return img

    # 读取标记图片
    def read_png(path):
        png = tf.io.read_file(path)
        png = tf.image.decode_png(png,channels=1)
        return png

    # 对图片进行归一化处理
    def normalization_image(input_images,input_pngs):
        input_images = tf.cast(input_images,tf.float32)
        input_images = input_images / 127.5 - 1
        input_pngs -= 1
        return input_images,input_pngs

    # 读取文件
    @tf.function
    def load_images(input_images_path,input_pngs_path):
        input_image = read_jpg(input_images_path)
        input_pngs = read_png(input_pngs_path)
        input_image = tf.image.resize(input_image,(224,224))
        input_pngs = tf.image.resize(input_pngs,(224,224))
        return normalization_image(input_image,input_pngs)

    autotune = tf.data.experimental.AUTOTUNE
    data_train = data_train.map(load_images,num_parallel_calls=autotune)
    data_test =  data_test.map(load_images,num_parallel_calls=autotune)

    batch_size = 8
    buffer_size = 256
    data_train = data_train.shuffle(buffer_size).batch(batch_size).repeat()
    data_test = data_test.batch(batch_size)

    print(data_train)
    print(data_test)

    # for img,png in data_train.take(1):
    #     # subplot(行，列，第几个)
    #     plt.subplot(2,2,1)
    #     plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
    #     plt.subplot(2,2,2)
    #     plt.imshow(tf.keras.preprocessing.image.array_to_img(png[0]))
    #     plt.subplot(2,2,3)
    #     plt.imshow(tf.keras.preprocessing.image.array_to_img(img[6]))
    #     plt.subplot(2,2,4)
    #     plt.imshow(tf.keras.preprocessing.image.array_to_img(png[6]))
    #     plt.show()

    # 初始化预训练网络
    conv_base = tf.keras.applications.VGG16(weights='imagenet',
                                            input_shape=(224,224,3),
                                            include_top=False)

    print(conv_base.summary())

    # 最后输出形状为7,7,512，取上采样，得到倒数第二个的大小形状为14,14,512，即反卷积，重复操作
    # 获取某一层的名字,调用output方法输出该层的形状
    shape1= conv_base.get_layer('block5_conv3').output
    print(shape1)

    shape2= conv_base.get_layer('block1_conv1').output
    print(shape2)
    # （None，224,224,64）


    # 创建一个子model,目的为更好的获取模型中间的输出
    # 该输入就是预训练模型
    # sub_model = tf.keras.models.Model(inputs=conv_base.input,
    #                                   outputs=conv_base.get_layer('block5_conv3').output)
    # print(sub_model.summary())


    # 创建一个多输出模型
    multi_model = tf.keras.models.Model(inputs=conv_base.input,
                                        outputs=[conv_base.get_layer('block5_conv3').output,
                                                 conv_base.get_layer('block4_conv3').output,
                                                 conv_base.get_layer('block3_conv3').output,
                                                 conv_base.get_layer('block5_pool').output])
    # print('hello world')
    print(multi_model.summary())

    # 这是一个已经训练好的网络，所以不需要训练，直接使用它的权重即可
    multi_model.trainable = False

    # print(conv_base.input)
    # print(conv_base.output)
    # print(conv_base.layers)

    inputs = tf.keras.layers.Input(shape=(224,224,3))
    print('inputs的形状为：',inputs.shape)

    # 多输出模型，multi_model有四个输出分别赋值
    output_block5_conv3,output_block4_conv3,output_block3_conv3,output_block5_pool = multi_model(inputs)

    # # 重新命名一下子
    out1 = output_block5_pool
    out2 = output_block5_conv3
    out3 = output_block4_conv3
    out4 = output_block3_conv3

    print(out1.shape)
    # (None, 7, 7, 512)
    print(out2.shape)
    # (None, 14, 14, 512)
    print(out3.shape)
    # (None, 28, 28, 512)
    print(out4.shape)
    # (None, 56, 56, 256)
    #
    # 最后一层做上采样，并与倒二层相加
    # 反卷积
    x1 = tf.keras.layers.Conv2DTranspose(filters=512,
                                         kernel_size=(3,3),
                                         strides=2,
                                         padding='same',
                                         activation='relu')(out1)
    print(x1.shape)
    # (None, 14, 14, 512)
    #
    # # 对最后一层上采样完成之后，还可以对改成进行卷积操作，步长为1，不改变形状，加深对特征的提取
    x1 = tf.keras.layers.Conv2D(filters=512,
                                kernel_size=(3,3),
                                strides=1,
                                padding='same',
                                activation='relu')(x1)
    print(x1.shape)
    # (None, 14, 14, 512)

    # 将采完样的最后一层和倒二相加，此时的形状是相同的
    x2 = tf.add(x1,out2)
    print(x2.shape)

    # 此时的倒数第二层变为‘最后一层’，将改成做上采样操作
    x2 = tf.keras.layers.Conv2DTranspose(filters=512,
                                         kernel_size=(3,3),
                                         strides=2,
                                         padding='same',
                                         activation='relu')(x2)
    print(x2.shape)

    # 同样对改成进行stride为1的卷积操作，提取更多特征
    x2 = tf.keras.layers.Conv2D(filters=512,
                                kernel_size=(3,3),
                                strides=1,
                                padding='same',
                                activation='relu')(x2)
    print(x2.shape)

    # 此时倒二的形状和他上一层的形状一样，做相加操作
    x3 = tf.add(x2,out3)
    print(out3.shape)
    # (None, 28, 28, 512)

    x3 = tf.keras.layers.Conv2DTranspose(filters=256,
                                         kernel_size=(3,3),
                                         strides=2,
                                         padding='same',
                                         activation='relu')(x3)

    x3 = tf.keras.layers.Conv2D(filters=256,
                                kernel_size=(3,3),
                                strides=1,
                                padding='same',
                                activation='relu')(x3)


    x4 = tf.add(x3,out4)

    print(x4.shape)
    # (None, 56, 56, 256)

    x4 = tf.keras.layers.Conv2DTranspose(filters=128,
                                         kernel_size=(3,3),
                                         strides=2,
                                         padding='same',
                                         activation='relu')(x4)

    print(x4.shape)
    # (None, 112, 112, 128)

    x4 = tf.keras.layers.Conv2D(filters=128,
                                kernel_size=(3,3),
                                strides=1,
                                padding='same',
                                activation='relu')(x4)

    print(x4.shape)
    # (None, 112, 112, 128)

    # 对x4操作完成后形状大小为语气目标大小的一半，而且此项目为识别分类问题，
    # 从之前图片可以看出png图片识别的分类有三种：环境、轮廓、身体
    # 对x4进行最后一次反卷积
    prediction = tf.keras.layers.Conv2DTranspose(filters=3,
                                                 kernel_size=3,
                                                 strides=2,
                                                 padding='same',
                                                 activation='softmax')(x4)
    # 输出的形状最终达到了预期效果
    print(prediction.shape)
    # (None, 224, 224, 3)

    # 创建好了model
    model = tf.keras.models.Model(inputs=inputs,outputs=prediction)
    print(model.summary())
    print(model.input_shape)
    print(model.output_shape)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    STEP_PER_EPOCH = train_count//batch_size
    VALIDATION_STEPS = test_count//batch_size

    his = model.fit(data_train,
              epochs=5,
              steps_per_epoch=STEP_PER_EPOCH,
              validation_data=data_test,
              validation_steps=VALIDATION_STEPS)

    loss = his.history['loss']
    val_loss = his.history['val_loss']
    plt.figure()
    plt.plot(range(5),loss,'r',label='Training_loss')
    plt.plot(range(5),loss,'bo',label='Validation_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.show()
    model.save('img_segment.h5')