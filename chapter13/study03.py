if __name__ == '__main__':

    # 创建输入管道
    # 应用到tf.data模块

    # 取出一张或两张图片显示

    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from lxml import etree
    import glob

    # 获取所有的图片
    imgs = glob.glob('../data_set/oxford-iiit-pet/images/*.jpg')

    # 获取所有的xml
    xmls = glob.glob('../data_set/oxford-iiit-pet/annotations/xmls/*.xml')

    # 取出xmls中的名字
    names = [x.split('\\')[-1].split('.xml')[0] for x in xmls]

    # 找出images中对应的xmls的名字作为train训练数据
    img_train = [img for img in imgs if(img.split('\\')[-1].split('.jpg')[0] in names)]
    img_test = [img for img in imgs if(img.split('\\')[-1].split('.jpg')[0] not in names)]

    # 排序
    img_train.sort(key=lambda x:x.split('\\')[-1].split('.jpg')[0])
    xmls.sort(key=lambda x:x.split('\\')[-1].split('.xml')[0])

    # 处理标签
    def to_labels(path):
        xml = open('{}'.format(path)).read()
        select = etree.HTML(xml)
        width = int(select.xpath('//size/width/text()')[0])
        height = int(select.xpath('//size/height/text()')[0])
        xmin = int(select.xpath('//bndbox/xmin/text()')[0])
        xmax = int(select.xpath('//bndbox/xmax/text()')[0])
        ymin = int(select.xpath('//bndbox/ymin/text()')[0])
        ymax = int(select.xpath('//bndbox/ymax/text()')[0])
        return [xmin/width,ymin/height,xmax/width,ymax/height]

    # 处理label# 使用to_labels处理所有的图片，得到标签
    labels = [to_labels(path) for path in xmls]

    # 将相同的目标值放在一起，使用zip的反向操作
    output1,output2,output3,output4 = list(zip(*labels))

    # tensor类型，转化为numpy类型
    output1 = np.array(output1)
    output2 = np.array(output2)
    output3 = np.array(output3)
    output4 = np.array(output4)

    print('output1',output1.shape)
    print(len(output1))
    # print(len(labels))

    # 创建label_dataset
    label_dataset = tf.data.Dataset.from_tensor_slices((output1,output2,output3,output4))

    # print(label_dataset)

    # 处理图片
    # @tf.function使用图运算模式，加快图的构建
    @tf.function
    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img,channels=3)
        # img = tf.image.decode_and_crop_jpeg(crop_window=img,contents=img,channels=3,ratio=4)
        img = tf.image.resize(img,[224,224])
        img = tf.cast(img, tf.float64)
        img = img / 255
        # 归一化到-1到1之间
        img = img * img - 1
        return img

    image_dataset = tf.data.Dataset.from_tensor_slices(img_train)
    autotune = tf.data.experimental.AUTOTUNE
    image_dataset = image_dataset.map(load_image,num_parallel_calls=autotune)

    print(image_dataset)

    # 得到最终训练的dataset
    print('Hello World')
    dataset = tf.data.Dataset.zip((image_dataset,label_dataset))

    print(dataset)

    dataset = dataset.shuffle(len(img_train)).batch(32).repeat()

    # for img,label in dataset.take(1):
    #     plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
    #     out1, out2, out3, out4 = label
    #     xmin, ymin, xmax, ymax = out1[0].numpy()*224,out2[0].numpy()*224,out3[0].numpy()*224,out4[0].numpy()*224
    #     rect = Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), fill=False, color='red')
    #     ax = plt.gca()
    #     ax.axes.add_patch(rect)
    #     plt.show()
    #
    #
    # test_count = int(len(imgs)*0.2)
    # train_count = len(imgs) - test_count
    # dataset_train = dataset.skip(test_count)
    # dataset_test = dataset.take(test_count)
    #
    # batch_size = 32
    # buffer_size = 256
    # train_step = train_count / batch_size
    # test_step = test_count / batch_size
    #
    # train_dataset = dataset_train.shuffle(buffer_size=buffer_size).batch(batch_size=batch_size).repeat()
    # train_dataset = train_dataset.prefetch(autotune)
    # test_dataset = dataset_test.batch(batch_size)
    #
    # xception = tf.keras.applications.Xception(weights='imagenet',
    #                                           include_top=False,
    #                                           input_shape=(224,224,3))
    #
    # inputs = tf.keras.layers.Input(shape=(224,224,3))
    #
    # x1 = xception(inputs)
    #
    # # 变为一个2维向量
    # x2 = tf.keras.layers.GlobalAveragePooling2D()(x1)
    #
    # print('x2形状为',x2.shape)
    #
    # x3 = tf.keras.layers.Dense(2048,activation='relu')(x2)
    # print('x3形状为',x3.shape)
    #
    # x4 = tf.keras.layers.Dense(256,activation='relu')(x3)
    # print('x4形状为',x3.shape)
    #
    # # outputs = tf.keras.layers.Dense(4,activation='softmax',name='输出')
    # output1 = tf.keras.layers.Dense(1)(x4)
    # output2 = tf.keras.layers.Dense(1)(x4)
    # output3 = tf.keras.layers.Dense(1)(x4)
    # output4 = tf.keras.layers.Dense(1)(x4)
    #
    # # print(output1.shape)
    # # print(output2.shape)
    #
    # prediction = [output1,output2,output3,output4]
    #
    # # 模型已完成创建
    # model = tf.keras.Model(inputs=inputs,outputs=prediction)
    #
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    #               loss='mse',
    #               metrics=['mae'])
    #
    # model.summary()
    #
    # STEP_PER_EPOCH = train_count//batch_size
    # VALIDATION_STEPS = test_count//batch_size
    #
    # history = model.fit(dataset,
    #                     epochs=2,
    #                     steps_per_epoch=STEP_PER_EPOCH,
    #                     validation_data=test_dataset,
    #                     validation_steps=VALIDATION_STEPS)
    #
    # plt.plot(range(2),history.history.get('acc'),'r',label='Training Loss')
    # plt.plot(range(2),history.history.get('val_acc'),'bo',label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss Value')
    # # 设置取值范围
    # # plt.ylim([0,1])
    # plt.legend()
    # plt.show()
    #
    #
    # 
