if __name__ == '__main__':

    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from lxml import etree
    import glob

    # 获取所有的图片
    images = glob.glob('../data_set/oxford-iiit-pet/images/*.jpg')
    # print(len(images))
    # print(images[:5])
    # print(images[-5:])

    # 获取所有的xmls
    xmls = glob.glob('../data_set/oxford-iiit-pet/annotations/xmls/*.xml')
    # print(len(xmls))
    # print(xmls[-5:])
    # ../ data_set / oxford - iiit - pet / annotations / xmls\\yorkshire_terrier_187.xml
    # 输出的xmls文件的长度和images的长度不一样，所以先将xmls中所有的名字取出，用split方法

    # 获取xmls中的名字
    names = [x.split('\\')[-1].split('.xml')[0] for x in xmls]
    # print(names[:5])
    # print(len(names))

    # 取出images中的名字和xmls的方法一样,名字和names中相同时则作为训练数据，否则作为测试数据
    imgs_train = [img for img in images if (img.split('\\')[-1].split('.jpg')[0] in names)]
    # print(len(imgs_train))
    # print(imgs_train[:5])

    # 取出测试数据
    imgs_test = [img for img in images if (img.split('\\')[-1].split('.jpg')[0] not in names)]
    # print(len(imgs_test))

    # 让images和xmls一一对应，做一个排序
    imgs_train.sort(key=lambda x:x.split('\\')[-1].split('.jpg')[0])
    xmls.sort(key=lambda x:x.split('\\')[-1].split('.xml')[0])
    # print(imgs_train[5:10])
    # print(xmls[5:10])

    # 数据读取预处理
    # 创建一个函数，对数据预处理，获得图片的xmax，xmin，ymin，ymax
    def to_labels(path):
        xml = open('{}'.format(path)).read()
        select = etree.HTML(xml)
        width = int(select.xpath('//size/width/text()')[0])
        height = int(select.xpath('//size/height/text()')[0])
        xmin = int(select.xpath('//bndbox/xmin/text()')[0])
        xmax = int(select.xpath('//bndbox/xmax/text()')[0])
        ymin = int(select.xpath('//bndbox/ymin/text()')[0])
        ymax = int(select.xpath('//bndbox/ymax/text()')[0])
        # 返回的label值有四种分类
        return [xmin/width,ymin/height,xmax/width,ymax/height]

    # 使用to_labels处理所有的图片，得到标签
    labels = [to_labels(path) for path in xmls]
    # print(labels[:3])

    # 将相同的目标值放在一起，使用zip的反向操作
    output1,output2,output3,output4 = list(zip(*labels))
    # print(output1,output2,output3,output4)
    # print(len(output1))

    # tensor类型，转化为numpy类型
    output1 = np.array(output1)
    output2 = np.array(output2)
    output3 = np.array(output3)
    output4 = np.array(output4)
    print('output1',output1.shape)
    print(output2.shape)
    print(output3.shape)
    print(output4.shape)

    # 创建label_dataset
    label_dataset = tf.data.Dataset.from_tensor_slices((output1,output2,output3,output4))
    # print(label_dataset)

    # 处理图片的函数
    @tf.function
    def load_image(path):
        # 读文件
        img = tf.io.read_file(path)
        # 解码
        img = tf.image.decode_jpeg(img,channels=3)
        # 更改大小
        img = tf.image.resize(img,(224,224))
        # 归一化，规定在了-1~1之间
        img = img / 127.5 - 1
        return img

    image_dataset = tf.data.Dataset.from_tensor_slices(imgs_train)
    autotune = tf.data.experimental.AUTOTUNE
    image_dataset = image_dataset.map(load_image, num_parallel_calls = autotune)
    # print(image_dataset)

    # 得到最终训练的dataset
    dataset = tf.data.Dataset.zip((image_dataset,label_dataset))
    # print(dataset)

    # print(len(images))

    for img,label in dataset.take(1):
        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
        out1, out2, out3, out4 = label
        xmin, ymin, xmax, ymax = out1[0].numpy()*224,out2[0].numpy()*224,out3[0].numpy()*224,out4[0].numpy()*224
        rect = Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), fill=False, color='red')
        ax = plt.gca()
        ax.axes.add_patch(rect)
        plt.show()

    # 划分训练和测试集
    test_count = int(len(images)*0.2)
    train_count = len(images) - test_count
    dataset_train = dataset.skip(test_count)
    dataset_test = dataset.take(test_count)

    BATCH_SIZE = 32
    BUFFER_SIZE = 300
    train_step = train_count / BATCH_SIZE
    test_step = test_count / BATCH_SIZE

    # 设置批次`
    # train_dataset = dataset.shuffle(300).batch(32).repeat()
    train_dataset = dataset_train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(autotune)
    test_dataset = dataset_test.repeat()


    # 设置图像定位模型，使用预训练模型
    xception = tf.keras.applications.Xception(weights='imagenet',
                                              include_top=False,
                                              input_shape=(224,224,3))

    inputs = tf.keras.layers.Input(shape=(224,224,3))
    # inputs = np.reshape(inputs,[None,224,224,3])

    x1 = xception(inputs)

    # 变为一个2维向量
    x2 = tf.keras.layers.GlobalAveragePooling2D()(x1)

    print('x2',x2.shape)

    x3 = tf.keras.layers.Dense(2048,activation='relu')(x2)
    x4 = tf.keras.layers.Dense(256,activation='relu')(x3)

    output1 = tf.keras.layers.Dense(1)(x4)
    output2 = tf.keras.layers.Dense(1)(x4)
    output3 = tf.keras.layers.Dense(1)(x4)
    output4 = tf.keras.layers.Dense(1)(x4)

    prediction = [output1,output2,output3,output4]

    # 模型已完成创建
    model = tf.keras.models.Model(inputs=inputs,outputs=prediction)

    model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='mse',
                  metrics=['mae'])

    history = model.fit(dataset,
                        epochs=50,
                        steps_per_epoch=len(imgs_train)//32,
                        validation_data=test_dataset,
                        validation_steps=test_step)

    plt.plot(history.epoch,history.history.get('acc'),'r',label='Training Loss')
    plt.plot(history.epoch,history.history.get('val_acc'),'bo',label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    # 设置取值范围
    # plt.ylim([0,1])
    plt.legend()
    plt.show()
