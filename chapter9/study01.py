if __name__ == '__main__':


    # 第九章 多输出模型实例

    # 对文件的预处理的代码来自chapter3中的study02

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import random
    #面向对象的路径管理工具
    import pathlib
    #提供的方法来显示图像
    # import IPython.display as display


    # 读取路径
    data_root = pathlib.Path('..//data_set//multi-output-classification//dataset')

    #对目录进行迭代
    for i in data_root.iterdir():
        print(i)

    # 得到所有文件的路径
    all_img_path = list(data_root.glob('*/*'))
    # 长度2525
    print(len(all_img_path))

    #获取文件的纯路径
    all_img_path = [str(path) for path in all_img_path]
    # print(all_img_path[10:12])

    #将数据集打乱
    random.shuffle(all_img_path)
    # print(all_img_path[10:12])

    #提取分类的名字
    label_names = sorted(item.name for item in data_root.glob('*/'))
    print(label_names)

    # 取出衣服的颜色
    label_names_colors = set(name.split('_')[0] for name in label_names)
    # print(label_names_colors)

    # 取出衣服的种类
    label_names_clothes = set(name.split('_')[1] for name in label_names)
    # print(label_names_clothes)


    #转换为编码格式
    colors_label_to_index = dict((name,index) for index,name in enumerate(label_names_colors))
    clothes_label_to_index = dict((name,index) for index,name in enumerate(label_names_clothes))
    # print(colors_label_to_index)
    # print(clothes_label_to_index)


    #获取所有数据的label
    all_img_labels = [pathlib.Path(i).parent.name for i in all_img_path ]

    print(all_img_labels)

    #遍历所有的数据，当遍历到i的时候获取它的path路径然后获取它父亲的名字，并转换为label_to_index索引，遍历完所有的数据，都会有一个label
    color_labels = [colors_label_to_index[i.split('_')[0]] for i in all_img_labels]
    clothe_labels = [clothes_label_to_index[i.split('_')[1]] for i in all_img_labels]

    #验证一下输出的类名和数字编码是否对应
    # print(all_img_labels[:5])
    # print(all_img_path[:5])

    #显示图片
    # for n in range(3):
    #     img_index = random.choice(range(len(all_img_path)))
    #     display.display(display.Image(all_img_path[img_index],width=100,height=100))
    #     print(all_img_labels[img_index])
    #     print()

    #加载预处理图片
    def load_preprocess_img(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img,channels=3)
        img = tf.image.resize(img,[224,224])
        img = tf.cast(img,tf.float32)
        img = img / 255
        # 归一化到-1到1之间
        img = img*img -1
        return img

    # 应用函数显示图片
    # img_path = all_img_path[0]
    # label = all_img_labels[0]
    # plt.imshow((load_preprocess_img(img_path) +1) /2)
    # plt.grid(False)
    # plt.xlabel(label)
    # plt.show()


    # 构造tf.data对象
    path_dataset = tf.data.Dataset.from_tensor_slices(all_img_path)
    # print(path_dataset)

    autotune = tf.data.experimental.AUTOTUNE
    img_dataset = path_dataset.map(load_preprocess_img,num_parallel_calls=autotune)

    #获取label的数据集
    label_dataset = tf.data.Dataset.from_tensor_slices((color_labels,clothe_labels))

    # for ele in label_dataset.take(3):
    #     print(ele[0].numpy(),ele[1].numpy())

    #合并img和label两个数据集,元组的形式传入
    dataset = tf.data.Dataset.zip((img_dataset,label_dataset))
    # print(dataset)

    # 得到数据集的大小，方便分割训练集和测试集
    len(all_img_path)
    # print(img_count)    #2525

    #划分为训练数据和测试数据
    train_count = int(len(all_img_path)*0.8)
    test_count = int(len(all_img_path) - train_count)
    # print(train_count,test_count)   #1120用于训练，280用于测试

    #由于并不是按照前1120是训练，后280是测试，用skip方法排除掉测试的数量
    train_dataset = dataset.skip(test_count)
    #运用take的方法取出test_count数量的数据作为test
    test_dataset = dataset.take(test_count)

    #打乱train和test的数据，测试集没必要打乱顺序

    BATCH_SIZE = 32

    train_dataset = train_dataset.shuffle(buffer_size=32).repeat().batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(autotune)

    test_dataset = test_dataset.batch(BATCH_SIZE)

    # 建立模型
    # 小型的预训练网络，部署在移动设备
    # 不包括头部的层，并没有使用权重
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False)

    inputs = tf.keras.Input(shape=(224,224,3))

    x = mobile_net(inputs)
    # 必须将图片从四维转换成二维的数据才可以用dense
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x1 = tf.keras.layers.Dense(1024,activation='relu')(x)
    # 多分类问题softmax，共有三种颜色，所以维度为3
    output_colors = tf.keras.layers.Dense(3,activation='softmax',name='output_color')(x1)

    x2 = tf.keras.layers.Dense(1024,activation='relu')(x)
    output_clothes = tf.keras.layers.Dense(4,activation='softmax',name='output_clothes')(x2)

    model = tf.keras.Model(inputs=inputs,outputs=[output_colors,output_clothes])

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  # 对应多个loss可以用字典的方式在每个输出定义一个name作为建，loss函数作为值
                  loss=dict({'output_color':'sparse_categorical_crossentropy',
                            'output_clothes':'sparse_categorical_crossentropy'}),
                  metrics=['acc'])


    train_step = train_count / BATCH_SIZE
    test_step = test_count / BATCH_SIZE

    model.fit(train_dataset,
                  epochs=5,
                  steps_per_epoch=train_step,
                  validation_data=test_dataset,
                  validation_steps=test_step)


    # 试验，只需要更改测试的路径

    my_img = load_preprocess_img(r'..\data_set\multi-output-classification\dataset\black_shoes')
    my_img = tf.expand_dims(my_img,axis=0)

    pred = model.predict(my_img)
    # print(pred[0])

    # training=False调用model的时候是以预测模式进行
    pred = model(my_img,training=False)

    # 将索引转换成颜色取出
    pred_color = colors_label_to_index(np.argmax(pred[0][0]))
    # 将索引转换成衣服种类取出
    pred_clothes = colors_label_to_index(np.argmax(pred[1][0]))

    pred = pred_color + '_' +pred_clothes

    plt.imshow((my_img[0] + 1) / 2)
    plt.xlabel(pred)
    plt.show()























