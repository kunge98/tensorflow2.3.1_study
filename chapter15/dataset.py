if __name__ == '__main__':

    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    import pathlib
    import os

    # os.environ["CUDA_DEVICE_ORDER"] = "0000: 01:00.0"


    # print(tf.__version__)

    data_root = pathlib.Path('..//data_set//NWPU-RESISC45')

    # 遍历该目录下的子目录
    # for i in data_root.iterdir():
    #     print(i)

    # 将所有的数据集里的图片转换成list形式，并输出长度
    all_img_path = list(data_root.glob('*/*'))
    print(len(all_img_path))

    # all_img_path[:3]
    # all_img_path[-3:]
    # 获取文件的纯路径
    all_img_path = [str(path) for path in all_img_path]
    # all_img_path[5:20]

    # img_count = len(all_img_path)

    # 将文件打乱
    random.shuffle(all_img_path)
    # all_img_path[5:20]

    img_count = len(all_img_path)

    # 提取分类的名字,45个分类,并标注数字
    label_names = sorted(i.name for i in data_root.glob('*/'))
    # label_names

    # 并标注数字
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    # label_to_index

    # 获取所有数据的label
    # 遍历所有的数据，当遍历到i的时候获取它的path路径然后获取它父亲的名字，并转换为label_to_index索引，遍历完所有的数据，都会有一个label
    all_img_label = [label_to_index[pathlib.Path(i).parent.name] for i in list(all_img_path)]

    # all_img_label[:3]

    # 输出的路径和分类一一匹配
    # all_img_path[:3]

    # 预处理图片
    def load_process_img(img_path):
        # 读取文件
        img_raw = tf.io.read_file(img_path)
        # 转换编码格式
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 更改图片的大小
        img_tensor = tf.image.resize(img_tensor, [256, 256])
        # 转换类型
        img_tensor = tf.cast(img_tensor, tf.float32)
        # 将图片归一化
        img = img_tensor / 255
        return img


    # 构造tf.data对象
    path_dataset = tf.data.Dataset.from_tensor_slices(all_img_path)
    # path_dataset

    # 使用预处理函数进行处理生成img的数据集
    img_dataset = path_dataset.map(load_process_img)
    # img_dataset

    # for image in img_dataset.take(2):
    #     plt.imshow(image)
    #     plt.show()

    label_dataset = tf.data.Dataset.from_tensor_slices(all_img_label)
    # label_dataset

    # for labels in label_dataset.take(2):
    #     labels.numpy()

    # 合并数据集
    dataset = tf.data.Dataset.zip((img_dataset, label_dataset))
    # dataset

    # 分割训练集和测试集
    # 训练集数量
    dataset_train_count = int(img_count * 0.7)
    print('训练集个数为', dataset_train_count)
    dataset_test_count = img_count - dataset_train_count
    print('测试集个数', dataset_test_count)

    # 得到训练集
    dataset_train = dataset.skip(dataset_test_count)
    # dataset_train

    # 得到测试集
    dataset_test = dataset.take(dataset_test_count)
    # dataset_test

    # 定义批次大小
    BATCH_SIZE = 16

    # 将训练集乱序
    dataset_train = dataset_train.repeat().shuffle(buffer_size=dataset_train_count).batch(BATCH_SIZE)
    # dataset_train

    # 将测试数据乱序
    dataset_test = dataset_test.repeat().batch(BATCH_SIZE)
    # dataset_test

    # 训练集步数
    STEPS_PER_EPOCH = dataset_train_count // BATCH_SIZE
    # STEPS_PER_EPOCH

    # 测试集步数
    VALIDATION_STEPS = dataset_test_count // BATCH_SIZE
    # VALIDATION_STEPS

    # 添加神经网络
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=(256, 256, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(45, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss='mse', metrics=['acc'])

    model.fit(dataset_train,
              epochs=50,
              validation_data=dataset_test,
              validation_steps=VALIDATION_STEPS,
              steps_per_epoch=STEPS_PER_EPOCH)