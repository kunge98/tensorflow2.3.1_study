if __name__ == '__main__':

    #卫星图像识别卷积综合案例（tf.data和CNN）
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import random
    #面向对象的路径管理工具
    import pathlib

    data_root = pathlib.Path('..//data_set//2_class')

    #glob得到所有目录的所有文件
    all_img_path = list(data_root.glob('*/*'))

    #获取文件的纯路径
    all_img_path = [str(path) for path in all_img_path]

    #将数据集打乱
    random.shuffle(all_img_path)

    #得到数据集的大小，方便分割训练集和测试集
    img_count = len(all_img_path)

    # 提取分类的名字
    label_names = sorted(i.name for i in data_root.glob('*/'))
    # print(label_names)

    # 转换为编码格式
    # 万一类特别多，就不能单个的自定义编码，采用循环label_names的方式来编码
    # 创建一个字典存放 类名 和 索引
    label_to_index = dict((name,index) for index,name in enumerate(label_names))
    # 只有两个类别分别为0和1
    # print(label_to_index)

    # 获取所有数据的label
    # 遍历所有的数据，当遍历到i的时候获取它的path路径然后获取它父亲的名字，并转换为label_to_index索引，遍历完所有的数据，都会有一个label
    all_img_label = [label_to_index[pathlib.Path(i).parent.name ] for i in all_img_path]

    # 加载预处理图片
    def load_preprocess_img(img_path):
        # 读取文件的路径
        img_raw = tf.io.read_file(img_path)
        # 对图片进行解码，_jepg专门处理jpg格式的图片。_image通用处理图片的格式
        img_tensor = tf.image.decode_jpeg(img_raw,channels=3)
        # 改变图像大小的方法，在此处图片的大小本来就是256*256，并没有什么用处，然后用_image处理之后不会返回图片的类型
        # 所以用来解决图片类型为unknow的问题
        img_tensor = tf.image.resize(img_tensor,[256,256])
        # 转换图片的类型
        img_tensor = tf.cast(img_tensor,tf.float32)
        # 标准化处理
        img = img_tensor / 255
        # 返回图片
        return img

    # img_path = all_img_path[3]
    # plt.imshow(load_preprocess_img(img_path))
    # plt.show()

    # 构造tf.data对象

    path_dataset = tf.data.Dataset.from_tensor_slices(all_img_path)
    # print(path_dataset)
    # 用上面变量的map函数，传递一个预处理图片的方法对图片进行操作，从而获取到图片的二进制数据存放在img_dataset
    img_dataset = path_dataset.map(load_preprocess_img)

    #获取label的数据集
    label_dataset = tf.data.Dataset.from_tensor_slices(all_img_label)

    # for label in label_dataset.take(10):
    #     print(label)
    # for image in img_dataset.take(10):
    #     print(image)

    # 合并img和label两个数据集,元组的形式传入
    dataset = tf.data.Dataset.zip((img_dataset,label_dataset))
    print(dataset)

    # 划分为训练数据和测试数据
    train_count = int(img_count*0.8)
    test_count = int(img_count - train_count)
    # print(train_count,test_count)   #1120用于训练，280用于测试

    # 由于并不是按照前1120是训练，后280是测试，用skip方法排除掉测试的数量
    train_dataset = dataset.skip(test_count)
    # 运用take的方法取出test_count数量的数据作为test
    test_dataset = dataset.take(test_count)

    # 打乱train和test的数据，测试集没必要打乱顺序
    train_dataset = train_dataset.shuffle(buffer_size=32).repeat().batch(batch_size=32)
    test_dataset = test_dataset.batch(batch_size=32)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3,3),
                                     input_shape=(256,256,3),
                                     activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(filters=1024,kernel_size=(3,3),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(1024,activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256,activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    #二分类问题，逻辑回归，输出个数为1
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))


    model.summary()

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

    steps_per_epoch = train_count / 32
    validation_steps = test_count / 32

    his = model.fit(train_dataset,
                    epochs=30,
                    validation_data=test_dataset,
                    validation_steps=validation_steps,
                    steps_per_epoch=steps_per_epoch)

    plt.plot(his.epoch,his.history.get('acc'),label='acc')
    plt.plot(his.epoch,his.history.get('val_acc'),label='val_acc')

    plt.legend()
    plt.show()












