if __name__ == '__main__':

    # Dataset实例

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    (train_img,train_label),(test_img,test_label) = tf.keras.datasets.mnist.load_data()

    train_img = train_img / 255
    test_img = test_img / 255

    #创建train的数据集
    dataset_train_img = tf.data.Dataset.from_tensor_slices(train_img)
    # print(dataset_train_img)
    dataset_train_label = tf.data.Dataset.from_tensor_slices(train_label)
    # print(train_label)
    dataset_train = tf.data.Dataset.zip((dataset_train_img,dataset_train_label))
    print(dataset_train)

    dataset_train = dataset_train.shuffle(buffer_size=60000).repeat().batch(batch_size=60)

    #创建test的数据集
    # dataset_test_img = tf.data.Dataset.from_tensor_slices(test_img)
    # dataset_test_label = tf.data.Dataset.from_tensor_slices(test_label)
    # dataset_test = tf.data.Dataset.zip(test_img,test_label)
    # dataset2 = dataset_test.batch(batch_size=50)
    dataset_test = tf.data.Dataset.from_tensor_slices((test_img,test_label))
    dataset_test = dataset_test.batch(batch_size=50)


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
    #设置迭代的步数
    train_steps = train_img.shape[0] / 60

    print(train_img.shape[0])
    print(train_img.shape[0]/60)


    # validation_steps = 10000 / 50
    # model.fit(dataset_train,epochs=30,steps_per_epoch = train_steps,
    #           validation_data=dataset_test,validation_steps=validation_steps)



