if __name__ == '__main__':

    # 第6章

    #tensorboard可视化

    import tensorflow as tf
    import datetime
    import os

    (train_img,train_label),(test_img,test_label) = tf.keras.datasets.mnist.load_data()

    train_img = tf.expand_dims(train_img,-1)
    test_img = tf.expand_dims(test_img,-1)

    train_img = tf.cast(train_img/255,tf.float32)
    train_label = tf.cast(train_label,tf.int64)
    test_img = tf.cast(test_img/255,tf.float32)
    test_label = tf.cast(test_label,tf.int64)

    dataset_train = tf.data.Dataset.from_tensor_slices((train_img,train_label))
    dataset_test = tf.data.Dataset.from_tensor_slices((test_img,test_label))

    dataset_train = dataset_train.shuffle(10000).repeat().batch(128)
    dataset_test = dataset_test.batch(128)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16,(3,3),input_shape=(None,None,1),activation='relu'))
    model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(10,activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])


    #tensorboard的log文件
    log_dir = os.path.join('logs',datetime.datetime.now().strftime('%Y%m%D-%H%M%S'))

    # histogram_freq记录直方图的评率
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=1)


    #学习率的writer文件

    #创建一个文件写入器,存储在log_dir变量下的learning_rate子文件夹下
    file_writer = tf.summary.create_file_writer(log_dir+'/learning_rate').set_as_default()
    #作为默认的编写器
    # file_writer.set_as_default()

    #编写一个学习率的函数
    def learning_rate_schedule(epoch):
        learning_rate = 0.2
        if epoch > 5 :
            learning_rate = 0.02
        if epoch > 10:
            learning_rate = 0.01
        if epoch > 20:
            learning_rate = 0.005

        #记录某一个标量值得变化
        tf.summary.scalar('learning_rate',data=learning_rate,step=epoch)
        return learning_rate

    #学习率回调函数
    learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)


    model.fit(dataset_train,
              epochs=30,
              steps_per_epoch=60000/128,
              validation_data=dataset_test,
              validation_steps=10000/128,
              # 用数组存储回调函数
              callbacks=[tensorboard_callback,learning_rate_callback])







