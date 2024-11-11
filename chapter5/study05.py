if __name__ == '__main__':
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    (train_img,train_label),(test_img,test_label) = tf.keras.datasets.mnist.load_data()

    train_img = tf.expand_dims(train_img,-1)
    test_img = tf.expand_dims(test_img,-1)

    train_img = tf.cast(train_img / 255,tf.float32)
    train_label = tf.cast(train_label,tf.int64)
    test_img = tf.cast(test_img / 255,tf.float32)
    test_label = tf.cast(test_label,tf.int64)

    dataset_train = tf.data.Dataset.from_tensor_slices((train_img,train_label))
    dataset_test = tf.data.Dataset.from_tensor_slices((test_img,test_label))

    dataset_train = dataset_train.shuffle(buffer_size=10000).repeat().batch(batch_size=32)
    dataset_test = dataset_test.batch(batch_size=32)


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16,(3,3),input_shape=(None,None,1),activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(10))

    # model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
    # model.fit(dataset_train,epochs=10,validation_data=dataset_test)

    optimizer = tf.keras.optimizers.Adam()
    loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    feature,label = next(iter(dataset_train))
    predictions = model(feature)

    train_loss = tf.keras.metrics.Mean('train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')


    # 传入三个参数，model、当前的训练数据。当前数据的label
    # 调用model方法传入当前的训练数据，得到一个预测的label
    # 在调用loss_fun方法，传入当前数据的label和预测的label得到一个交叉熵损失
    def loss(model,data_train,data_label):
        pre_data_label = model(data_train)
        return loss_fun(data_label,pre_data_label)


    def train_one_step(model,img,label):
        with tf.GradientTape() as t:
            pre = model(img)
            loss_step = loss_fun(label,pre)
        grad = t.gradient(loss_step,model.trainable_variables)
        optimizer.apply_gradients(zip(grad,model.trainable_variables))
        train_loss(loss_step)
        train_accuracy(label,pre)

    def test_one_step(model,img,label):
        pre = model(img)
        loss_step = loss_fun(label,pre)
        train_loss(loss_step)
        train_accuracy(label,pre)

    def train():
        for epoch in range(10):

            for (batch,(img,label)) in enumerate(dataset_train):
                train_one_step(model, img, label)
            print('epoch train_loss is {},train_accuracy is {}'.format(epoch,
                            train_loss.result(),
                            train_accuracy.result()))

            for (batch,(img,label)) in enumerate(dataset_test):
                test_one_step(model, img, label)
            print('epoch test_loss is {},test_accuracy is {}'.format(epoch,
                            test_loss.result(),
                            test_accuracy.result()))

            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()


    train()




