if __name__ == '__main__':

    # 第11章  自定义模型保存
    # 代码来自 chapter5 study03

    import tensorflow as tf
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

    dataset_train = dataset_train.shuffle(buffer_size=10000).batch(32)
    dataset_test = dataset_test.batch(32)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(10))

    optimizer = tf.keras.optimizers.Adam()
    loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean('train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

    # 创建一个目录保存
    cp_dir = 'training_save/custome'
    # 设置前缀
    cp_prefix = os.path.join(cp_dir,'ckpt')
    # 需要保存的参数
    check_point = tf.train.Checkpoint(optimizer=optimizer,model=model)

    # 得到最新的检查点并重建
    check_point.restore(tf.train.latest_checkpoint(cp_dir))


    def loss(model,data_train,data_label):
        pre_data_label = model(data_train)
        return loss_fun(pre_data_label,data_label)

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
        test_loss(loss_step)
        test_accuracy(label,pre)


    def train():
        for epoch in range(10):
            for (batch,(img,label)) in enumerate(dataset_train):
                train_one_step(model, img, label)
            print(''.format(epoch,
                            train_loss.result(),
                            train_accuracy.result()))

            check_point.save(file_prefix=cp_prefix)

            for (batch,(img,label)) in enumerate(dataset_test):
                test_one_step(model, img, label)
            print(''.format(epoch,
                            test_loss.result(),
                            test_accuracy.result()))
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
    train()

    # 正确率
    (tf.argmax(model(train_img,training=False),axis=-1).numpy() == train_label).sum() / len(train_label)






