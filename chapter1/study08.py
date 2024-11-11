if __name__ == '__main__':

    #函数式API,把每一层看成一个函数来调用这一次

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    (train_img,train_label),(test_img,test_label) = tf.keras.datasets.fashion_mnist.load_data()

    train_img = train_img / 255
    test_img = test_img / 255

    #建立输入
    input = tf.keras.Input(shape=(28,28))

    series =tf.keras.layers.Flatten()(input)

    dense1 = tf.keras.layers.Dense(128,activation='relu')(series)

    dropout = tf.keras.layers.Dropout(0.5)(dense1)

    output = tf.keras.layers.Dense(256,activation='softmax')(dropout)

    model = tf.keras.Model(inputs=input,outputs=output)

    # model.summary()

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
    model.fit(train_img,train_label,epochs=50)

