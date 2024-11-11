if __name__ == '__main__':

    # 图像的语义分割

    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # 反卷积
    # tf.keras.layers.Conv2DTranspose()

    # list = os.listdir(r'../data_set/oxford-iiit-pet/annotations/trimaps')
    # print(list[-5:])

    png = tf.io.read_file('../data_set/oxford-iiit-pet/annotations/trimaps/yorkshire_terrier_95.png')
    png = tf.image.decode_png(png)
    print(png.shape)

    # 压缩维度变为二维
    png = tf.squeeze(png)
    print(png.shape)

    plt.imshow(png)


    img = tf.io.read_file(r'../data_set/oxford-iiit-pet/annotations/trimaps/yorkshire_terrier_95.jpg')
    # img = tf.image.decode_jpeg(img)

    # plt.imshow(img)

    plt.show()