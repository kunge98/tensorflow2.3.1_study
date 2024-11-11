if __name__ == '__main__':

    #Dataset的基础知识以及用法

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    #创建一个一维的dataset
    dataset1 = tf.data.Dataset.from_tensor_slices([6,2,8,4,3,6,5])
    # print(dataset)
    # for ele in dataset1:
    #     print(ele)              #tf.Tensor类型
    #     print(ele.numpy())      #直接将转化为numpy类型

    #创建一个二维的dataset
    dataset2 = tf.data.Dataset.from_tensor_slices([[1,2],[3,4],[5,6]])
    # for ele in dataset2:
    #     print(ele)              #tf.Tensor类型
    #     print(ele.numpy())      #直接将转化为numpy类型

    #创建一个字典的dataset
    dataset3 = tf.data.Dataset.from_tensor_slices({'a':[1,2,3,4],'b':[5,6,7,8],'c':[9,10,11,12]})
    # for ele in dataset3:
    #     print(ele)              #tf.Tensor类型

    #用numpy创建一个一维的dataset
    dataset4 = tf.data.Dataset.from_tensor_slices(np.array(range(12)))
    # print(dataset)

    #take()表示取出数组的前几个数据
    # for ele in dataset4.take(4):
        # print(ele)              #tf.Tensor类型
        # print(ele.numpy())      #直接将转化为numpy类型

    #shuffle将数据打乱
    #repeat将数据重复x次
    #batch批次的请求为x
    dataset5 = dataset1.shuffle(len(dataset1)).repeat(count=3).batch(batch_size=3)
    # print(dataset5)
    # for i in dataset5:
    #     print(i.numpy())

    #使用函数对数据进行变换
    #tf.square对每个元素进行平方的计算
    dataset6 = dataset1.map(tf.square)
    for i in dataset6:
        print(i.numpy())