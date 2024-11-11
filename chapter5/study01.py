if __name__ == '__main__':

    #
    #eager,在执行模型时，可以使用宿主语言的所有的功能

    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    #判断是否处于eager模式。默认是true
    # print(tf.executing_eagerly())

    x = [[2],]
    #数组相乘
    m = tf.matmul(x,x)
    # print(m)

    #转换为numpy类型
    # print(m.numpy())
    # print(m.numpy().astype)

    #创建一个常量,二维数组
    a = tf.constant([[1,2],[3,4]])

    # print(a)
    # print(a.numpy())
    # print(a.numpy().astype)
    b = tf.add(a,2)
    # print(b)
    c = tf.matmul(a,b)
    # print(c)

    #d是ndarray对象,a是tensor对象
    d = np.array([[5,6],[7,8]])
    #两个不同类型的对象也可以进行计算
    #转换为tensor对象
    print(a+d)
    e = (a+d).numpy()
    #转换为ndarray对象
    print(e)
    print(e.astype)


    #使用python控制流建立tf的运算

    #将10个数转换为tensor对象
    # num = tf.convert_to_tensor(10)
    # print(num)

    # for i in range(num.numpy()):
    #     #i为tf创建的tensor对象，当进行python语言的计算时，自动转换为python的方式
    #     i = tf.constant(i)
    #     if int(i%2) == 0:
    #         print('even')
    #     else:
    #         print('odd')


