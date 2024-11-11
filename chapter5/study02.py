if __name__ == '__main__':

    #变量和自动微分运算

    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    v = tf.Variable(0.0)    #创建一个变量
    # print(v)                #Variable类型的
    # print(v+1)              #tensor类型的
    # print((v+1).numpy())    #numpy类型
    # print(v.assign(6))      #Variable类型的,值为6
    # print(v.assign_add(1))  #Variable类型的,值为7
    # print(v.read_value())   #tensor类型的,值为7

    #自动求解微分
    #注意：对于变量和常量的值要求是float类型的

    w = tf.Variable(2.0)
    #建立一个上下文管理器（变量）
    with tf.GradientTape() as t:
        #进行跟踪运算，求导一次
        loss =  w * w * w - 6 * w * w - 6
    #求loss对w的导数
    grad_w = t.gradient(loss,w)
    # print(grad_w)

    #建立一个上下文管理器（常量）
    v = tf.constant(3.0)
    with tf.GradientTape() as t:
        #让t去跟踪常量v
        t.watch(v)
        #进行跟踪运算，求导一次
        loss = 2 * v * v - 6 * v
    #求loss对w的导数
    grad_v = t.gradient(loss,v)
    # print(grad_v)

    x = tf.Variable(3.0)
    #建立一个上下文管理器（变量）
    with tf.GradientTape(persistent=True) as t:
        #进行跟踪运算，求导一次
        y = x * x
        z = y * y
    #求y对x的导数
    dy_dx = t.gradient(y,x)
    print(dy_dx)
    #求z对y的导数
    dz_dy = t.gradient(z,y)
    print(dz_dy)


