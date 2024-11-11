if __name__ == '__main__':

    # 入门第一节
    import tensorflow as tf
    print(tf.__version__)
    a = tf.constant([1,2],name='a')
    b = tf.constant([1,2],name='b')
    #相加的方法，计算两个数组
    res = tf.add(a,b)
    print('res为',res)
