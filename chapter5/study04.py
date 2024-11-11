if __name__ == '__main__':

    # tf.keras.metrics汇总计算模块,应用在study03中

    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    m = tf.keras.metrics.Mean()
    m(10)
    m(20)
    m(30)
    m(40)
    m(87)
    print(m.result().numpy())
    # 重置对象,将之前m存放的数值m清空
    m.reset_states()
    m(15)
    print(m.result().numpy())


