if __name__ == '__main__':
    # 梯度下降算法:深度学习的核心算法
    # 是一种以利于找到函数极值点的算法，所谓“学习”就是改进模型参数，以便通过大量训练步骤将损失最小化。
    #tf.keras实现线性回归
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "0000: 01:00.0"

    data = pd.read_csv('../data_set/Income1.csv')
    print(data)
    #生成散点图
    plt.scatter(data.Education,data.Income)
    x = data.Education
    y = data.Income
    #1.初始化模型
    model = tf.keras.Sequential()
    #2.添加层
    #参数1：输出的维度是1，
    #参数2：输入数据的维度是1，用一个元组表示
    model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
    print(model.summary())
    #3.使用梯度下降算法，
    #optimizer为优化方法
    #loss损失函数
    model.compile(optimizer='adam',
                  loss='mse')
    #4.epochs=1000表示对所有的数据进行训练1000次
    history = model.fit(x,y,epochs=1000)
    #5.预测
    pre = model.predict(x)
    print(pre)
    # plt.show()