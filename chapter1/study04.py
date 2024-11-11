if __name__ == '__main__':

    #多层感知器及激活函数
    #继续使用神经网络解决这种不具备线性可分性的问题，采取在神经网络的输入端和输出单之间差入更多的神经元
    #在输入层和输出层之间嵌入了多层的隐含层，而这些隐含层就是算法
    #常用的激活函数：relu函数、sigmoid函数、tanh函数、leak relu函数（了解，常用在生成网络中）

    #问题探究三列TV、radio、newspaper对于sales的影响

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    data = pd.read_csv('../data_set/Advertising.csv')
    #一共有索引列、TV、radio、newspaper、sales五列
    # print(data.info)
    # print(data.head(1))

    #取第二列到最后一列的切片（左闭右开）
    x = data.iloc[:,1:-1]
    #取最后一列
    y = data.iloc[:,-1]
    # 1.初始化模型
    model = tf.keras.Sequential()
    #2.添加层数，输入的数据是三列（tv、radio、newspaper），用到的激活函数是relu
    model.add(tf.keras.layers.Dense(10,input_shape=(3,),activation='relu'))
    #添加输出层
    model.add(tf.keras.layers.Dense(1))
    model.summary()
    # 3.使用梯度下降算法，
    # optimizer为优化方法
    # loss损失函数
    model.compile(optimizer='adam',loss='mse')
    # 4.epochs=1000表示对所有的数据进行训练100次
    model.fit(x,y,epochs=100)
    # 5.预测
    #取出三列数据进行预测
    data_test = data.iloc[:10, 1:-1]
    print(model.predict(data_test))
    #三列对应实际的数值
    sales_test = data.iloc[:10, -1]
    print(sales_test)

    # plt.scatter(data.newspaper,data.sales)
    # plt.show()



