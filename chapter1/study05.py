if __name__ == '__main__':

    #逻辑回归：sigmoid函数是一个概率分布函数，给定某个输入，它将输出为一个概率值
    #逻辑回归损失函数：对于分类问题，我们最好的使用交叉熵损失函数会更有效交叉熵会输出一个更大的“损失”
    #平方差所惩罚的是与损失为同一数量级的情形对于分类问题，我们最好的使用交叉熵损失函数会更有效交叉熵会输出一个更大的“损失”
    #在keras里，我们使用binary_crossentropy来计算二元交叉熵

    #解决二分类问题

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    # CUDA_VISIBLE_DEVICES = 0,1
    os.environ["CUDA_DEVICE_ORDER"] = "0000: 01:00.0"


    #header=None去掉索引的数组采用默认的形式
    data = pd.read_csv('../data_set/credit-a.csv', header=None)
    print(data.head(5))
    #查看最后一列不同数值的数量（只有1和-1两种）
    print(data.iloc[:,-1].value_counts())
    #取出除了最后一列的数据
    x = data.iloc[:,:-1]
    #将最后一列中值为-1的替换为0再取出
    y = data.iloc[:,-1].replace(-1,0)
    #创建模型
    model = tf.keras.Sequential()
    #设置了100个隐藏层
    model.add(tf.keras.layers.Dense(4,input_shape=(15,),activation='relu'))
    #添加第二层100个隐藏层
    model.add(tf.keras.layers.Dense(4,activation='relu'))
    #添加输出层
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    model.summary()
    #逻辑回归用到的loss函数为binary_crossentropy
    #optimizer='adam'使用adam优化方法。metrics=['acc']为准确率
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    exercise = model.fit(x,y,epochs=100)
    #将训练的数据保存在exercise中输出时字典类型的
    print(exercise.history.keys())
    #选择其一语句运行查看训练次数与acc或者loss的关系
    plt.plot(exercise.epoch,exercise.history.get('acc'))
    # plt.plot(exercise.epoch,exercise.history.get('loss'))
    plt.show()


