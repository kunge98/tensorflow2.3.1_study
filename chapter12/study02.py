import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import datetime
from datetime import datetime

if __name__ == '__main__':

    # 循环神经网络实例

    # 北京空气污染预测

    # 每小时记录一次数据，共43824条数据


    data = pd.read_csv('..//data_set//PRSA_data_2010.1.1-2014.12.31.csv')

    print(data.head(5))
    print(data.tail(5))
    print(data.info())

    # 看到数据pm2.5一列有nan值，对数据做一个填充
    # 一共有2067条nan数据
    print(data['pm2.5'].isna().sum())

    # 第一天的数据全为nan，直接跳过当天的,每小时记录一次所以略过前24条数据
    data = data.iloc[24:]

    # 对nan'值进行填充，使用ffill方法为前向填充，使用
    data = data.fillna(method='ffill')

    # 0
    print(data['pm2.5'].isna().sum())

    # 将分布在多列的时间合成一个时间,引入datatime模块
    datetime = datetime.datetime(year=2010,month=6,day=12,hour=13)
    # 2010-06-12 13:00:00
    print(datetime)

    data['time'] = data.apply(lambda x: datetime.datetime(year=x['year'],
                                           month=x['month'],
                                           day=x['day'],
                                           hour=x['hour']),
               # 按照行进行计算
               axis=1)

    # 将时间组合成一段时间点
    print(data)

    # 删除多列
    data.drop(columns=['year','month','day','hour','No'],inplace=True)
    print(data)

    # 把时间序列作为索引
    # data = data.set_index('time')
    # print(data)

    # 风向['SE' 'cv' 'NW' 'NE']
    # print(data['cbwd'].unique())
    # 将风向独热编码化，作为列，然后删除cbwd那一列
    # data = data.join(pd.get_dummies(data.cbwd))
    # print(data)

    # del data['cbwd']
    # print(data)

    # 观察
    # data['pm2.5'][-1000:].plot()
    # data['TEMP'][-1000:].plot()
    # plt.show()

    # 要观测前面5天的数据
    # seq_length = 5*24
    # 预测未来一天的数据
    # delay = 24

    # data_ = []

    # 减去是因为到最后观测数据不够 ，采样到倒数第七天
    # for i in range(len(data) -seq_length -delay):
    #     # 一次采样六天的数据，前五天用作训练，后一天用作预测
    #     data_.append(data.iloc[i:i + seq_length + delay])

    # 采样了144条数据，11个列
    # print(data_[0].shape)

    # data_ = np.array([df.values for df in data_])
    # (43656, 144, 11) 有43656数据，每条数据长度44，每条观测数据的特征值有11个
    # print(data_.shape)

    # 乱序
    # np.random.shuffle(data_)
    # print(data_[0].shape)
    # print(data_.shape)

    # train
    # x = data_[:,:120,:]
    # # print(x)
    # # # label
    # y = data_[:,-1,0]
    #
    # # 训练数据
    # split_b = int(data_.shape[0] * 0.8)
    # # 训练数据
    # train_x = x[:split_b]
    # train_y = y[:split_b]
    # # print(train_x.shape,train_y.shape)
    #
    # # 测试数据
    # test_x = x[split_b:]
    # test_y = y[split_b:]
    # # print(test_x.shape,test_y.shape)
    #
    # # 计算训练数据的列的均值
    # mean = train_x.mean(axis=0)
    # # 求方差
    # std = train_x.std(axis=0)
    #
    # # 标准化
    # train_x = (train_x -mean) / std
    # test_x = (test_x -mean) / std
    #
    # BATCH_SIZE = 128
    #
    # model = tf.keras.Sequential()
    # # batch  长度  特征值
    # model.add(tf.keras.layers.LSTM(32,input_shape=(120,11),return_sequences=True))
    # model.add(tf.keras.layers.LSTM(32,return_sequences=True))
    # model.add(tf.keras.layers.LSTM(32,return_sequences=True))
    # model.add(tf.keras.layers.LSTM(32))
    # # 回归问题不需要激活
    # model.add(tf.keras.layers.Dense(1))
    #
    # model.summary()
    #
    # # 监控对象，如果在3次之内val_loss不降低，则把学习速率乘以factor,最小的学习速率底线
    # # 作为回调函数传入fit中
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau('val_loss',
    #                                      patience=3,
    #                                      factor=0.5,
    #                                      min_lr=0.000005)
    #
    # model.compile(optimizer=tf.keras.optimizers.Adam(),
    #               # 回归问题的loss为均方差mse，metrices也更换为mae平均绝对误差
    #               loss='mse',
    #               metrics=['mae'])
    #
    # his = model.fit(train_x,
    #                 train_y,
    #                 batch_size=BATCH_SIZE,
    #                 epochs=100,
    #                 validation_data=(test_x,test_y),
    #                 callbacks=[reduce_lr])
    #
    # plt.plot(his.epoch,his.history.get('loss'),label='Training_loss')
    # plt.plot(his.epoch,his.history.get('val_loss'),label='Test_loss')
    #
    # plt.legend()
    # plt.show()


    # pred = model.evaluate(test_x,test_y)
    # print(pred)











