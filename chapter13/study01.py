if __name__ == '__main__':

    # 主要应用图像的识别，遥感信息定位
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mping
    from lxml import etree
    import glob

    # 引入矩形框
    from matplotlib.patches import Rectangle

    # 使用之前的tf.io.read_file老是报错UnicodeDecodeError尚未解决
    lean = mping.imread('../data_set/oxford-iiit-pet/images/Abyssinian_1.jpg')
    print(lean.shape)
    plt.imshow(lean)
    # plt.show()

    # 解析
    xml = open('../data_set/oxford-iiit-pet/annotations/xmls/Abyssinian_1.xml').read()

    select = etree.HTML(xml)
    # 根据图片的对应xml文件查看前端源代码
    width = int(select.xpath('//size/width/text()')[0])
    height = int(select.xpath('//size/height/text()')[0])
    xmin = int(select.xpath('//bndbox/xmin/text()')[0])
    xmax = int(select.xpath('//bndbox/xmax/text()')[0])
    ymin = int(select.xpath('//bndbox/ymin/text()')[0])
    ymax = int(select.xpath('//bndbox/ymax/text()')[0])
    # print(width,height,xmin,xmax,ymin,ymax)

    # 制作矩形的参数：左下角的坐标，宽度，高度,是否填充，颜色框
    rect = Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),fill=False,color='yellow')
    # 获取当前图像
    ax = plt.gca()
    # 在图像上添加矩形框框
    ax.axes.add_patch(rect)
    plt.show()





