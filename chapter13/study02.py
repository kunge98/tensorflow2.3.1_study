if __name__ == '__main__':

    # 图片的缩放与目标值的规范

    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from lxml import etree
    import matplotlib.image as mping
    import glob


    img = mping.imread('../data_set/oxford-iiit-pet/images/Abyssinian_1.jpg')

    xml = open('../data_set/oxford-iiit-pet/annotations/xmls/Abyssinian_1.xml').read()

    select = etree.HTML(xml)

    width = int(select.xpath('//size/width/text()')[0])
    height = int(select.xpath('//size/height/text()')[0])
    xmin = int(select.xpath('//bndbox/xmin/text()')[0])
    xmax = int(select.xpath('//bndbox/xmax/text()')[0])
    ymin = int(select.xpath('//bndbox/ymin/text()')[0])
    ymax = int(select.xpath('//bndbox/ymax/text()')[0])


    # 对图片进行缩放
    img = tf.image.resize(img,[224,224])
    img = img / 255

    # 按照原来的比例对数值重新赋值
    xmin = (xmin/width)*224
    xmax = (xmax/width)*224
    ymin = (ymin/height)*224
    ymax = (ymax/height)*224

    rect = Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),fill=False,color='red')

    # 获取当前图像
    ax = plt.gca()
    # 在图像上添加矩形框框
    ax.axes.add_patch(rect)
    plt.imshow(img)
    plt.show()



