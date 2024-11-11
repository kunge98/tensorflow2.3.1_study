# 函數：1.loss函數
def compute_loss_function(b, w, points):
    total_error = 0
    for i in range(0, len(points)):
        # 读取的文件一共两列，x读第一列，y读第二列
        x = points[i, 0]
        y = points[i, 1]
        total_error += ((w * x + b) - y) ** 2
    return total_error / float(len(points))

    # 函数：2.求每个点的梯度计算
def compute_gradient(b_current, w_current, points, learning_rate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        b_gradient += (2 / N) * ((w_current * x + b_current) - y)
        w_gradient += (2 / N) * x * (w_current * x + b_current - y)
    b_new = b_current - (learning_rate * b_gradient)
    w_new = w_current - (learning_rate * w_gradient)
    return [b_new, w_new]

    # 函数：3.更新每个点的梯度
def update_gradient(points, b_start, w_start, learning_rate, num_iterations):
    b = b_start
    w = w_start
    for i in range(num_iterations):
        b, w = compute_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


def run():
    points = np.genfromtxt("../data_set/data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_w = 0  # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_loss_function(initial_b, initial_w, points))
          )
    print("Running...")
    [b, w] = update_gradient(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_loss_function(b, w, points))
          )


if __name__ == '__main__':
    import numpy as np

    run()
