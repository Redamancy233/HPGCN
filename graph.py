import matplotlib.pyplot as plt
import numpy as np

def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")   # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asfarray(data, float)

def graph_train_loss(train_loss_path):
    train_loss_path = train_loss_path  # 存储文件路径

    y_train_loss = data_read(train_loss_path)  # loss值，即y轴
    x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴

    figure1 = plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epoch')  # x轴标签
    plt.ylabel('loss')  # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.legend()
    plt.title('Loss curve')
    plt.savefig('./fig/loss.png')
    plt.show()
    return figure1

def graph_acc(acc_path):
    acc_path = acc_path  # 存储文件路径

    y_acc = data_read(acc_path)  # acc值，即y轴
    x_acc = range(len(y_acc))  # acc的数量，即x轴

    figure2 = plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epoch')  # x轴标签
    plt.ylabel('acc')  # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_acc, y_acc, linewidth=1, linestyle="solid", label="acc")
    plt.legend()
    plt.title('acc curve')
    plt.savefig('./fig/acc.png')
    plt.show()
    return figure2
