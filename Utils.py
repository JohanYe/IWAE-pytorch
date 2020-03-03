import matplotlib.pyplot as plt
import numpy as np

def Plot_loss_curve(train_list, test_dict):
    x_tst = list(test_dict.keys())
    y_tst = list(test_dict.values())
    train_x_vals = np.arange(len(train_list))
    plt.figure(2)
    plt.xlabel('Num Steps')
    plt.ylabel('Negative Log Likelihood')
    plt.title('ELBO Loss Curve')
    plt.plot(train_x_vals, train_list, label='train')
    plt.plot(x_tst, y_tst, label='tst')
    plt.legend(loc='best')
    plt.locator_params(axis='x', nbins=10)

    plt.show()
    return

def create_canvas(x):
    rows = 10
    columns = 10

    plt.figure(1)
    canvas = np.zeros((28 * rows, columns * 28))
    for i in range(rows):
        for j in range(columns):
            idx = i % columns + rows * j
            canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = x[idx].reshape((28, 28))
    return canvas
