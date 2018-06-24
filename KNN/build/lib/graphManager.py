import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# result = pd.DataFrame(chlorides, density)
# # plt.plot(x_axis, y_axis, 'r.', label='predicted')
# # plt.show()
# sns.pairplot(result, kind="reg")
# plt.show()

# colors = {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}


def color(x):
    return {
        3: 'b',
        4: 'g',
        5: 'r',
        6: 'c',
        7: 'm',
        8: 'y',
        9: 'k',
        10: 'w'
    }.get(x, 'b')


# for i in range(0, len(chlorides)):
#     colors = color(quality[i])
#     plt.scatter(chlorides[i], density[i], c=colors, alpha=0.5)


def plot_line(x, y, x_label, y_label):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()




