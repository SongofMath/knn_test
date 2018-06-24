from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from math import pow
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# use the "train" data to predict the "test" data
def test(train, test):
    frames = [train, test]
    result = pd.concat(frames)
    # get the nearest neighbors (brute, ball_tree, kd_tree, auto)
    nbrs = NearestNeighbors(n_neighbors=8, algorithm='auto').fit(result)
    # return distances and indices
    distances, indices = nbrs.kneighbors(result)
    print(distances)
    print("=====")
    print(indices)
    return indices[len(train):len(result)-1]


def train(X, y, k):
    # train the data (X = features, y = classifer, k = number of neighbors)
    return KNeighborsClassifier(n_neighbors=k).fit(X, y)


def predict(X_test, knn):
    # predict the result
    return knn.predict(X_test)


def score(predict, y_test):
    # give out accuracy_score
    return accuracy_score(predict, y_test)


def get_optimal_k(k, X_train, y_train, cv_num):
    scoresList = []
    for i in range(1, k):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=cv_num, scoring='accuracy')
        scoresList.append(scores.mean())
    print(scoresList)
    return scoresList.index(min(scoresList))+1


def optimal_k(k, data, cv_num, classfierName):
    # get the optimal value of k which leads to the lowest error rate in n fold cross validation
    # k = number of neighbours in knn
    # cv_num = number of fold in a cross validation
    # cv = percent of each portion in the whole data
    cv = 1/cv_num

    # cv_split : to store the "split points"
    # score_list : to store the score (correction rate) for each k
    cv_split = []
    score_list = []

    # this is made to split the data
    for i in range(1, cv_num):
        cv_split.append(i * int(cv * len(data)))
    # the following is where the actual split is taking place
    train_data = np.split(data.sample(frac=1), cv_split)
    # the last column is the classifer (y)
    test_data = train_data[cv_num-1]
    # here we like to do for loop to get the score list for knn with every integers < k
    for j in range(1, k + 1):
        scores = 0
        for n in range(0, len(train_data) - 1):
            knn = KNeighborsClassifier(n_neighbors=j).fit(train_data[n].drop(classfierName, axis=1), train_data[n][classfierName].values)
            pred = predict(test_data.drop(classfierName, axis=1), knn)
            scores += score(pred, test_data[classfierName].values)
        score_list.append(scores/len(train_data))
    # we need to plus one because list always starts with 0 index but k starts from 1
    return score_list.index(max(score_list)) + 1, score_list


def label_quality(indices, order):
    # label the quality
    label_list = []
    for i in range(0, len(indices)):
        label = [indices[i][0]]
        for j in range(1, len(indices[i])):
            label.append(order[indices[i][j]])
        label_list.append(label)
    return label_list


def max_count(label_list):
    # find the maximum frequency in an array
    count_list = []
    for i in range(0, len(label_list)):
        cnt = [label_list[i][0]]
        ls = label_list[i][1:len(label_list)]
        cnt.append(max(ls, key=ls.count))
        count_list.append(cnt)
    return count_list


def check_correction_two(label, y):
    # check the correction number
    cor = 0
    incor = 0
    bias_sum = 0;
    variance_sum = 0;
    for i in range(0, len(label)):
        real = y[i]
        pred = label[i]
        if real == pred:
            cor += 1
        else:
            incor += 1
        bias_sum += (pred - real)
        variance_sum += pow(pred - real, 2)
    cor_pre = (cor)/(cor+incor)
    incor_pre = (incor)/(cor+incor)
    bias = bias_sum / len(label)
    variance = variance_sum / len(label)
    return cor_pre, incor_pre, bias, variance


def check_correction(label, y):
    # check the correction number
    cor = 0
    incor = 0
    for i in range(0, len(label)):
        real = y[label[i][0]]
        pred = label[i][1]
        if real == pred:
            cor += 1
        else:
            incor += 1
    cor_pre = (cor)/(cor+incor)
    incor_pre = (incor)/(cor+incor)
    return cor_pre, incor_pre


# plot knn graph
def plotKNNgraph(data1, data2, result):
    plt.figure(np.random.randint(100, 1000))
    # fig, ax = plt.subplots()
    # ax.set_xlabel('Smarts')
    # ax.set_ylabel('Probability density')
    for i in range(0, len(data1)):
        colors = color(result[i])
        plt.scatter(data1[i], data2[i], c=colors, alpha=0.5)
    plt.show()
    return


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
