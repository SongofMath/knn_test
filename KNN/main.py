import knnManager as knn
import pandas as pd
import graphManager as gm
import features_select as fs
import numpy as np
from sklearn.model_selection import train_test_split


# #######################################################################################
# read data
data = pd.read_csv("/Users/bill/Documents/Project/resource/winequality-red.csv", sep=";")
X = data.drop('quality', axis=1)
y = data["quality"].values

# get variance
variance_by_columns = fs.var_columns(X)
print(variance_by_columns)
# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

k = 100
best_k, score_list = knn.optimal_k(k, data, 3, 'quality')
print("best k : " + str(best_k))
x = [i for i in range(1, k + 1)]
gm.plot_line(x, score_list, 'Numbers of Neighbours', 'Scores')

handler = knn.train(X_train, y_train, best_k)
predict = knn.predict(X_test, handler)
score = knn.score(predict, y_test)
print("score : " + str(score))

cor, incor, bias, variance = knn.check_correction_two(predict, y_test)

print("correct : " + str(cor))
print("incor : " + str(incor))
print("bias : " + str(bias))
print("variance : " + str(variance))
#######################################################################################




# #######################################################################################
# # read data
# data = pd.read_csv("/Users/bill/Documents/Tutor/Kenji/source/IrisFlower/Iris", sep=",")
# X = data.drop('class', axis=1)
# y = data["class"].values
# # split train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
# k = 50
# best_k, score_list = knn.optimal_k(k, data, 2, 'class')
# print("best k : " + str(best_k))
# x = [i for i in range(1, k + 1)]
#
# # train knn
# handler = knn.train(X_train, y_train, best_k)
# # make prediction
# predict = knn.predict(X_test, handler)
# # calculate the score (0 <= score <= 1)
# score = knn.score(predict, y_test)
# print("score : " + str(score))
# # plot graph score against k
# gm.plot_line(x, score_list, 'Numbers of Neighbours', 'Scores')
#
# #######################################################################################



#######################################################################################
# read data
data = pd.read_csv("/Users/bill/Documents/Tutor/Kenji/source/blood-transfusion/bloodDonation", sep=",")
X = data.drop('donate', axis=1)
y = data["donate"].values
# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

k = 100
best_k, score_list = knn.optimal_k(k, data, 4, 'donate')
print("best k : " + str(best_k))
x = [i for i in range(1, k + 1)]
gm.plot_line(x, score_list, 'Numbers of Neighbours', 'Scores')

# train knn
handler = knn.train(X_train, y_train, best_k)
# make prediction
predict = knn.predict(X_test, handler)
# calculate the score (0 <= score <= 1)
score = knn.score(predict, y_test)
print("score : " + str(score))
#######################################################################################





#######################################################################################
# read data
# data = pd.read_csv("/Users/bill/Documents/Project/resource/creditcard.csv")
# data = data.drop('Time', axis=1)
# data = data[:30000]
# # split train and test
# train, test = train_test_split(data, test_size=0.1)
# # train set
# y_train = train["Class"].values
# x_train = train.drop('Class', axis=1)
# # test set
# y_test = test["Class"].values
# x_test = test.drop('Class', axis=1)
#
# index = knn.test(x_train, x_test)
# # print(index)
# label_list = knn.label_quality(index, data["Class"].values)
# # print(label_list)
# max_list = knn.max_count(label_list)
# # print(max_list)
# cor, incor = knn.check_correction(max_list, data["Class"].values)
# print(cor)
# print(incor)
#######################################################################################


#######################################################################################
# # read data
# data = pd.read_csv("/Users/bill/Documents/Project/resource/BreastCancerWisconsinDataSet.csv")
# data = data.drop('id', axis=1)
# # split train and test
# train, test = train_test_split(data, test_size=0.1)
# # train set
# y_train = train["diagnosis"].values
# x_train = train.drop('diagnosis', axis=1)
# # test set
# y_test = test["diagnosis"].values
# x_test = test.drop('diagnosis', axis=1)
#
# index = knn.test(x_train, x_test)
# # print(index)
# label_list = knn.label_quality(index, data["diagnosis"].values)
# # print(label_list)
# max_list = knn.max_count(label_list)
# # print(max_list)
# cor, incor = knn.check_correction(max_list, data["diagnosis"].values)
# print(cor)
# print(incor)
#######################################################################################
