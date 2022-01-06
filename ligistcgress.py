# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import random
import matplotlib.pyplot as plt

class Logisticregress():
    def __init__(self,features=None,labels=None,batch_size=None,epoch=None,beta=None,eta=None):
        """
        :param features: n*p
        :param labels: n*1
        :param batch_size: 1
        :param epoch: 1
        :param beta: p*1
        :param eta: 1
        """
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.epoch = epoch
        self.beta = beta
        self.eta = eta

    def synthetic_data(self, num_examples = None, dim = None):
        """
        synthetic data containing noise for test the model
        :param num_examples:
        :param beta: p*1
        :param dim: len(beta)
        :return: features ,labels
        """
        beta = np.random.normal(2, 1, (dim, 1)).astype((int))
        features = np.random.normal(0, 1, (num_examples, dim))
        labels = 1/(1+np.exp(np.dot(features,beta)))
        noise = np.random.normal(0, 0.01, labels.shape)
        labels = labels + noise
        labels = np.array(labels-0.5 > 0).astype(int)
        return features, labels,beta

    def plot_data(self,features, labels):
        """
        visual data
        :return:
        """
        plt.scatter(features,labels)
        return

    def iter_data(self,batch_size, features, labels):
        """
        randomly generate batch size data
        :param batch_size:
        :param features:
        :param labels:
        :return:features[batch size] , labels[batch size]
        """
        num_examples = len(features)
        index = list(range(num_examples))
        random.shuffle(index)
        for i in range(0,num_examples,batch_size):
            batch_index = np.array(index[i:min(i+batch_size,num_examples)])
            yield features[batch_index], labels[batch_index]

    def init_para(self,beta= np.random.normal(2, 1,(4,1)).astype(int),eta=0.001): # how to set defalt paras
        return beta,eta

    def logisreg(self,features, beta):
        """
        define model function
        :param features:
        :param beta:

        :return: 1
        """
        return 1/(1+np.exp(-np.dot(features,beta)))

    def zero_one_loss(self,y_hat,y_true):
        """
        calculate the loss between the predicts and the ground trues.
        :param y_hat:
        :param y_true:
        :return: 1
        """
        return np.array(-y_true *np.log(y_hat)-(1-y_true)*np.log(1-y_hat)).mean()

    def bgd(self,X,y_true,y_hat):
        """
        derivation of loss function to parameter
        :param X:
        :param y_true:
        :param y_hat:
        :return:
        """
        n = len(X)
        return np.array(X).T@(y_hat-y_true)/n

    def train_data(self,epoches,features, labels,batch_size):
        """
        epoches is the stop condition, for every epoch , output the loss
        :return:
        """
        self.Loss=[]
        beta, eta = self.init_para()
        #print("beta is:",beta,"\neta is:",eta)
        for epoch in range(epoches):
            for X,y_true  in self.iter_data(batch_size, features, labels):
                # 在这个数据集合下进行反向传播，直到收敛
                y_hat = self.logisreg(X, beta)
                loss = self.zero_one_loss(y_hat, y_true)
                self.Loss.append(loss)
                # Gradient calculation of lSoss at beta # P*1
                grad_beta = self.bgd(X,y_true,y_hat)
                # Update parameters
                beta = beta - eta * grad_beta
                print("\n epoch:",epoch,"loss:",loss)
        # Judgment convergence
        if len(self.Loss)>10:
            t= 0
        for i in range(-1,-10,-1):
            t += self.Loss[i]-self.Loss[i-1]
            if np.abs(t/10)<0.1:
                break
        return beta


lr= Logisticregress()
# load the data  the size of examples is num_examples and the size of features is dim
features, labels, beta_true = lr.synthetic_data(num_examples = 100, dim = 4)
# trian the model
beta_predict = lr.train_data(1000,features, labels,batch_size= 20)
# plot the curve of the true beta
fig, ax = plt.subplots(2,2)
ax[0][0].plot(range(len(beta_true)),beta_true)
ax[0][0].set_title(" the curve of the true beta")
print("the beta_true is:",beta_true)
# plot the curve of the predicted beta
ax[0][1].plot(range(len(beta_predict)),beta_predict)
ax[0][1].set_title(" the curve of the predicted beta")
print("the beta_predict is:",beta_predict)
# plot loss change curve
ax[1][0].scatter(range(len(lr.Loss)),lr.Loss)
ax[1][0].set_title("The curve of loss function")
plt.show()











