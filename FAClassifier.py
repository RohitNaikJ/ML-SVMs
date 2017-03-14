from cvxpy import *
import numpy as np
import math
from svmutil import *
import matplotlib.pyplot as plt

class FacialAttClassifier:
    def __init__(self, fileData, fileLabels):
        # m -> #training examples
        # n -> #features
        labels = []
        self.m = 0;
        for line in open(fileData):
            x = [float(x) for x in line.rstrip().split(",")]
            labels.append(x)
            self.m += 1;
            self.n = len(x)
        self.X = np.mat(labels)
        self.Y = np.array(np.loadtxt(fileLabels))
        for id,y in enumerate(self.Y):
            if(y==2):
                self.Y[id] = -1
            else:
                self.Y[id] = 1;
        self.Y = np.mat(self.Y).reshape(self.m,1)
        self.C = 500


    def LinearLearnAlpha(self):
        A = Variable(self.m)
        paramC = Parameter(sign="positive", value=self.C)
        Q = np.zeros((self.m,self.m))
        for i in range(self.m):
            for j in range(self.m):
                Q[i][j] = ((self.X[i,:] * self.X[j,:].T))[0,0]
                Q[i][j] *= self.Y[i]
                Q[i][j] *= self.Y[j]
        # Q = Q.tolist()
        objective = Maximize((-0.5)*quad_form(A,Q) + sum_entries(A))
        constraints = [0 <= A, A <= paramC, (A.T * self.Y)[0,0] == 0]
        prob = Problem(objective, constraints)
        optValue = prob.solve()
        spVec = []
        self.epsilon = 1
        for x in A.value:
            if x[0,0]>self.epsilon:
                spVec.append(x[0,0])
        self.linearAlpha = A.value
        self.spVecCt = len(spVec)
        print("Optimal Value: {:.3f}".format(optValue))
        print("No. of Support Vectors: {}".format(len(spVec)))
        print("Support Vectors: ", spVec)

    def LinearWB(self):
        # Computing Weight vector, w
        w = np.zeros((1, self.n))
        for i in range(self.m):
           w += (self.X[i,:]*self.Y[i,0]*self.linearAlpha[i,0])
        self.linearW = w.T

        # Computing Intercept b
        B = 0
        for i in range(0, self.m):
            if (499 > self.linearAlpha[i,0] > self.epsilon):
                B += self.Y[i,0] - (self.X[i,:]*self.linearW)[0,0]
                break
        self.linearB = B
        print("Weight Vector", self.linearW.T)
        print("Intercept", self.linearB)

    # Making Prediction on testing set
    def LinearPred(self, fileFeatures, fileLabels):
        self.LinearWB()
        data = []
        self.testm = 0
        for line in open(fileFeatures):
            x = [float(x) for x in line.rstrip().split(",")]
            data.append(x)
            self.testm += 1
        self.testX = np.mat(data)
        self.testY = np.array(np.loadtxt(fileLabels))
        for id, y in enumerate(self.testY):
            if (y == 2):
                self.testY[id] = -1
            else:
                self.testY[id] = 1
        self.testY = np.mat(self.testY).reshape(self.testm, 1)
        correct = 0.0
        for i in range(self.testm):
            if self.testY[i,0]*((self.testX[i,:] * self.linearW)[0,0] + self.linearB) > 0:
                correct += 1
        print("Average Accuracy on testing set using Linear Model: {:.3f}".format(correct*100/self.testm))

    # Gaussian Kernel Function
    def Kernel(self, X, Z):
        # x = math.exp(-2.5 * ( ((X-Z).T * (X-Z))[0,0] ))
        x = math.exp(-2.5 * (np.linalg.norm(X - Z)**2))
        return x

    # Learning using Gaussian Kernel
    def Gaussain(self):
        A = Variable(self.m)
        paramC = Parameter(sign="positive", value=self.C)
        Q = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                Q[i][j] = self.Kernel(self.X[i,:], self.X[j,:])
                Q[i][j] *= self.Y[i]
                Q[i][j] *= self.Y[j]
        # Q = Q.tolist()
        objective = Maximize((-0.5) * quad_form(A, Q) + sum_entries(A))
        constraints = [0 <= A, A <= paramC, (A.T * self.Y)[0, 0] == 0]
        prob = Problem(objective, constraints)
        optValue = prob.solve()
        spVec = []
        for x in A.value:
            if x[0, 0] > self.epsilon:
                spVec.append(x[0, 0])
        self.gaussAlpha = A.value
        self.gausspVecCt = len(spVec)
        print("Optimal Value: {:.3f}".format(optValue))
        print("No. of Support Vectors: {}".format(self.gausspVecCt))
        print("Support Vectors: ", spVec)

    def GaussianPred(self):
        # Computing Intercept b
        B = 0
        for k in range(0, self.m):
            if (499 > self.gaussAlpha[k, 0] > 1):
                for i in range(0, self.m):
                    B += self.gaussAlpha[i, 0] *  self.Y[i,0] * self.Kernel(self.X[i,:], self.X[k,:])
                B = self.Y[k,0] - B
                break
        self.gaussB = B
        print("Intercept for Gaussian Model: {}".format(self.gaussB))

        # Making Predictions
        correct = 0.0
        for i in range(self.testm):
            P = 0
            for j in range(self.m):
                P += self.gaussAlpha[j,0] * self.Y[j,0] * self.Kernel(self.X[j,:], self.testX[i,:])
            if self.testY[i, 0] * (P + self.gaussB) > 0:
                correct += 1
        print("Average Accuracy on testing set using Gaussian Model: {:.3f}".format(correct * 100 / self.testm))

    def LibSVM(self):
        trainFile = open("lib_train.txt", "w")
        for i in range(self.m):
            trainFile.write(str(self.Y[i,0])+" ")
            for j in range(self.n):
                trainFile.write(str(j+1)+":"+str(self.X[i,j])+" ")
            trainFile.write("\n");

        testFile = open("lib_test.txt", "w")
        for i in range(self.testm):
            testFile.write(str(self.testY[i,0])+" ")
            for j in range(self.n):
                testFile.write(str(j+1)+":"+str(self.testX[i,j])+" ")
            testFile.write("\n");

        # Linear Model
        self.trainy, self.trainx = svm_read_problem("lib_train.txt")
        self.testy, self.testx = svm_read_problem("lib_test.txt")
        
        print("LIBSVM: Linear Model")
        m = svm_train(self.trainy, self.trainx, "-t 0 -c 500")
        p_labels, p_acc, p_vals = svm_predict(self.testy, self.testx, m)
        sv_indices = m.get_sv_indices()

        print("No. of Support Vectors: {}".format(len(sv_indices)))
        print("Indices of Support Vectors: ", sv_indices)

        # Guassian Model
        print("LIBSVM: Gaussain Model")
        model_g = svm_train(self.trainy, self.trainx, '-g 2.5 -c 500')
        p_labels, p_acc, p_vals = svm_predict(self.testy, self.testx, model_g)
        sv_indices = model_g.get_sv_indices()

        print("No. of Support Vectors: {}".format(len(sv_indices)))
        print("Indices of Support Vectors: ", sv_indices)

    def crossValid(self, CList):
        xList = []
        yList = []
        ytList = []

        maxA = -1
        maxC = -1
        for c in CList:
            m = svm_train(self.trainy, self.trainx, ('-t 2 -c {} -g {} -v 10 -q').format(c,2.5))
            model = svm_train(self.trainy, self.trainx, '-t 2 -c {} -g {} -q'.format(c,2.5))
            xList.append(math.log(c, 10))
            yList.append(m)
            r = svm_predict(self.testy, self.testx, model)
            # print(r[1][0])
            ytList.append(r[1][0])

            if(m > maxA):
                maxA = m
                maxC = c

        print("For best validation accuracy C: {}".format(maxC))

        plt.plot(xList, yList)
        plt.plot(xList, ytList)
        plt.legend(['Validation Set','Test Set'])
        plt.xlabel('log(C)')
        plt.ylabel('Accuracy%')
        plt.show()


FAC = FacialAttClassifier("traindata.txt", "trainlabels.txt")
FAC.LinearLearnAlpha()
FAC.LinearPred("testdata.txt", "testlabels.txt")
FAC.gamma = 2.5
# FAC.Gaussain()
# FAC.GaussianPred()
FAC.LibSVM()
CList = [1, 10, 10**2, 10**3, 10**4, 10**5, 10**6]
# FAC.crossValid(CList)

