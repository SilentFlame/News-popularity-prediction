from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
import numpy as np
from sklearn.svm import SVC
from decimal import Decimal


data_train = loadtxt('data_train.txt', delimiter=',')
X_train = data_train[:,:8]
Y_train = data_train[:,8]

val = Y_train.tolist()

# for two class classification

for i in xrange(len(val)):
    if(val[i] > 3000):
            val[i]=1
    else:
        val[i]=0     


# For 3 class Classification

# for i in xrange(len(val)):
#     if(val[i] > 4000):
#             val[i]=2
#     elif(val[i] > 1500 and val[i] <= 4000):
#             val[i]=1
#     else:
#         val[i]=0


        
Y=np.array(val)

clf = SVC()
clf = clf.fit(X_train,Y)


data_test = loadtxt('data_test.txt', delimiter=',')
X_test = data_test[:,:8]
Y_test = data_test[:,8]

labels = Y_test.tolist()

# for two class classification

for j in xrange(len(labels)):
	if(labels[j] > 3000):
        	labels[j]=1
    	else:
        	labels[j]=0

#Mean: 3467.4497807097032
#Accuracy: 77.8%


#for 3 class classification
# for j in xrange(len(labels)):
# 	if(labels[j] > 4000):
#         	labels[j]=2
#     	elif(labels[j] > 1500 and labels[j] <= 4000):
#         	labels[j]=1	
#     	else:
#         	labels[j]=0
#Mean: 3467.4497807097032
#Accuracy: 54.1%


test_inputs = X_test.tolist()

correct = 0
for i in xrange(len(test_inputs)):
	predict = clf.predict(test_inputs[i])
	predict.tolist()
	if(predict == labels[i]):
		correct += 1


print "Correct classified: "+str(correct)
accuracy = Decimal(correct)/Decimal(len(test_inputs))
accuracy = round(accuracy,3)
print "Accuracy= "+str(accuracy*100)+"%"