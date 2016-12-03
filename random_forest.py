from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel, axis


data_train = loadtxt('data_train.txt', delimiter=',')
X_train = data_train[:,:8]
Y_train = data_train[:,8]

mean = Y_train.mean()
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


X = X_train.tolist()

clf = RandomForestClassifier(n_estimators=20)
clf = clf.fit(X,val)


data_test = loadtxt('data_test.txt', delimiter=',')
X_test = data_test[:,:8]
Y_test = data_test[:,8]

labels = Y_test.tolist()

labels_true = open('labels_true-3.txt', 'w+')
labels_derived = open('labels_der-3.txt', 'w+')

# for two class classification
for j in xrange(len(labels)):
	if(labels[j] > 3000):
        	labels[j]=1
    	else:
        	labels[j]=0


#Mean: 3467.449
#Accuracy:77.1%



#for 3 class classification
# for j in xrange(len(labels)):
# 	if(labels[j] > 4000):
#         	labels[j]=2
#     	elif(labels[j] > 1500 and labels[j] <= 4000):
#         	labels[j]=1	
#     	else:
#         	labels[j]=0

#Mean: 3467.449
#Accuracy: 53.5%


labels = np.array(labels)
np.savetxt(labels_true, labels,  fmt=['%d'])
labels_true.close()


test_inputs = X_test.tolist()
derived = test_inputs
correct = 0
for i in xrange(len(test_inputs)):
	predict = clf.predict(test_inputs[i])
	derived[i] = predict.tolist()
	predict.tolist()
	if(predict == labels[i]):
		correct += 1




derived = np.array(derived)
np.savetxt(labels_derived, derived, fmt=['%d'])
labels_derived.close()


print "Mean: "+str(mean)
print "Correct classified: "+str(correct)
accuracy = Decimal(correct)/Decimal(len(test_inputs))
accuracy = round(accuracy,3)
print "Accuracy= "+str(accuracy*100)+"%"


final_dict = dict(zip(Y_train.tolist(), derived.tolist()))
plot(final_dict.keys(), final_dict.values(), 'ro')
axis([-10, 100000, -1, 2])
xlabel('shares')
ylabel('class')
show()