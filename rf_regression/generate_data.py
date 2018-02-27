# This generates a simple noisy data of a function defined as y = 5+x for -5 < x < 0 and y = 5-x for 0 < x < 5.

import numpy

num_train = 100
num_test = 20

with open("data_train.csv",'a') as file:
	x = 0
	for i in range(num_train):
		y = 5 - x + numpy.random.normal(0,1)
		file.write(str(x)+","+str(y)+"\n")
		x = x + 5/num_train

	x = 0
	for i in range(num_train):
		y = x + 5 + numpy.random.normal(0,1)
		file.write(str(x)+","+str(y)+"\n")
		x = x - 5/num_train

with open("data_test.csv",'a') as file:
	x = 0
	for i in range(num_test):
		y = 5 - x + numpy.random.normal(0,1)
		file.write(str(x)+","+str(y)+"\n")
		x = x + 5/num_test

	x = 0
	for i in range(num_test):
		y = x + 5 + numpy.random.normal(0,1)
		file.write(str(x)+","+str(y)+"\n")
		x = x - 5/num_test
