import csv
import math
import random

import numpy
import numpy.linalg as la

def get_data(name):
	data = []
	with open(name, 'r') as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
			for i in range(len(row)):
				row[i] = float(row[i])
			data = data + [row]
	return data

def log_det_cov(data): # Variant of Unlabelled Shannon Entropy
	data_ = numpy.array(data).T
	cov = numpy.cov(data_)
	return math.log(la.det(cov))

def information_gain(left, right): # Will have to extremely correct this
  if len(left) <= 2 or len(right) <= 2: # 2, because if a set has 1 point, covariance is undefined
    return 0
  se_total = log_det_cov(left+right)
  se_left  = log_det_cov(left)
  se_right = log_det_cov(right)
  ig = se_total - ( len(left)*se_left + len(right)*se_right ) / len(left)+len(right)
  return ig

# We can use any of the below functions, they should give the same results. 
# First one is a black-box approach and the second one is more clear mathematically.

def get_linear_fit_coefs(data): # This is to give the leaf a value, return type : list
	import numpy
	from sklearn import linear_model
	x = [row[0:-1] for row in data]
	y = [row[-1] for row in data]
	x = numpy.asarray(x)
	y = numpy.asarray(y)
	reg = linear_model.LinearRegression()
	reg.fit(x, y)
	coef = list(reg.coef_) + [reg.intercept_]
	return coef

# def get_linear_fit_coefs(data): # This is to give the leaf a value, return type : list
# 	x = [row[0:-1] for row in data]
# 	y = [row[-1] for row in data]
# 	x = numpy.asarray(x)
# 	y = numpy.asarray(y)
# 	return numpy.matmul(numpy.linalg.pinv(x),y) # using T = inv((transpost(X)*X))*transpose(X)*Y

def split_data(data_train, s): # Splits with respect to a line aligned to one axis
 	left = []
 	right = []
 	for point in data_train:
 		if point[s[0]] < s[1]:
 			left = left + [point]
 		else:
 			right = right + [point]
 	return left, right

def get_best_attribute(data_train, attributes):
 	max_info = 0
 	best_att = [-1, 0]
 	for att in attributes:
 		left, right = split_data(data_train, att)
 		i_g = information_gain(left, right)
 		if i_g >= max_info:
 			max_info = i_g
 			best_att = att
 	return best_att

def train_tree(tree, data_train, curr_index):
	if curr_index > len(tree) or len(data_train) <= 1:
		return
	if len(data_train) < min_elems or 2*curr_index > len(tree):
		tree[curr_index-1][2] = get_linear_fit_coefs(data_train)
		return
	attributes = []
	for i in range(randomness):
 		dim = random.randint(0, len(data_train[0])-2)
 		x_s = random.uniform(ranges[dim][0], ranges[dim][1])
 		attributes = attributes + [[dim, x_s]]
	s_final = get_best_attribute(data_train, attributes)
	tree[curr_index-1][0] = s_final[0]
	tree[curr_index-1][1] = s_final[1]
	left_split, right_split = split_data(data_train, s_final)
	# print("At node " + str(curr_index) + " " + str(len(left_split)) + " " + str(len(right_split)))
	train_tree(tree, left_split, 2*curr_index)
	train_tree(tree, right_split, 2*curr_index+1)

def train_forest(forest, data_train):
	for i in range(len(forest)):
		train_tree(forest[i], data_train, 1)

def calculate_value(point, h):
	value = float(0)
	for i in range(len(point)):
		value = value + h[i]*point[i]
	value = value + h[-1]
	return value

def predict_tree(tree, point, curr_index):
 	if tree[curr_index-1][0]  == -1:
 	  print("here")
 		return calculate_value(point, tree[curr_index-1][2])
 	if point[tree[curr_index-1][0]] < tree[curr_index-1][1]:
 		return predict_tree(tree, point, 2*curr_index)
 	return predict_tree(tree, point, 2*curr_index + 1)

def predict(forest, point):
	prediction = float(0)
	for tree in forest:
		prediction = prediction + predict_tree(tree, point,1)
	prediction = prediction/float(len(forest))
	return prediction

def get_variance(forest, data_test):
 	variance = float(0)
 	for point in data_test:
 		prediction = predict(forest, point[0:-1])
 		print("Actual: " + str(point[-1]) + " Predicted: " + str(prediction))
 		variance = variance + (point[-1] - prediction)**2	
 	return variance/float(len(data_test))


forest_size = 5
max_depth = 5
min_elems = 2
randomness = 20

forest = []
for i in range(forest_size):
	tree = []
	for j in range(2**(max_depth+1)-1):
		tree = tree + [[-1, 0, []]]
	forest = forest + [tree]

data_train = get_data('data_train.csv')
data_test = get_data('data_test.csv')
ranges = [[-100, 100]]*(len(data_train[0])-1) # Depends upon number of dimensions of data and their ranges, defined manually as of now
train_forest(forest, data_train)
variance = get_variance(forest, data_test)
print("Variance: " + str(variance))