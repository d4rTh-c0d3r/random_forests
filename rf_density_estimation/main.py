import csv
import math
import random

from scipy.stats import multivariate_normal as mvn # To calculate value of pdf of a Multivariate Gaussian
import numpy as np
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

def get_gaussian_fit(data):
	data_ = np.array(data).T
	mean = np.mean(data, axis=0)
	cov = np.cov(data_)
	return [mean, cov]

def log_det_cov(data): # Variant of Unlabelled Shannon Entropy
	data_ = np.array(data).T
	cov = np.cov(data_)
	return math.log(la.det(cov))

def information_gain(left, right):
  if len(left) <= 2 or len(right) <= 2: # 2, because if a set has 1 point, covariance is undefined
    return 0
  se_total = log_det_cov(left+right)
  se_left  = log_det_cov(left)
  se_right = log_det_cov(right)
  ig = se_total - ( len(left)*se_left + len(right)*se_right ) / len(left)+len(right)
  return ig

def split_data(data, s): # Splits with respect to a line aligned to one axis
 	left = [] 	
 	right = []
 	for point in data:
 		if point[s[0]] < s[1]:
 			left = left + [point]
 		else:
 			right = right + [point]
 	return left, right

def get_best_attribute(data, attributes):
 	max_info = 0
 	best_att = [-1, 0]
 	for att in attributes:
 		left, right = split_data(data, att)
 		i_g = information_gain(left, right)
 		if i_g >= max_info:
 			max_info = i_g
 			best_att = att
 	return best_att

def train_tree(tree, data, curr_index):
	if curr_index > len(tree) or len(data) == 0:
		if(len(data) == 0):
			tree[curr_index-1][0] = -2
		return
	if len(data) < min_elems or 2*curr_index > len(tree):
		tree[curr_index-1][2] = get_gaussian_fit(data)+[float(len(data))/total_size] # Partition function still missing, sad.
		return
	attributes = []
	for i in range(randomness):
 		dim = random.randint(0, len(data[0])-1)
 		x_s = random.uniform(ranges[dim][0], ranges[dim][1])
 		attributes = attributes + [[dim, x_s]]
	s_final = get_best_attribute(data, attributes)
	left_split, right_split = split_data(data, s_final)
	tree[curr_index-1][0] = s_final[0]
	tree[curr_index-1][1] = s_final[1]
	tree[curr_index-1][3] = [float(len(left_split)), float(len(right_split))] # For sampling function
	train_tree(tree, left_split, 2*curr_index)
	train_tree(tree, right_split, 2*curr_index+1)

def train_forest(forest, data):
	for i in range(len(forest)):
		train_tree(forest[i], data, 1)

def calculate_value(point, h): # Needs to be changed to multiply Partition Function, later :( (h = [mean, cov_matrix, pi])
	x = np.array(point)
	return h[2]*(mvn.pdf(x,h[0],h[1]))

def predict_tree(tree, point, curr_index):
 	if tree[curr_index-1][0]  == -1:
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

def sample_point(tree, curr_index):
	print(tree[curr_index-1])
	if tree[curr_index-1][0] == -2:
		print("Hit the end")
	elif tree[curr_index-1][0] == -1:
		return np.random.multivariate_normal(tree[curr_index-1][2][0],tree[curr_index-1][2][1])
	else:
		r = np.random.uniform(0)
		ratio = tree[curr_index-1][3][0]/(tree[curr_index-1][3][0]+tree[curr_index-1][3][1]) 
		print(r,ratio)
		if r < ratio:
			return sample_point(tree,2*curr_index)
		else:
			return sample_point(tree,2*curr_index+1)


def main_function():
	# choice = int(input("0: Value of PDF at a point \n1: Sample a point from the learnt distribution\n"))
	choice = int(input())
	if choice == 0:
		point = []
		for _ in range(len(data[0])):
			val = float(input())
			point = point+[val]
		print(predict(forest,point))
	elif choice == 1:
		t = random.randint(0, len(forest)-1)
		print(sample_point(forest[t],1))
	else:
		print("Invalid Input")
	main_function()


forest_size = 10
max_depth = 5
min_elems = 5
randomness = 20
ranges = [[-100, 100],[-100,100]] # Depends upon number of dimensions of data and their ranges, defined manually as of now

forest = []
for i in range(forest_size):
	tree = []
	for j in range(2**(max_depth+1)-1):
		tree = tree + [[-1, 0, [],[]]]
	forest = forest + [tree]

data = get_data('data_ul.csv')
ranges = [[-100, 100]]*len(data) # Depends upon number of dimensions of data and their ranges, defined manually as of now
total_size = float(len(data))
train_forest(forest, data)
print("Training Over")

main_function()