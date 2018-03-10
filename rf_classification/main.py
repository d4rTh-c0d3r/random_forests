import csv
import math
import random

def get_data(name):
	data = []
	with open(name, 'r') as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
			for i in range(len(row)-1):
				row[i] = float(row[i])
			row[-1] = int(row[-1]) + shifting_factor
			data = data + [row]
	return data


def shannon_entropy(set_c):
	if(len(set_c) == 0):
		return 0
	set_info = [float(0)]*n_colors
	for row in set_c:
		set_info[row[-1]-1] =  set_info[row[-1]-1] + 1
	for i in range(n_colors):
		set_info[i] = set_info[i]/len(set_c)
	s_e = 0
	for num in set_info:
		if num != 0:
			s_e = s_e - num*math.log(num)
	return s_e

def information_gain(set_1, set_2):
	h_s = shannon_entropy(set_1 + set_2)
	hs1 = shannon_entropy(set_1)
	hs2 = shannon_entropy(set_2)
	l_1 = float(len(set_1))
	l_2 = float(len(set_2))
	i_g = h_s - ((l_1/(l_1 + l_2))*hs1 + (l_2/(l_1 + l_2))*hs2)
	return i_g

def get_prediction(data): # This is to give the leaf a value
	data_info = [0]*n_colors
	for row in data:
		data_info[row[-1]-1] = data_info[row[-1]-1] + 1
	max_n = 0
	max_i = -1 # Can't predict if data set is empty
	for i in range(n_colors):
		if data_info[i] > max_n:
			max_i = i
			max_n = data_info[i]
	return max_i + 1

def calculate_value(point, h):
	value = point[h[0][0]] + float(h[1][0])*point[h[0][1]] + float(h[1][1])
	return value

def split_data(data_train, h): # Splits with respect to a line in 2 Dimensions
	left = []
	right = []
	for point in data_train:
		if calculate_value(point, h) < 0:
			left = left + [point]
		else:
			right = right + [point]
	return left, right

def get_best_attribute(data_train, attributes):
	max_info = 0
	best_att = [[0,0],[0,0]]
	for att in attributes:
		left, right = split_data(data_train, att)
		i_g = information_gain(left, right)
		if i_g >= max_info:
			max_info = i_g
			best_att = att
	return best_att

def train_tree(tree, data_train, curr_index):
	if curr_index > len(tree) or len(data_train) == 0:
		return
	if len(data_train) < min_elems or 2*curr_index > len(tree):
		tree[curr_index-1][0] = get_prediction(data_train)
		return
	attributes = []
	for i in range(randomness):
		dim_1 = random.randint(0, len(data_train[0])-2)
		dim_2 = random.randint(0, len(data_train[0])-2)
		while dim_2 == dim_1:
			dim_2 = random.randint(0, len(data_train[0])-2)
		m = random.uniform(-1000, 1000)
		c = random.uniform(-1000, 1000)
		attributes = attributes + [[[dim_1, dim_2], [m, c]]]
	h_final = get_best_attribute(data_train, attributes)
	tree[curr_index-1][1] = h_final[0]
	tree[curr_index-1][2] = h_final[1]
	left_split, right_split = split_data(data_train, h_final)
	# print("At node " + str(curr_index) + " " + str(len(left_split)) + " " + str(len(right_split)))
	train_tree(tree, left_split, 2*curr_index)
	train_tree(tree, right_split, 2*curr_index+1)

def train_forest(forest, data_train):
	for i in range(len(forest)):
		train_tree(forest[i], data_train, 1)

def predict_tree(tree, point, curr_index):
	if tree[curr_index-1][1]  == [0,0]:
		return tree[curr_index-1][0]
	v = calculate_value(point, [tree[curr_index-1][1], tree[curr_index-1][2]])
	if v < 0:
		return predict_tree(tree, point, 2*curr_index)
	return predict_tree(tree, point, 2*curr_index + 1)

def predict(forest, point):
	data_forest = [0]*n_colors
	for tree in forest:
		predict_temp = predict_tree(tree, point,1)-1
		if predict_temp >= 0:
			data_forest[predict_temp] =  data_forest[predict_temp] + 1
	max_n = 0
	max_i = 0
	for i in range(n_colors):
		if data_forest[i] > max_n:
			max_n = data_forest[i]
			max_i = i
	return max_i + 1

def get_accuracy(forest, data_test):
	correct = 0
	for point in data_test:
		prediction = predict(forest, point[0:-1])

		##########################
		# temp = str(point[0])+","+str(point[1])+","+str(prediction-shifting_factor)+"\n"
		# with open('data_predicted.csv','a') as file:
		# 	file.write(temp)
		##########################

		print("Actual: " + str(point[-1]-shifting_factor) + " Predicted: " + str(prediction-shifting_factor))
		if  prediction == point[-1]:
			correct = correct + 1
	return float(correct)/float(len(data_test))

shifting_factor = 1 # Modify this according to the data lables. The smallest label should correspond to 1.
# Parameters
forest_size = 20
max_depth = 10
min_elems = 10
randomness = 20
n_colors = 2

data_train = get_data('data_train.csv')
data_test = get_data('data_test.csv')

forest = []
for i in range(forest_size):
	tree = []
	for j in range(2**(max_depth+1)-1):
		tree = tree + [[0, [0, 0], [0, 0]]]
	forest = forest + [tree]
train_forest(forest, data_train)
accuracy = get_accuracy(forest, data_test)
print(accuracy)

# with open('temp','a') as file:
# 	for k in range(1,21):
# 		max_depth = k
# 		forest = []
# 		for i in range(forest_size):
# 			tree = []
# 			for j in range(2**(max_depth+1)-1):
# 				tree = tree + [[0, [0, 0], [0, 0]]]
# 			forest = forest + [tree]
# 		train_forest(forest, data_train)
# 		accuracy = get_accuracy(forest, data_test)
# 		print(k,accuracy)
# 		file.write(str(k) + " " + str(accuracy) + "\n")	


