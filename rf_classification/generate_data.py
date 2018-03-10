# Generates gausians with means in the means vector and with sigma as in sigma vector

import numpy

means =[[4, 4, 4, 4, 4], [-4, -4, -4, -4, -4]]
sigma = [4, 4, 4, 4, 4]
n_1 = 100
n_2 = 20
j = 0

for mean in means:
	i = 0
	while i < n_1:
		p = numpy.random.normal(mean, sigma)
		s = ""
		for k in p:
			s = s + str(k) + ","
		with open('data_train.csv', 'a') as the_file:
			the_file.write(s+str(j)+"\n")
		i = i + 1
	i = 0
	while i < n_2:
		p = numpy.random.normal(mean, sigma)
		s = ""
		for k in p:
			s = s + str(k) + ","
		with open('data_test.csv', 'a') as the_file:
			the_file.write(s+str(j)+"\n")
		i = i + 1

	j = j + 1

# generates data in squares, inner square 0 and its border square 1

# import random
# import numpy

# n_1 = 500
# n_2 = 100
# with open('data_train.csv', 'a') as the_file:
# 	for _ in range(n_1):
# 		x = random.randint(-10,10)
# 		y = random.randint(-10,10)
# 		if abs(x) <= 7 and abs(y) <= 7:
# 			the_file.write(str(x)+","+str(y)+",0\n")
# 		else:
# 			the_file.write(str(x)+","+str(y)+",1\n")
# with open('data_test.csv', 'a') as the_file:
# 	for x in range(-10,11):
# 		for y in range(-10,11):
# 			if abs(x) <= 7 and abs(y) <= 7:
# 				the_file.write(str(x)+","+str(y)+",0\n")
# 			else:
# 				the_file.write(str(x)+","+str(y)+",1\n")

