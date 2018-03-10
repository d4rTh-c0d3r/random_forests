# Gaussian around means

import numpy.random as rm
import numpy

means = [[-5,-5],[5,5],[-5,5],[5,-5]] # Can give as many as you want
# cov_s = [[[1,0],[0,1]],[[1,0],[0,1]]] # Corrsponding covariance matrices
cov_s = [[1,1],[1,1],[1,1],[1,1]] # Comment this out and uncomment above to use actual multivariate normal
num_points = 100 # Number of points in each gaussian

for i in range(len(means)):
	mean = means[i]
	cova = cov_s[i]
	with open('data_ul.csv', 'a') as the_file:
		for _ in range(num_points):
			# random_pt = rm.multivariate_normal(mean,cova) # Use this for actual multivariate normal
			random_pt = numpy.random.normal(mean, cova) # Temporary fix as above function doesn't work in my PC for some reason
			pt = ""
			for val in random_pt[:-1]:
				pt=pt+str(val)+","
			pt = pt+str(random_pt[-1])+"\n"
			the_file.write(pt)

# Uniform Distribution centered around vector(0)

# import numpy.random as rm

# maxes = [[-10,10],[-10,10]]
# n = 200 # Number of Points

# with open('data_ul.csv','a') as file:
# 	for _ in range(n):
# 		temp = ""
# 		for m in maxes[:-1]:
# 			p = rm.uniform(m[0],m[1])
# 			temp = temp + str(p) + ","
# 		p = rm.uniform(maxes[-1][0],maxes[-1][1])
# 		temp = temp + str(p) + "\n"
# 		file.write(temp)