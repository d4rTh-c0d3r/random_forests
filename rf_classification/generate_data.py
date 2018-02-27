import numpy

means =[[4, 4, 4, 4, 4], [-4, -4, -4, -4, -4], [0,0,0,0,0]]
sigma = [1, 1, 1, 1, 1]
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