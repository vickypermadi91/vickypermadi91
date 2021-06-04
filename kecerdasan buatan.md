from sklearn import tree
Parameter = [
		[170, 53, 32, 2,  95],
		[165, 51, 31, 14, 92],
		[180, 56, 33, 1,  101],
		[176, 55, 33, 2,  99],
		[182, 56, 34, 3,  102],
		[160, 64, 29, 15, 90],
		[154, 62, 28, 14, 86],
		[170, 68, 31, 16, 95],
		[155, 62, 28, 14, 87],
		[161, 64, 29, 15, 90]
	    ]
Gender = [ 	
		'laki-laki',
		'laki-laki',
		'laki-laki',
		'laki-laki',
		'laki-laki',
		'perempuan',
		'perempuan',
		'perempuan',
		'perempuan',
		'perempuan'
	]
#Memanggil metode Decision Tree dari Library 
metode = tree.DecisionTreeClassifier()
#Training data
metode = metode.fit(Parameter, Gender)
