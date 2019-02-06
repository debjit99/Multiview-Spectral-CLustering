import numpy as np 
from scipy.linalg import block_diag

#----------------------------------
c = 1/2         						# only parameter 
#-----------------------------------


def sym(A):              				# making the matrix symetric 
	A = (A + A.T)/2
	return A

def getmatrix(m):         				# generates symetric matrix with row sum 1 the parameter m is size of the matrix

	M = np.zeros((m, m))

	rows = np.zeros((1,m))

	for i in range(0, m):
		
		s = 0
		for j in range(i, m):

			M[i][j] = np.abs(np.random.randn()) + 0.0001
			
			s = s + M[i][j]

		for j in range(i, m):
			M[i][j] = (M[i][j]/s)*(1 - rows[0][i])

		for j in range(i, m):
			M[j][i] = M[i][j]
			rows[0][j] = rows[0][j] + M[i][j] 

	return M

def blockmatrix(ar, k):                 #make block diagonal matrix acording to the sizes in ar and with k blocks

	M = getmatrix(ar[0])

	for i in range(1, k):
		N = getmatrix(ar[i])
		M = block_diag(M, N)

	return M

def keigen(M, k):						#k+1 th eigenvalue of the matrix M
	eigVals, eigVecs = np.linalg.eig(M)
	eigVals = np.sort(eigVals)

	if np.size(eigVals) >= k + 1:
		return eigVals[-(k + 1)]
	else:
		return 0

def perturbation(M, l, n):         		# adding perturbation of l to the matrix M with size n 

	for i in range(0, n):
		for j in range(0, n):
			if M[i][j] == 0:
				M[i][j] = l*(np.random.rand())

	return M


def generator(k , l):   				# k is the number of clusters and l is the largest cluster size 

	a = []

	for i in range(0, k):
		a.append(np.random.random_integers(1, 2*l))

	ar = np.array(a)
	n = np.sum(ar)

	M = blockmatrix(ar, k)

	while(keigen(M, k) >= c):
		M = blockmatrix(ar, k)

	U = perturbation(M, (c - keigen(M, k)/n), n)
	V = perturbation(M, (c - keigen(M, k)/n), n)
	return (U, V) 
