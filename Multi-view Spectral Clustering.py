import numpy as np
from Test_data import generator
from sklearn.cluster import KMeans

#-------------------------------------------------------

k = 100		 			  			# number of cluster centers 
it = 10					  			# number of time the co-traning iterates 

(U, V) = generator(k, 10)            # similarity matrix wrt 1 st view
                    				# similarity matrix wrt 2 nd view
n,m= np.shape(U)              		# number of points to cluster 


#---------------------------------------------------------


def laplacian(K):   		  # laplacian norm to normalize the similarity matix, K is a numpy array   

	row_sum = np.sum(K, axis = 1)

	#print(row_sum)
	row_sum = np.array(row_sum)
	row_sum = np.sqrt(row_sum)

	D = np.diag(row_sum)

	D = np.linalg.pinv(D)
	
	L = D.dot(K)
	L = L.dot(D)

	#print(L)
	return L


def svd_matrix(L):      	  #the first k eigenvector 
	
	eigVals, eigVecs = np.linalg.eig(L)
	eigVecs = -1.0 * eigVecs
	sortedEigInds = np.argsort(eigVals)
	U = eigVecs[:,sortedEigInds[-k:]]

	#print(U)
	return U


def sym(A):              	  # making the matrix symetric 
	A = (A + A.T)/2
	return A


def fmatrix(A): 			  # final martix to work on
	
	A = laplacian(A)
	A = svd_matrix(A)

	return A


def row_normalize(A):          # row normalized for clustering 

	row_sum = np.sum(A*A, axis = 1)
	row_sum = np.diag(row_sum)
	D = np.linalg.pinv(row_sum)	
	
	#print(D)
	
	A = D.dot(A*A)
	A = np.sqrt(A)
	return A


def iterate(A, B): 			  # A and B are the similarity matrix w.r.t view 1 and view 2

	U = fmatrix(A)
	V = fmatrix(B)

	for i in range(0, it):
		U = sym(U.dot(U.T.dot(A)))
		V = sym(V.dot(V.T.dot(B)))

		U = fmatrix(U)
		V = fmatrix(V)

	U = row_normalize(U)
	V = row_normalize(V)

	ans = row_normalize(U*V)

	return ans

def kmeans(A):                # k-means algorithm on the final normalized-similarity matrix
	
	km = KMeans(n_clusters = k, init = 'random').fit(A)

	M = np.full((n,n), 0)

	lab = km.labels_

	for i in range(0, n):
		for j in range(0, n):
			if lab[i] == lab[j]:
				M[i][j] = 1

	return M

S = iterate(U, V)

ans = kmeans(S)

print(ans)

