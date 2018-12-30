import numpy as np
import scipy.sparse as sp
import ctypes

slp = ctypes.cdll.LoadLibrary('slp.dll')

# returns recovered graph signal by using Sparse Label Propagation (SLP)

# W - sparse weight matrix of an undirected graph. elements must be in row-major order.
# indices, labels - indices of known graph nodes and their labels
# numIter - number of iteration of the SLP algorithm
def recoverSLP(W, indices, labels, numIter):
	
	# check input format

	if not isinstance(W, sp.coo_matrix): 
		raise TypeError("Weight matrix must be in sparse COO format, row-major order")

	if W.ndim != 2 or W.size == 0: 
		raise TypeError("Weight matrix must be non-empty 2D matrix")

	if W.shape[0] != W.shape[1]: 
		raise TypeError("Matrix must be square")

	if not isinstance(indices, np.ndarray): 
		raise TypeError("Indices must be numpy.array")

	if not isinstance(labels, np.ndarray): 
		raise TypeError("Labels must be numpy.array")

	if len(indices) != len(labels) or (indices.ndim != 1) or (labels.ndim != 1) or len(indices) == 0: 
		raise Exception("Indices and labels arrays must be nonempty 1D arrays of the same length")

	if len(indices) > W.shape[0]:
		raise Exception("Sample size should be smaller or equal to the signal length")

	if  np.min(indices) < 0 or np.max(indices) >= W.shape[0]: 
		raise ValueError("Sample indices are out of range")

	if not isinstance(numIter, int): 
		raise TypeError("numIter must be int")

	if numIter < 1: 
		return None

	# prepare for C++ processing, wrap pointers

	#
	out = np.zeros(shape=(W.shape[0],))
	val = W.data.astype('float64')

	#
	rowP = ctypes.c_void_p(W.row.ctypes.data)
	colP = ctypes.c_void_p(W.col.ctypes.data)
	valP = ctypes.c_void_p(val.ctypes.data)
	sparseLen = ctypes.c_int(len(W.row))

	#
	outP = ctypes.c_void_p(out.ctypes.data)
	numNodes = ctypes.c_int(W.shape[0])

	#
	indicesP = ctypes.c_void_p(indices.ctypes.data)
	labelsP = ctypes.c_void_p(labels.ctypes.data)
	numSamples = ctypes.c_int(len(labels))
	
	#
	numIter = ctypes.c_int(numIter)

	# run SLP
	slp.recoverSLP(rowP, colP, valP, sparseLen, outP, numNodes, indicesP, labelsP, numSamples, numIter)

	return out