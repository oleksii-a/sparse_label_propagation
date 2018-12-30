#include <memory>
#include <cstring>
#include <cmath>
#include "slp.h"

// these helper functions have been found to be faster than the built-in implementations
template <class T>
inline T max(T a, T b)
{
	return a > b ? a : b;
}

template <class T>
inline T min(T a, T b)
{
	return a < b ? a : b;
}

// This function implements Sparse Label Propagation (SLP) algorithm for graph signal recovery with large sparse matrices.
// The implementation follows the publication: https://arxiv.org/abs/1612.01414

// row,col,val - sparse weight matrix in COO format. must be row-major order.
// sparseSize - number of non-zero elements of the sparse matrix
// xhat - resulting recovered graph signal
// numNodes - total number of nodes in the graph. this is also length of xhat
// samplingIndices, samplingLabels - indices of nodes in the sampling set and their labels
// numSamples - number of samples in the sampling set
// numIter - number of iterations of the SLP algorithm

void recoverSLP(const unsigned int *row, const unsigned int *col, const double *val, unsigned int sparseSize, double *xhat, unsigned int numNodes, const unsigned int* samplingIndices, const double* samplingLabels, unsigned int numSamples, unsigned int numIter)
{
	// some fundamental checks
	if (!sparseSize || !numNodes || !numSamples || !numIter || (numSamples >= numNodes)) return;

	// allocate buffers
	std::unique_ptr<double[]> xbuff(new double[numNodes]);
	std::unique_ptr<double[]> gamma(new double[numNodes]);
	std::unique_ptr<double[]> lambda(new double[sparseSize]);
	std::unique_ptr<double[]> y(new double[sparseSize]);

	// keeps starting indices of rows in the sparse matrix
	// (required for parallel processing)
	std::unique_ptr<int[]> indices(new int[numNodes]);

	// initialize
	memset(xhat, 0, numNodes * sizeof(double));
	memset(xbuff.get(), 0, numNodes * sizeof(double));
	memset(gamma.get(), 0, numNodes * sizeof(double));
	memset(y.get(), 0, sparseSize * sizeof(double));

	// precompute starting indices of rows in the sparse matrix
	// precompute gamma and lambda
	for (unsigned int l = 0, i = -1; l < sparseSize; l++)
	{
		if (row[l] != i)
		{
			++i = row[l];
			indices[i] = l;
		}
		lambda[l] = (double)1 / (2 * val[l]);
		gamma[row[l]] += val[l];
	}

	for (unsigned int i = 0; i < numNodes; i++)
		gamma[i] = 1 / gamma[i];

	// iterations of the SLP
	for (unsigned int k = 0; k < numIter; k++)
	{
		memcpy(xbuff.get(), xhat, numNodes * sizeof(double));

		#pragma omp parallel for
		for (unsigned int i = 0; i < numNodes; i++)
		{
			unsigned int lstart = indices[i];
			unsigned int lend = (i != numNodes - 1) ? indices[i + 1] : sparseSize;

			double sumPlus = 0, sumMinus = 0;
			for (unsigned int l = lstart; l < lend; l++)
			{
				unsigned int j = col[l];
				if (j > i) sumPlus += (val[l] * y[l]);
				else if (j < i) sumMinus += (val[l] * y[l]);
			}
			xhat[i] -= gamma[i] * (sumPlus - sumMinus);
			xbuff[i] = 2 * xhat[i] - xbuff[i];
		}

		// force nodes belonging to the sampling set to their true labels
		for (unsigned int i = 0; i < numSamples; i++)
		{
			unsigned int idx = samplingIndices[i];
			xbuff[idx] = xbuff[idx] - 2 * (xhat[idx] - samplingLabels[i]);
			xhat[idx] = samplingLabels[i];
		}

		#pragma omp parallel for
		for (unsigned int l = 0; l < sparseSize; l++)
		{
			unsigned int src = min(row[l], col[l]);
			unsigned int dst = max(row[l], col[l]);
			y[l] += lambda[l] * (xbuff[src] - xbuff[dst]);
			y[l] /= max((double)1, fabs(y[l]));
		}
	}
}