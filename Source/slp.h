#pragma once

#ifdef BUILD_DLL
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __declspec(dllimport)
#endif


// This function defines Sparse Label Propagation (SLP) algorithm for graph signal recovery with large sparse matrices.
// The implementation follows the publication: https://arxiv.org/abs/1612.01414

// row,col,val - sparse weight matrix in COO format. must be row-major order.
// sparseSize - number of non-zero elements of the sparse matrix
// xhat - resulting recovered graph signal
// numNodes - total number of nodes in the graph. this is also length of xhat
// samplingIndices, samplingLabels - indices of nodes in the sampling set and their labels
// numSamples - number of samples in the sampling set
// numIter - number of iterations of the SLP algorithm

extern "C" EXPORT_API void recoverSLP(const unsigned int *row, const unsigned int *col, const double *val, unsigned int sparseSize, double *xhat, unsigned int numNodes, const unsigned int* samplingIndices, const double* samplingLabels, unsigned int numSamples, unsigned int numIter);
