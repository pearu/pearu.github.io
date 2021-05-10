# CSR format notations

|            |                 |
| ---------- | --------------- |
| Author     | Pearu Peterson  |
| Created    | 2021-05-10      |

The aim of this blog post is to review the notations used in different software implementing CSR format support.

The CSR format, originating from mid-1960, was introduced to represent two-dimensional arrays (matrices) by three one-dimensional arrays:
- explicitly specified values, dimension is `nnz`
- extents of rows, dimension is `nrows + 1`
- column indices, dimension is `nnz`

where `nrows` denotes the number of array rows and `nnz` denotes the number of specified values.

Note: the notation `nnz` is an abreviaton of "number of non-zero" elements but this should not be
taken literally because nothing in the CSR format specification does not require that the specified
values must be non-zero. The more appropiate term is the "number of specified elements" with notation `nse` but
many software still use `nnz` while allowing explicit zero values.

The following table summarizes the CSR format notations implemented in existing software and as used in various papers

| Software | `nse` | values | extents of rows | column indices |
| -------- | ----- | ------ | --------------- | -------------- |
| [PyTorch (Python)](https://pytorch.org/docs/master/generated/torch.sparse_csr_tensor.html?highlight=csr#torch.sparse_csr_tensor) | `nnz` | `values` | `crow_indices` | `col_indices` |
| [scipy.sparse (Python)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) | `nnz` | `data` | `indptr` | `indices` |
| [PyData Sparse (Python)](https://sparse.pydata.org/en/stable/generated/sparse.GCXS.html) | `nnz` | `data` | `indptr` | `indices` |
| [cuSparse (C)](https://docs.nvidia.com/cuda/cusparse/index.html#csr-format) | `nnz` | `csrValA` | `csrRowPtrA` | `csrColIndA` |
| [Intel MKL solvers (C)](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/appendix-a-linear-solvers-basics/sparse-matrix-storage-formats/sparse-blas-csr-matrix-storage-format.html) | | `values` | `rowIndex` | `columns` | |
| [Intel MKL CSR format (C)](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/inspector-executor-sparse-blas-routines/matrix-manipulation-routines/mkl-sparse-create-csr.html) | | `values` | `rows_start`/`rows_end` |  `col_indx` |
| [GNU GSL (C)](https://www.gnu.org/software/gsl/doc/html/spmatrix.html) | `nnz` | `data` | `row_ptr` | `col` |
| [AOCL-sparse](https://github.com/amd/aocl-sparse) | `nnz` | `val` | `row_ptr` | `col_ind` |
| [SPARSEKIT (Fortran)](https://people.sc.fsu.edu/~jburkardt/f77_src/sparsekit/sparsekit.html) | `n` | `a` | `ia` | `ja` |
| [SparseM (R)](https://cran.r-project.org/web/packages/SparseM/vignettes/SparseM.pdf) | `nnz` | `ra` | `ia` | `ja` | 
| [https://arxiv.org/abs/1511.02494](https://arxiv.org/abs/1511.02494) | `N` | `val` | `rowptr` | `colind` |
| [Wikipedia Sparse Matrix](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) | `NNZ` | `V` | `ROW_INDEX` | `COL_INDEX` |

