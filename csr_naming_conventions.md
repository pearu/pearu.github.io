# CSR format naming conventions

|            |                 |
| ---------- | --------------- |
| Author     | Pearu Peterson  |
| Created    | 2021-05-10      |

The aim of this blog post is to review naming conventions used in various software that implement CSR format support.

The CSR format, originating from mid-1960, was introduced to represent two-dimensional arrays (matrices) by three one-dimensional arrays:
- explicitly specified values, dimension is `nnz`
- extents of rows, dimension is `nrows + 1`
- column indices, dimension is `nnz`

where `nrows` denotes the number of array rows and `nnz` denotes the number of specified values.

Note: the notation `nnz` is an abreviaton from the "number of non-zero" elements. However, the "non-zero" part
should not be taken literally because nothing in the CSR format specification requires that the specified
values must be non-zero. The more appropiate term would be the "number of specified elements" (NSE) but
many software still use `nnz` while allowing explicit zero values.

The following table summarizes the CSR format naming conventions used in existing software as well as elsewhere (ordering is arbitrary):

| Software | NSE | values | extents of rows | column indices |
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
| [MathDotNet (C#)](https://numerics.mathdotnet.com/api/MathNet.Numerics.LinearAlgebra.Storage/SparseCompressedRowMatrixStorage%601.htm) | `ValueCount` | `Values` | `RowPointers` | `ColumnIndices` |
| [Stan Math Library (C++)](https://mc-stan.org/math/dc/d79/group__csr__format.html) | `NNZE` | `w` | `u` | `v` |
| [Magma Sparse (C)](https://icl.cs.utk.edu/projectsfiles/magma/doxygen/_m_a_g_m_a-sparse.html) | `nnz` | `val` | `row` | `col` |
| [https://arxiv.org/abs/1511.02494](https://arxiv.org/abs/1511.02494) | `N` | `val` | `rowptr` | `colind` |
| [Wikipedia Sparse Matrix](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) | `NNZ` | `V` | `ROW_INDEX` | `COL_INDEX` |
| [Sputnik](https://github.com/google-research/sputnik) | `nonzeros` | `values` | `row_offsets` | `column_indices` |

Notes:
- Using `NNZ` for the number of specified elements is dominant. Documentation of various software define it as the "number of non-zero" elements and in next sentense these may mention that explicit zero values are allowed. In summary, the usage of NNZ can be characterized as "the most consistently used inconsistency between the notation and the actual definition".
- Some software just don't care about the clarity of the used naming convention and appear to consider the naming convention as implementation detail: `ia`, `w`, etc
- Software that use `row` or `row_index` or `rowIndex` for naming the "extents of rows" array are not really good role models for choosing naming conventions (IMHO) because the given namings are misleading in the sense that the values of the "extents of rows" array are never the row indices (but are input parameters to row index generators).
- There is a dominant naming convenion for the "extents of rows" that is derived from the phrase "row/index pointers": `RowPtr`, `rowptr`, `row_ptr`, `indptr`. However, the usage of "pointers" may be confusing for C/C++ programmers because in C/C++ language the term is used as "a memory address of a variable".
- PyTorch currently uses `crow_indices` for the "extents of rows" and is derived from phrase "Compressed ROW INDICES". However, unwitting user may relate "crow" to a bird [Crow](https://en.wikipedia.org/wiki/Crow).

## Conclusions

The current choice of PyTorch naming convention is satisfactory (IMHO) but not ideal mainly because of `crow_indices` choice that is not used elsewhere and has birdish flavor. On the other hand, there appears to be no naming convention that would be ideal in general and therefore I think that PyTorch has a freedom as well as opportinity to introduce better naming convention from other software with respect to sparse tensor formats. The naming convention must be
- pythonic as most PyTorch users are Python users
- acceptable for C++ programs as PyTorch code base is C++ heavy
- accurate in the sense that naming will match with the actual definition/constraints of the notations, or at least, the naming choice should not be misleading.
