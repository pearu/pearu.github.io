# BSR tensor invariants in PyTorch

|            |                 |
| ---------- | --------------- |
| Author     | Pearu Peterson  |
| Created    | 2022-05-06      |

The aim of this blog post is to define the invariants of PyTorch
tensors with BSR layout. As a first instance, we assume that a BSR
tensor has zero batch dimensions and is a non-hybrid tensor (tensor
values are scalars), and then generalize this to batch hybrid BSR
tensor.

## Sparse tensor BSR format

A sparse tensor BSR format is a generalization of a [sparse tensor CSR
format](csr_tensor_invariants.html) where BSR indices refer to
2-dimensional dense blocks in the tensor rather than elements as in
the case of a CSR tensor. For example:
```python
>>> dense = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]])
>>> bsr = dense.to_sparse_bsr(blocksize=(2, 3))
>>> bsr
tensor(crow_indices=tensor([0, 2, 4]),
       col_indices=tensor([0, 1, 0, 1]),
       values=tensor([[[ 0,  1,  2],
                       [ 6,  7,  8]],

                      [[ 3,  4,  5],
                       [ 9, 10, 11]],

                      [[12, 13, 14],
                       [18, 19, 20]],

                      [[15, 16, 17],
                       [21, 22, 23]]]), size=(4, 6), nnz=4,
       layout=torch.sparse_bsr)
```
Notice that `crow_indices` and `col_indices` contain indices of a CSR
tensor with shape `(2, 2)` while the shape of BSR tensor is `(4, 6)`
because it incorporates the shape of `values` blocks: a block is a
strided tensor with shape `(2, 3)`.

The sparse tensor CSR format is equivalent to sparse tensor BSR format
with `blocksize=(1, 1)`.

### BSR tensor members

A tensor with BSR layout has the following members (as defined by constructor `sparse_bsr_tensor`):
- `crow_indices` contains the compressed block row indices information
- `col_indices` contains block column indices
- `values` contains the values of tensor elements organized in a 3-dimensional array of blocks
- `size` defines the shape of tensor
- `dtype` defines the dtype of tensor elements
- `layout` holds the layout parameter
- `device` holds the device of values storage
- `pin_memory` defines if cuda storage uses pinned memory

### Type invariants

1.1 `crow_indices.dtype == indices_dtype`

1.2 `col_indices.dtype == indices_dtype`

1.3 `indices_dtype` is `int32` or `int64` (default)

1.4 `values.dtype == dtype`

1.5 `dtype` is `float32` (default if `values` contains floats),
    `float64`, `int8`, `int16`, `int32`, `int64` (default if `values`
    contains ints only), `bool` (default if `values` contains only
    boolean values), `complex32` (unsupported ATM), `complex64`
    (default if `values` contains complex numbers), or `complex128`.

### Layout invariants

2.1 `crow_indices.layout == torch.strided`

2.2 `col_indices.layout == torch.strided`

2.3 `values.layout == torch.strided`

2.4 `layout == torch.sparse_bsr`

### Shape and strides invariants

3.1 `size == (nrows * blocksize[0], ncols * blocksize[1])`, that is, a
    BSR tensor represents a two dimensional tensor.

3.2 `crow_indices.dim() == 1`

3.3 `col_indices.dim() == 1`

3.4 `values.dim() == 3`

3.5 `crow_indices.is_contiguous()`

3.6 `col_indices.is_contiguous()`

3.7 `values.is_contiguous()` or `values.transpose(-2, -1).is_contiguous()`

3.8 `crow_indices.shape == (nrows + 1,)`

3.9 `col_indices.shape == (nnz,)`

3.10 `values.shape == (nnz, blocksize[0], blocksize[1])`

3.11 `numel() == nrows * ncols * blocksize[0] * blocksize[1]` is the number of indexable elements (note: indexing CSC, BSR, and BSC tensors is not yet supported)

### Device invariants

4.1 `device` is `CPU` or `CUDA`

4.2 `crow_indices.device == device`

4.3 `col_indices.device == device`

4.4 `values.device == device`

### Indices invariants

5.1 `crow_indices[0] == 0`

5.2 `crow_indices[nrows] == nnz`

5.3 `0 <= crow_indices[i] - crow_indices[i-1] <= ncols` for all `i=1,...,nrows`

5.4 `0 <= col_indices.min()`

5.5 `col_indices.max() < ncols`

5.6 `col_indices[crow_indices[..., i-1]:crow_indices[..., i]]`
    must be sorted and with distinct values for all `i=1,...,nrows`,
    that is, CSR tensor is coalesced. This is required by
    [cuSparse](https://docs.nvidia.com/cuda/cusparse/index.html#csr-format)
    and SciPy calls such CSR tensors being in [canonical
    form](https://github.com/scipy/scipy/blob/8a64c938ddf1ae4c02a08d2c5e38daeb8d061d38/scipy/sparse/sparsetools/csr.h#L313-L324).


## BSR transpose as a view operation

CSR tensor transpose returns a CSC tensor so that CSR transpose
becomes a view operations with respect to `values` member. This is
achieved by reinterpreting the compressed and plain indices as column
and row indices, respectively. The resulting CSC tensor shared the
storage of indices and values of the original CSR tensor:
```python
>>> csr = torch.tensor([[0, 1], [2, 3]]).to_sparse_csr()
>>> csc = csr.transpose(-2, -1)
>>> csc.values().storage().data_ptr() == csr.values().storage().data_ptr()
True
>>> csc.ccol_indices().storage().data_ptr() == csr.crow_indices().storage().data_ptr()
True
```

Similarly, we require that BSR transpose is a view operation by
reinterpreting the compressed and plain indices. While doing so, we
must take into account the following invariant:
```
dense.to_sparse_bsr(blocksize=(bdim1, bdim2)).transpose(-2, -1) == dense.transpose(-2, -1).to_sparse_bsc(blocksize=(bdim2, bdim1))
```
that is, the conversions from dense to blocked tensors and transpose
opetation must commute.

Notice that the dimensions of blocksizes of BSR and BSC tensors are
swapped. This invariant is satisfied if BSR transpose is defined as
```
bsr.transpose(-2, -1) = torch.sparse_bsc_tensor(bsr.crow_indices(), bsr.col_indices(), bsr.values().transpose(-2, -1))
```

At the moment of writing this blog, `torch.sparse_bsc_tensor` requires
row-wise contiguous `values` input but `bsr.values().transpose(-2,
-1)` is a column-wise contiguous tensor. Here follows are workaround:
```python
>>> dense = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]])
>>> bsr = dense.to_sparse_bsr(blocksize=(2, 3))
>>> bsc = torch._sparse_bsc_tensor_unsafe(bsr.crow_indices(), bsr.col_indices(), bsr.values().transpose(-2, -1), size=(bsr.shape[-1], bsr.shape[-2]))
>>> bsc
tensor(ccol_indices=tensor([0, 2, 4]),
       row_indices=tensor([0, 1, 0, 1]),
       values=tensor([[[ 0,  6],
                       [ 1,  7],
                       [ 2,  8]],

                      [[ 3,  9],
                       [ 4, 10],
                       [ 5, 11]],

                      [[12, 18],
                       [13, 19],
                       [14, 20]],

                      [[15, 21],
                       [16, 22],
                       [17, 23]]]), size=(6, 4), nnz=4,
       layout=torch.sparse_bsc)
```
The above means that we must generalize the row-wise contiguous input
requirement to row- or column-wise contiguous input requirement of
sparse blocked tensors.

## Generalization of BSR format to batched and hybrid BSR formats.

While a canonical BSR tensor is a two dimensional tensor with a shape
`(nrows * blocksize[0], ncols * blocksize[1])` then a batched BSR
tensor introduces so-called batch dimensions that appear in the left
of the canonical dimensions. Moreover, a hybrid BSR tensor introduces
so-called dense dimensions that appear in the right of the canonical
dimensions. So, the shape of a batched hybrid BSR tensor is
```python
(batchsize[0], ..., batchsize[M-1], nrows * blocksize[0], ncols * blocksize[1], densesize[0], ..., densesize[N-1])
```
where `M` is the number of batch dimensions and `N` is the number of
dense dimensions. The batch dimensions extend the BSR indices and
values shapes from the left. This means that different batches of BSR
tensors may have different sparsity patterns as long as the number of
specified elements is the same for all batches. The dense dimensions
extend the BSR values shape from the right while the shapes of BSR
indices are unaffected. This means that hybrid BSR tensors can be
interpreted as tensor-valued tensors as is the case with hybrid COO
tensors.

The batched and hybrid BSR tensors have the following invariants:

### Type, layout, and device invariants

Same as 1.1-1.5, 2.1-2.4, and 4.1-4.4 above.

### Shape and strides invariants

3.1 `size == batchsize + (nrows * blocksize[0], ncols *
    blocksize[1]) + densesize` where `batchsize` and `densesize` are
    tuples of non-negative integers.

3.2 `crow_indices.dim() == 1 + M`

3.3 `col_indices.dim() == 1 + M`

3.4 `values.dim() == M + 3 + N`

3.5 `crow_indices.is_contiguous()`

3.6 `col_indices.is_contiguous()`

3.7 `values.is_contiguous()` or `values.transpose(-2 - N, -1 - N).is_contiguous()` (see BSR transpose section below)

3.8 `crow_indices.shape == batchsize + (nrows + 1,)`

3.9 `col_indices.shape == batchsize + (nnz,)`

3.10 `values.shape == batchsize + (nnz,) + blocksize + densesize`

3.11 `numel() == nrows * ncols * prod(batchsize + blocksize + densesize)` is the number of indexable elements

### Indices invariants

5.1 `crow_indices[..., 0] == 0`

5.2 `crow_indices[..., nrows] == nnz`

5.3 `0 <= crow_indices[..., i] - crow_indices[..., i-1] <= ncols` for all `i=1,...,nrows`

5.4 `0 <= col_indices.min()`

5.5 `col_indices.max() < ncols`

5.6 `col_indices[..., crow_indices[..., i-1]:crow_indices[..., i]]`
    must be sorted and with distinct values for all `i=1,...,nrows`,
    that is, each batch is coalesced.

In the above, ellipses `...` reads `i1, ..., iM` and the invariants
apply for all `0 <= i1 < batchsize[0], ..., 0 <= iM < batchsize[M-1]`.

### Estimating the shape of a BSR tensor from the shapes of indices and values

PyTorch supports two factory functions for creating sparse (COO, CSR,
CSC, BSR, and BSC) tensors (i) from indices, values, and shape, and
(ii) from indices and values while the tensor shape is estimated from
the shapes of indices and values. In the following, a solution to the
shape estimation problem is provided for a BSR tensor.

The general form of a BSR tensor shape is
```
batchsize + (nrows * blocksize[0], ncols * blocksize[1]) + densesize
```

From 3.8, 3.9, and 3.10 follows that
```python
batchsize = crow_indices.shape[:-1]
M = len(batchsize)
blocksize = values.shape[M + 1: M + 3]
densesize = values.shape[M + 3:]
nrows = crow_indices.shape[-1] - 1
ncols = max(col_indices.max() - 1, crow_indices.diff(dim=-1).max())
```
