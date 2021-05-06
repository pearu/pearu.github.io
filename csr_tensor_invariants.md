# CSR tensor invariants in PyTorch

|            |                 |
| ---------- | --------------- |
| Author     | Pearu Peterson  |
| Created    | 2021-05-06      |

The aim of this blog post is to define the invariants of PyTorch tensors with CSR layout.

## CSR tensor members

A tensor with CSR layout has the following members (as defined by constructor `sparse_csr_tensor`):
- `crow_indices` contains the compressed row indices information
- `col_indices` contains column indices
- `values` contains the values of tensor elements
- `size` defines the shape of tensor
- `dtype` defines the dtype of tensor elements
- `layout` holds the layout parameter
- `device` holds the device of values storage
- `pin_memory` defines if cuda storage uses pinned memory

## Type invariants

1.1. `crow_indices.dtype == col_indices.dtype = indices_dtype` where `indices_dtype` is `int32` (default) or `int64`

1.2. `values.dtype == dtype` where `dtype` is `float32` (default), or `float64`, or `int8`, ..., or `int64`

## Layout invariants

2.1 `crow_indices.layout == col_indices.layout == values.layout == torch.strided`

2.2 `layout == torch.sparse_csr`

## Shape and strides invariants

3.1 `size == (nrows, ncols)`, that is, CSR tensor represents a 2 dimensional tensor

3.2 `crow_indices.dim() == col_indices.dim() == values.dim() == 1`

3.3 `crow_indices.stride() == col_indices.stride() == values.stride() == (1,)`, that is, all member tensors are 1D contiguous tensors

3.4 `crow_indices.size() == (nrows+1,)`

3.5 `col_indices.size() == values.size() == nnz`

3.6 `numel() == nrows * ncols` is the number of indexable elements [NOT IMPLEMENTED]

## Device invariants

4.1 `device` is `CPU` or `CUDA`

4.2 `crow_indices.device == col_indices.device == values.device == device`, that is, the storage devices of all member tensors are the same

## Indices invariants

5.1 `crow_indices[0] == 0`

5.2 `crow_indices[nrows] == nnz`

5.3 `crow_indices[i-1] <= crow_indices[i]` for all `i=1,...,nrows`

5.4 `0<=col_indices.min()`

5.5 `col_indices.max() < ncols`

## Invariant checks

According to [PR 57274](https://github.com/pytorch/pytorch/pull/57274), creating a CSR tensor has the following function calling tree with the corresponding invariant checks:

- `sparse_csr_tensor(crow_indices, col_indices, values, size)`
   - checks 4.1, 2.2
   - `check_csr_invariants(crow_indices, col_indinces, values)`
     - checks `crow_indices.numel() >= 1`, 3.2, 5.2, 5.1
   - checks 3.4
   - `new_csr_tensor`
     - checks ...
   - `resize_and_clear_`
     - checks ...
   - `set_member_tensors`
     - checks ...
    

