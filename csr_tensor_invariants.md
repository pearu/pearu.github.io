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

1.1 `crow_indices.dtype == indices_dtype`

1.2 `col_indices.dtype = indices_dtype`

1.3 `indices_dtype` is `int32` (default) or `int64`

1.4 `values.dtype == dtype`

1.5 `dtype` is `float32` (default), or `float64`, or `int8`, ..., or `int64`

## Layout invariants

2.1 `crow_indices.layout == torch.strided`

2.2 `col_indices.layout == torch.strided`

2.3 `values.layout == torch.strided`

2.4 `layout == torch.sparse_csr`

## Shape and strides invariants

3.1 `size == (nrows, ncols)`, that is, CSR tensor represents a 2 dimensional tensor

3.2 `crow_indices.dim() == 1`

3.3 `col_indices.dim() == 1`

3.4 `values.dim() == 1`

3.5 `crow_indices.stride() == (1,)` or `crow_indices.is_contiguous()`

3.6 `col_indices.stride() == (1,)` or `col_indices.is_contiguous()`

3.7 `values.stride() == (1,)` or `values.is_contiguous()`

3.8 `crow_indices.size() == (nrows+1,)` or `crow_indices.numel() == nrows + 1`

3.9 `col_indices.size() == (nnz,)` or `col_indices.numel() == nnz`

3.10 `values.size() == (nnz,)` or `values.numel() == nnz`

3.11 `numel() == nrows * ncols` is the number of indexable elements [NOT IMPLEMENTED]

## Device invariants

4.1 `device` is `CPU` or `CUDA`

4.2 `crow_indices.device == device`

4.3 `col_indices.device == device`

4.4 `values.device == device`

## Indices invariants

5.1 `crow_indices[0] == 0`

5.2 `crow_indices[nrows] == nnz`

5.3 `crow_indices[i-1] <= crow_indices[i]` for all `i=1,...,nrows`

5.4 `0 <= col_indices.min()`

5.5 `col_indices.max() < ncols`

## Invariant checks

According to [PR 57274](https://github.com/pytorch/pytorch/pull/57274), creating a CSR tensor has the following function calling tree with the corresponding invariant checks:

- `sparse_csr_tensor(crow_indices, col_indices, values, size)`
   - checks 4.1, 2.2
   - `check_csr_invariants(crow_indices, col_indices, values)`
     - checks `crow_indices.numel() >= 1`, 3.2, 5.2, 5.1
   - checks 3.4
   - `new_csr_tensor`
     - checks ...
   - `resize_and_clear_`
     - checks ...
   - `set_member_tensors(crow_indices, col_indices, values)`
     - checks 1.1, 1.2, 1.3, 2.1, 3.3
