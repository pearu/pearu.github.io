---
author: Pearu Peterson
created: 2022-03-07
---

# PyTorch masked operations on sparse tensors

## Introduction

This blog post inspired by the need to define useful (masked) array
operations on PyTorch sparse tensors.


## Dense and sparse arrays

A multi-dimensional array is a data structure that defines a regular
structure to a set of values with the same data-type: each array value
is labeled with an index (a sequence of non-negative integers) so that
one can imagine the array values being distributed as the nodes of an
multi-dimensional regular grid.

When all nodes of the grid have assigned a value then this defines the
so-called *dense array*. Dense arrays are the most commonly used array
structures as these allow efficient storage of array elements as a
contiguous sequence of values while storing array indices is not
required (recall, an array element is a pair of index and the
corresponding value): the array indices can be computed from the
memory locations of the corresponding array values.  With dense
arrays, computationally very efficient algorithms can be implemented
that take advantage of the memory structure of the computational
processor unit(s) if it matches with the memory structure of arrays.

A generalization of dense arrays is the so-called *sparse array* that
is an array that may leave some grid nodes undefined (the
corresponding nodes have no value assigned). From this follows that in
addition to storing array values, the indices of specified array
elements must be stored as well. A number of storage formats exists
for storing the elements of sparse arrays. These formats are designed
to be memory and processor efficient for particular sparsity patterns
as well as for particular operations on such arrays.

It is important to notice that we have introduced the sparse array as
generalization of the dense array. On the other hand, sparse arrays
can also be viewed as dense arrays that have majority of elements with
the same value, say, zero, and the concept of a sparse array becomes a
certain compression method of the dense array elements.  In fact, this
view is very common in applications using linear algebra operations
when the number of zeros in arrays is an order of magnitute larger
than the number of non-zero values, and hence, for large sized arrays
a considerable portion of memory can be saved if one only stores the
non-zero elements of the array.


## Dense and sparse tensors in PyTorch

PyTorch currently defines the following tensor storage layouts:
- strided layout (``torch.strided``) is used in so-called *strided
  tensors* that represent dense array.  Strided tensors are
  particularly efficient in performing slicing operations as the array
  slice can be defined as a strides operation alone.
- sparse COO layout (``torch.sparse_coo``) is used in so-called
  *sparse tensors* that store the array elements a a pair of strided
  tensors: indices and values.
- sparse CSR layout (``torch.sparse_csr``) is similar to sparse COO
  layout but it uses certain compression for storing the row indices
  of array elements. Only 2-D tensors can use this layout.

According to the latest development, PyTorch considers a sparse tensor
layout as a compression method for storing dense tensor elements (here
we consider "dense" and "strided" as distinct concepts).  Of course,
the compression has positive effect only for dense tensors that
elements are mostly zeros.

All elementwise tensor operations defined in ``torch`` namespace
support sparse tensor inputs only if the corresponding operation maps
a zero value to zero. For this reason, ``torch.sin`` can be applied to
sparse tensors but ``torch.cos`` can't, for instance.

Historically, ``torch.sparse`` defines few operations (``sum``,
``softmax``, ``softmin``, etc) that treat unspecified elements as
masked-out elements. However, if one requires a consistent tool for
working with masked-out elements, then one should use operations
defined in the ``torch._masked`` namespace. The operations in
``torch._masked`` namespace use the same API as operations in
``torch`` namespace but the API is extended with an extra keyword
argument ``mask`` that contains a tensor that defines which values in
the input are masked-out while performing the operations. Note that
the input and mask arguments to the masked operations are tensors with
arbitrary layouts: masked operations are layout invariant and the
unspecified elements of sparse tensors are treated as zeros.


## Masked reductions on tensors with arbitrary storage layout

Consider a regular reduction ``torch.reduction_op``
(e.g. ``torch.sum``, ``torch.amax``, ``torch.argmin``, etc) that
supports strided tensor inputs. On the other hand, the corresponding
masked reduction ``torch._masked.reduction_op`` is designed to support
both strided and sparse inputs. In the following, we'll define masked
reductions on tensors with arbitrary storage layout using the PyTorch
design principle of layout invariance of operations and the definition
of existing operations on strided tensors.

For strided inputs, the regular and masked reductions
are related via
```python
torch._masked.reduction_op(strided, ..., mask=None) == torch.reduction_op(strided)
```
(here the equality ``x == y`` is defined as ``assertEqual(x,
y)``). We'll generalize this relation to inputs with arbitrary layout
as
```python
torch._masked.reduction_op(input, ..., mask=None).to_dense() == torch.reduction_op(input.to_dense()))
```
The behaviour of masked reduction when mask is not specified (the
``mask is None`` case), corresponds to masked reduction with so-called
*default mask* tensor. For strided inputs, the default mask is
```python
input.new_ones(input.shape, dtype=bool)
```
that generalizes for sparse inputs as
```python
torch.ones(input.shape, dtype=bool)
```

Masked reductions on strided tensors typically use the ``mask``
argument by converting the ``input`` argument to
```python
input_mask = torch.where(mask, input, reduction_op_identity)
```
and then apply regular reduction to ``input_mask``.

In the case of sparse tensor inputs, it is desirable that masked
reduction algorithms would operate on the sparse tensors data only,
that is, sparse tensors should not be materialized as strided tensors
because it would be very memory inefficient method for large sized
tensors. On the other hand, to allow optimizations, we want to provide
masked reduction implementations on sparse tensors that assume that
the masked-in pattern of input elements (as defined by the mask
tensor) would match with the sparsity pattern of the sparse input,
that is, for all specified elements of the sparse input the
corresponding mask value is ``True`` and for all unspecified elements
the mask value is ``False``.

To define a masked operation on sparse tensors, we defined so-called *normalized* mask:
```python
normalized_mask = mask.to_sparse().coalesce().to(dtype=bool, layout=mask.layout)
```

such that ``normalized_mask.values().all() == True``. Next, apply
normalized mask to input (pseudo-code follows, actual implementations
will vary):
```python
mask_input = torch.empty(input.shape, dtype=input.dtype)
for index in indices(normalized_mask):
    if index in indices(input):
        mask_input[index] = input[index]
    else:
        mask_input[index] = 0

for index not in indices(normalized_mask):
    mask_input[index] = reduction_op_identity
```
The result of masked reduction operation is obtained by applying
regular reduction to ``mask_input``.


