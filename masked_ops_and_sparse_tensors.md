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
corresponding mask value is ``True``, and for all unspecified elements
the mask value is ``False``.

To compute a masked reduction of a sparse input tensor, we first define using pseudo-code:
```python
input_mask = torch.empty(input.shape, dtype=input.dtype)  # this is strided tensor!
for index in itertools.product(*map(range, input.shape)):
    if index in mask and mask[index]:
        if index in input:
            input_mask[index] = input[index]
        else:
            input_mask[index] = 0  # follows from layout invariance of tensor operations
    else:
        input_mask[index] = reduction_op_identity  # masked-out elements are treated as reduction identities
```
The result of masked reduction operation is obtained by applying
regular reduction to ``input_mask``.

While the above defines the masked reduction on sparse tensors, it
would be impractical to implement because ``input_mask`` tensor is
strided tensor. Next we'll adjust the definintion of ``input_mask``
tensor such that it will be a sparse tensor and we could implement the
masked reduction operator on sparse tensors such that
```python
torch._masked.reduction_op(input, ..., mask=mask) == _sparse_masked_reduction_op(input_mask, ...)
```
holds.

First, let's analyze what should be the sparsity pattern of
``input_mask``: should it match with the sparsity pattern of
``input``, or of ``mask``, or something else?
For each element of the ``input_mask`` tensor, we have four
possibilities:
1. If ``input`` specifies an element and ``mask`` defines it as being
   masked-out, then the corresponding ``input_mask`` element value is
   ``reduction_op_identity``. [Here we have a choice of not specifying
   the corresponding element in ``input_mask``, or we can specify it by
   assigning reduction identity value to it.]
2. If ``input`` specifies an element and ``mask`` defines it as being
   masked-in, then the corresponding ``input_mask`` element
   value is the value of the input element.
3. If ``input`` does not specify a value and ``mask`` defines it as
   being masked-out, then the corresponding ``input_mask`` element
   will not be specified. [This case enables ``input_mask`` being a
   sparse tensor when ``input`` is sparse].
4. If ``input`` does not specify a value and ``mask`` defines it as
   being masked-in, then the corresponding ``input_mask`` element
   value is zero. [This case requires that ``input_mask`` must specify
   elements that are not specified in ``input`` (unless the reduction
   identity value is zero).]
   
So, the case 1 allows reducing the set of ``input_mask`` indices while
the case 3 may increase the set of indices with respect to the set of
``input`` indices. We don't consider the question with respect to the
set of ``mask`` indices because it would be impractical, for example,
when ``mask`` is the default mask then ``input_mask`` would become a
dense array.

To visualize the possible options for composing ``input_mask`` tensor,
let us consider the following 1-D sparse arrays:
```python
input = [*, *, *, x, y, z]
mask  = [*, F, T, *, F, T]
```
where ``*`` denotes unspecified values, ``F``/``T`` are the boolean
values defining the mask, and ``x``/``y``/``z`` are arbitrary values
of the input. In addition, let ``I`` denote the reduction identity
value. Then there exists four general constructions of the
``mask_input`` tensors:
```python
input_mask1 = [*, *, 0, *, *, z]
input_mask2 = [*, I, 0, *, I, z]
input_mask3 = [*, I, 0, I, I, z]
input_mask4 = [I, I, 0, I, I, z]
```
The first option ``input_mask1`` is minimal in the sense that all
specified values are masked-in and its sparsity pattern is defined by
``mask.to_dense().to_sparse()``.  The second option ``input_mask2``
has the same sparsity pattern as ``mask``.  The third option
``input_mask3`` has the sparsity pattern of the union of the sparsity
patterns of ``input`` and ``mask``.  Finally, the forth option
``input_mask4`` corresponds to dense array as defined in the
pseudo-code above.

For certain reductions that result can be quickly determined if one of
the operands has zero value, we can define reduction-dependent
constructions for ``input_mask``:
```python
input_mask5 = [*, *, 0, *, *, *]
input_mask6 = [*, *, *, *, *, z]
```
where ``input_mask5`` would be optimal for ``prod`` reduction, for
instance, where all input values can be discarded if unspecified input
value is masked-in.
The option ``input_mask6`` would be appropiate for reductions that
reduction identity is zero so that only masked-in and specified input
values determine the result, as is the case for ``sum``, for instance.





Let's assume that we have an implementation to masked reduction
operation for sparse input tensors that sparsity pattern matches with
the pattern of masked-in elements:
``_sparse_masked_reduction_op(sparse, ...)``.  Next we'll adjust the
definition of ``input_mask`` such that it will be a sparse tensor that
sparsity pattern matches with the sparsity pattern of input tensor
such that we would have:
```python
torch._masked.reduction_op(input, ..., mask=mask) == _sparse_masked_reduction_op(input_mask, ...)
```
where ``input`` is a sparse tensor and it uses the same layout as
``input_mask`` (to ensure that the input and result layouts will be
the same).

Let us assume that the indices set of ``input_mask`` tensor matches with
the indices set of ``input`` even when some elements would be masked
out by ``mask`` elements: the corresponding masked out elements will
be replaced by reduction identity values. On the other hand, when some
unspecified elements of ``input`` tensor are masked-in as defined by
the ``mask`` tensor, then the corresponding elements should be treated
as zeros. This will contradict with the assumption as the
``input_mask`` input would need to specify elements that are not
specified in the ``input`` tensor.

Next, let us assume that the indices set of ``input_mask`` tensor matches with the indices set of ``mask``: the specified elements of ``input``


An alternative would be that the indices set of ``input_mask`` matches
with the indices set of the corresponding ``mask`` elements that have
value ``True`` --- if so then masked-in elements that are not
specified in input would be replaced by zero within ``input_mask``
tensor.

Warning: At the moment, it is not completely clear what either of
these choices mean with respect to PyTorch autograd support.




This
indices set may or may not intersect with the indices set of ``input``.

If ``mask`` is a strided tensor then ``input_mask`` indices can be
obtained via ``normalized_mask = mask.to(dtype=bool).to_sparse()`` if input is sparse
COO tensor, or via ``normalized_mask = mask.to(dtype=bool).to_sparse_csr()`` if the
input is sparse CSR tensor.  The values of ``input_mask`` can be
computed as follows (unoptimized code follows):
```python
values = normalized_mask.values().zeros_like(dtype=input.values().dtype)
a = flatten_indices(normalized_mask.to_sparse().indices())
b = flatten_indices(input.to_sparse().indices())
for i, index in enumerate(b):
    if index in a:
        j = a.index(index)
        values[j] = input.values()[i]
```
As a result, we have:
```
input_mask = torch.sparse_coo_tensor(normalized_mask.indices(), values)
input_mask = torch.sparse_csr_tensor(normalized_mask.crow_indices(), normalized_mask.col_indices(), values)
```

If ``mask`` is already a sparse tensor, we'll need to take into
account possible masked-out elements:
```python
values = mask.values().zeros_like(dtype=input.values().dtype)
a = flatten_indices(mask.to_sparse().indices())
b = flatten_indices(input.to_sparse().indices())
for i, index in enumerate(b):
    if index in a:
        j = a.index(index)
        if mask[index]:
            values[j] = input.values()[i]
        else:
            values[j] = reduction_op_identity
```
