---
author: Pearu Peterson
created: 2022-04-22
---

# Compressed storage of variable length arrays and generalizations

## Introduction

This blog post inspired by the need to create and store [variable
length arrays](https://en.wikipedia.org/wiki/Variable-length_array) (varlen arrays, in short)
in a memory and processor efficient way. The problem is
raised from the HeavyAI project where variable length buffers
(tables of strings, arrays, etc) are stored using a contiguous buffer of values
accompanied with integer valued buffer containing a cumulative sum of
the lengths of varlen arrays. For example, the following
collection of varlen arrays ([a jagged array](https://en.wikipedia.org/wiki/Jagged_array))
```
[ 1, 2, 3], [], [4, 5], [6]
```
that have lengths 3, 0, 2, and 1, respectively, is stored as a pair of
contiguous buffers:
```
values = [1, 2, 3, 4, 5, 6]
compressed_indices = [0, 3, 3, 5, 6]
```

Notice that ``compressed_indices[i]`` gives the index of the first
item in the ``i``-th varlen array in the ``values`` buffer.

### Connection to sparse CSR format

The above representation of varlen array buffers is analogous to
storing a sparse matrix in a [Compressed Sparse Row
(CSR)](https://en.wikipedia.org/wiki/Sparse_matrix) format, that is,
the same pair of ``values`` and ``compressed_indices`` represents the
following sparse matrix:

```
1 2 3 * * *
* * * * * *
* * * 4 5 *
* * * * * 6
```

where we have specified ``column_indices=[0, 1, 2, 3, 4, 5]`` and
asterisks (``*``) denote unspecified elements. That is, existing
algorithms operating on sparse CSR matricies can be used on the
storage data structure of variable length arrays. For instance,
one-dimensional reduction operations on CSR matrices would correspond
to applying reductions operations to variable length arrays.

### Null varlen arrays

When storing varlen arrays in a DB engine such as HeavyDB, the storage
schema needs to support so-called null varlen arrays that correspond
to unspecified row values in a DB table (null varlen array is not the
same as the varlen array with zero size!). Introducing null varlen
array to a DB table will diverge the storage format from the sparse
CSR matrix format by introducing negative values to the array of
``compressed_indices``. The negative value indicates the presence of
null varlen array at the given index as well as information to compute
the step sizes to the next non-null varlen arrays.

For example, the following collection of varlen arrays
```
[ 1, 2, 3], NULL, [4, 5], [6]
```
contains null varlen array at the second position and it has the
following storage data:
```
values = [1, 2, 3, 4, 5, 6]
compressed_indices = [0, -4, 3, 5, 6]
```

where the index ``-4`` value indicates that the corresponding varlen
array is null varlen array and the next non-null varlen array starts
at the index ``-(-4 + 1) = 3`` of the ``values`` buffer.

### Statement of the problem

The examples above illustrate how to store varlen arrays using a pair
of ``values`` and ``compressed_indices`` buffers, and how null varlen
arrays can be represented in this storage format. Creating such a
storage of a collection of varlen arrays assumes that we know the
lengths of all varlen arrays in the collection. However, this
constraint does not allow creating such a storage of varlen arrays
when the varlen arrays are created at runtime (the lengths of arrays
are unknown) and possibly in random order.

The aim of this blog post is to discuss an efficient method for
creating a storage of varlen arrays in runtime that will support
random storage order of varlen arrays. For that, we'll extend the
above described strorage format as will be discussed below.

We also provide a Python prototype of the method in a module
[vla.py](vla.py). This module provides a Python class ``JaggedArray``
that implements the new storage format for demonstation. Here follows
a quick usage guide of the method:
```python
>>> from vla import JaggedArray
>>> a = JaggedArray.fromlist([[1, 2, 3], None, [4, 5], [6]])
>>> print(a)
JaggedArray[data=[1, 2, 3, 4, 5, 6], compressed_indices=[0, -4, 3, 5, 6], indices=[0, 1, 2, 3]]
```
where ``None`` represents the null varlen array and the details about
``indices`` will be given below.

## Random access storage format of varlen arrays

In the following, we propose a storage format for varlen arrays that
supports adding varlen arrays with arbitrary lengths to the storage in
random order. As an input to the method, we assume the total number of
varlen arrays is pre-specified as well as an upper bound of the number
of all array values is given. These assumptions are required for
pre-allocating the buffers for storing varlen arrays.

The specification of the storage format constist of:

- ``size`` - the total number of varlen arrays, pre-specified
- ``max_buffer_size`` - the upper bound to the number of values in all
  varlen arrays, pre-specified
- ``values`` - a buffer of arrays, pre-allocated with size
  ``max_buffer_size`` (here assuming ``sizeof(value_type) == 1`` for
  simplicity))
- ``compressed_indices`` - an integer buffer of cumsum on varlen array
  sizes, pre-allocated with size ``size + 1`` and filled with zeros
- ``indices`` - an integer buffer of indices in arbitrary
    order, pre-allocated with size ``size`` and filled
    with negative one (``-1``)
- ``array_count`` - an integer counting the number of specified varlen
  array, variable, initialized with zero
- ``setnull(index)`` - a method that specifies a null varlen array
  with the given index
- ``setitem(index, arr_values, arr_length)`` - a method that stores a non-null
  varlen array with the given index and values with lenght
- ``getitem(index) -> (arr_values, arr_length)`` - a method that returns the
  values and size of a specified varlen array with the given index.

The algorithms for the three methods are as follows (using pseudo-code
with Python syntax):

```python
def setnull(index):
    indices[array_count] = index
    ptr = compressed_indices[array_count]
    compressed_indices[array_count + 1] = ptr
    compressed_indices[array_count] = -(ptr + 1)
    array_count += 1

def setitem(index, arr_values, arr_length):
    indices[array_count] = index
    ptr = compressed_indices[array_count]
    compressed_indices[array_count + 1] = ptr + arr_length
    values[ptr:ptr+lenght] = arr_values
    array_count += 1

def getitem(index):
    storage_index = indices.index(index)
    ptr = compressed_indices[storage_index]
    if ptr < 0:
        return (None, None)  # represents null varlen array
    next_ptr = compressed_indices[storage_index + 1]
    if next_ptr < 0:
        arr_lenght = -(next_ptr + 1) - ptr
    else:
        arr_length = next_ptr - ptr
    arr_values = values[ptr:ptr + arr_lenght]
    return (arr_values, arr_length)
```

### Complexity and storage normalization

Notice that the complexity of ``setnull`` and ``setitem`` methods is
``O(1)`` while the complexity of ``getitem`` is ``O(size)`` because of
the ``indices.index`` call. If we can assume that varlen arrays are
added to the storage in a linear way, that is, ``indices ==
range(size)`` then the complexity of ``getitem`` can be reduced to
``O(1)``. Therefore, after creating a storage of varlen arrays with
possibly random order, it would be advantageous to normalize the
storage so that the indices of varlen arrays will be strictly ordered.

The naive normalization procedure would be as follows. Let ``arr`` be
a storage of varlen arrays and ``arr_normalized`` be the normalized
storage of the same set of varlen arrays computed as follows:
```python
for i in range(size):
    arr_normalized.setitem(i, arr.getitem(i))
```
that complexity would be ``O(size^2)``.

TODO: develop a more efficent normalization method that avoids
``.index`` calls.
