---
author: Pearu Peterson
created: 2022-04-22
---

# Compressed storage of variable length arrays and generalization to random storage order

## Introduction

This blog post inspired by the need to create and store [variable
length arrays](https://en.wikipedia.org/wiki/Variable-length_array) (varlen arrays, in short)
in a memory and processor efficient way. The problem is
raised from the [HeavyAI project](https://www.heavy.ai/) where variable length buffers
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
asterisk (``*``) denotes unspecified elements. This connection allows
to use existing algorithms operating on sparse CSR matrices also on
the storage data structure of variable length arrays. For instance,
one-dimensional reduction operations on CSR matrices would correspond
to applying reductions operations to varlen arrays.

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
array is null varlen array (``NULL``) and the next non-null varlen
array starts at the index ``-(-4 + 1) = 3`` of the ``values`` buffer.

### Statement of the problem

The examples above illustrate how to store varlen arrays using a pair
of ``values`` and ``compressed_indices`` buffers, and how null varlen
arrays can be represented in this storage format. Creating such a
storage of a collection of varlen arrays assumes that we know the
lengths of all varlen arrays prior creating the storage stucture and
the varlen arrays are stored in subsequent order. If the lenghts of
varlen arrays are to be determined at runtime, or the order of
inserting the varlen arrays to the storage needs to be arbitrary, the
above described storage format will not be applicable.

The aim of this blog post is to discuss an efficient method for
creating a storage of varlen arrays in runtime that will support
random storage order of varlen arrays. For that, we shall extend the
above described strorage format with an additional integer buffer that
records the ordering of stored arrays.

We also provide a Python prototype of the method in a module
[vla.py](vla.py). This module provides a Python class ``JaggedArray``
that implements the extended storage format for demonstation. Here
follows a quick usage guide of the module:
```python
>>> from vla import JaggedArray
>>> a = JaggedArray.fromlist([[1, 2, 3], None, [4, 5], [6]])
>>> print(a)
JaggedArray[data=[1, 2, 3, 4, 5, 6], compressed_indices=[0, -4, 3, 5, 6], storage_indices=[0, 1, 2, 3]]
```
where ``None`` represents a null varlen array and the details about
``storage_indices`` will be given below.

## Random access storage format of varlen arrays

In the following, we propose a storage format for varlen arrays that
supports storing varlen arrays with arbitrary lengths in arbitrary
order. As an input to the method, we assume the total number of varlen
arrays is pre-specified as well as an upper bound of the number of all
array values is given. These assumptions are required for
pre-allocating the buffers for storing varlen arrays.

The specification of the storage format of varlen arrays constist of
the following attibutes and methods:

- ``size`` - the total number of varlen arrays, pre-specified
- ``max_buffer_size`` - the upper bound to the number of values in all
  varlen arrays, pre-specified
- ``values`` - a buffer of arrays, pre-allocated with size
  ``max_buffer_size`` (here assuming ``sizeof(value_type) == 1`` for
  simplicity))
- ``compressed_indices`` - an integer buffer of cumsum on varlen array
  sizes, pre-allocated with size ``size + 1`` and filled with zeros
- ``storage_indices`` - an integer buffer of storage indices,
  pre-allocated with size ``size`` and filled with negative one
  (``-1``). Unless ``-1``, the value of ``storage_index[index] + 1``
  gives the order of storing varlen array with the given index.
- ``storage_count`` - an integer counting the number of specified varlen
  array, variable, initialized with zero.
- ``setnull(index)`` - a method that specifies a null varlen array
  with the given index
- ``setitem(index, arr_values, arr_length)`` - a method that stores a non-null
  varlen array with the given index and values with lenght
- ``getitem(index) -> (arr_values, arr_length)`` - a method that returns the
  values and size of a specified varlen array with the given index.

The algorithms for the methods ``setnull``, ``setitem``, and
``getitem`` are as follows (using pseudo-code with Python syntax):

```python
def setnull(index):
    storage_indices[index] = storage_count
    ptr = compressed_indices[storage_count]
    compressed_indices[storage_count + 1] = ptr
    compressed_indices[storage_count] = -(ptr + 1)
    storage_count += 1

def setitem(index, arr_values, arr_length):
    storage_indices[index] = storage_count
    ptr = compressed_indices[storage_count]
    compressed_indices[storage_count + 1] = ptr + arr_length
    values[ptr:ptr+lenght] = arr_values
    storage_count += 1

def getitem(index):
    storage_index = storage_indices[index]
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

### Complexity

The computational complexity of all the three methods is ``O(1)``.

## Example

To demonstrate the storage format of varlen arrays in action, let us
consider the following collection of varlen arrays
```
[ 1, 2, 3], NULL, [4, 5], [6]
```
that are stored in the jagged array in a non-linear fashion, say,
first store the 3rd array, second store the 2nd array, then the last
array, and finally, the 1st array:
```python
>>> a = JaggedArray(4)
>>> print(a)
JaggedArray[values=[], compressed_indices=[0, 0, 0, 0, 0], storage_indices=[-1, -1, -1, -1]]
>>> a[2] = [4, 5]
>>> print(a)
JaggedArray[values=[4, 5], compressed_indices=[0, 2, 0, 0, 0], storage_indices=[-1, -1, 0, -1]]
>>> a[1] = None  # null varlen array
>>> print(a)
JaggedArray[values=[4, 5], compressed_indices=[0, -3, 2, 0, 0], storage_indices=[-1, 1, 0, -1]]
>>> a[3] = [6]
>>> print(a)
JaggedArray[values=[4, 5, 6], compressed_indices=[0, -3, 2, 3, 0], storage_indices=[-1, 1, 0, 2]]
>>> a[0] = [1, 2, 3]
>>> print(a)
JaggedArray[values=[4, 5, 6, 1, 2, 3], compressed_indices=[0, -3, 2, 3, 6], storage_indices=[3, 1, 0, 2]]
```

We can eliminate the need for using ``storage_indices`` if varlen
arrays are stored in order. After the jagged array is constructed, we
can normalize the storage such that the storage indices become sorted:
```python
>>> print(a.normalize())
JaggedArray[values=[1, 2, 3, 4, 5, 6], compressed_indices=[0, -4, 3, 5, 6], storage_indices=[0, 1, 2, 3]]
```

Notice that normalization will also update the values and compressed
indices buffers. Since the normalization procedure is essentially a
piece-wise copy of the original array buffers, it has computational
complexity of ``O(size)``.
