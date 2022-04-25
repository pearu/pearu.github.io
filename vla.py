"""A prototype of storing Variable Lenght Arrays (VLAs).

A variable lenght array (varlen array) is an array of the same type of
elements while the number of elements is not fixed and determined at
the time of storing a varlen array. This Python module provides a
storage model of a jagged array (an array of varlen arrays) that uses
a single buffer of values for storing all the elements of all varlen
arrays. In addition, the storage model is required to support
so-called null varlen arrays and storing varlen arrays in arbitrary
order to the jagged array.

See https://pearu.github.io/variable_length_arrays.html for more
information.

"""
# Author: Pearu Peterson
# Created: April 16, 2022


import itertools


class Unspecified(object):
    """Represents a value of unspecified item.
    """
    def __str__(self):
        return 'Unspecified'
    __repr__ = __str__


UNSPECIFIED = Unspecified()


class JaggedArray:
    """Jagged array is an array of varlen arrays.
    """

    def __init__(self, size, max_buffer_size=100):
        """
        Parameters
        ----------
        size : int
          The number of var-length arrays.
        max_buffer_size : int
          The size of values buffer.
        """
        # The pre-allocated size of values buffer
        self.max_buffer_size = max_buffer_size

        # The total number of var-length arrays in the jagged array
        self.size = size

        # If no null var-length array is specified then
        # compressed_indices is the cumsum of var-lenght array sizes:
        #
        #   self.compressed_indices[i+1] - self.compressed_indices[i]
        #
        # is the size of var-length array that storage index is i.
        # The last element is the number of values in all specified
        # var-length arrays. In addition,
        #
        #   self.compressed_indices[i]
        #
        # defines that the buffer location of the first var-length
        # array element.
        #
        # Null var-length arrays take no buffer storage space and is
        # defined by negative value of self.compressed_indices[i] such that
        #
        #  -(self.compressed_indices[i] + 1) - self.compressed_indices[i - 1]
        #
        # is the length of previous non-null var-length array.
        #
        self.compressed_indices = [0] * (size + 1)

        # The total number of specified values in all var-length arrays
        #
        #   0 <= self.storage_count <= size
        #
        #  self.storage_count is the storage index of the next var-length array
        self.storage_count = 0

        # The storage indices map of varlen arrays:
        #
        #   self.storage_indices[index]
        #
        # is the storage index of varlen array with the given index.
        self.storage_indices = [-1] * size

        # The storage of the values of all var-length array elements
        self.values = [0] * max_buffer_size

    def __str__(self):
        return (f'{type(self).__name__}[values={self.values[:self.compressed_indices[self.storage_count]]},'
                f' compressed_indices={self.compressed_indices}, storage_indices={self.storage_indices}]')

    def __repr__(self):
        return f'{type(self).__name__}.fromlist({self.tolist()})'

    def _unsafe_setnull(self, index):
        storage_index = self.storage_count
        self.storage_indices[index] = storage_index
        ptr = self.compressed_indices[storage_index]
        self.compressed_indices[storage_index + 1] = ptr
        self.compressed_indices[storage_index] = -(ptr + 1)
        self.storage_count += 1

    def _unsafe_setitem(self, index, values):
        size = len(values)
        storage_index = self.storage_count
        self.storage_indices[index] = storage_index
        ptr = self.compressed_indices[storage_index]
        assert ptr + size <= len(self.values)
        self.compressed_indices[storage_index + 1] = ptr + size
        self.values[ptr:ptr + size] = values
        self.storage_count += 1

    def _unsafe_getitem(self, index):
        storage_index = self.storage_indices[index]
        if storage_index == -1:
            return UNSPECIFIED
        ptr = self.compressed_indices[storage_index]
        if ptr < 0:
            return None  # null varlen array
        next_ptr = self.compressed_indices[storage_index + 1]
        if next_ptr < 0:
            size = -(next_ptr + 1) - ptr
        else:
            size = next_ptr - ptr
        return self.values[ptr:ptr + size]

    def setnull(self, index):
        self[index] = None

    def __setitem__(self, index, varlenarray):
        """Insert a new var-length array as the index-th element of jagged
        array.

        Parameters
        ----------
        index : int
          The index of var-lenght array.
        varlenarray : {VarLengthArray, list, None}
          A var-lenght array or a list of var-length array elements or
          a null var-length array represented by None value.
        """
        if index < 0:
            index += self.size
        assert self.storage_indices[index] == -1  # varlen array can be inserted exactly once
        assert 0 <= index and index < self.size

        if varlenarray is None:
            self._unsafe_setnull(index)
            return
        if isinstance(varlenarray, list):
            size = len(varlenarray)
            values = varlenarray
        elif isinstance(varlenarray, VarLengthArray):
            size = varlenarray.size
            values = varlenarray.values
        elif varlenarray is UNSPECIFIED:
            return
        else:
            raise TypeError(type(varlenarray))

        self._unsafe_setitem(index, values)

    def __getitem__(self, index):
        if index < 0:
            index += self.size
        assert 0 <= index and index < self.size
        values = self._unsafe_getitem(index)
        return values
        if values is None or values is UNSPECIFIED:
            return values
        return VarLengthArray(values, len(values))

    def tolist(self):
        """Return jagged array as a list of lists.
        """
        return [self[index] for index in range(self.size)]

    @classmethod
    def fromlist(cls, lst):
        """Construct jagged array from a list of lists or None values.

        None values are interpreted as null varlen arrays.
        """
        buffer_size = sum(len(arr) for arr in lst if isinstance(arr, list))
        jarr = cls(len(lst), max_buffer_size=buffer_size)
        for i, arr in enumerate(lst):
            jarr[i] = arr
        return jarr

    def finalize(self):
        """
        Set unspecified elements as null var-length arrays.
        """
        for index, storage_index in enumerate(self.storage_indices):
            if storage_index == -1:
                self._unsafe_setnull(index)
        return self

    def normalize(self, unspecified_is_null=True):
        """Return a copy of jagged array with sorted storage indices.
        """
        jarr = type(self)(self.size, max_buffer_size=self.compressed_indices[self.storage_count])
        for index, storage_index in enumerate(self.storage_indices):
            if storage_index == -1:
                if unspecified_is_null:
                    jarr._unsafe_setnull(index)
            else:
                values = self._unsafe_getitem(index)
                if values is None:
                    jarr._unsafe_setnull(index)
                else:
                    jarr._unsafe_setitem(index, values)
        return jarr


def test():
    jarr = JaggedArray(5)
    print(jarr.tolist())

    #jarr[2] = [1, 2, 3]
    jarr[1] = []
    #jarr[0] = None
    jarr[0] = [7, 8, 9, 10]
    jarr[-1] = None #[11]
    jarr[-2] = [5, 6]
    print(jarr.tolist())
    print(jarr, jarr.tolist())
    print(jarr.normalize(), jarr.normalize().tolist())
    print(jarr.finalize().tolist())
    jarr = JaggedArray.fromlist([[1,2], None, [3, 4, 5]])
    print(jarr)


def test_full():
    items = [[1, 2], [], [3], None, UNSPECIFIED]
    for size in range(0, len(items)):
        for inp in itertools.product(items, repeat=size):
            inp = list(inp)
            for perm in itertools.permutations(list(range(size))):
                jarr = JaggedArray(size)
                for i in perm:
                    if inp[i] is UNSPECIFIED:
                        continue
                    jarr[i] = inp[i]
                assert jarr.tolist() == inp
                narr = jarr.normalize(unspecified_is_null=False)
                assert narr.tolist() == inp
                narr = jarr.normalize(unspecified_is_null=True)
                assert narr.tolist() == [(item if item is not UNSPECIFIED else None) for item in inp]
                jarr.finalize()
                assert jarr.tolist() == narr.tolist()


def test_normalize():
    import time
    import random
    import numpy
    for size in [10, 100, 1000, 10000]:
        elapsed = []
        for count in range(10):
            jarr = JaggedArray(size, max_buffer_size=10*size)
            indices = list(range(size))
            random.shuffle(indices)
            for i in indices:
                jarr[i] = list(range(random.randint(0, 10)))
            start = time.time()
            jarr.normalize()
            elapsed.append(time.time() - start)
        # The elapsed timings should scale linearly wrt size:
        print('%s: %.1f +- %.1f um' % (size, numpy.mean(elapsed)*1e6, numpy.std(elapsed)*1e6))
