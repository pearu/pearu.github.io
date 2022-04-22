"""A prototype of storing Variable Lenght Arrays (VLAs).

A variable lenght array (var-length array) is an array of same type
elements while the number of elements is not fixed. This Python module
provides a storage model of a jagged array (an array of variable
length arrays) that uses a single buffer of values for storing all the
elements of all variable length arrays. In addition, the storage model
is required to support null var-length arrays and random order of
insertion of var-length arrays.

See https://pearu.github.io/variable_length_arrays.html for more
information.
"""
# Author: Pearu Peterson
# Created: April 16, 2022


import itertools


class VarLengthArray:

    def __init__(self, values, size):
        self.values = values
        self.size = size

    def __repr__(self):
        return f'{type(self)}(self.values, self.size)'

    def tolist(self):
        return list(self.values)

    def __len__(self):
        return self.size


class Unspecified(object):

    def __str__(self):
        return 'Unspecified'
    __repr__ = __str__


UNSPECIFIED = Unspecified()


class JaggedArray:

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

        # The storage of the values of all var-length array elements
        self.values = [0] * max_buffer_size

        # The total number of var-length arrays in the jagged array
        self.size = size

        # The total number of specified values in all var-length arrays
        #
        #   0 <= self.array_count <= size
        #
        #  self.array_count is the storage index of the next var-length array
        self.array_count = 0

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

        # The indices of var-lenght arrays:
        #
        #   self.indices[i]
        #
        # is the index of var-length array that storage index is i.
        # In the case of sequential storage of var-length arrays,
        # self.indices is range(size), otherwise, self.indices is a
        # permutation of range(size).
        self.indices = [-1] * size

    def __str__(self):
        return f'{type(self).__name__}[values={self.values[:self.compressed_indices[self.array_count]]}, compressed_indices={self.compressed_indices}, indices={self.indices}]'

    def __repr__(self):
        return f'{type(self).__name__}.fromlist({self.tolist()})'

    def set_null(self, index):
        if index < 0:
            index += self.size
        assert index not in self.indices  # var-length array can be inserted exactly once
        assert 0 <= index and index < self.size

        storage_index = self.array_count
        self.array_count += 1
        self.indices[storage_index] = index
        ptr = self.compressed_indices[storage_index]
        self.compressed_indices[storage_index + 1] = ptr
        self.compressed_indices[storage_index] = -(ptr + 1)

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
        if varlenarray is None:
            self.set_null(index)
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

        if index < 0:
            index += self.size
        assert index not in self.indices  # var-length array can be inserted exactly once
        assert 0 <= index and index < self.size

        storage_index = self.array_count
        self.array_count += 1
        self.indices[storage_index] = index
        ptr = self.compressed_indices[storage_index]
        assert ptr + size <= len(self.values)
        self.compressed_indices[storage_index + 1] = ptr + size
        self.values[ptr:ptr + size] = values

    def _get_ptr_and_size(self, storage_index):
        ptr = self.compressed_indices[storage_index]
        if ptr < 0:
            return None, None
        next_ptr = self.compressed_indices[storage_index + 1]
        if next_ptr < 0:
            return ptr, -(next_ptr + 1) - ptr
        return ptr, next_ptr - ptr

    def __getitem__(self, index):
        if index < 0:
            index += self.size
        assert 0 <= index and index < self.size
        if index not in self.indices:
            return UNSPECIFIED
        storage_index = self.indices.index(index)
        ptr, size = self._get_ptr_and_size(storage_index)
        if ptr is None:
            return None  # null var-length array
        values = self.values[ptr:ptr + size]
        return VarLengthArray(values, size)

    def tolist(self):
        lst = [UNSPECIFIED] * self.size
        for storage_index in range(self.array_count):
            index = self.indices[storage_index]
            arr = self[index]
            if arr is not None:
                lst[index] = arr.tolist()
            else:
                lst[index] = None
        return lst

    @classmethod
    def fromlist(cls, lst):
        jarr = cls(len(lst), max_buffer_size=sum(len(arr) for arr in lst if isinstance(arr, (list, VarLengthArray))))
        for i, arr in enumerate(lst):
            jarr[i] = arr
        return jarr

    def normalize(self, unspecified_is_null=True):
        jarr = type(self)(self.size, max_buffer_size=self.compressed_indices[self.array_count])
        for i in range(self.size):
            arr = self[i]
            if arr is UNSPECIFIED and unspecified_is_null:
                arr = None
            jarr[i] = arr
        return jarr

    def finalize(self):
        """
        Set unspecified elements as null var-length arrays.
        """
        if self.array_count <= self.size:
            unspecified_indices = list(range(self.size))
            for index in self.indices[:self.array_count]:
                unspecified_indices[index] = -1
            for index in unspecified_indices:
                if index != -1:
                    self.set_null(index)
        return self

    def fast_normalize(self, unspecified_is_null=True):
        jarr = type(self)(self.size, max_buffer_size=self.compressed_indices[self.array_count])
        storage_index = 0
        while storage_index < self.array_count:

            
            start_storage_index = storage_index
            while self.indices[storage_index + 1] == self.indices[storage_index] + 1:
                storage_index += 1
                if storage_index == self.array_count:
                    break
            # values in range start_storage_index:storage_index+1 is contiguous
    
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
