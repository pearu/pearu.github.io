---
author: Pearu Peterson
created: 2021-05-14
---

# What is array?

This blog post is motivated from a question what is a 0-dimensional array, it's relation to array values, and how to represent such arrays in a programming language. Here the answer will be given only to the first parts of the question (sorry..).

While almost all programming languages implement multidimensional array objects and various operations on these, implementing indexing operations
that result in an array object with reduced dimensionality require handling the degenerate case where the resulting dimensionality is 0: should the result
be a 0-dimensional array object (of array type) or should the result be an array value object (of value type, e.g. scalar type)?
This question has been discussed a lot but there appears to be no definite answer. I think one reason for this originates from "practicality beats purity"
based decisions that array implementations are often based on.

In this blog post I'll deliberetly distance the discussion from implementation details and try to answer the question what
is an array in general. I hope that it will support the discussions on the implementation details as well.

## Array definition

Let me define:

> An array is a collection of values that can be referenced.

and discuss this as follows.

Array, as a collection, is different from other collections, such as sets or lists, in two aspects:
1. all array values have the same type
2. array values are referenced

The type of an array values does not necessarily need to be a scalar type. The values could be anything, including arrays, as long as they have the same type.
This property allows various optimizations at the implementation level, for instance, the common value type information can be stored as an array
metadata instead of storing the type information together with each value.

The "referencing of values" means that one can perform array manipulations with references to values rather than with values directly.
For instance, to take a slice of an array (that is a specific subset of array) one can avoid (possibly expensive) retrieval of the array
values from storage, and instead, one can just restrict the manipulations on the set of references (e.g. via strides manipulations).

In general, a reference to an array value can be any object that represents a unique label (within the given array definition) that is attached to
each array value. Hence, *array is a bijective mapping* from a _set of references_ to a _set of array values_. (Replacing the adjective "bijective" with "surjective"
would apply to sparse array storage formats that use the so-called fill-value which is a value that that most array values are equal to.)

> An empty array is an array with no values.

## Multi-dimensional array

> A multi-dimensional array is an array that reference set is a set of tuples with the same length, called *dimensionality* of the array.

From this definition follows:

> a 0-dimensional array is a non-empty array that reference set consists of a single element: a zero-length tuple.

and

> the set of 0-dimensional arrays is isomorphic to a set of array values.

That's it.

## Bonus part

### Multi-dimensional array with integer indices

Restricting the set of references to tuples of (a compact set of) integers, called _indices_, allows many optimizations. For instance, one does not need to store the reference set when the array values are stored in memory contiguously: for a given tuple of integers one can compute the location of array value in memory storage very efficiently; and vice-versa, given the location of the array value in memory, one can compute the corresponding index as efficiently. In addition,
the operation of slicing an array boils down to an operation on shape/strides only.
