
# What is array?

|            |                 |
| ---------- | --------------- |
| Author     | Pearu Peterson  |
| Created    | 2021-05-14      |

This blog post is triggered from a question what is a 0-dimensional array and how to represent this concept in a programming language.
While almost all programming languages implement multidimensional array objects and various operations on these, implementing indexing operations
that result in an array object with reduced dimensionality require handling the degenerate case where the resulting dimensionality is 0: should the result
be a 0-dimensional array object (of array type) or should the result be an array value object (of value type, e.g. scalar type)?
This question has been discussed a lot but there appears to be no definite answer. I think one reason for this originates from "practicality beats purity"
based decisions that array implementations are often based on. In this blog post I'll deliberetly distance the discussion from implementation details
and try to answer the question what is an array in general and with the hope that it will help the later discussions on the level of implementation details as well.

## Array definition

Let me define:

> An array is a collection of values that can be referenced.

and discuss this as follows.

Array as a collection is different from other collections of values suchs as sets or lists in two aspects:
1. all array values have the same type
2. array values are referenced

The value type does not necessarily need to be some scalar type. The values could be anything, including arrays, as long as they have the same type.
This property allows various optimizations at the implementation level, for instance, the common value type information can be stored as an array
metadata instead of storing the type information together with each value.

The referencing of values means that one can perform array manipulations with references to values rather than with values directly.
For instance, to take a slice of an array (that is a specific subset of array) one can avoid (possibly expensive) retrieval of the array
values from storage, and instead, one can just restrict the set of references for the array slice (e.g. via strides manipulations).

In general, a reference to an array value can be any object that represents a unique label (within the given array definition) that is attached to
each array value. Hence, *array is a bijective mapping* from a set of references to a set of array values. (Replacing the adjective "bijective" with "surjective"
would apply to sparse array storage formats that use the so-called fill-value which is a value that that most array values are equal to.)

