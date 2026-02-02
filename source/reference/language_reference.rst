Dynamic System Modelling Language Reference
============================================

To formulate a Markov decision problem, a formalization of the state and transition functions is required. In DynaPlex, *states* of a sequential problem are expressed as objects of a user-defined class, while *transitions* between states are formalized as modifications of those objects by transition functions. DynaPlex imposes a certain formalism for expressing these states and transition functions, that is referred to as Dynamic System Modeling Language (DynaML).

DynaML is designed to achieve two core requirements: 

1. being an expressive and readable language that is easy to learn, write, and read; 
2. enabling automatic, efficient analysis and execution of the models it describes. 

To achieve these design goals, classes (states, MDPs) in DynaML are expressed as python dataclasses, whereas transition functions are represented as python methods on an MDP class that operate on objects of another dataclass: a State class.

Classes and functions expressed in DynaML are also valid python code, i.e. DynaML is a subset of python, making it easy to learn. However, python is also a very expressive language, which gives a lot of freedom in expressing this. To make it easy for algorithms to *reason* about problems and implement efficient solutions, DynaML imposes certain structural and semantic properties, that are described in this document.

How to use this document
------------------------

This document describes what can and cannot be accepted in DynaPlex MDP methods. Summary:

- Annotate parameters and return types.
- Use DataClasses; avoid dictionaries, tuples. 
- Use homogeneous lists; avoid heterogeneous lists.
- Use functions; avoid lambdas.
- Use pure python; no packages apart from limited numpy support. 
- avoid strings; use enums instead. 
- When you need to load data using external files and packages, do so in the `__init__` method of the MDP class, not in the transition functions, and load the results into supported fields of the state class. 

Test your code in CPython. Validate in pyright - if that complains, then it is likely that the code is not valid DynaML. If something fails to compile, use an LLM, pointing it to this reference document and to your code, and it will likely tell you what is wrong. If not, post a question on `GitHub <https://github.com/WillemvJ/DynaPlex-docs/issues>`_.

DynaML Data types
-----------------

DynaML supports data primitives and objects of user-defined classes. In DynaML, any object must have a type that can be determined before running the code.

Primitives
~~~~~~~~~~

DynaML programs manipulate a small collection of well-defined data primitives. Primitive values are scalars (``bool``, ``int``, ``float``) and lists of such scalars. These primitives can appear as parameters to functions, as local variables in expressions, and as fields of state objects.

Classes and objects
~~~~~~~~~~~~~~~~~~~

On top of these primitives, DynaML uses Python dataclasses to define structured objects. State and MDP classes in DynaML are classes whose fields are either primitive values or other DynaML objects (nested classes), or lists of such objects. During execution, states are concrete instances of these dataclasses, and transition functions are methods that read and update their fields.

Assignable values and support for None
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scalars (``float``, ``int``, ``bool``) can be assigned following standard Python typing rules (as enforced by tools such as Pyright). In particular, a ``bool`` value is silently accepted wherever an ``int`` is expected, and an ``int`` value is silently accepted wherever a ``float`` is expected. Floats are not silently accepted where an ``int`` is expected, but we can always cast.

If annotated as ``Optional(T)`` or ``T | None``, then the corresponding object field can be assigned None. The type system does *not* support None values for scalars, enums, or lists.

Syntax for class definition and defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example of a simple DynaML class (throughout, we suppress include statements):

.. code-block:: python

   @dataclass
   class Node:
       product_inventories: list[int]
       on_hold: bool = False
       parent: "Node | None" = None
       children: list["Node"] = field(default_factory=list)

The listing shows that DynaML supports defaults. For objects, the only default supported is None, whereas the only default supported for lists is an empty list (via default factory pattern as list is mutable).

Enums
~~~~~

DynaML supports Python enums for representing finite sets of named integer values. Only standard ``enum.Enum`` is supported (not ``IntEnum``, ``Flag``, or other enum variants). Enum values must be consecutive integers starting from ``1``; this can be easily achieved using ``auto``:

.. code-block:: python

   from enum import Enum, auto

   class Light(Enum):
       RED = auto()
       ORANGE = auto()
       GREEN = auto()

Enums can be used as function parameters, return types, and as fields in dataclasses. DynaML supports enum comparison using ``is`` and ``is not`` (e.g., ``x is Light.RED``), constructing enums from integer values (e.g., ``Light(1)``), and accessing the integer value via the ``.value`` attribute.

.. important::

   Enums and dataclasses must be defined or imported at module level (not inside functions). When DynaPlex analyzes a function's type annotations, it resolves types from the function's module globals using ``get_type_hints()``. Types defined or imported only within a function's local scope can not always be resolved and are unsupported.

Functions and Methods signatures
--------------------------------

DynaML supports both free functions and methods. Here, we discuss the allowed syntax and semantics for free functions.

Free Function Signature
~~~~~~~~~~~~~~~~~~~~~~~

- Every parameter must have a type hint.
- Every function must have a return annotation. Use ``-> None`` for void.
- Multi-return is expressed with tuple annotations: ``-> tuple[int, float]``.
- Parameters may have default values (literal constants only). Default values for list parameters are not supported inside function calls.
- Calls to other registered functions are supported with both positional and keyword arguments.
- Returns: a single typed value or a typed tuple, e.g. ``-> int``, ``-> tuple[int, float]``
- Arguments and return types must be valid DynaML types, as described in the DynaML data type section above.

**Example of a free function:**

.. code-block:: python

   def calculate_distance(x: int, y: int) -> float:
       return (x**2 + y**2) ** 0.5

Method Signatures
~~~~~~~~~~~~~~~~~

Methods can be added to dataclasses. The following rules apply:

- The first parameter (``self``) should **not** have a type annotation (it's inferred from the dataclass)
- All other parameters must have type hints
- Return type annotations work the same as for free functions
- Methods are called using dot notation: ``obj.method(args)``
- Special methods (e.g., ``__init__``, ``__post_init__``, ``__eq__``, ``__lt__``, etc.) are not supported; must make use of default initialization and eq machinery of dataclasses. That is, ``__init__`` and ``__post_init__`` are unsupported insofar as the object is constructed in any functions that are parsed as DynaML. (Since MDP is passed as an argument, the ``__init__`` of the MDP is supported, and there, any valid cpython classes and libraries can be used.)

**Example of a dataclass with a method:**

.. code-block:: python

   @dataclass
   class Point:
       x: int
       y: int
       
       def distance_from_origin(self) -> float:
           return (self.x**2 + self.y**2) ** 0.5

Function bodies
---------------

DynaML supports most mathematical constructs that are also supported by python for scalars. A few specific requirements are worth mentioning. We summarize below.

Parameters
~~~~~~~~~~

Parameters work as in normal python. A minor restriction is that you cannot assign to parameters, but you can of course assign to fields and indexes:

.. code-block:: python

   def give_age(node: Node, age: int) -> None:
       node.on_hold = True       # accepted
       node.children[0] =  node  # accepted
       age = 42           # ERROR: assignment to parameter 'age' is not permitted

Types and casting
~~~~~~~~~~~~~~~~~

DynaML supports explicit casts between any two scalar types: ``int``, ``float``, and ``bool``. These casts follow the usual Python semantics, e.g. ``int(1.7) == 1``, ``float(True) == 1.0``, and ``bool(0) == False``. In addition, any object type can be cast to ``bool``, with the usual python semantics (``None`` becomes false). Implicit casts are allowed only where described in the data-type section (for example, using an ``int`` where a ``float`` is expected).

Arithmetic
~~~~~~~~~~

DynaML supports the usual arithmetic operators on scalar types (``bool``, ``int``, ``float``), with behavior matching Python.

The supported binary operators are ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, and ``**``. For ``+``, ``-``, ``*``, ``//``, ``%``, and ``**``, the result type is ``float`` if either operand is a ``float``, and ``int`` otherwise, while ``/`` always produces a ``float``, even when both operands are ``int``. Boolean values participate in arithmetic as ``0`` (``False``) or ``1`` (``True``), conforming to python values. Augmented assignment on numeric locals and fields is also supported for all of these operators: ``+=``, ``-=``, ``*=``, ``/=``, ``//=``, ``%=``, and ``**=``.

.. code-block:: python

   def arithmetic_examples(inv: int, sold: int, price: float, factor: float, flag: bool) -> None:
       inv -= sold                  # valid: int -= int
       discounted = price * factor  # valid: float * float -> float
       discounted /= 100.0          # valid: float /= float -> float
       # REJECTED by pyright; INVALID in DynaML:
       # flag = inv - sold               
       # instead, use: 
       flag = bool(inv - sold)

Boolean operations
~~~~~~~~~~~~~~~~~~

Boolean expressions in DynaML are built from comparisons. Comparisons (``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``) between scalar values produce boolean results, which are stored as ``bool``.

Boolean connectives
~~~~~~~~~~~~~~~~~~~

Short-circuiting ``and`` and ``or`` are supported. They are supported when all operands are bool, and also when all operands have the same type. (Mixing of ``Object`` and ``Object | None`` is allowed, as long as the object are of the same type.) The unary operator ``not`` is supported with the usual Python semantics and always produces a ``bool``.

Control flow
~~~~~~~~~~~~

Control flow in DynaML is restricted to structured constructs: ``if`` / ``if-else``, ``while``, and ``for`` loops over ``range(...)``. If expressions are also allowed; for example:

.. code-block:: python

   def sign(x: int) -> int:
       return 1 if x >= 0 else -1

You can use if-expressions (the "ternary" operator) to assign the result of a conditional directly. The form is ``A if condition else B``, just as in Python; note that types of A and B must be the same. Nested if-expressions are also supported, as shown above.

Loop bodies may use ``break`` and ``continue`` to alter control flow. Conditions in ``if`` and ``while`` may be explicit booleans or any expression that Python would treat as truthy/falsy: for example, ``if obj:`` tests whether an object reference is non-``None``. The only allowed iterable in ``for`` loops is ``range(...)`` with one, two, or three integer arguments: ``range(stop)``, ``range(start, stop)``, or ``range(start, stop, step)``. A simple ``while`` loop example is:

.. code-block:: python

   def countdown(start: int) -> int:
       steps = 0
       while start != 0: # or while start:
           start -= 1
           steps += 1
       return steps

``for`` loops iterating over lists are also supported. An example using the ``Point`` dataclass defined above:

.. code-block:: python

   def sum_points(points: list[Point]) -> tuple[int, int]:
       """Sum the first and second elements of a list of (x, y) points."""
       sum_x = 0
       sum_y = 0
       for point in points:
           sum_x += point.x
           sum_y += point.y
       return sum_x, sum_y

Local variables
~~~~~~~~~~~~~~~

- Locals are created on first assignment and can be annotated for clarity (``x: int = ...``).
- A variable's *first* assignment must occur in a scope that syntactically encloses all of its uses. In particular, a variable that is first assigned inside an ``if`` or loop body cannot be read outside that ``if`` or loop; it must be introduced before the control-flow construct.

For example, the first function below is rejected, because the return statement uses ``x``, but ``x`` is first defined only inside the ``if`` statement. To sidestep this, declare ``x`` before the ``if``:

.. code-block:: python

   def rejected(flag: bool) -> int:    
       if flag:
           x = 1      
       else:
           x = 10    
       return x # rejected - x is not defined in this scope.
   def accepted(flag: bool) -> int:
       x = 0
       if flag:
           x = 1      
       else:
           x = 10    
       return x

Once a variable has a type in a given (or enclosing) scope, it must not be reused with a different type. For example, the following is rejected because ``x`` is first used as an ``int`` and later as a ``float``:

.. code-block:: python

   def bad_retype(flag: bool) -> None:
       x = 0        # x: int
       if flag:
           x = 1.0  # INVALID: cannot reuse x as float

Global variables are not supported.

Accessing and modifying fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Field access/assignment on dataclass instances is supported, including augmented assignment, e.g. ``location.inventory -= 10``.

Calling functions
~~~~~~~~~~~~~~~~~

Tuple unpacking is supported only one level deep: ``a, b = f()``; this is the only way in which multiple return values can be captured by the calling function. Nested tuple unpacking is not supported.

Functions may be called with positional or keyword arguments. Arguments with defaults may be omitted from the call. Type checking is performed at compile time, ensuring passed arguments match the expected types.

Objects
~~~~~~~

Objects can be constructed following the usual syntax, and methods can be called on those objects.

For example:

.. code-block:: python

   def create_and_use_point() -> float:
       """Create a Point object and call its method."""
       p = Point(x=3, y=4)
       distance = p.distance_from_origin()
       return distance

Lists
~~~~~

DynaML supports single-level lists of primitives (``list[int]``, ``list[float]``, ``list[bool]``) and objects (``list[MyClass]``). List literals can be constructed using standard Python syntax: ``[1, 2, 3]`` or ``[]``. Lists support indexing (including negative indices: ``my_list[-1]``), the ``len()`` function, and the ``.append()`` method.

At present, list of list types (``list[list[int]]``) are not supported; this will be added at a later stage.

NumPy arrays
~~~~~~~~~~~~

There is support for simple NumPy ``NDArray``, but only for NumPy arrays and operations that enable the determination of the dimension of types by looking at the code. Also, the data type must always be annotated, and one of ``np.bool_``, ``np.float64``, ``np.int64``, e.g.:

.. code-block:: python

   @dataclass(slots=True)
   class State:
       weight_vector: NDArray[np.int64]  
       upcoming_weight: int              
       category: StateCategory = StateCategory.AWAIT_EVENT

Precisely which NumPy operations will be supported is being determined, but the most common operations will likely be included, including multi-dimensional arrays.

For a complete example using NumPy arrays in an infinite horizon MDP, see the :doc:`binpacking MDP tutorial <../tutorial/binpacking_mdp>`.

Other unsupported Python syntax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following Python features are presently unsupported, and not planned for future support in DynaML: dictionaries; tuples (except for returning multiple objects, which must be immediately bound to individual variables by the caller, see above); lambdas.

Common pitfalls
---------------

- Missing type hints on parameters or return.
- Using default values for list parameter.
- Optional primitives or list types (``Optional[int]``, etc.), or assigning ``None`` to any list or scalar type.
- Assigning None to an object that is not annotated as with ``|None``.
