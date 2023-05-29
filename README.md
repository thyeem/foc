# fof

`FOF` or `Francis' Odd Functions`.

Here 'odd' probably does not mean the opposite of 'even'.


`Python`'s standard library is great. However some notations are a bit painful and confusing for personal use, so I created this odd collection of functions.

I never reinvented the wheel. All functions are minor modifications from the `Python` standard lib.



```bash
# install
$ pip install -U fof

# import (python 3.6+ maybe)
>>> from fof import *
```

### functional

#### identity function
```python
>>> id("francis")
# 'francis'
```

#### getting n-th element from iterators
```python
>>> fst(["sofia", "maria", "claire"])
# 'sofia'

>>> snd(("sofia", "maria", "claire",))
# 'maria'

>>> nth(["sofia", "maria", "claire"], 3)
# 'claire'
```

#### partial application of function: `f_`
```python
>>> def add(a, b): return a + b

# '_' at the end of function name indicates that the function is not fully evaluated yet.
>>> add_five = f_(add, 5)

>>> add_five(2)
# 7

>>> def print_args(a, b, c): print(f"{a}-{b}-{c}")

# f_ is the same as functools.partial: "currying arguments from the left"
>>> munch_left_two = f_(print_args, "a", "b")

>>> munch_left_two("c")
# a-b-c
```

>>>
