<p align="center"> <img src="https://raw.githubusercontent.com/thyeem/foc/main/foc.png" height="180"/></p>

[![foc](https://img.shields.io/pypi/v/foc)](https://pypi.org/project/foc/)

# foc

_Func-Oriented Code_ or _Francis' Odd Collection_.

`foc` is a _non-frilled_ and _seamlessly integrated_ functional `Python` tool.


- provides a collection of ___higher-order functions___ and ___placeholder lambda syntax___ (`_`)
- provides an easy way to ___compose functions with symbols___. (`.` and `|`)

```python
>>> from foc import *

# in standard Python, we normally use, 
>>> sum(map(lambda x: x+1, range(10))) 
55

# 'foc' allows three more things:
>>> cf_(sum, map(_ + 1))(range(10))    # using the 'cf_' compose function
55

>>> (sum . map(_ + 1))(range(10))      # using '.' mathematical symbol (:P)
55

>>> range(10) | map(_ + 1) | sum       # using '|' Unix pipeline style
55

# Scala-style placeholder syntax lambda expression 
>>> (_ + 7)(3)                  # same as (lambda x: x + 7)(3)
10

# (3 + 4) * 6
>>> cf_(_ * 6, _ + 4)(3)        # function.
42                             
                               
>>> 3 | _ + 4 | _ * 6           # pipeline.
42                             
                               
>>> ((_ * 6) . fx(_ + 4))(3)    # dot. Wrap 'lambda expression' in 'fx' when using '.'.
42
```

Remember that it is only necessary to use `fx` _when using lambda expressions and `.`_.    
That's all.   

For more examples, see the [documentation](https://github.com/thyeem/foc/blob/main/foc/__init__.py#L150) provided with each function.

```python
>>> (rev . filter(even) . range)(10)  # list(reversed(filter(even, range(10))))
[8, 6, 4, 2, 0]

>>> ((_ * 5) . nth(3) . range)(5)  # range(5)[3] * 5
10

>>> (collect . filter(_ == "f"))("fun-on-functions")  # list(filter(lambda x: x == "f", "fun-on-functions"))
['f', 'f']

# To use built-ins 'list' on the fly, 
>>> (fx(list) . map(abs) . range)(-2, 3)  # list(map(abs, range(-2, 3)))
[2, 1, 0, 1, 2]

>>> range(73, 82) | map(chr) | unchars  # unchars(map(chr, range(73, 82)))
'IJKLMNOPQ'
```

> To see all the functions provided by `foc`, 
```python
>>> from ouch import pp
>>> catalog() | pp
```

## Install
```bash
$ pip install -U foc
```

## What is `fx`?
> `fx` (_Function eXtension_) is the backbone of `foc` and provides a _new syntax_ when composing functions.  
> `fx` basically maps every function in Python to a _monadic function_ in **`fx` monad**.  
> More precisely, `fx` is a _lift function_, but here, I also refer to the functions generated by `fx` as `fx`.

### 1. **`fx` is a _composable_ function using symbols.**

There are two ways to compose functions with symbols as shown in _the previous section_.
| Symbol                | Description                                             | Evaluation Order | Same as in Haskell |
|:---------------------:|:-------------------------------------------------------:|:----------------:|:------------------:|
| **`.`** (_dot_)       | Same as dot(`.`) _mathematical symbol_                  | _Right-to-Left_  | `(<=<)`            |
| **`\|`** (_pipeline_) | In _Unix_ pipeline manner                               | _Left-to-Right_  | `(>=>)`            |
| `fx`                  | _Lift function_. Convert functions into _monadic_ forms | -                | `(pure .)`         |

> _If you don't like function composition using symbols, use **`cf_`**._   
> _In fact, it's the most reliable and safe way to use it for all functions._

### 2. **`fx` is really easy to make.**
`fx` is just a function decorated by `@fx`.  
**Wrap any function in `fx`** when you need function composition on the fly.

```python
>>> [1, 2, 3] | sum | (lambda x: x * 7)    # error, lambda is not a 'fx'
TypeError: unsupported operand ...

>>> [1, 2, 3] | sum | fx(lambda x: x * 7)  # just wrap it in 'fx'.
42

>>> @fx
... def func(arg):    # place @fx above the definition or bind 'g = fx(func)'
...     ...           # 'func' is now 'composable' with symbols
```
> Most of the functions provided by `foc` are `fx` functions.   
> If you don't have one, you can just create one and use it.

### 3. **`fx` is a curried function.**

```python
# currying 'map' -> map(predicate)(iterable)
>>> map(_ * 8)(seq(1,...)) | takel(5)   # seq(1,...) == [1,2,3,..], 'infinite' sequence
[8, 16, 24, 32, 40]                    

# bimap := bimap(f, g, tuple), map over both 'first' and 'second' argument
>>> bimap(_ + 3)(_ * 7)((5, 7))
(8, 49)
>>> foldl(op.sub)(10)(range(1, 5))
0
>>> @fx
... def args(a, b, c, d):
...     return f"{a}-{b}-{c}-{d}"
>>> args(1)(2)(3)(4) == args(1,2)(3,4) == args(1,2,3)(4) == args(1)(2,3,4) == args(1,2,3,4)
True
```
> You can get the curried function of `g` with `fx(g)`.   
> But if you want to get a curried function other than `fx`, use `curry(g)`.  

### 4. **_Lambdas_ with `_` are `fx`.**

```python
>>> [1, 2, 3] | sum | (_ * 7)    # Use '_' lambda instead.
42
>>> ((_ * 6) . fx(_ + 4))(3)     # (3 + 4) * 6
42
>>> 2 | (_ * 7) | (60 % _) | (_ // 3)   # (60 % (2 * 7)) // 3
1
```

**Partial application** driven by `_` is also possible when accessing `dict`, `object` or `iterable`, or even _calling functions_. 

| Operator       | Equivalent Function      |
|----------------|--------------------------|
| `_[_]`         | `op.getitem`             |
| `_[item]`      | `op.itemgetter(item)`    |
| `_._`          | `getattr`                |
| `_.attr`       | `op.attrgetter(attr)`    |
| ``_(_)``       | ``apply``                |
| ``_(*a, **k)`` | ``lambda f: f(*a, **k)`` |


```python
# dict
>>> d = dict(one=1, two=2, three="three")
>>> _[_](d)("two")  # curry(lambda a, b: a[b])(d)("two")
2
>>> _["one"](d)  # (lambda x: x["one"])(d)
1
>>> cf_(_[2:4], _["three"])(d)  # d["three"][2:4]
're'

# iterable
>>> r = range(5)
>>> _[_](r)(3)  # curry(lambda a, b: a[b])(r)(3)
3
>>> _[3](r)     # (lambda x: x[3])(r)
3

# object
>>> o = type('', (), {"one": 1, "two": 2, "three": "three"})()
>>> _._(o)("two")  # curry(lambda a, b: getattr(a, b))(o)("two")
2
>>> _.one(o)  # (lambda x: x.one)(o)
1
>>> o | _.three | _[2:4]  # o.three[2:4]
're'

# function caller 
>>> _(_)(foldl)(op.add)(0)(range(5))
10
>>> _(7 * _)(mapl)(range(1, 10))
[7, 14, 21, 28, 35, 42, 49, 56, 63]

# Not seriously, this creates multiplication table.
>>> [ mapl(f)(range(1, 10)) for f in _(_ * _)(map)(range(1, 10)) ]
```

## Don't forget that `foc` is a collection, _albeit very odd_.

[Everything in one place](https://github.com/thyeem/foc/blob/main/foc/__init__.py).

- `fx` _pure basic functions_ `id`, `const`, `take`, `drop`, `repeat`, `replicate`..
- _higher-order_ functions like `cf_`, `f_`, `ob`, `curry`, `uncurry`, `map`, `filter`, `zip`,.. 
-  useful yet very fundamental like `seq`, `force`, `trap`, `error`, `guard`,..

## Real-World Example
A _causal self-attention_ of the `transformer` model based on `pytorch` can be described as follows.  
_Some_ claim that this helps follow the workflow of tensor operation without distraction. (_plus, 3-5% speed-up_)

```python
    def forward(self, x):
        B, S, E = x.size()  # size_batch, size_block (sequence length), size_embed
        N, H = self.config.num_heads, E // self.config.num_heads  # E == (N * H)

        q, k, v = self.c_attn(x).split(self.config.size_embed, dim=2)
        q = q.view(B, S, N, H).transpose(1, 2)  # (B, N, S, H)
        k = k.view(B, S, N, H).transpose(1, 2)  # (B, N, S, H)
        v = v.view(B, S, N, H).transpose(1, 2)  # (B, N, S, H)

        # Attention(Q, K, V)
        #   = softmax( Q*K^T / sqrt(d_k) ) * V
        #         // q*k^T: (B, N, S, H) x (B, N, H, S) -> (B, N, S, S)
        #   = attention-prob-matrix * V
        #         // prob @ v: (B, N, S, S) x (B, N, S, H) -> (B, N, S, H)
        #   = attention-weighted value (attention score)

        return cf_(
            self.dropout,  # dropout of layer's output
            self.c_proj,  # linear projection
            ob(_.view)(B, S, E),  # (B, S, N, H) -> (B, S, E)
            torch.Tensor.contiguous,  # contiguos in-memory tensor
            ob(_.transpose)(1, 2),  # (B, S, N, H)
            _ @ v,  # (B, N, S, S) x (B, N, S, H) -> (B, N, S, H)
            self.dropout_attn,  # attention dropout
            f_(F.softmax, dim=-1),  # softmax
            ob(_.masked_fill)(mask == 0, float("-inf")),  # no-look-ahead
            _ / math.sqrt(k.size(-1)),  # / sqrt(d_k)
            _ @ k.transpose(-2, -1),  # Q @ K^T -> (B, N, S, S)
        )(q)
```

