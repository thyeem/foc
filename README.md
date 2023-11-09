# foc

![foc](https://img.shields.io/pypi/v/foc)

`fun oriented code` or `francis' odd collection`.


Functions from the `Python` standard library are great. But some notations are a bit painful and confusing for personal use, so I created this _odd collection of functions_.


## Tl;dr

- `foc` provides a collection of _higher-order functions_ and some (_pure_) helpful functions
- `foc` respects the `Python` standard library. _Never reinvented the wheel_.
- _Take a look at the examples below._

## Use
```bash
# install
$ pip install -U foc

# import
>>> from foc import *
```

> To list all available functions, call `flist()`.

## Ground rules
- Followed `Haskell`-like function names and arguments order
- Considered using generators first if possible. (_lazy-evaluation_)
> `map`, `filter`, `zip`, `range`, `flat` ...
- Provide the functions that unpack generators in `list` as well. (annoying to unpack with `[*]` or `list` every time)
- Function names that end in `l` indicate the result will be unpacked in a list.
> `mapl`, `filterl`, `zipl`, `rangel`, `flatl`, `takewhilel`, `dropwhilel`, ...
- Function names that end in `_` indicate that the function is a **partial application** (_not-fully-evaluated function_) builder.
> `f_`, `ff_`, `c_`, `cc_`, `m_`, `v_`, `u_`, ...
- Most function implementations _should be less than 5-lines_.
- No dependencies except for the `Python` standard library
- No unnessary wrapping objects.

## Examples
__Note__: `foc`'s functions are valid for any _iterable_ such as `list`, `tuple`, `deque`, `set`, `str`, ...
```python
>>> id("francis")
'francis'

>>> fst(["sofia", "maria", "claire"])
'sofia'

>>> snd(("sofia", "maria", "claire"))
'maria'

>>> nth(3, ["sofia", "maria", "claire"])    # not list index, but literally n-th
'claire'

>>> take(3, range(5, 10))
[5, 6, 7]

>>> list(drop(3, "github"))   # `drop` returns a generator
['h', 'u', 'b']

>>> head(range(1,5))          # range(1, 5) = [1, 2, 3, 4]
1

>>> last(range(1,5))
4

>>> list(init(range(1,5)))    # `init` returns a generator
[1, 2, 3]

>>> list(tail(range(1,5)))    # `tail` returns a generator
[2, 3, 4]

>>> pred(3)
2

>>> succ(3)
4

>>> odd(3)
True

>>> even(3)
False

>>> null([]) == null(()) == null({}) == null("")
True

>>> elem(5, range(10))
True

>>> words("fun on functions")
['fun', 'on', 'functions']

>>> unwords(['fun', 'on', 'functions'])
'fun on functions'

>>> lines("fun\non\nfunctions")
['fun', 'on', 'functions']

>>> unlines(['fun', 'on', 'functions'])
("fun\non\nfunctions")

>>> take(3, repeat(5))        # repeat(5) = [5, 5, ...]
[5, 5, 5]

>>> take(5, cycle("fun"))     # cycle("fun") = ['f', 'u', 'n', 'f', 'u', 'n', ...]
['f', 'u', 'n', 'f', 'u']

>>> replicate(3, 5)           # the same as 'take(3, repeat(5))'
[5, 5, 5]

>>> take(3, count(2))         # count(2) = [2, 3, 4, 5, ...]
[2, 3, 4]

>>> take(3, count(2, 3))      # count(2, 3) = [2, 5, 8, 11, ...]
[2, 5, 8]
```

### Build partial application: `f_` and `ff_`
`f_` takes arguments _from the left_ (left-associative) while `ff_` takes them _from the right_ (right-associative).
> _`_` in function names indicates that it is a partial application (not-fully-evaluated function) builder._

```python
>>> f_("+", 5)(2)    # the same as `(5+) 2` in Haskell
7                    # 5 + 2

>>> ff_("+", 5)(2)   # the same as `(+5) 2 in Haskell`
7                    # 2 + 5

>>> f_("-", 5)(2)    # the same as `(5-) 2`
3                    # 5 - 2

>>> ff_("-", 5)(2)   # the same as `(subtract 5) 2`
-3                   # 2 - 5

# with N-ary function
>>> def print_args(a, b, c, d): print(f"{a}-{b}-{c}-{d}")

>>> f_(print_args, 1, 2)(3, 4)                # partial-eval from the left
1-2-3-4                                       # print_args(1, 2, 3, 4)

>>> f_(print_args, 1, 2, 3)(4)                # patial-eval with different args number
1-2-3-4                                       # print_args(1, 2, 3, 4)

>>> ff_(print_args, 1, 2)(3, 4)               # partial-eval from the right
4-3-2-1                                       # print_args(4, 3, 2, 1)
```

### Build curried functions: `c_` and `cc_`
When currying a given function, `c_` takes arguments _from the left_ while `cc_` takes them _from the right_.
> _`_` in function names indicates that it is a partial application (not-fully-evaluated function) builder._

```python
# currying from the left args
>>> c_("+")(5)(2)    # 5 + 2
7

>>> c_("-")(5)(2)    # 5 - 2
3

# currying from the right args
>>> cc_("+")(5)(2)   # 2 + 5
7

>>> cc_("-")(5)(2)   # 2 - 5
-3

# with N-ary function
>>> c_(print_args)(1)(2)(3)(4)    # print_args(1, 2, 3, 4)
1-2-3-4

>>> cc_(print_args)(1)(2)(3)(4)   # print_args(4, 3, 2, 1)
4-3-2-1
```

### Build composition of functions: `cf_` and `cfd`
`cf_` (_composition of function_) composes functions using the given list of functions. On the other hand, `cfd` (_composing-function decorator_) decorates a function with the given list of functions.

> _`_` in function names indicates that it is a partial application (not-fully-evaluated function) builder._

```python
>>> square = ff_("**", 2)     # the same as (^2) in Haskell
>>> add_by_5 = ff_("+", 5)    # the same as (+5)
>>> mul_by_7 = ff_("*", 7)    # the same as (*7)

>>> cf_(mul_by_7, add_by_5, square)(3)   # (*7) . (+5) . (^2) $ 3
98                            # mul_by_7(add_by_5(square(3))) = ((3 ^ 2) + 5) * 7

>>> @cfd(mul_by_7, add_by_5, square)
... def even_num_less_than(x):
...     return len(list(filter(even, range(x))))

>>> even_num_less_than(7)     # even numbers less than 7 = len({0, 2, 4, 6}) = 4
147                           # mul_by_7(add_by_5(square(4))) = ((4 ^ 2) + 5) * 7

# the meaning of decorating a function with a composition of functions
g = cfd(a, b, c, d)(f)   # g = (a . b . c . d)(f)

# the same
cfd(a, b, c, d)(f)(x)    # g(x) = a(b(c(d(f(x)))))

cf_(a, b, c, d, f)(x)    # (a . b . c . d . f)(x) = a(b(c(d(f(x))))) = g(x)
```

`cfd` is very handy and useful to recreate previously defined functions by composing functions. All you need is to write a basic functions to do fundamental things.

### Partial application of `map`: `m_` and `mm_`
`m_` builds partial application of `map` (left-associative) while `mm_` builds partial application from right to left (right-associative).

> _`_` in function names indicates that it is a partial application (not-fully-evaluated function) builder._

Compared to `Haskell`,
- `f <$> xs == map(f, xs)`
- `(f <$>) == f_(map, f) == m_(f)`
- `(<$> xs) == f_(flip(map), xs) == mm_(xs)`

Unpacking with `list(..)` or `[* .. ]` is sometimes very annoying. So often use `mapl` for low memory consuming tasks.

> _Hereafter, function names that end in `l` indicate the result will be unpacked in a list._
>
> See also, `filterl`, `zipl`, `rangel`, `enumeratel`, `reversel`, `flatl` ... and so on

```python
# mapl(f, xs) == [* map(f, xs)] == list(map(f, xs))
>>> mapl = cfd(list)(map)

# so 'm_' and 'mm_' do
>>> ml_ = cfd(list)(m_)
>>> mml_ = cfd(list)(mm_)
```

```python
# The same as [ (lambda x: 8*x)(x) for x in range(1, 6) ]
>>> list(map(f_("*", 8), range(1, 6)))   # (8*) <$> [1..5]
[8, 16, 24, 32, 40]

# tha same: shorter using 'mapl'
>>> mapl(f_("*", 8), range(1, 6))        # (8*) <$> [1..5]
[8, 16, 24, 32, 40]

# the same: partial application (from left)
>>> ml_(f_("*", 8))(range(1, 6))         # ((8*) <$>) [1..5]
[8, 16, 24, 32, 40]

# the same: partial application (from right)
>>> mml_(range(1, 6))(f_("*", 8))        # (<$> [1..5]) (8*)
[8, 16, 24, 32, 40]
```

### Partial application of `filter`: `v_` and `vv_`
`v_` builds partial application of `filter` (left-associative) while `vv_` builds partial application from right to left (right-associative).

The same as `map` (mapping functions over iterables) except for filtering iterables using predicate function.


> _`_` in function names indicates that it is a partial application (not-fully-evaluated function) builder._
>
> The name of `v_` comes from the shape of 'funnel'.

```python
# filterl(f, xs) == [* filter(f, xs)] == list(filter(f, xs))
>>> filterl = cfd(list)(filter)

>>> vl_ = cfd(list)(v_)      # v_ = f_(filter, f)
>>> vvl_ = cfd(list)(vv_)    # vv_ = ff_(filter, xs)
```

```python
# generate a filter to select only even numbers
>>> even_nums = vl_(even)

>>> even_nums(range(10))
[0, 2, 4, 6, 8]

>>> even_nums({2, 3, 5, 7, 11, 13, 17, 19, 23})
[2]

# among prime numbers less than 50
>>> primes_lt_50 = vvl_([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])

# numbers greater than 20 (among prime numbers less than 50)
>>> primes_lt_50(ff_(">", 20))    # (> 20)
[23, 29, 31, 37, 41, 43, 47]

# the same, used a lambda function
>>> primes_lt_50(lambda x: x % 3 == 2)
[2, 5, 11, 17, 23, 29, 41, 47]

# the same, used function composition
>>> primes_lt_50(cf_(ff_("==", 2), ff_("%", 3)))    # ((== 2) . (% 3))
[2, 5, 11, 17, 23, 29, 41, 47]
```

### Other higher-order functions
> To see all available functions, use `flist()` to print to `stdout` or `usage = flist(True)`.

```python
>>> bimap(f_("+", 3), f_("*", 7), (5, 7))       # bimap (3+) (7*) (5, 7)
(8, 49)                                         # (3+5, 7*7)

>>> first(f_("+", 3), (5, 7))                   # first (3+) (5, 7)
(8, 7)                                          # (3+5, 7)

>>> second(f_("*", 7), (5, 7))                  # second (7*) (5, 7)
(5, 49)                                         # (5, 7*7)

>>> take(5, iterate(lambda x: x**2, 2))         # [2, 2**2, (2**2)**2, ((2**2)**2)**2, ...]
[2, 4, 16, 256, 65536]

>>> [* takewhile(even, [2, 4, 6, 1, 3, 5]) ]    # `takewhile` returns a generator
[2, 4, 6]

>>> takewhilel(even, [2, 4, 6, 1, 3, 5])
[2, 4, 6]

>>> [* dropwhile(even, [2, 4, 6, 1, 3, 5]) ]    # `dropwhile` returns a generator
[1, 3, 5]

>>> dropwhilel(even, [2, 4, 6, 1, 3, 5])
[1, 3, 5]

# fold with a given initial value from the left
>>> foldl("-", 10, range(1, 5))                 # foldl (-) 10 [1..4]
0

# fold with a given initial value from the right
>>> foldr("-", 10, range(1, 5))                 # foldr (-) 10 [1..4]
8

# `foldl` without an initial value (used first item instead)
>>> foldl1("-", range(1, 5))                    # foldl1 (-) [1..4]
-8

# `foldr` without an initial value (used first item instead)
>>> foldr1("-", range(1, 5))                    # foldr1 (-) [1..4]
-2

# accumulate reduced values from the left
>>> scanl("-", 10, range(1, 5))                 # scanl (-) 10 [1..4]
[10, 9, 7, 4, 0]

# accumulate reduced values from the right
>>> scanr("-", 10, range(1, 5))                 # scanr (-) 10 [1..4]
[8, -7, 9, -6, 10]

# `scanl` but no starting value
>>> scanl1("-", range(1, 5))                    # scanl1 (-) [1..4]
[1, -1, -4, -8]

# `scanr` but no starting value
>>> scanr1("-", range(1, 5))                    # scanr1 (-) [1..4]
[-2, 3, -1, 4]

# See also 'concat' that returns a generator
>>> concatl(["sofia", "maria"])
['s', 'o', 'f', 'i', 'a', 'm', 'a', 'r', 'i', 'a']
# Note that ["sofia", "maria"] = [['s','o','f','i','a'], ['m','a','r','i','a']]

# See also 'concatmap' that returns a generator
>>> concatmapl(str.upper, ["sofia", "maria"])   # concatmapl = cfd(list, concat)(map)
['S', 'O', 'F', 'I', 'A', 'M', 'A', 'R', 'I', 'A']
```

### Lazy Evaluation: `lazy` and `force`
To defers the evaluation of a function(or expression), just use `lazy`.

In order to generate a lazy expression, use `lazy(function-name, *args, **kwargs)`

`force` forces the deferred-expression to be fully evaluated when needed.
> it reminds `Haskell`'s `force x = deepseq x x`.

```python
# strictly generate a random integer between [1, 10)
>>> randint(1, 10)

# generate a lazy expression for the above
>>> deferred = lazy(randint, 1, 10)

# evaluate it when it need
>>> force(deferred)

# the same as 'force(deferred)'
>>> deferred()
```

Are those evaluations with `lazy` really deferred?

```python
>>> long_list = randint(1, 100000, 100000)    # a list of one million random integers

>>> %timeit sort(long_list)
142 ms ± 245 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# See the evaluation was deferred
>>> %timeit lazy(sort, long_list)
1.03 µs ± 2.68 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each
```

When to use? Let me give an example.

For given a function `randint(low, high)`, how can we generate a list of random integers?

```python
[ randint(1, 10) for _ in range(5) ]    # exactly the same as 'randint(1, 10, 5)'
```

It's the simplest way but what about using `replicate`?
```python
# generate a list of random integers using 'replicate'
>>> replicate(5, randint(1, 10))
[7, 7, 7, 7, 7]        # ouch, duplication of the first evaluated item.
```
Wrong! This result is definitely not what we want. We need to defer the function evaluation till it is _replicated_.

Just use `lazy(randint, 1, 10)` instead of `randint(1, 10)`

```python
# replicate 'deferred expression'
>>> randos = replicate(5, lazy(randint, 1, 10))

# evaluate when needed
>>> mforce(randos)      # mforce = ml_(force), map 'force' over deferred expressions
[6, 2, 5, 1, 9]         # exactly what we wanted
```

Here is the simple secret: if you complete `f_` or `ff_` with a function name and its arguments, and leave it unevaluated (not called), they will act as a _deferred expression_.

Not related to `lazy` operation, but you do the same thing with `uncurry`

```python
# replicate the tuple of arguments (1, 10) and then apply to uncurried function
>>> ml_(u_(randint))(replicate(5, (1,10)))    # u_ == uncurry
[7, 6, 1, 7, 2]
```

### Normalize containers: `flat`
`flat` flattens all kinds of iterables except for string-like object, _regardless of the number of arguments_.

`flat(*args)`
```python
# Assume that we regenerate 'data' every time in the examples below
>>> data = [1,2,[3,4,[[[5],6],7,{8},((9),10)],range(11,13)], (x for x in [13,14,15])]

# 'flat' returns a generator. flatl = cfd(list)(flat)
>>> flatl(data)    # list
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

>>> flatt(data)    # tuple
(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

>>> flats(data)    # set
{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

>>> flatd(data)    # deque
deque([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

# regardless of the number of arguments
>>> flatl(1,[2,{3}],[[[[[4]],5]]], "sofia", "maria")
[1, 2, 3, 4, 5, 'sofia', 'maria']
```

### Handy File Tools: `ls` and `grep`

`Path` from `pathlib` and `glob` are great and useful.

But, personally I feel like it's still complicated and I'm not likely to use it. Using `os.path.expanduser("~")` is very painful every time and not intuitive at all.

They never understand `~/francis/foc` and are not tolerable `foc//__init__` (typo `/`).

I needed more handy one that controls everything with only `glob` and `regex` patterns.

`ls(path=PATH, grep=REGEX, i=BOOL, r=BOOL)`
```python
# couldn't be simpler! expand "~" automatically
>>> ls("~")    # the same as `ls -1 ~`: returns a list of $HOME

# support glob patterns (*, ?, [)
>>> ls("./*/*.py")
```
```
['foc/__init__.py', 'tests/__init__.py', 'tests/test_foc.py']
```
```python
# list up recursively, like "find .git"
>>> ls(".git", r=True)
```
```
...
 '.git/hooks/update.sample',
 '.git/index',
 '.git/info/exclude',
 '.git/logs/HEAD',
 ...
```
```python
# search recursivley and matching a pattern with `grep`
>>> ls(".", r=True, i=True, grep=".PY")    # 'i=True' turns on case-insensitive (-i flag)
```
```
...
 '.pytest_cache/v/cache/stepwise',
 'foc/__init__.py',
 'foc/__pycache__/__init__.cpython-310.pyc',
 'tests/__init__.py',
 ...
```
```python
# regex patterns comes in
>>> ls(".", r=True, grep=".py$")
```
```
['./setup.py', 'foc/__init__.py', 'tests/__init__.py', 'tests/test_foc.py']
```
```python
# that's it!
>>> ls(".", r=True, grep="^(foc).*py$")
```
```
['foc/__init__.py']
```
`grep(REGEX, i=BOOL)` yields a function: `[STRING] -> [STRING]`

```python
# 'grep' builds filter with regex patterns
>>> grep(r"^(foc).*py$")(ls(".", r=True))
```
```
['foc/__init__.py']
```
There are several fundamental functions prepared as well such as: `HOME`, `cd`, `pwd`, `mkdir`, `rmdir`, `exists`, `dirname`, `basename` and so on.

### Neatify data structures: `neatly` and `nprint`
`neatly` generates neatly formatted string of the complex data structures of `dict` and `list`.

`nprint` (_neatly-print_) prints data structures to `stdout` using `neatly` formatter."""

`nprint(...) = print(neatly(...))`

`nprint(DICT, _cols=INDENT, _width=WRAP, **kwargs)`


```python
>>> o = dict(name="yunchan lim", age=19, profession="pianist")
>>> mozart=["piano concerto no.22 in E-flat Major, k.482", "sonata No.9 in D Major, k.311"]
>>> beethoven=["piano concerto no.3 in C minor, op.37", "eroica variations, Op.35"]
>>> nprint(o, cliburn=dict(mozart=mozart, beethoven=beethoven))
```
```
      name  |  'yunchan lim'
       age  |  19
profession  |  'pianist'
   cliburn  |     mozart  -  'piano concerto no.22 in E-flat Major, k.482'
            :             -  'sonata No.9 in D Major, k.311'
            :  beethoven  -  'piano concerto no.3 in C minor, op.37'
            :             -  'eroica variations, Op.35'
```
```python
>>> o = dict(widget=dict(debug="on",
...                      settings=["log", "0xff",
...                                dict(window=dict(title="sample", name="main", width=480, height=360))]),
...          image=dict(src="sun.png", align="center", kind=["data", "size", dict(hOffset=250, vOffset=100)]))
>>> nprint(o)
```
```
widget  |     debug  |  'on'
        :  settings  -  'log'
        :            -  '0xff'
        :            -  window  |   title  |  'sample'
        :                       :    name  |  'main'
        :                       :   width  |  480
        :                       :  height  |  360
 image  |    src  |  'sun.png'
        :  align  |  'center'
        :   kind  -  'data'
        :         -  'size'
        :         -  hOffset  |  250
        :            vOffset  |  100
```


### Dot-accessible dictionary: `dmap`
`dmap` is a _yet another_ `dict`. It's exactly the same as `dict` but it enables to access its nested structure with '_dot notations_'.

`dmap(DICT, **kwargs)`

```python
>>> d = dmap()    # empty dict
>>> d = dmap(dict(...))
>>> d = dmap(name="yunchan lim", age=19, profession="pianist")    # or dmap({"name":.., "age":..,})

# just put the value in the desired keypath
>>> d.cliburn.semifinal.mozart = "piano concerto no.22"
>>> d.cliburn.semifinal.liszt = "12 transcendental etudes"
>>> d.cliburn.final.beethoven = "piano concerto no.3"
>>> d.cliburn.final.rachmaninoff = "piano concerto no.3"
>>> nprint(d)
```
```
      name  |  'yunchan lim'
       age  |  19
profession  |  'pianist'
   cliburn  |  semifinal  |  mozart  |  'piano concerto no.22'
            :             :   liszt  |  '12 transcendental etudes'
            :      final  |     beethoven  |  'piano concerto no.3'
            :             :  rachmaninoff  |  'piano concerto no.3'
```
```python
>>> del d.cliburn.semifinal
>>> d.profession = "one-in-a-million talent"
>>> nprint(d)
```
```
      name  |  'yunchan lim'
       age  |  19
profession  |  'one-in-a-million talent'
   cliburn  |  final  |     beethoven  |  'piano concerto no.3'
            :         :  rachmaninoff  |  'piano concerto no.3'
```
```python
# No such keypath
>>> d.bach.chopin.beethoven
{}
```


### `raise` and `assert` with _expressions_: `error` and `guard`

Raise any kinds of exception in `lambda` expression as well.

```python
>>> error(MESSAGE, e=EXCEPTION_TO_RAISE)    # by default, e=SystemExit

>>> error("Error, used wrong type", e=TypeError)

>>> error("out of range", e=IndexError)

>>> (lambda x: x if x is not None else error("Error, got None", e=ValueError))(None)
```
Likewise, use `guard` if there need _assertion_ not as a statement, but as an _expression_.

```python
>>> guard(PREDICATE, MESSAGE, e=EXCEPTION_TO_RAISE)    # by default, e=SystemExit

>>> guard("Almost" == "enough", "'Almost' is never 'enough'")

>>> guard(rand() > 0.5, "Assertion error occurs with a 0.5 probability")

>>> guard(len(x := range(11)) == 10, f"length is not 10: {len(x)}")
```

### Other utils
_Documents will be updated_


### Real-world Example
A causal self-attention of the `transformer` model based on `pytorch` can be described as follows. _Somebody_ insists that this helps to follow the process flow without distraction.

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
            ff_(torch.Tensor.view, *_r(B, S, E)),  # (B, S, N, H) -> (B, S, E)
            torch.Tensor.contiguous,  # contiguos in-memory tensor
            ff_(torch.transpose, *_r(1, 2)),  # (B, S, N, H)
            ff_(torch.matmul, v),  # (B, N, S, S) x (B, N, S, H) -> (B, N, S, H)
            self.dropout_attn,  # attention dropout
            ff_(torch.masked_fill, *_r(mask == 0, 0.0)),  # double-check masking
            f_(F.softmax, dim=-1),  # softmax
            ff_(torch.masked_fill, *_r(mask == 0, float("-inf"))),  # no-look-ahead
            ff_("/", math.sqrt(k.size(-1))),  # / sqrt(d_k)
            ff_(torch.matmul, k.transpose(-2, -1)),  # Q @ K^T -> (B, N, S, S)
        )(q)
```
