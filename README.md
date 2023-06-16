# foc

![foc](https://img.shields.io/pypi/v/foc)

`fun oriented code` or `francis' odd collection`.


Functions from the `Python` standard library are great. But some notations are a bit painful and confusing for personal use, so I created this _odd collection of functions_.


### Tl;dr

- `foc` provides a collection of _higher-order functions_ and some (_pure_) helpful functions
- `foc` respects the `Python` standard library. _Never reinvented the wheel_.
- _Take a look at the examples below._

### Use
```bash
# install
$ pip install -U foc

# import
>>> from foc import *
```

> To list all available functions, call `flist()`.

### Ground rules
- Followed `Haskell`-like function names and arguments order
- Considered using generators first if possible. (_lazy-evaluation_)
> `map`, `filter`, `zip`, `range`, `flat` ...
- Provide the functions that unpack generators in `list` as well. (annoying to unpack with `[*]` or `list` every time)
- Function names that end in `l` indicate the result will be unpacked in a list.
> `mapl`, `filterl`, `zipl`, `rangel`, `flatl`, ...
- Function names that end in `_` indicate that the function is a **partial application** (_not-fully-evaluated function_) builder.
> `f_`, `ff_`, `c_`, `cc_`, `m_`, `v_`, ...
- Most function implementations _should be less than 5-lines_.
- No dependencies except for the `Python` standard library
- No unnessary wrapping objects.

### Examples
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

#### Build partial application: `f_` and `ff_`
`f_` takes arguments _from the left_ (left-associative) while `ff_` takes them _from the right_ (right-associative).

> When using `ff_`, passing arguments in reverse order for a long-args function is painful.
>
> `ff_` takes arguments _in order_ by default (`sgra=False`). It will take args _in reverse order_ when the `sgra` keyword is on.
>
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
3-4-1-2                                       # print_args(3, 4, 1, 2)

>>> ff_(print_args, 1, 2, sgra=True)(3, 4)    # partial-eval from the right (sgra=True)
4-3-2-1                                       # print_args(4, 3, 2, 1)
```

#### Build curried functions: `c_` and `cc_`
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

#### Build composition of functions: `cf_` and `cfd`
`cf_` (_composition of function_) composes functions using the given list of functions. On the other hand, `cfd` (_composing-function decorator_) decorates a function with the given list of functions.

> _`_` in function names indicates that it is a partial application (not-fully-evaluated function) builder._

```python
>>> square = ff_("**", 2)     # the same as (^2) in Haskell
>>> add_by_5 = ff_("+", 5)    # the same as (+5)
>>> mul_by_7 = ff_("*", 7)    # the same as (*7)

>>> cf_(mul_by_7, add_by_5, square)(3)   # (*7) . (+5) . (^2) $ 2
98                            # mul_by_7(add_by_5(square(2))) = ((3 ^ 2) + 5) * 7

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

#### Partial application of `map`: `m_` and `mm_`
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
>>> list(map(f_("*", 8), range(1, 6)))   # (8*) <$> [1..5]
[8, 16, 24, 32, 40]                         # [ (lambda x: 8*x)(x) for x in range(1, 6) ]

>>> mapl(f_("*", 8), range(1, 6))        # (8*) <$> [1..5]
[8, 16, 24, 32, 40]

>>> ml_(f_("*", 8))(range(1, 6))         # ((8*) <$>) [1..5]
[8, 16, 24, 32, 40]

>>> mml_(range(1, 6))(f_("*", 8))        # (<$> [1..5]) (8*)
[8, 16, 24, 32, 40]
```

#### Partial application of `filter`: `v_` and `vv_`
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
>>> even_nums = vl_(even)

>>> even_nums(range(10))
[0, 2, 4, 6, 8]

>>> even_nums({2, 3, 5, 7, 11, 13, 17, 19, 23})
[2]

>>> primes_lt_50 = vvl_([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])

>>> primes_lt_50(ff_(">", 20))    # (> 20)
[23, 29, 31, 37, 41, 43, 47]

>>> primes_lt_50(lambda x: x % 3 == 2)
[2, 5, 11, 17, 23, 29, 41, 47]

>>> primes_lt_50(cf_(ff_("==", 2), ff_("%", 3)))    # ((== 2) . (% 3))
[2, 5, 11, 17, 23, 29, 41, 47]
```

#### Other higher-order functions
> To see all available functions, use `flist()` to print to `stdout` or `usage = flist(True)`.

```python
>>> bimap(f_("+", 3), f_("*", 7), (5, 7))    # bimap (3+) (7*) (5, 7)
(8, 49)                                            # (3+5, 7*7)

>>> first(f_("+", 3), (5, 7))                   # first (3+) (5, 7)
(8, 7)                                             # (3+5, 7)

>>> second(f_("*", 7), (5, 7))                  # second (7*) (5, 7)
(5, 49)                                            # (5, 7*7)

>>> take(5, iterate(lambda x: x**2, 2))            # [2, 2**2, (2**2)**2, ((2**2)**2)**2, ...]
[2, 4, 16, 256, 65536]

>>> [* takewhile(even, [2, 4, 6, 1, 3, 5]) ]       # `takewhile` returns a generator
[2, 4, 6]

>>> takewhilel(even, [2, 4, 6, 1, 3, 5])
[2, 4, 6]

>>> [* dropwhile(even, [2, 4, 6, 1, 3, 5]) ]       # `dropwhile` returns a generator
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

# See also 'concatmap' that returns a generator
>>> concatmapl(str.upper, ["sofia", "maria"])      # concatmapl = cfd(list, concat)(map)
['S', 'O', 'F', 'I', 'A', 'M', 'A', 'R', 'I', 'A']
```

#### Lazy Evaluation: `lazy` and `force`
`lazy` delays the evaluation of a function(or expression) using `python`s generator. `force` forces the delayed-expression to be fully evaluated.

```python
>>> %timeit pow(2, 12345)    # 2 ** 12345
24 µs ± 33.5 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

# See the evaluation was delayed
>>> %timeit lazy(pow, 2, 12345)
1.46 µs ± 14.9 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each

>>> r = lazy(pow, 2, 12345)
>>> r()       # fully evaluate it!

# or
>>> force(r)  # like Haskell's "force", x `seq` x.

>>> replicate(5, random_int(1, 10))    # wrong. not wanted.
[7, 7, 7, 7, 7]         # evaluation is not delayed. duplication of the same elements.

>>> randos = replicate(5, lazy(random_int, 1, 10))    # [ delayed_fn, delayed_fn .., ]

>>> ml_(force)(randos)  # map 'force' over list of delayed functions
[6, 2, 5, 1, 9]

>>> mforce(randos)      # the same as ml_(force)(randos)
[7, 3, 9, 1, 3]         # expected result
```

#### Normalize containers: `flat`
`flat` flattens all kinds of iterables except for string-like object, _regardless of the number of arguments_.

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

`fread` and `fwrite` are used to easily read and write _flattened_ data using `flat`.

```python
>>> fwrite("family.dat", ["maria",[[[["sofia"]]],[["claire"]],"francis"]])
'family.dat'

# 'bytes' indicates 'filename'
>>> [* fread(b"family.dat", [[1,{2}],[[[3],(4,5)]], (x for x in range(6,11))]) ]
['maria', 'sofia', 'claire', 'francis', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# with not-exists files
>>> [* fread(b"fruits", ...[[1,{2}],[[[3],(4,5)]], (x for x in range(6,11))]) ]
# ...
# Exception: Error, not found file: fruits
```

#### Dot-accessible dictionary: `dmap`
`dmap` is a _yet another_ `dict`. It's exactly the same as `dict` but it enables to access its nested structure with '_dot notations_'.

```python
>>> d = dmap(name="yunchan lim", age=19, profession="pianist")
>>> nprint(d)    # neatly print 'dict' or 'dict-items'
        name  |  yunchan lim
         age  |  19
  profession  |  pianist

# just put the value in the desired key path
>>> d.cliburn.semifinal.mozart = "piano concerto no.22"
>>> d.cliburn.semifinal.liszt = "12 transcendental etudes"
>>> d.cliburn.final.beethoven = "piano concerto no.3"
>>> d.cliburn.final.rachmaninoff = "piano concerto no.3"
>>> nprint(d)
        name  |  yunchan lim
         age  |  19
  profession  |  pianist
     cliburn  |   semifinal  |   mozart  |  piano concerto no.22
              |              |    liszt  |  12 transcendental etudes
              |       final  |      beethoven  |  piano concerto no.3
              |              |   rachmaninoff  |  piano concerto no.3

>>> del d.cliburn
>>> d.profession = "one-in-a-million talent"
>>> nprint(d)
        name  |  yunchan lim
         age  |  19
  profession  |  one-in-a-million talent

# No such key path
>>> d.bach.chopin.beethoven
{}
```


#### `raise` with a function(_expression_): `error`

Raise any kinds of exception in `lambda` expression as well.

```python
>>> error("Error, nice!", e=TypeError)   # by default, e=Exception

>>> error("out of range", e=IndexError)

>>> lambda x, y: x if x is not None else error("Error, got None", e=ValueError)
```


#### Other utils
_Documents will be updated_


#### Real-world Example
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
            self.resid_dropout,  # residual dropout
            self.c_proj,  # linear projection
            ff_(torch.Tensor.view, B, S, E),  # (B, S, N, H) -> (B, S, E)
            torch.Tensor.contiguous,  # contiguos in-memory tensor
            ff_(torch.transpose, 1, 2),  # (B, S, N, H)
            ff_(torch.matmul, v),  # (B, N, S, S) x (B, N, S, H) -> (B, N, S, H)
            self.attn_dropout,  # attention dropout
            ff_(F.softmax, dim=-1),  # softmax
            ff_(torch.masked_fill, self.mask[:,:,:S,:S] == 0, float("-inf")),  # mask
            ff_("/", math.sqrt(k.size(-1))),  # / sqrt(d_k)
            ff_(torch.matmul, k.transpose(-2, -1)),  # Q @ K^T -> (B, N, S, S)
        )(q)
```
