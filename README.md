# foc

![foc](https://img.shields.io/pypi/v/foc)

`fun oriented code` or `francis' odd collection`.


Functions from the `Python` standard library are great. But some notations are a bit painful and confusing for personal use, so I created this _odd collection of functions_.


## Tl;dr

- `foc` provides a collection of ___higher-order functions___, some helpful (_pure_) _functions_, and ___piper___.
- `foc` respects the `python` standard library. _Never reinvented the wheel_.


## How to use
```bash
# install
$ pip install -U foc

# import
>>> from foc import *
```
> To list all available functions, call `catalog()`.

## Ground rules
- _No dependencies_ except for the `python` standard library
- _No unnessary wrapping_ objects.
- Most function implementations _should be less than 5-lines_.
- Followed `haskell`-like function names and arguments order
- Used `python` generator first if possible. (_lazy-evaluation_)
  > `map`, `filter`, `zip`, `range`, `flat` ...

- Provide the functions that unpack generators in `list` as well. 
- Function names that end in `l` indicate the result will be unpacked in a list.
  > `mapl`, `filterl`, `zipl`, `rangel`, `flatl`, `takewhilel`, `dropwhilel`, ...
- Function names that end in `_` indicate that the function is a _partial application_ builder.
  > `cf_`, `f_`, `ff_`, `c_`, `cc_`, `u_`, ...

## Quickstart
`foc` collection consists of `piper` and `regular functions`. `foc`'s functions are mostly `piper`.  


### As a `regular function`
No further explanations are needed. Just use them as we used.

```python
>>> id("francis")
'francis'

>>> even(3)
False

>>> take(3, range(5, 10))
[5, 6, 7]
```
The above functions are all `piper` and are:
```python
>>> "francis" | id
'francis'

>>> 3 | even
False

>>> range(5, 10) | take(3)
[5, 6, 7]
```

### As a `piper`
`piper` is a special type of function that facilitate function composition in a pipeline-like manner.
> To list all available `piper`, call `catalog(piper=True)`.

```python
>>> range(10) | length
10

>>> range(10) | filter(even) | unpack    # 'unpack' unpacks generators
[0, 2, 4, 6, 8]

>>> range(10) | map(f_("+", 5)) | sum    # f_("+", 5) == lambda x: x+5
95

>>> rangel(11) | shuffle | sort | last
10
```

If you want to make a function `piper` on the fly, just wrap the function with `piper`. That's it.
```python
>>> 7 | piper(lambda x: x * 6)
42
```

Try binding a function to a new reference:
```python
>>> foo = piper(func)
```

or use `piper` decorator. All the same. 
```python
>>> @piper            # @piper pipefies `func`
... def func(arg):    # arg | func == func(arg)
...    ...
```


## Examples

### Simple Functions
```python
>>> id("francis")
>>> "francis" | id
'francis'

>>> const(5, "no-matther-what-comes-here")
>>> 'whatever' | const(5)
5

>>> seq("only-returns-the-following-arg", 5)
>>> 5 | seq('whatever')
5

>>> void(randbytes(256))
>>> randbytes(256) | void
... (returns None)

>>> fst(["sofia", "maria", "claire"])
>>> ["sofia", "maria", "claire"] | fst
'sofia'
```

Likewise, the below are all `piper`.

```python
>>> snd(("sofia", "maria", "claire"))
'maria'

>>> nth(3, ["sofia", "maria", "claire"]) 
'claire'

>>> take(3, range(5, 10))
[5, 6, 7]

>>> drop(3, "github") | unpack 
['h', 'u', 'b']

>>> head(range(1,5)) 
1

>>> last(range(1,5))
4

>>> init(range(1,5)) | unpack
[1, 2, 3]

>>> tail(range(1,5)) | unpack 
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
"fun\non\nfunctions"

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
### Higher-Order Functions
```python
>>> flip(pow)(7, 3)                             # the same as `pow(3, 7) = 3 ** 7`
2187

>>> bimap(f_("+", 3), f_("*", 7), (5, 7))       # bimap (3+) (7*) (5, 7)
(8, 49)                                         # (3+5, 7*7)

>>> first(f_("+", 3), (5, 7))                   # first (3+) (5, 7)
(8, 7)                                          # (3+5, 7)

>>> second(f_("*", 7), (5, 7))                  # second (7*) (5, 7)
(5, 49)                                         # (5, 7*7)

>>> take(5, iterate(lambda x: x**2, 2))         # [2, 2**2, (2**2)**2, ((2**2)**2)**2, ...]
[2, 4, 16, 256, 65536]

>>> [* takewhile(even, [2, 4, 6, 1, 3, 5]) ]    
[2, 4, 6]

>>> takewhilel(even, [2, 4, 6, 1, 3, 5])
[2, 4, 6]

>>> [* dropwhile(even, [2, 4, 6, 1, 3, 5]) ]    
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

>>> concatl(["sofia", "maria"])
['s', 'o', 'f', 'i', 'a', 'm', 'a', 'r', 'i', 'a']
# Note that ["sofia", "maria"] = [['s','o','f','i','a'], ['m','a','r','i','a']]

>>> concatmapl(str.upper, ["sofia", "maria"])   
['S', 'O', 'F', 'I', 'A', 'M', 'A', 'R', 'I', 'A']
```


### Real-World Example
A causal self-attention of the `transformer` model based on `pytorch` can be described as follows.  
_Somebody_ insists that this helps to follow the process flow without distraction.

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
            ff_(torch.Tensor.view, *rev(B, S, E)),  # (B, S, N, H) -> (B, S, E)
            torch.Tensor.contiguous,  # contiguos in-memory tensor
            ff_(torch.transpose, *rev(1, 2)),  # (B, S, N, H)
            ff_(torch.matmul, v),  # (B, N, S, S) x (B, N, S, H) -> (B, N, S, H)
            self.dropout_attn,  # attention dropout
            ff_(torch.masked_fill, *rev(mask == 0, 0.0)),  # double-check masking
            f_(F.softmax, dim=-1),  # softmax
            ff_(torch.masked_fill, *rev(mask == 0, float("-inf"))),  # no-look-ahead
            ff_("/", math.sqrt(k.size(-1))),  # / sqrt(d_k)
            ff_(torch.matmul, k.transpose(-2, -1)),  # Q @ K^T -> (B, N, S, S)
        )(q)
```

## In Detail
### Get binary functions from `python` operators: `sym`
`sym(OP)` converts `python`'s _symbolic operators_ into _binary functions_.  
The string forms of operators like `+`, `-`, `/`, `*`, `**`, `==`, `!=`, .. represent the corresponding binary functions.
> To list all available symbol operators, call `sym()`.

```python
>>> sym("+")(5, 2)                 # 5 + 2
7

>>> sym("==")("sofia", "maria")    # "sofia" == "maria"
False

>>> sym("%")(123456, 83)           # 123456 % 83
35
```

### Build partial application: `f_` and `ff_`
- `f_` build left-associative partial application,  
where the given function's arguments partially evaluation _from the left_.
- `ff_` build right-associative partial application,  
where the given function's arguments partially evaluation _from the right_.

> `f_(fn, *args, **kwargs)`  
> `ff_(fn, *args, **kwargs) == f_(flip(fn), *args, **kwargs)`  

```python
>>> f_("+", 5)(2)    # the same as `(5+) 2` in Haskell
7                    # 5 + 2

>>> ff_("+", 5)(2)   # the same as `(+5) 2 in Haskell`
7                    # 2 + 5

>>> f_("-", 5)(2)    # the same as `(5-) 2`
3                    # 5 - 2

>>> ff_("-", 5)(2)   # the same as `(subtract 5) 2`
-3                   # 2 - 5
```

### Build curried functions: `c_` and `cc_`
When currying a given function, 
- `c_` takes the function's arguments _from the left_ 
- while `cc_` takes them _from the right_.

> `c_(fn) == curry(fn)`   
> `cc_(fn) == c_(flip(fn))`

See also `uncurry`

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
```

### Build composition of functions: `cf_` and `cfd`
- `cf_` (_composition of function_) composes functions using the given list of functions. 
- `cfd` (_composing-function decorator_) decorates a function with the given list of functions.

> `cf_(*fn, rep=None)`  
> `cfd(*fn, rep=None)`

```python
>>> square = ff_("**", 2)        # the same as (^2) in Haskell
>>> add5 = ff_("+", 5)           # the same as (+5) in Haskell
>>> mul7 = ff_("*", 7)           # the same as (*7) in Haskell

>>> cf_(mul7, add5, square)(3)   # (*7) . (+5) . (^2) $ 3
98                               # mul7(add5(square(3))) = ((3 ^ 2) + 5) * 7

>>> cf_(square, rep=3)(2)        # cf_(square, square, square)(2) == ((2 ^ 2) ^ 2) ^ 2 = 256
256
```

`cfd` is very handy and useful to recreate previously defined functions by composing functions. All you need is to write a basic functions to do fundamental things.

### Extend and Pipefy Built-ins: `map`, `filter` and `zip`
- Extend usability while _maintaining full compatibility_
- __Pipefied__, __symbolified__ and __partially-applied__ automatically
- __Never use these functions__ if you don't know what you're doing!

> `map(f, *xs)`   
> `mapl(f, *xs)`
```python
>>> map(abs, range(-2, 3)) | unpack
[2, 1, 0, 1, 2]

>>> map("*", [1, 2, 3], [4, 5, 6]) | unpack    # the same as `zipwith`
[4, 10, 18]

>>> [4, 5, 6] | map("*", [1, 2, 3]) | unpack   # pipefied
[4, 10, 18]
```

> `filter(p, xs)`  
> `filterl(p, xs)`
```python
>>> filter(f_("==", "f"))("fun-on-functions") | unpack
['f', 'f']

>>> primes = [2, 3, 5, 7, 11, 13, 17, 19]
>>> primes | filter(cf_(ff_("==", 2), ff_("%", 3))) | unpack
[2, 5, 11, 17]
```

### Lazy Evaluation: `lazy` and `force`
- `lazy` defers the evaluation of a function(or expression) and returns the _deferred expression_.
- `force` forces the deferred-expression to be fully evaluated when needed.
  > it reminds `Haskell`'s `force x = deepseq x x`.

> `lazy(function-name, *args, **kwargs)`  
> `force(expr)`  
> `mforce([expr])`  

```python
# strictly generate a random integer between [1, 10)
>>> randint(1, 10)

# generate a lazy expression for the above
>>> deferred = lazy(randint, 1, 10)

# evaluate it when it need
>>> force(deferred)

# the same as above
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

#### when to use
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
>>> mforce(randos)      # mforce = map(force), map 'force' over deferred expressions
[6, 2, 5, 1, 9]         # exactly what we wanted
```

Here is the simple secret: if you complete `f_` or `ff_` with a function name and its arguments, and leave it unevaluated (not called), they will act as a _deferred expression_.

Not related to `lazy` operation, but you do the same thing with `uncurry`

```python
# replicate the tuple of arguments (1, 10) and then apply to uncurried function
>>> map(u_(randint))(replicate(5, (1,10)))    # u_ == uncurry
[7, 6, 1, 7, 2]
```

### Raise and assert with _expressions_: `error` and `guard`

Raise any kinds of exception in `lambda` expression as well.

> `error(MESSAGE, e=EXCEPTION_TO_RAISE)`    
```python
>>> error("Error, used wrong type", e=TypeError)

>>> error("out of range", e=IndexError)

>>> (lambda x: x if x is not None else error("Error, got None", e=ValueError))(None)
```
Likewise, use `guard` if there need _assertion_ not as a statement, but as an _expression_.

> `guard(PREDICATE, MESSAGE, e=EXCEPTION_TO_RAISE)` 
```python

>>> guard("Almost" == "enough", "'Almost' is never 'enough'")

>>> guard(rand() > 0.5, "Assertion error occurs with a 0.5 probability")

>>> guard(len(x := range(11)) == 10, f"length is not 10: {len(x)}")
```

### Exception catcher builder: `trap`

_Document will be updated_

## Utilities
### Normalize containers: `flat`
`flat` flattens all kinds of iterables except for _string-like object_ (`str`, `bytes`).

> `flat(*args)`   
> `flatl(*args)`

```python
# Assume that we regenerate 'data' every time in the examples below
>>> data = [1,2,[3,4,[[[5],6],7,{8},((9),10)],range(11,13)], (x for x in [13,14,15])]

>>> flatl(data)    
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# regardless of the number of arguments
>>> flatl(1,[2,{3}],[[[[[4]],5]]], "sofia", "maria")
[1, 2, 3, 4, 5, 'sofia', 'maria']
```
### Shell Command: `shell`
`shell` executes shell commands _synchronosly_ and _asynchronously_ and capture their outputs.

> `shell(CMD, sync=True, o=True, *, executable="/bin/bash")`

```
  --------------------------------------------------------------------
    o-value  |  return |  meaning
  --------------------------------------------------------------------
     o =  1  |  [str]  |  captures stdout/stderr (2>&1)
     o = -1  |  None   |  discard (&>/dev/null)
  otherwise  |  None   |  do nothing or redirection (2>&1 or &>FILE)
  --------------------------------------------------------------------
```
`shell` performs the same operation as `ipython`'s magic command `!!`. However, it can also be used within a `python` script.

```python
>>> output = shell("ls -1 ~")    
>>> output = "ls -1 ~" | shell           # the same

>>> shell("find . | sort" o=-1)          # run and discard the result  
>>> "find . | sort") | shell(o=-1)    

>>> shell("cat *.md", o=writer(FILE))    # redirect to FILE
>>> "cat *.md" | shell(o=writer(FILE))   # redirect to FILE
```
### Neatify data structures: `neatly` and `nprint`
`neatly` generates neatly formatted string of the complex data structures of `dict` and `list`.

`nprint` (_neatly-print_) prints data structures to `stdout` using `neatly` formatter.   
`nprint(...)` = `print(neatly(...))`  

> `nprint(DICT, _cols=INDENT, _width=WRAP, _repr=BOOL, **kwargs)`

```python
>>> o = {
...   "$id": "https://example.com/enumerated-values.schema.json",
...   "$schema": "https://json-schema.org/draft/2020-12/schema",
...   "title": "Enumerated Values",
...   "type": "object",
...   "properties": {
...     "data": {
...       "enum": [42, True, "hello", None, [1, 2, 3]]
...     }
...   }
... }
>>> nprint(o)
```
```
       $id  |  'https://example.com/enumerated-values.schema.json'
   $schema  |  'https://json-schema.org/draft/2020-12/schema'
properties  |  data  |  enum  +  42
            :        :        -  True
            :        :        -  'hello'
            :        :        -  None
            :        :        -  +  1
            :        :           -  2
            :        :           -  3
     title  |  'Enumerated Values'
      type  |  'object'
```

### Dot-accessible dictionary: `dmap`
`dmap` is a _yet another_ `dict`. It's exactly the same as `dict` but it enables to access its nested structure with '_dot notations_'.

> `dmap(DICT, **kwargs)`

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
       age  |  19
   cliburn  |      final  |     beethoven  |  'piano concerto no.3'
            :             :  rachmaninoff  |  'piano concerto no.3'
            :  semifinal  |   liszt  |  '12 transcendental etudes'
            :             :  mozart  |  'piano concerto no.22'
      name  |  'yunchan lim'
profession  |  'pianist'
```
```python
>>> del d.cliburn.semifinal
>>> d.profession = "one-in-a-million talent"
>>> nprint(d)
```
```
       age  |  19
   cliburn  |  final  |     beethoven  |  'piano concerto no.3'
            :         :  rachmaninoff  |  'piano concerto no.3'
      name  |  'yunchan lim'
profession  |  'one-in-a-million talent'
```
```python
# No such keypath
>>> d.bach.chopin.beethoven
{}
```
### Handy File Tools: `ls` and `grep` 
Use `ls` and `grep` in the same way you use in your terminal every day.   
_This is just a more intuitive alternative to_ `os.listdir` and `os.walk`. When applicable, use `shell` instead. 

> `ls(*paths, grep=REGEX, i=BOOL, r=BOOL, f=BOOL, d=BOOL, g=BOOL)`
```python
# couldn't be simpler!
>>> ls()       # the same as ls("."): get contents of the curruent dir

# expands "~" automatically
>>> ls("~")    # the same as `ls -a1 ~`: returns a list of $HOME

# support glob patterns (*, ?, [)
>>> ls("./*/*.py")

# with multiple filepaths
>>> ls(FILE, DIR, ...)
```
```python
# list up recursively and filter hidden files out
>>> ls(".git", r=True, grep="^[^\.]")
```
```python
# only files in '.git' directory
>>> ls(".git", r=True, f=True)

# only directories in '.git' directory
>>> ls(".git", r=True, d=True)
```
```python
# search recursivley and matching a pattern with `grep`
>>> ls(".", r=True, i=True, grep=".Py")    # 'i=True' for case-insensitive grep pattern
```
```
[ ..
 '.pytest_cache/v/cache/stepwise',
 'foc/__init__.py',
 'foc/__pycache__/__init__.cpython-310.pyc',
 'tests/__init__.py',
.. ]
```
```python
# regex patterns come in
>>> ls(".", r=True, grep=".py$")
```
```
['foc/__init__.py', 'setup.py', 'tests/__init__.py', 'tests/test_foc.py']
```
```python
# that's it!
>>> ls(".", r=True, grep="^(foc).*py$")

# the same as above
>>> ls("foc/*.py")
```
```
['foc/__init__.py']
```

`grep` build a filter to select items matching `REGEX` pattern from _iterables_.
> `grep(REGEX, i=BOOL)`

```python
# 'grep' builds filter with regex patterns
>>> grep(r"^(foc).*py$")(ls(".", r=True))
```
```
['foc/__init__.py']
```
See also: `HOME`, `cd`, `pwd`, `mkdir`, `rmdir`, `exists`, `dirname`, and `basename`.

### Flexible Progress Bar: `taskbar`

_Document will be updated_
