# foc

<p align="center"> <img src="https://raw.githubusercontent.com/thyeem/foc/main/foc.png" height="180"/></p>

[![foc](https://img.shields.io/pypi/v/foc)](https://pypi.org/project/foc/)

_Func-Oriented Code_ or _Francis' Odd Collection_.

`foc` is a _non-frilled_ and _seamlessly integrated_ functional `Python` tool.


- provides a collection of ___higher-order functions___ and ___placeholder lambda syntax___ (`_`)
- provides an easy way to ___compose functions with symbols___. (`^` and `|`)


## Install
```bash
$ pip install -U foc
```

## Use

For more information, see the [examples](https://github.com/thyeem/foc/blob/main/foc/__init__.py#L150) provided with each function.

```python
>>> from foc import *

>>> (_ + 7)(3)  # (lambda x: x + 7)(3)
10

>>> 3 | (_ + 4) | (_ * 6)  # (3 + 4) * 6
42

>>> (length ^ range)(10)  # length(range(10))
10

>>> cf_(collect, filter(even), range)(10)  # (collect . filter(even . range)(10)
[0, 2, 4, 6, 8]

>>> ((_ * 5) ^ nth(3) ^ range)(5)  # range(5)[3] * 5
10

>>> cf_(sum, map(_ + 1), range)(10)  # sum(map((+1), [0..9]))
55

>>> map((_ * 3) ^ (_ + 2))(range(5)) | sum  #  sum(map((*3) . (+2)), [0..4])
60

>>> range(73, 82) | map(chr) | unchars  # unchar(map(chr, range(73, 82)))
'IJKLMNOPQ'
```

`foc` provides two ways to __compose functions with symbols__
> If you don't like function composition using symbols, use `cf_`.   
> In fact, it's the most reliable and safe way to use it for all functions.

| Symbol          | Description                            | Evaluation order | Available               |
|-----------------|----------------------------------------|------------------|-------------------------|
| `^` (caret)     | same as dot(`.`) _mathematical symbol_ | backwards        | every (first only `fx`) |
| `\|` (pipeline) | in _Unix_ pipeline manner              | in order         | `fx`-functions          |

`fx`-functions (or _Function eXtension_) are `fx`-decorated (`@fx`) functions.  
If you want to make a function the `fx` function on the fly, ___just wrap the function___ in `fx`. 

```python
>>> 7 | fx(lambda x: x * 6) 
42

>>> @fx
... def fn(x, y):
...     ...
```


_Partial application_ using placeholders is also possible when accessing items
in `dict`, `object` or `iterable`.

| Operator  | Equiv Function        |
|-----------|-----------------------|
| `_[_]`    | `op.getitem`          |
| `_[item]` | `op.itemgetter(item)` |
| `_._`     | `getattr`             |
| `_.attr`  | `op.attrgetter(attr)` |

```python
# dict
>>> d = dict(one=1, two=2, three="three")
>>> (_[_])(d)("two")  # curry(lambda a, b: a[b])(d)("two")
2
>>> (_["one"])(d)  # (lambda x: x["one"])(d)
1
>>> cf_(_[2:4], _["three"])(d)  # d["three"][2:4]
're'

# iterable
>>> r = range(5)
>>> (_[_])(r)(3)  # curry(lambda a, b: a[b])(r)(3)
True
>>> (_[3])(r)  # (lambda x: x[3])(r)
True

# object
>>> o = type('', (), {"one": 1, "two": 2, "three": "three"})()
>>> (_._)(o)("two")  # curry(lambda a, b: getattr(a, b))(o)("two")
True
>>> (_.one)(o)  # (lambda x: x.one)(o)
True
>>> o | _.three | _[2:4]  # o.three[2:4]
're'
```


## Real-World Example
A _causal self-attention_ of the `transformer` model based on `pytorch` can be described as follows.  
_Somebody_ insists that this helps to follow the process flow without distraction. (_plus, 3-5% speed-up_)

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
            g_(_.view)(B, S, E),  # (B, S, N, H) -> (B, S, E)
            torch.Tensor.contiguous,  # contiguos in-memory tensor
            g_(_.transpose)(1, 2),  # (B, S, N, H)
            _ @ v,  # (B, N, S, S) x (B, N, S, H) -> (B, N, S, H)
            self.dropout_attn,  # attention dropout
            f_(F.softmax, dim=-1),  # softmax
            g_(_.masked_fill)(mask == 0, float("-inf")),  # no-look-ahead
            _ / math.sqrt(k.size(-1)),  # / sqrt(d_k)
            _ @ k.transpose(-2, -1),  # Q @ K^T -> (B, N, S, S)
        )(q)
```
