import itertools as it
import operator as op
import os
import random as rd
import re
import sys
import threading
import zipfile
from collections import deque
from datetime import datetime, timedelta
from functools import partial, reduce, wraps
from glob import glob
from inspect import signature
from itertools import accumulate, count, cycle, dropwhile, islice
from itertools import product as cprod
from itertools import takewhile, tee
from shutil import rmtree
from textwrap import fill

__all__ = [
    "safe",
    "id",
    "const",
    "seq",
    "void",
    "fst",
    "snd",
    "nth",
    "take",
    "drop",
    "head",
    "tail",
    "init",
    "last",
    "ilen",
    "pred",
    "succ",
    "odd",
    "even",
    "null",
    "chars",
    "unchars",
    "words",
    "unwords",
    "lines",
    "unlines",
    "elem",
    "nub",
    "repeat",
    "replicate",
    "cycle",
    "count",
    "tee",
    "islice",
    "product",
    "deque",
    "flip",
    "f_",
    "ff_",
    "curry",
    "c_",
    "cc_",
    "uncurry",
    "u_",
    "cf_",
    "cfd",
    "m_",
    "mm_",
    "ml_",
    "mml_",
    "v_",
    "vv_",
    "vl_",
    "vvl_",
    "mapl",
    "filterl",
    "zipl",
    "rangel",
    "enumeratel",
    "reverse",
    "sort",
    "takewhile",
    "takewhilel",
    "dropwhile",
    "dropwhilel",
    "_not",
    "_and",
    "_or",
    "_in",
    "_is",
    "_is_not",
    "_t",
    "_l",
    "_s",
    "_d",
    "_r",
    "bimap",
    "first",
    "second",
    "until",
    "iterate",
    "apply",
    "foldl",
    "foldl1",
    "foldr",
    "foldr1",
    "scanl",
    "scanl1",
    "scanr",
    "scanr1",
    "permutation",
    "combination",
    "cartprod",
    "cartprodl",
    "concat",
    "concatl",
    "concatmap",
    "concatmapl",
    "intersperse",
    "intercalate",
    "flat",
    "flatl",
    "flatt",
    "flatd",
    "flats",
    "lazy",
    "force",
    "mforce",
    "reader",
    "writer",
    "split_at",
    "chunks_of",
    "capture",
    "captures",
    "guard",
    "guard_",
    "error",
    "HOME",
    "cd",
    "pwd",
    "ls",
    "grep",
    "normpath",
    "exists",
    "dirname",
    "basename",
    "mkdir",
    "rmdir",
    "bytes_to_int",
    "int_to_bytes",
    "randbytes",
    "rand",
    "randn",
    "randint",
    "choice",
    "shuffle",
    "dmap",
    "fn_args",
    "singleton",
    "polling",
    "neatly",
    "nprint",
    "pbcopy",
    "pbpaste",
    "timestamp",
    "flist",
]


def safe(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return

    return wrapper


def id(x):
    return x


def const(x):
    return lambda _: x


def seq(_):
    return id


def void(_):
    return


@safe
def fst(x):
    return nth(1, x)


@safe
def snd(x):
    return nth(2, x)


def nth(n, x):
    return x[n - 1] if hasattr(x, "__getitem__") else next(islice(x, n - 1, None))


def take(n, x):
    return [*islice(x, n)]


def drop(n, x):
    return islice(x, n, None)


@safe
def head(x):
    return fst(x)


@safe
def tail(x):
    return drop(1, x)


@safe
def init(x):
    it = iter(x)
    o = next(it)
    for i in it:
        yield o
        o = i


@safe
def last(x):
    return deque(x, maxlen=1)[0]


def ilen(x):
    c = count()
    deque(zip(x, c), maxlen=0)
    return next(c)


def pred(x):
    return x - 1


def succ(x):
    return x + 1


def odd(x):
    return x % 2 == 1


def even(x):
    return x % 2 == 0


def null(x):
    """check if a given collection is empty"""
    return len(x) == 0


def chars(x):
    """split string 'x' into `chars`: the same as (:[]) <$> x"""
    return list(x)


def unchars(x):
    """inverse operation of 'chars': the same as 'concat'"""
    return "".join(x)


def words(x):
    return x.split()


def unwords(x):
    return " ".join(x)


def lines(x):
    return x.splitlines()


def unlines(x):
    return "\n".join(x)


def elem(x, xs):
    return x in xs


def nub(x):
    return cf_(list, dict.fromkeys)(x)


def repeat(x, nf=True):
    """create an infinite list with x value of every element
    if nf (NF, Normal Form) is set, callable objects will be evaluated.
    """
    return (x() if nf and callable(x) else x for _ in count())


def replicate(n, x, nf=True):
    """get a list of length `n` from an infinite list with `x` values"""
    return take(n, repeat(x, nf=nf))


def product(x):
    """product of the elements of given iterable"""
    return foldl1(op.mul, x)


def flip(f):
    """flip(f) takes its arguments in the reverse order of f:
    `f :: a -> b -> ... -> c -> d -> o`
    `flip(f) :: d -> c -> ... -> b -> a -> o`
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args[::-1], **kwargs)

    return wrapper


def f_(f, *args, **kwargs):
    """build left-associative partial application,
    where the given function's arguments partially evaluation from the left
    """
    return partial(o_(f), *args, **kwargs)


def ff_(f, *args, **kwargs):
    """build left-associative partial application,
    where the given function's arguments partially evaluation from the right
    """
    return f_(flip(o_(f)), *args, **kwargs)


def curry(f, n=None):
    """build curried function that takes arguments from the left.
    The currying result is simply a nested unary function.

    This function takes positional arguments only when currying.
    Use partial application `f_` before currying if you need to change kwargs
    """
    f = o_(f)
    n = n if n else len(fn_args(f))

    @wraps(f)
    def wrapper(x):
        return f(x) if n <= 1 else curry(f_(f, x), n=pred(n))

    return wrapper


# for decreasing verbosity
c_ = curry


def cc_(f):
    """build curried function that takes arguments from the right"""
    return c_(flip(o_(f)))


def uncurry(f):
    """convert a uncurried normal function to a unary function of a tuple args.
    This is not exact reverse operation of `curry`. Here `uncurry` simply does:
    `uncurry :: (a -> ... -> b -> o) -> (a, ..., b, o) -> o`
    """
    f = o_(f)

    @wraps(f)
    def wrapper(x):
        return f(*x)

    return wrapper


# for decreasing verbosity
u_ = uncurry


def cf_(*fs, rep=None):
    """compose a given list of functions then return the composed function"""

    def compose(f, g):
        return lambda x: f(g(x))

    return reduce(compose, fs * rep if rep else fs)


def cfd(*fs, rep=None):
    """decorator using the composition of functions:
    decorate a function using the composition of the given functions.
    """

    def cfdeco(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return cf_(*fs, rep=rep)(f(*args, *kwargs))

        return wrapper

    return cfdeco


def m_(f):
    """builds partial application of `map` (left-associative)
    map(f, xs) == f <$> xs

    (f <$>) == map(f,)  == f_(map, f) == m_(f)
    (<$> xs) == map(,xs) == f_(flip(map), xs)
    """
    return f_(map, f)


def mm_(xs):
    """builds partial application of `map` (right-associative)
    See also 'm_'.

    (f <$>) == map(f,)  == f_(map, f) == m_(f)
    (<$> xs) == map(,xs) == f_(flip(map), xs)
    """
    return ff_(map, xs)


def ml_(f):
    """unpacks the result in list after `m_`"""
    return cfd(list)(m_(f))


def mml_(f):
    """unpacks the result in list after `mm_`"""
    return cfd(list)(mm_(f))


def v_(f):
    """builds partial application of `filter` (left-associative)
    f: predicate or filter funtion
    """
    return f_(filter, f)


def vv_(xs):
    """builds partial application of `filter` (right-associative)
    xs: iterable
    """
    return ff_(filter, xs)


def vl_(f):
    """unpacks the result in list after `v_`"""
    return cfd(list)(v_(f))


def vvl_(f):
    """unpacks the result in list after `vv_`"""
    return cfd(list)(vv_(f))


mapl = cfd(list)(map)
mapl.__doc__ = "unpacks the result in list after `map`"


filterl = cfd(list)(filter)
filterl.__doc__ = "unpacks the result in list after `filter`"


zipl = cfd(list)(zip)
zipl.__doc__ = "unpacks the result in list after `filter`"


rangel = cfd(list)(range)
rangel.__doc__ = "unpacks the result in list after `range`"


enumeratel = cfd(list)(enumerate)
enumeratel.__doc__ = "unpacks the result in list after `enumerate`"


def reverse(x):
    """returns reversed sequence"""
    return list(x)[::-1]


# "for clarity of function names"
sort = sorted


takewhilel = cfd(list)(takewhile)
takewhilel.__doc__ = "unpacks the result in list after `takewhile`"


dropwhilel = cfd(list)(dropwhile)
dropwhilel.__doc__ = "unpacks the result in list after `dropwhile`"


def _not(x):
    """`not` as a function"""
    return not x


def _and(a, b):
    """`and` as a function"""
    return a and b


def _or(a, b):
    """`and` as a function"""
    return a or b


# `in` as a function
_in = flip(op.contains)


# `is` as a function
_is = op.is_


# `is not` as a function
_is_not = op.is_not


def _t(*args):
    """functional form of tuple constructor"""
    return args


_l = cfd(list)(_t)
_l.__doc__ = "functional form of list constructor"


_s = cfd(set)(_t)
_s.__doc__ = "functional form of set constructor"


_d = cfd(deque)(_t)
_d.__doc__ = "functional form of deque constructor"


_r = cfd(lambda x: x[::-1])(_t)
_r.__doc__ = "generate args in the reverse order of '_t'"


def bimap(f, g, x):
    """map over both 'first' and 'second' arguments at the same time
    bimap(f, g) == first(f) . second(g)
    """
    return f(fst(x)), g(snd(x))


def first(f, x):
    """map covariantly over the 'first' argument"""
    # return f(fst(x)), snd(x)
    return bimap(f, id, x)


def second(g, x):
    """map covariantly over the 'second' argument"""
    # return fst(x), g(snd(x))
    return bimap(id, g, x)


def until(p, f, x):
    while not p(x):
        x = f(x)
    return x


def iterate(f, x):
    while True:
        yield x
        x = f(x)


def apply(f, *args, **kwargs):
    return o_(f)(*args, **kwargs)


def foldl(f, initial, xs):
    """left-associative fold of an iterable. The same as 'foldl' in Haskell"""
    return reduce(o_(f), xs, initial)


def foldl1(f, xs):
    """`foldl` without initial value. The same as 'foldl1' in Haskell"""
    return reduce(o_(f), xs)


def foldr(f, inital, xs):
    """right-associative fold of an iterable. The same as 'foldr' in Haskell"""
    return reduce(flip(o_(f)), xs[::-1], inital)


def foldr1(f, xs):
    """`foldr` without initial value. The same as 'foldr1' in Haskell"""
    return reduce(flip(o_(f)), xs[::-1])


@cfd(list)
def scanl(f, initial, xs):
    """returns a list of successive reduced values from the left
    The same as `scanl` in Haskell"""
    return accumulate(xs, o_(f), initial=initial)


@cfd(list)
def scanl1(f, xs):
    """`scanl` without starting value. The same as 'scanl1' in Haskell"""
    return accumulate(xs, o_(f))


@cfd(reverse)
def scanr(f, initial, xs):
    """returns a list of successive reduced values from the right
    The same as `scanr` in Haskell
    """
    return accumulate(xs[::-1], flip(o_(f)), initial=initial)


@cfd(reverse)
def scanr1(f, xs):
    """`scanr` without starting value. The same as 'scanr1' in Haskell"""
    return accumulate(xs[::-1], flip(o_(f)))


def capture(p, string):
    x = captures(p, string)
    if x:
        return fst(x)


def captures(p, string):
    return re.compile(p).findall(string)


def o_(f):
    """get a function from symbolic operators"""
    return {
        "+": op.add,
        "-": op.sub,
        "*": op.mul,
        "/": op.truediv,
        "//": op.floordiv,
        "**": op.pow,
        "@": op.matmul,
        "%": op.mod,
        "&": op.and_,
        "|": op.or_,
        "^": op.xor,
        "<<": op.lshift,
        ">>": op.rshift,
        "==": op.eq,
        "!=": op.ne,
        ">": op.gt,
        ">=": op.ge,
        "<": op.lt,
        "<=": op.le,
        "!!": op.getitem,
        ",": lambda a, b: (a, b),
        "..": rangel,
        "[]": _l,
        "()": _t,
        "{}": _s,
        "-<": _in,
        "~<": capture,
        "~~<": captures,
    }.get(f, f)


def permutation(x, r, rep=False):
    return cprod(x, repeat=r) if rep else it.permutations(x, r)


def combination(x, r, rep=False):
    return it.combinations_with_replacement(x, r) if rep else it.combinations(x, r)


cartprod = f_(cprod, repeat=1)
cartprod.__doc__ = "returns Cartesian product"


cartprodl = cfd(list)(cartprod)
cartprodl.__doc__ = "unpacks the result in list after `cartprod`"


# concatenates all elements of iterables"
concat = it.chain.from_iterable


concatl = cfd(list)(concat)
concatl.__doc__ = "unpacks the result in list after `concat`"


concatmap = cfd(concat)(map)
concatmap.__doc__ = "map a function over the given iterable then concat it"


concatmapl = cfd(list)(concatmap)
concatmapl.__doc__ = "unpacks the result in list after `concatmap`"


def intersperse(sep, x):
    """inserts an element between the elements of the list"""
    return concatl(zip(repeat(sep), x))[1:]


intercalate = cfd(concatl)(intersperse)
intersperse.__doc__ = "inserts the given list between the lists then concat it"


def flat(*args):
    """flatten all kinds of iterables (except for string-like object)"""

    def ns_iter(x):
        return (
            hasattr(x, "__iter__")
            and not isinstance(x, str)
            and not isinstance(x, bytes)
        )

    def go(xss):
        if ns_iter(xss):
            for xs in xss:
                yield from go([*xs] if ns_iter(xs) else xs)
        else:
            yield xss

    return go(args)


flatl = cfd(list)(flat)
flatl.__doc__ = "flatten iterables into list"

flatt = cfd(tuple)(flat)
flatt.__doc__ = "flatten iterables into tuple"

flatd = cfd(deque)(flat)
flatd.__doc__ = "flatten iterables into deque"

flats = cfd(set)(flat)
flats.__doc__ = "flatten iterables into set"


# easy-to-use alias for lazy operation
lazy = f_


def force(x, *args, **kwargs):
    """forces the delayed-expression to be fully evaluated"""
    return x(*args, **kwargs) if callable(x) else x


mforce = ml_(force)
mforce.__doc__ = "map 'force' over iterables of delayed-evaluation"


def reader(f=None, mode="r", zipf=False):
    """get ready to read stream from a file or stdin, then returns the handle"""
    if f is not None:
        guard(exists(f, "f"), f"Error, not found such a file: {f}")
    return (
        sys.stdin
        if f is None
        else zipfile.ZipFile(normpath(f), mode)
        if zipf
        else open(normpath(f), mode)
    )


def writer(f=None, mode="w", zipf=False):
    """get ready to write stream to a file or stout, then returns the handle"""
    return (
        sys.stdout
        if f is None
        else zipfile.ZipFile(normpath(f), mode)
        if zipf
        else open(normpath(f), mode)
    )


def split_at(ix, x):
    """split iterables at the given splitting-indices"""
    s = flatl(0, ix, None)
    return ([*it.islice(x, begin, end)] for begin, end in zip(s, s[1:]))


def chunks_of(n, x, fillvalue=None, fill=True):
    if not fill:
        x = list(x)
        x = x[: len(x) // n * n]
    """split interables into the given `n-length` pieces"""
    return it.zip_longest(*(iter(x),) * n, fillvalue=fillvalue)


def guard(p, msg="guard", e=SystemExit):
    """'assert' as a function or expression"""
    if not p:
        error(msg=msg, e=e)


def guard_(f, msg="guard", e=SystemExit):
    """partial application builder for 'guard':
    the same as 'guard', but the positional predicate is given
    as a function rather than an boolean expression"""
    return lambda x: seq(guard(f(x), msg=msg, e=e))(x)


def error(msg="error", e=SystemExit):
    """'raise' an exception with a function or expression"""
    raise e(msg)


def HOME():
    """get the current user's home directory: the same as '$HOME'"""
    return os.getenv("HOME")


def cd(path=None):
    """change directories: similar to the shell-command 'cd'"""
    if path:
        os.chdir(normpath(path, abs=True))
    else:
        os.chdir(HOME())
    return pwd()


def pwd():
    """get the current directory: similar to the shell-command 'pwd'"""
    return os.getcwd()


def ls(path=".", grep=None, i=False, r=False):
    """list the given <path>'s contents of files and directories: 'ls -1 <path>'
    - glob patterns (*,?,[) in <path> are allowed.
    - if 'grep=<regex>' is given, it's similar to 'ls -1 <path> | grep <regex>'
    - if i is set, it maks 'grep' case-insensitive (turns on grep's -i flag)
    - if r is set, it behaves like 'find <path>', recursively listing <path>
    """
    paths = (
        glob(normpath(path))
        if re.search(r"[\*\+\?\[]", path)
        else cf_(
            _l,
            guard_(exists, f"Error, no such file or directory: {path}"),
            normpath,
        )(path)
    )
    fs = flat(
        [f"{path}/{f}" for f in os.listdir(path)] if exists(path, "d") else path
        for path in paths
    )
    finish = sort if grep is None else cf_(sort, globals()["grep"](grep, i=i))
    return (
        cf_(finish, flat)(
            ls(f, grep=grep, i=i, r=r) if exists(f, "d") else f for f in fs
        )
        if r
        else finish(fs)
    )


@safe
def grep(regex, i=False):
    return vl_(f_(re.search, regex, flags=re.IGNORECASE if i else 0))


def normpath(path, abs=False):
    return cf_(
        os.path.abspath if abs else id,
        os.path.normpath,
        os.path.expanduser,
    )(path)


def exists(path, kind=None):
    path = normpath(path)
    if kind == "f":
        return os.path.isfile(path)
    elif kind == "d":
        return os.path.isdir(path)
    else:
        return os.path.exists(path)


def dirname(*args, prefix=False, abs=False):
    if len(args) > 1:
        args = [normpath(a, abs=True) for a in args]
        return os.path.commonprefix(args) if prefix else os.path.commonpath(args)
    else:
        args = [normpath(a, abs=abs) for a in args]
        d = os.path.dirname(*args)
        return d if d else "."


def basename(path):
    return cf_(os.path.basename, normpath)(path)


def mkdir(path, mode=0o755):
    path = normpath(path)
    os.makedirs(path, mode=mode, exist_ok=True)
    return path


def rmdir(path, rm_rf=False):
    path = normpath(path)
    if rm_rf:
        rmtree(path)
    else:
        os.removedirs(path)


def bytes_to_int(x, byteorder="big"):
    return int.from_bytes(x, byteorder=byteorder)


def int_to_bytes(x, size=None, byteorder="big"):
    if size is None:
        size = (x.bit_length() + 7) // 8
    return x.to_bytes(size, byteorder=byteorder)


def randbytes(n):
    """generate cryptographically secure random bytes"""
    return os.urandom(n)


def rand(x=None, high=None, size=None):
    return (
        [rd.uniform(x, high) for _ in range(size)]  # #args == 3
        if size is not None
        else rd.uniform(x, high)  # #args == 2
        if high is not None
        else rd.uniform(0, x)  # #args == 1
        if x is not None
        else rd.random()  # #args == 0
    )


def randn(mu=0, sigma=1, size=None):
    return (
        [rd.gauss(mu, sigma) for _ in range(size)]
        if size is not None
        else rd.uniform(mu, sigma)
    )


def randint(x=None, high=None, size=None):
    """generate random integer cryptographically secure and faster than numpy's.
    return random integer(s) in range of [low, high)
    """

    def rint(high=1 << 256, low=0):
        guard(low < high, f"Error, low({low}) must be less than high({high})")
        x = high - low
        return low + (bytes_to_int(randbytes((x.bit_length() + 7) // 8)) % x)

    return (
        [rint(high, x) for _ in range(size)]  # #args == 3
        if size is not None
        else rint(high, x)  # #args == 2
        if high is not None
        else rint(x)  # #args == 1
        if x is not None
        else rint()  # #args == 0
    )


def shuffle(x):
    """Fisher-Yates shuffle in a cryptographically secure way"""
    for i in range(len(x) - 1, 0, -1):
        j = randint(0, i)
        x[i], x[j] = x[j], x[i]
    return x


def choice(x, size=None, replace=False, p=None):
    """Generate a sample with/without replacement from a given iterable"""

    def fromp(x, probs, e=1e-6):
        guard(
            len(x) == len(probs),
            f"Error, not the same size: {len(x)}, {len(probs)}",
        )
        guard(
            1 - e < sum(probs) < 1 + e,
            f"Error, sum of probs({sum(probs)}) != 1",
        )
        r = rand()
        for y, p in zip(x, scanl1(f_("+"), probs)):
            if r < p:
                return y

    if p is not None:
        return fromp(x, p)
    if size is None:
        return x[randint(len(x))]
    else:
        size = int(len(x) * size) if 0 < size < 1 else size
        return [
            x[i]
            for i in (
                randint(0, len(x), size)
                if len(x) < size or replace
                else shuffle(rangel(len(x)))[:size]
            )
        ]


class dmap(dict):
    """dot-accessible dict(map)"""

    def __init__(self, *args, **kwargs):
        super(dmap, self).__init__(*args, **kwargs)
        for key, val in self.items():
            self[key] = self._g(val)

    def _g(self, val):
        if isinstance(val, dict):
            return dmap(val)
        elif isinstance(val, list):
            return [self._g(x) for x in val]
        return val

    def __getattr__(self, key):
        if key.startswith("__"):  # disabled for compat with ipython
            return
        if key not in self and key != "_ipython_canary_method_should_not_exist_":
            self[key] = dmap()
        return self[key]

    def __setattr__(self, key, val):
        self[key] = self._g(val)

    def __delattr__(self, key):
        if key in self:
            del self[key]

    def __or__(self, o):
        self.update(o)
        return self


def fn_args(f):
    """get positional arguments of a given function"""
    return [
        *filter(
            lambda x: bool(x.strip()) and x[0] != "*" and x[0] != "/",
            re.sub(
                r"(\w+=[\=\(\)\{\}\:\'\[\]\w,\s]*|\*\*\w+)",
                "",
                signature(f).__str__()[1:-1],
            ).split(", "),
        )
    ]


def singleton(cls):
    instances = {}

    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


def polling(f, sec, args=None, kwargs=None):
    def go():
        polling(f, sec, args, kwargs)
        if args and kwargs:
            f(*args, **kwargs)
        elif kwargs:
            f(**kwargs)
        elif args:
            f(*args)
        else:
            f()

    t = threading.Timer(sec, go, args, kwargs)
    t.start()
    return t


def neatly(x={}, _cols=None, _width=10000, _root=True, **kwargs):
    """create neatly formatted string for data structure of 'dict' and 'list'"""

    def munch(x):
        return (
            f"-  {x[3:]}"
            if x and x[0] == "|"
            else f"   {x[3:]}"
            if x and x[0] == ":"
            else f"-  {x}"
        )

    def bullet(o, s):
        return (
            (munch(x) for x in s)
            if isinstance(o, list)
            else (f":  {x}" if i else f"|  {x}" for i, x in enumerate(s))
        )

    def filine(x, width, initial, subsequent):
        return fill(
            x,
            width=width,
            break_on_hyphens=False,
            drop_whitespace=False,
            initial_indent=initial,
            subsequent_indent=subsequent,
        )

    if isinstance(x, dict):
        d = x | kwargs
        if not d:
            return ""
        _cols = _cols or max(map(len, d.keys()))
        return unlines(
            filine(v, _width, f"{k:>{_cols}}  ", f"{' ':>{_cols}}     ")
            for a, o in d.items()
            for k, v in [
                ("", b) if i else (a, b)
                for i, b in enumerate(bullet(o, lines(neatly(o, _root=0))))
            ]
        )
    elif isinstance(x, list):
        if _root:
            return neatly({"'": x}, _root=0)
        return unlines(
            filine(v, _width, "", "   ")
            for o in x
            for v in bullet(o, lines(neatly(o, _root=0)))
        )
    else:
        return repr(x)


def nprint(x={}, _cols=None, _width=10000, **kwargs):
    """neatly print data structures of 'dict' and 'list' using `neatly`"""
    print(neatly(x, _cols=_cols, _width=_width, **kwargs))


def pbcopy(x):
    import subprocess

    subprocess.Popen("pbcopy", stdin=subprocess.PIPE).communicate(x.encode())


def pbpaste():
    import subprocess

    return subprocess.Popen("pbpaste", stdout=subprocess.PIPE).stdout.read().decode()


def timestamp(
    origin=None,
    w=0,
    d=0,
    h=0,
    m=0,
    s=0,
    from_str=None,
    to_str=False,
):
    if from_str:
        t = datetime.strptime(from_str, "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
    else:
        dt = timedelta(
            weeks=w,
            days=d,
            hours=h,
            minutes=m,
            seconds=s,
        ).total_seconds()
        if origin is None:
            origin = datetime.utcnow().timestamp()
            t = origin + dt

    return to_str and f"{datetime.fromtimestamp(t).isoformat()[:26]}Z" or t


def __sig__():
    def sig(o):
        try:
            return signature(o).__str__()
        except:
            return " is valid, but live-inspect not available"

    return dmap({x: x + sig(eval(x)) for x in __all__})


def flist(to_dict=False):
    if to_dict:
        return __sig__()
    else:
        nprint(__sig__(), _cols=14)
