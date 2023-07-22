import itertools as it
import operator as op
import os
import re
import threading
from collections import deque
from datetime import datetime, timedelta
from functools import partial, reduce, wraps
from inspect import signature
from itertools import count, cycle, dropwhile, takewhile, tee
from shutil import rmtree
from textwrap import fill

__all__ = [
    "flist",
    "deque",
    "safe",
    "id",
    "const",
    "fst",
    "snd",
    "nth",
    "take",
    "drop",
    "head",
    "tail",
    "last",
    "init",
    "pred",
    "succ",
    "odd",
    "even",
    "null",
    "words",
    "unwords",
    "lines",
    "unlines",
    "elem",
    "repeat",
    "replicate",
    "cycle",
    "count",
    "tee",
    "product",
    "flip",
    "f_",
    "ff_",
    "curry",
    "c_",
    "cc_",
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
    "tup",
    "not_",
    "and_",
    "or_",
    "in_",
    "is_",
    "is_not_",
    "bop",
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
    "lazy",
    "force",
    "mforce",
    "flat",
    "flatl",
    "flatt",
    "flatd",
    "flats",
    "fread",
    "fwrite",
    "split_at",
    "chunks_of",
    "capture",
    "captures",
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
    "random_bytes",
    "random_int",
    "choice",
    "shuffle",
    "dmap",
    "fn_args",
    "singleton",
    "polling",
    "neatly",
    "nprint",
    "timestamp",
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


@safe
def fst(x):
    return nth(1, x)


@safe
def snd(x):
    return nth(2, x)


def nth(n, x):
    return x[n - 1] if hasattr(x, "__getitem__") else next(it.islice(x, n - 1, None))


def take(n, x):
    return [*it.islice(x, n)]


def drop(n, x):
    return it.islice(x, n, None)


@safe
def head(x):
    return fst(x)


@safe
def tail(x):
    return drop(1, x)


@safe
def last(x):
    return deque(x, maxlen=1)[0]


@safe
def init(x):
    it = iter(x)
    o = next(it)
    for i in it:
        yield o
        o = i


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


def words(x):
    return x.split()


def unwords(x):
    return " ".join(x)


def lines(x):
    return x.split("\n")


def unlines(x):
    return "\n".join(x)


def elem(x, xs):
    return x in xs


def repeat(x):
    return (x for _ in it.count())


def replicate(n, x):
    return take(n, repeat(x))


def product(x):
    """product of the elements of given iterable"""
    return foldl1(op.mul, x)


def flip(f):
    """flip(f) takes its arguments in the reverse order of f"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args[::-1], **kwargs)

    return wrapper


def f_(f, *args, **kwargs):
    """build left-associative partial application,
    where the given function's arguments partially evaluation from the left"""

    return partial(bop(f), *args, **kwargs)


def ff_(f, *args, sgra=False, **kwargs):
    """build left-associative partial application,
    where the given function's arguments partially evaluation from the right

    Passing arguments in reverse order for a function is painful.
    `ff_` takes arguments _in order_ by default (`sgra=False`).
    When 'sgra=True', it will take arguments in reverse order.

    naming: 'sgra' == 'args'[::-1]
    """
    if sgra:
        return f_(flip(bop(f)), *args, **kwargs)
    else:
        return flip(f_(flip(bop(f)), *args[::-1], **kwargs))


def curry(f, n=None):
    """build curried function that takes arguments from the left.
    The currying result is simply a nested unary function.

    This function takes positional arguments only when currying.
    Use partial application `f_` before currying if you need to change kwargs"""
    f = bop(f)
    n = n if n else len(fn_args(f))

    @wraps(f)
    def wrapper(x):
        return f(x) if n <= 1 else curry(f_(f, x), n=pred(n))

    return wrapper


# for decreasing verbosity
c_ = curry


def cc_(f):
    """build curried function that takes arguments from the right"""
    return c_(flip(bop(f)))


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
    f: predicate or filter funtion"""
    return f_(filter, f)


def vv_(xs):
    """builds partial application of `filter` (right-associative)
    xs: iterable"""
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


def tup(*args):
    """construct tuple with given arguments: this mirros `dict(**kwargs)`"""
    return args


def not_(x):
    """`not` as a function"""
    return not x


def and_(a, b):
    """`and` as a function"""
    return a and b


def or_(a, b):
    """`and` as a function"""
    return a or b


# `in` as a function
in_ = op.contains


# `is` as a function
is_ = op.is_


# `is not` as a function
is_not_ = op.is_not


def bop(f):
    """symbolic binary operators"""
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
        "[]": op.getitem,
        ",": lambda a, b: (a, b),
    }.get(f, f)


def bimap(f, g, x):
    """map over both 'first' and 'second' arguments at the same time
    bimap(f, g) == first(f) . second(g)
    """
    return cf_(f_(first, f), f_(second, g))(x)


def first(f, x):
    """map covariantly over the 'first' argument"""
    return f(fst(x)), snd(x)


def second(g, x):
    """map covariantly over the 'second' argument"""
    return fst(x), g(snd(x))


def until(p, f, x):
    while not p(x):
        x = f(x)
    return x


def iterate(f, x):
    while True:
        yield x
        x = f(x)


def apply(f, *args):
    return f(*args)


def foldl(f, initial, xs):
    """left-associative fold of an iterable. The same as 'foldl' in Haskell"""
    return reduce(bop(f), xs, initial)


def foldl1(f, xs):
    """`foldl` without initial value. The same as 'foldl1' in Haskell"""
    return reduce(bop(f), xs)


def foldr(f, inital, xs):
    """right-associative fold of an iterable. The same as 'foldr' in Haskell"""
    return reduce(flip(bop(f)), xs[::-1], inital)


def foldr1(f, xs):
    """`foldr` without initial value. The same as 'foldr1' in Haskell"""
    return reduce(flip(bop(f)), xs[::-1])


@cfd(list)
def scanl(f, initial, xs):
    """returns a list of successive reduced values from the left
    The same as `scanl` in Haskell"""
    return it.accumulate(xs, bop(f), initial=initial)


@cfd(list)
def scanl1(f, xs):
    """`scanl` without starting value. The same as 'scanl1' in Haskell"""
    return it.accumulate(xs, bop(f))


@cfd(reverse)
def scanr(f, initial, xs):
    """returns a list of successive reduced values from the right
    The same as `scanr` in Haskell"""
    return it.accumulate(xs[::-1], flip(bop(f)), initial=initial)


@cfd(reverse)
def scanr1(f, xs):
    """`scanr` without starting value. The same as 'scanr1' in Haskell"""
    return it.accumulate(xs[::-1], flip(bop(f)))


def permutation(x, r, rep=False):
    return it.product(x, repeat=r) if rep else it.permutations(x, r)


def combination(x, r, rep=False):
    return it.combinations_with_replacement(x, r) if rep else it.combinations(x, r)


cartprod = ff_(it.product, repeat=1)
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


def lazy(f, *args, **kwargs):
    """delays the evaluation of a function(or expression) using generator"""

    f = bop(f)

    def g(*a, **k):
        yield f(*a, **k) if callable(f) else f

    return cfd(next)(f_(g, *args, **kwargs))


def force(x, *args, **kwargs):
    """forces the delayed-expression to be fully evaluated"""
    return x(*args, **kwargs) if callable(x) else x


mforce = ml_(force)
mforce.__doc__ = "map 'force' over iterables of delayed-evaluation"


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


@cfd(flat)
def fread(*args):
    """flat-read: read iterables from objects or files then flatten them"""
    f = cf_(normpath, bytes.decode)

    for x in flat(args):
        if isinstance(x, bytes):
            if exists(f(x), "f"):
                yield open(f(x), "r").read().splitlines()
            else:
                error(f"Error, not found file: {f(x)}")
        else:
            yield x


def fwrite(f, *args):
    """flat-write: get iterables flattened then write it to a file"""
    with open(f, "w") as fh:
        for line in flat(args):
            fh.write(f"{line}\n")
    return f


def split_at(ix, x):
    """split iterables at the given splitting-indices"""
    s = flatl(0, ix, None)
    return ([*it.islice(x, begin, end)] for begin, end in zip(s, s[1:]))


def chunks_of(n, x, fill=None):
    """split interables into the given `n-length` pieces"""
    return it.zip_longest(*(iter(x),) * n, fillvalue=fill)


def capture(p, string):
    x = captures(p, string)
    if x:
        return fst(x)


def captures(p, string):
    return re.compile(p).findall(string)


def error(str, e=Exception):
    raise e(str)


def HOME():
    return os.getenv("HOME")


def cd(path=None):
    if path:
        os.chdir(normpath(path, abs=True))
    else:
        os.chdir(HOME())
    return pwd()


def pwd():
    return os.getcwd()


def ls(d=".", path=False, abs=False):
    d = normpath(d, abs=abs)
    return [f"{d}/{f}" for f in os.listdir(d)] if path else os.listdir(d)


@safe
def grep(regex):
    return vl_(f_(re.search, regex))


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


def random_bytes(n):
    """generate cryptographically secure random bytes"""
    return os.urandom(n)


def random_int(*args):
    """generate random integer cryptographically secure and faster than numpy.
    return random integer(s) in range of [low, high)"""

    def rint(high, low=0):
        assert high > low, "Error, low >= high"
        x = high - low
        return low + (
            bytes_to_int(
                random_bytes((x.bit_length() + 7) // 8),
            )
            % x
        )

    if not args:
        return rint(1 << 256)
    elif len(args) < 3:
        return rint(*args[::-1])
    elif len(args) == 3:
        return [rint(*args[:2][::-1]) for _ in range(args[-1])]
    else:
        error(f"Error, wrong number of args: {len(args)}", e=SystemExit)


def shuffle(x):
    """Fisher-Yates shuffle in a cryptographically secure way"""
    x = list(x)
    for i in range(len(x) - 1, 0, -1):
        j = random_int(0, i)
        x[i], x[j] = x[j], x[i]
    return x


def choice(x, size=None, replace=False):
    """Generate a sample with/without replacement from a given iterable"""
    x = list(x)
    if size is None:
        return x[random_int(len(x))]
    else:
        size = int(len(x) * size) if 0 < size < 1 else size
        replace = True if len(x) < size else replace
        return [
            x[i]
            for i in (
                random_int(0, len(x), size)
                if replace
                else shuffle(range(len(x)))[:size]
            )
        ]


class dmap(dict):
    """dot-accessible dict(map)"""

    __delattr__ = dict.__delitem__

    def __getattr__(self, key):
        if key not in self and key != "_ipython_canary_method_should_not_exist_":
            self[key] = dmap()
        o = self[key]
        return dmap(o) if type(o) is dict else o

    def __setattr__(self, key, val):
        if isinstance(val, dict):
            self[key] = dmap(val)
        else:
            self[key] = val

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


def neatly(x={}, _width=100, _indent=None, **kwargs):
    """generate justified string of 'dict' or 'dict-items'"""
    d = {**x, **kwargs}
    _indent = _indent or max(map(len, d.keys())) + 1
    if _width < _indent:
        error(f"Error, neatly print with invalid width: {_width}")

    def go(x):
        if isinstance(x, dict):
            return neatly(**x, _width=_width - _indent - 6)
        else:
            return x

    def lf(k, v):
        return [
            (" ", v) if i else (k, v)
            for i, v in enumerate(filter(cf_(not_, null), lines(f"{v}")))
        ]

    return "\n".join(
        fill(
            f"{v}",
            width=_width,
            break_on_hyphens=False,
            drop_whitespace=False,
            initial_indent=f"{k.replace('_','-'):>{_indent}}  |  ",
            subsequent_indent=f"{' ':>{_indent}}  |  ",
        )
        for k, v in d.items()
        for k, v in lf(k, go(v))
    )


def nprint(x={}, _width=100, _indent=None, **kwargs):
    """neatly print dictionary using `neatly` formatter"""
    print(neatly(**x, _width=_width, _indent=_indent, **kwargs))


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

    return dmap({x: x + sig(eval(x)) for x in __all__[2:]})


def flist(to_dict=False):
    if to_dict:
        return __sig__()
    else:
        nprint(__sig__(), _indent=14)
