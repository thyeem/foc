import itertools as it
import operator as op
import os
import re
import threading
from collections import deque
from datetime import datetime, timedelta
from functools import cache, partial, reduce, wraps
from inspect import signature
from itertools import count, cycle, dropwhile, takewhile
from shutil import rmtree
from textwrap import fill

__all__ = [
    "safe",
    "not_",
    "id",
    "take",
    "drop",
    "head",
    "last",
    "init",
    "tail",
    "fst",
    "snd",
    "nth",
    "pred",
    "succ",
    "odd",
    "even",
    "null",
    "words",
    "unwords",
    "lines",
    "unlines",
    "iterate",
    "repeat",
    "replicate",
    "cycle",
    "count",
    "takewhile",
    "dropwhile",
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
    "bimap",
    "first",
    "second",
    "fold",
    "fold1",
    "scan",
    "scan1",
    "permutation",
    "combination",
    "cprod",
    "concat",
    "concatl",
    "concatmap",
    "concatmapl",
    "flat",
    "flatl",
    "flatt",
    "flatd",
    "flats",
    "fread",
    "fwrite",
    "split_at",
    "chunk_of",
    "capture",
    "captures",
    "error",
    "HOME",
    "pwd",
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
    "dmap",
    "fn_args",
    "singleton",
    "polling",
    "fmt",
    "timestamp",
]


def safe(msg=None):
    def run(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except:
                if msg:
                    print(f"{msg}")

        return wrapper

    return run


# `not` as a function
not_ = op.not_


def id(x):
    return x


def take(n, x):
    return [*it.islice(x, n)]


def drop(n, x):
    return it.islice(x, n, None)


def head(x):
    return fst(x)


@safe()
def last(x):
    return x[-1]


def init(x):
    return it.islice(x, len(x) - 1)


def tail(x):
    return drop(1, x)


def fst(x):
    return nth(x, 1)


def snd(x):
    return nth(x, 2)


def nth(x, n):
    assert n > 0, f"Error, not a positive integer: {n}"
    if hasattr(x, "__getitem__"):
        return op.itemgetter(n - 1)(x)
    else:
        for _ in range(n - 1):
            x.__next__()
        return x.__next__()


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


def iterate(f, x):
    while True:
        yield x
        x = f(x)


def repeat(x):
    return (x for _ in it.count())


def replicate(n, x):
    return (x for _ in range(n))


def product(x):
    """product of the elements of given iterable"""
    return fold1(op.mul, x)


def flip(f):
    """flip(f) takes its arguments in the reverse order of f"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args[::-1], **kwargs)

    return wrapper


# Partial evaluation of function
f_ = partial


def ff_(f, *args, sgra=False, **kwargs):
    """Partial application of a flipped-function:
    'flipped' means the given funtion is partially applied from the right.
    (or apply arguments from the left after the function is 'flipped')

    Passing arguments in reverse order for a function is painful.
    When 'sgra=True', args can be given in the forward direction
    even if the flipped funtion is applied from the right.

    'sgra' == 'args'[::-1]
    """
    if sgra:
        return f_(flip(f), *args, **kwargs)
    else:
        return flip(f_(flip(f), *args[::-1], **kwargs))


def curry(f, n=None):
    """Curried function that takes arguments from the left.
    The currying result is simply a nested unary function.

    This function takes positional arguments only when currying.
    Use partial application `f_` before currying if you need to change kwargs"""
    n = n if n else len(fn_args(f))

    @wraps(f)
    def wrapper(x):
        return f(x) if n <= 1 else curry(f_(f, x), n=pred(n))

    return wrapper


# for decreasing verbosity
c_ = curry


def cc_(f, *args, **kwargs):
    """Curried function that takes arguments from the right"""
    return c_(flip(f), *args, **kwargs)


def cf_(*fs, rep=None):
    """Composing functions using the given list of functions"""

    def wrapper(f, g):
        return lambda x: f(g(x))

    args = fs * rep if rep else fs
    return reduce(wrapper, args, id)


def cfd(*fs, rep=None):
    """Compose-function decorator:
    decorate a function using the composition of the given functions.
    """

    def comp(g):
        @wraps(g)
        def wrapper(*args, **kwargs):
            return cf_(*fs, rep=rep)(g(*args, *kwargs))

        return wrapper

    return comp


def m_(f):
    """builds partial application of `map`
    map(f, xs) == f <$> xs

    (f <$>) == map(f,)  == f_(map, f) == m_(f)
    (<$> xs) == map(,xs) == f_(flip(map), xs)
    """
    return f_(map, f)


def mm_(xs):
    """builds flipped-partial application of `map`
    See also 'm_'.

    (f <$>) == map(f,)  == f_(map, f) == m_(f)
    (<$> xs) == map(,xs) == f_(flip(map), xs)
    """
    return ff_(map, xs)


def ml_(f):
    """list-decorated `m_`"""
    return cfd(list)(m_(f))


def mml_(f):
    """list-decorated `mm_`"""
    return cfd(list)(mm_(f))


def v_(f):
    """builds partial application of `filter`.
    f: predicate or filter funtion"""
    return f_(filter, f)


def vv_(xs):
    """builds flipped-partial application of `filter`
    xs: iterable
    """
    return ff_(filter, xs)


def vl_(f):
    """list-decorated `v_`"""
    return cfd(list)(v_(f))


def vvl_(f):
    """list-decorated `vv_`"""
    return cfd(list)(vv_(f))


# list-decorated `map`
mapl = cfd(list)(map)


# list-decorated `filter`
filterl = cfd(list)(filter)


# list-decorated `filter`
zipl = cfd(list)(zip)


# list-decorated `range`
rangel = cfd(list)(range)


# list-decorated `enumerate`
enumeratel = cfd(list)(enumerate)


# (list) for personal clarity
reverse = cfd(list)(reversed)


# (list) for personal clarity
sort = sorted


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


def fold(f, initial, xs):
    """folding an foldable object from the left. The same as 'foldl' in Haskell"""
    return reduce(f, xs, initial)


def fold1(f, xs):
    """equivalent to 'foldl1' in Haskell"""
    return reduce(f, xs)


def scan(f, initial, xs):
    """accumulation from the left. Equivalent to 'scanl' in Haskell"""
    return it.accumulate(xs, f, initial=initial)


def scan1(f, xs):
    """equivalent to 'scanl1' in Haskell"""
    return scan(f, None, xs)


def permutation(x, r, rep=False):
    return it.product(x, repeat=r) if rep else it.permutations(x, r)


def combination(x, r, rep=False):
    return it.combinations_with_replacement(x, r) if rep else it.combinations(x, r)


# Cartesian product
cprod = ff_(it.product, repeat=1)


# concatenation of all elements of iterables
concat = it.chain.from_iterable


# list-decorated `concat`
concatl = cfd(list)(concat)


# map a function over the given iterable then concat it
concatmap = cfd(concat)(map)


# list-decorated `concatmap`
concatmapl = cfd(list)(concatmap)


def _is_ns_iter(x):
    """Check if the given is a non-string-like iterable"""
    return all(
        (
            hasattr(x, "__iter__"),
            not isinstance(x, str),
            not isinstance(x, bytes),
        )
    )


def flat(*args):
    """flatten all kinds of iterables (except for string-like object)"""

    def go(xss):
        if _is_ns_iter(xss):
            for xs in xss:
                yield from go([*xs] if _is_ns_iter(xs) else xs)
        else:
            yield xss

    return go(args)


# flatten iterables into list
flatl = cfd(list)(flat)

# flatten iterables into tuple
flatt = cfd(tuple)(flat)

# flatten iterables into deque
flatd = cfd(deque)(flat)

# flatten iterables into set
flats = cfd(set)(flat)


def fread(*args):
    """flat-read: read iterables from objects or files then flatten them"""
    f = cf_(normpath, bytes.decode)
    return flatl(
        open(f(x), "r").readlines()
        if isinstance(x, bytes) and exists(f(x), "f")
        else error(f"Error, not found file: {f(x)}")
        for x in flat(args)
    )


def fwrite(f, *args):
    """flat-write: get iterables flattened then write it to a file"""
    with open(f, "w") as fh:
        for line in flat(args):
            fh.write(f"{line}\n")
    return f


def split_at(ix, x):
    """split iterables at the given splitting-indices"""
    s = flatl(0, ix, None)
    return [[*it.islice(x, begin, end)] for begin, end in zip(s, s[1:])]


def chunk_of(n, x, fill=None):
    """split interables into the given `n-length` pieces"""
    return it.zip_longest(*(iter(x),) * n, fillvalue=fill)


def capture(p, string):
    x = captures(p, string)
    if x:
        return x.pop()


def captures(p, string):
    return re.compile(p).findall(string)


def error(str, e=Exception):
    raise e(str)


def HOME():
    return os.getenv("HOME")


def pwd():
    return os.getcwd()


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
        return os.path.dirname(*args)


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
    return os.urandom(n)


def random_int(*args):
    def rint(n):
        return bytes_to_int(random_bytes((n.bit_length() + 7) // 8)) % n

    if not args:
        return rint(2 << 256 - 1)
    elif len(args) == 1:
        return rint(fst(args))
    else:  # [low, high)]
        low, high, *_ = args
        return low + rint(high - low)


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

    def __ror__(self, o):
        return self.__or__(o)


@cache
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
    def wrapper():
        polling(f, sec, args, kwargs)
        if args and kwargs:
            f(*args, **kwargs)
        elif kwargs:
            f(**kwargs)
        elif args:
            f(*args)
        else:
            f()

    t = threading.Timer(sec, wrapper, args, kwargs)
    t.start()
    return t


def fmt(*args, width=100, indent=12):
    return "\n".join(
        fill(
            f"{v}",
            width=width,
            break_on_hyphens=False,
            drop_whitespace=False,
            initial_indent=f"{k:>{indent}}  |  ",
            subsequent_indent=f"{' ':>{indent}}  |  ",
        )
        for k, v in args
    )


def timestamp(
    origin=None,
    w=0,
    d=0,
    h=0,
    m=0,
    s=0,
    from_fmt=None,
    to_fmt=False,
):
    if from_fmt:
        t = datetime.strptime(from_fmt, "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
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

    return to_fmt and f"{datetime.fromtimestamp(t).isoformat()[:26]}Z" or t
