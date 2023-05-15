import os
import re
import threading
from datetime import datetime, timedelta
from functools import partial, reduce, wraps
from itertools import accumulate, islice, zip_longest
from operator import itemgetter
from textwrap import fill


def id(x):
    return x


def fst(x):
    return nth(x, 0)


def snd(x):
    return nth(x, 1)


def nth(x, n):
    if hasattr(x, "__getitem__"):
        return itemgetter(n)(x)
    else:
        for _ in range(n):
            x.__next__()
        return x.__next__()


def flip(f):
    """flip(f) takes its arguments in the reverse order of f"""

    @wraps(f)
    def wrapper(*args):
        return f(*args[::-1])

    return wrapper


# Partial evaluation of function
f_ = partial


def cf_(*fs, rep=None):
    """Composing functions using the given list of functions"""

    # def check(f):
    # defs = 0 if f.__defaults__ is None else len(f.__defaults__)
    # params = f.__code__.co_argcount - defs
    # assert params == 1, f"Given non-unary function: {f.__name__}"

    def wrapper(f, g):
        return lambda x: f(g(x))

    args = fs * rep if rep else fs
    return reduce(wrapper, args, id)


def cfd(*fs, rep=None):
    """Compose-function decorator:
    generate a decorator using the given functions' composition.
    """

    def comp(g):
        @wraps(g)
        def wrapper(*args, **kwargs):
            return cf_(*fs, rep=rep)(g(*args, *kwargs))

        return wrapper

    return comp


def m_(f):
    """Builds partially applied map
    map(f, xs) == f <$> xs

    (f <$>) == map(f,)  == f_(map, f) == m_(f)
    (<$> xs) == map(,xs) == f_(flip(map), xs)
    """
    return f_(map, f)


def mf_(xs):
    """Builds partially-applied flipped map
    See also 'm_'.

    (f <$>) == map(f,)  == f_(map, f) == m_(f)
    (<$> xs) == map(,xs) == f_(flip(map), xs)
    """
    return f_(flip(map), xs)


def ft_(f):
    """Builds partially applied filter
    using f, predicate or filter funtion
    """
    return f_(filter, f)


def ftf_(xs):
    """Builds partially-applied flipped map
    using xs, an iterable object to be filtered
    """
    return f_(flip(filter), xs)


def fold(f, initial, xs):
    """Folding an foldable object from LEFT. The same as 'foldl' in Haskell"""
    return reduce(f, xs, initial)


def fold1(f, xs):
    """Equivalent to 'foldl1' in Haskell"""
    return reduce(f, xs)


def scan(f, initial, xs):
    """Accumulation from LEFT. Equivalent to 'scanl' in Haskell"""
    return accumulate(xs, f, initial=initial)


def scan1(f, xs):
    """Equivalent to 'scanl1' in Haskell"""
    return scan(f, None, xs)


def singleton(cls):
    instances = {}

    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


def bytes_to_int(x, byteorder="big"):
    return int.from_bytes(x, byteorder=byteorder)


def int_to_bytes(x, size=None, byteorder="big"):
    if size is None:
        size = (x.bit_length() + 7) // 8
    return x.to_bytes(size, byteorder=byteorder)


def random_bytes(n):
    return os.urandom(n)


def random_int(n):
    return bytes_to_int(random_bytes((n.bit_length() + 7) // 8)) % n


def flatten(xss):
    if isinstance(xss, tuple) or isinstance(xss, list):
        for xs in xss:
            yield from flatten(xs)
    else:
        yield xss


def split_by(o, ix):
    """Split list/tuple by the given splitting-indices"""
    i = [0] + list(ix) + [None]
    return [[*islice(o, begin, end)] for begin, end in zip(i, i[1:])]


def group_by(o, size, fill=None):
    """Group list/tuple by the given group size"""
    return [*zip_longest(*(iter(o),) * size, fillvalue=fill)]


def capture(p, string):
    o = captures(p, string)
    if o:
        return o.pop()


def captures(p, string):
    return re.compile(p).findall(string)


def mkdir(dname, path=None, mode=0o755):
    d = f"{path}/{dname}" if path else dname
    os.makedirs(d, mode=mode, exist_ok=True)
    return d


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
            initial_indent=f"{k:>{indent}}" + "  |  ",
            subsequent_indent="" * indent + "  |  ",
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
