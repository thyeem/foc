import os
import re
import threading
from collections import deque
from datetime import datetime, timedelta
from functools import partial, reduce, wraps
from itertools import accumulate, islice, zip_longest
from operator import itemgetter
from pathlib import Path
from shutil import rmtree
from textwrap import fill


def id(x):
    return x


def fst(x):
    return nth(x, 1)


def snd(x):
    return nth(x, 2)


def nth(x, n):
    assert n > 0, f"Error, must be a positive integer: {n}"
    if hasattr(x, "__getitem__"):
        return itemgetter(n - 1)(x)
    else:
        for _ in range(n - 1):
            x.__next__()
        return x.__next__()


def flip(f):
    """flip(f) takes its arguments in the reverse order of f"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args[::-1], **kwargs)

    return wrapper


# Partial evaluation of function
f_ = partial


def ff_(f, *args, sgra=False, **kwargs):
    """Partial application of a flipped-function and arguments:
    'flipped' means the given funtion is partially applied from the right
    after being 'flipped'.

    Passing arguments in reverse for a function is painful.
    When 'sgra=True', args can be given in the forward direction
    even if the flipped funtion is applied from the right.

    'sgra' == 'args'[::-1]
    """
    if sgra:
        return f_(flip(f), *args, **kwargs)
    else:
        return flip(f_(flip(f), *args[::-1], **kwargs))


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
    decorate a function using the given functions' composition.
    """

    def comp(g):
        @wraps(g)
        def wrapper(*args, **kwargs):
            return cf_(*fs, rep=rep)(g(*args, *kwargs))

        return wrapper

    return comp


def bimap(f, g, o):
    return f(fst(o)), g(snd(o))


def first(f, o):
    return bimap(f, id, o)


def second(g, o):
    return bimap(id, g, o)


def ma_(f):
    """Builds partial application of `map`
    map(f, xs) == f <$> xs

    (f <$>) == map(f,)  == f_(map, f) == ma_(f)
    (<$> xs) == map(,xs) == f_(flip(map), xs)
    """
    return f_(map, f)


def am_(xs):
    """Builds flipped-partial application of `map`
    See also 'ma_'.

    (f <$>) == map(f,)  == f_(map, f) == ma_(f)
    (<$> xs) == map(,xs) == f_(flip(map), xs)
    """
    return f_(flip(map), xs)


def ft_(f):
    """Builds partially applied filter
    using f, predicate or filter funtion
    """
    return f_(filter, f)


def tf_(xs):
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


def singleton(cls):
    instances = {}

    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


def safe(header=None):
    def run(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if header:
                    print(f"\n{header} {e}\n")

        return wrapper

    return run


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


def is_ns_iter(x):
    """Check if the given is a non-string-like iterable"""
    return all(
        (
            hasattr(x, "__iter__"),
            not isinstance(x, str),
            not isinstance(x, bytes),
        )
    )


@cfd(deque)
def flat(*args):
    """Flatten all kinds of iterables (except for string-like object)"""

    def go(xss):
        if is_ns_iter(xss):
            for xs in xss:
                yield from go([*xs] if is_ns_iter(xs) else xs)
        else:
            yield xss

    return go(args)


flatl = cfd(list)(flat)
flatg = cfd(iter)(flat)


def fitr(*args):
    return flat(
        iter(open(x, "r").readlines())
        if isinstance(x, bytes) and exists(x.decode(), "f")
        else x
        for x in flat(args)
    )


def fitw(f, *args):
    with open(f, "w") as fh:
        for line in flat(args):
            fh.write(f"{line}\n")
    return f


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


def HOME():
    return os.getenv("HOME")


def exists(path, kind=None):
    o = Path(path)
    if kind == "f":
        return o.is_file()
    elif kind == "d":
        return o.is_dir()
    else:
        return o.exists()


def mkdir(path, mode=0o755):
    os.makedirs(path, mode=mode, exist_ok=True)
    return path


def rmdir(path, rm_rf=False):
    if rm_rf:
        rmtree(path)
    else:
        os.removedirs(path)


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
