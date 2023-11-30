import builtins
import itertools as it
import operator as op
import os
import random as rd
import re
import sys
import time
import zipfile
from collections import deque
from datetime import datetime, timedelta
from functools import lru_cache, partial, reduce, wraps
from glob import glob
from inspect import signature
from itertools import accumulate, count, cycle, dropwhile, islice
from itertools import product as cprod
from itertools import takewhile
from multiprocessing import Process
from shutil import rmtree
from subprocess import DEVNULL, PIPE, STDOUT, Popen
from textwrap import fill
from threading import Thread, Timer

__version__ = "0.4.0"

__all__ = [
    "composable",
    "fx",
    "trap",
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
    "pair",
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
    "not_elem",
    "nub",
    "repeat",
    "replicate",
    "cycle",
    "count",
    "islice",
    "deque",
    "product",
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
    "map",
    "mapl",
    "filter",
    "filterl",
    "zip",
    "zipl",
    "unzip",
    "unzipl",
    "rangel",
    "enumeratel",
    "rev",
    "takewhile",
    "takewhilel",
    "dropwhile",
    "dropwhilel",
    "_not",
    "_and",
    "_or",
    "_in",
    "_is",
    "_isnt",
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
    "lazy",
    "force",
    "mforce",
    "reader",
    "writer",
    "split_at",
    "chunks_of",
    "sym",
    "capture",
    "captures",
    "guard",
    "guard_",
    "error",
    "HOME",
    "cd",
    "pwd",
    "normpath",
    "exists",
    "dirname",
    "basename",
    "mkdir",
    "rmdir",
    "ls",
    "grep",
    "split",
    "echo",
    "tee",
    "bytes_to_int",
    "int_to_bytes",
    "bytes_to_bin",
    "bin_to_bytes",
    "randbytes",
    "rand",
    "randn",
    "randint",
    "choice",
    "shuffle",
    "dmap",
    "singleton",
    "thread",
    "proc",
    "polling",
    "shell",
    "pbcopy",
    "pbpaste",
    "timer",
    "neatly",
    "nprint",
    "timestamp",
    "taskbar",
    "nfpos",
    "catalog",
    "rev",
    "uniq",
    "unpack",
    "xargs",
    "zipwith",
    "sort",
    "reverse",
    "length",
    "abs",
    "sum",
    "min",
    "max",
    "ord",
    "chr",
    "all",
    "any",
]


class composable:
    """Lifts the given function to be 'composable' by symbols.
    'composable' allows functions to be composed in two intuitive ways.

    +----------+---------------------------------+------------+
    |  symbol  |         description             | eval-order |
    +----------+---------------------------------+------------+
    | . (dot)  | same as the mathematical symbol | backwards  |
    | | (pipe) | in Unix pipeline manner         | in order   |
    +----------+---------------------------------+------------+

    >>> fx = composable

    'fx' makes a function `composable` on the fly.
    `fx` stands for 'Function eXtension'.

    >>> (length . range)(10)
    10
    >>> range(10) | length
    10

    >>> (unpack . filter(even) . range)(10)
    [0, 2, 4, 6, 8]
    >>> range(10) | filter(even) | unpack
    [0, 2, 4, 6, 8]

    >>> (unpack . map(pred . succ) . range)(5)
    [0, 1, 2, 3, 4]
    >>> (sum . map(f_("+", 5)) . range)(10)
    95
    >>> range(10) | map(f_("+", 5)) | sum
    95

    >>> (last . sort . shuffle . unpack . range)(11)
    10
    >>> range(11) | unpack | shuffle | sort | last
    10

    >>> (unchars . map(chr))(range(73, 82))
    'IJKLMNOPQ'
    >>> range(73, 82) | map(chr) | unchars
    'IJKLMNOPQ'

    >>> (fx(lambda x: x * 6) . fx(lambda x: x + 4))(3)
    42
    >>> 3 | fx(lambda x: x + 4) | fx(lambda x: x * 6)
    42
    """

    def __init__(self, f=lambda x: x):
        self.f = f
        wraps(f)(self)

    def __ror__(self, other):
        return self.f(other)

    def __call__(self, *args, **kwargs):
        npos = nfpos(self.f)
        if len(args) < npos or (not args and kwargs):
            if not npos:
                return self.f(*args, **kwargs)
            return fx(f_(self.f, *args, **kwargs))
        return self.f(*args, **kwargs)

    def __getattr__(self, key):
        g = globals().get(key, getattr(builtins, key, None))
        guard(callable(g), f"fx, no such callable: {key}", e=AttributeError)
        if capture(r"\bcomposable|fx\b", key):
            return lambda g: fx(cf_(self.f, g))
        if capture(r"\bf_|ff_|c_|cc_|u_|curry|uncurry|mapl?|filterl?\b", key):
            return lambda *a, **k: fx(cf_(self.f, g(*a, **k)))
        return fx(cf_(self.f, g))


# for the sake of brevity
fx = composable


def trap(callback, e=None):
    """decorator factory that creates exception catchers."""

    def catcher(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if e is None:
                try:
                    return f(*args, **kwargs)
                except:
                    return callback(sys.exc_info()[1])
            else:
                try:
                    return f(*args, **kwargs)
                except e as err:
                    return callback(err)

        return wrapper

    return catcher


def safe(f):
    """make a given function return 'None' instead of raising an exception.

    >>> safe(error)("never-throw-errors")
    """
    return trap(callback=void, e=None)(f)


@fx
def id(x):
    """identity function

    >>> id("francis")
    'francis'
    >>> id("francis") == "francis" | id
    True
    """
    return x


@fx
def const(x, _):
    """build an id function that returns a given 'x'.

    >>> const(5, "no-matther-what-comes-here")
    5
    >>> 'whatever' | const(5)
    5
    """
    return x


@fx
def seq(_, x):
    """return the id function after consuming the given argument.

    >>> seq("only-returns-the-following-arg", 5)
    5
    >>> 5 | seq('whatever')
    5
    """
    return x


@fx
def void(_):
    """return 'None' after consuming the given argument.

    >>> void(randbytes(256))
    >>> randbytes(256) | void
    """
    return


@fx
@safe
def fst(x):
    """get the first component of a given iterable.

    >>> fst(["sofia", "maria", "claire"])
    'sofia'
    >>> ["sofia", "maria", "claire"] | fst
    'sofia'
    """
    return nth(1, x)


@fx
@safe
def snd(x):
    """get the second component of a given iterable.

    >>> snd(("sofia", "maria", "claire"))
    'maria'
    >>> ("sofia", "maria", "claire") | snd
    'maria'
    """
    return nth(2, x)


@fx
def nth(n, x):
    """get the 'n'-th component of a given iterable 'x'.

    >>> nth(3, ["sofia", "maria", "claire"])
    'claire'
    >>> ["sofia", "maria", "claire"] | nth(3)
    'claire'
    """
    return x[n - 1] if hasattr(x, "__getitem__") else next(islice(x, n - 1, None))


@fx
def take(n, x):
    """take 'n' items from a given iterable 'x'.

    >>> take(3, range(5, 10))
    [5, 6, 7]
    >>> range(5, 10) | take(3) | unpack
    [5, 6, 7]
    """
    return islice(x, n) | unpack


@fx
def drop(n, x):
    """return items of the iterable 'x' after skipping 'n' items.

    >>> list(drop(3, 'github'))
    ['h', 'u', 'b']
    >>> 'github' | drop(3) | unpack
    ['h', 'u', 'b']
    """
    return islice(x, n, None)


@fx
@safe
def head(x):
    """extract the first element of a given iterable: the same as 'fst'.

    >>> head(range(1, 5))
    1
    >>> range(1, 5) | head
    1
    """
    return fst(x)


@fx
@safe
def tail(x):
    """extract the elements after the 'head' of a given iterable.

    >>> list(tail(range(1, 5)))
    [2, 3, 4]
    >>> range(1, 5) | tail | unpack
    [2, 3, 4]
    """
    return drop(1, x)


@fx
@safe
def init(x):
    """return all the elements of an iterable except the 'last' one.

    >>> list(init(range(1, 5)))
    [1, 2, 3]
    >>> range(1, 5) | init | unpack
    [1, 2, 3]
    """
    it = iter(x)
    try:
        o = next(it)
    except StopIteration:
        return
    for i in it:
        yield o
        o = i


@fx
@safe
def last(x):
    """extract the last element of a given iterable.

    >>> last(range(1, 5))
    4
    >>> range(1, 5) | last
    4
    """
    return deque(x, maxlen=1)[0]


@fx
def ilen(x):
    """get the length of a given iterator.

    >>> ilen((x for x in range(100)))
    100
    >>> (x for x in range(100)) | ilen
    100
    """
    c = count()
    deque(zip(x, c), maxlen=0)
    return next(c)


@fx
def pair(a, b):
    """make the given two arguments a tuple pair.

    >>> pair("sofia", "maria")
    ('sofia', 'maria')
    >>> "maria" | pair("sofia")
    ('sofia', 'maria')
    """
    return (a, b)


@fx
def pred(x):
    """return the predecessor of a given value: it substracts 1.

    >>> pred(3)
    2
    >>> 3 | pred
    2
    """
    return x - 1


@fx
def succ(x):
    """return the successor of a given value: it adds 1.

    >>> succ(3)
    4
    >>> 3 | succ
    4
    """
    return x + 1


@fx
def odd(x):
    """check if the given number is odd.

    >>> odd(3)
    True
    >>> 3 | odd
    True
    """
    return x % 2 == 1


@fx
def even(x):
    """check if the given number is even.

    >>> even(3)
    False
    >>> 3 | even
    False
    """
    return x % 2 == 0


@fx
def null(x):
    """check if a given collection is empty.

    >>> null([]) == null(()) == null({}) == null('')
    True
    >>> [] | null == () | null == {} | null == '' | null
    True
    """
    return len(x) == 0


@fx
def chars(x):
    """split string 'x' into `chars`: the same as (:[]) <$> x.

    >>> chars("sofimarie")
    ['s', 'o', 'f', 'i', 'm', 'a', 'r', 'i', 'e']
    >>> chars("sofimarie") == "sofimarie" | chars
    True
    """
    return list(x)


@fx
def unchars(x):
    """inverse operation of 'chars': the same as 'concat'.

    >>> unchars(['s', 'o', 'f', 'i', 'm', 'a', 'r', 'i', 'e'])
    'sofimarie'
    >>> ['s', 'o', 'f', 'i', 'm', 'a', 'r', 'i', 'e'] | unchars
    'sofimarie'
    """
    return "".join(x)


@fx
def words(x):
    """joins a list of words with the blank character.

    >>> words("fun on functions")
    ['fun', 'on', 'functions']
    >>> 'fun on functions' | words
    ['fun', 'on', 'functions']
    """
    return x.split()


@fx
def unwords(x):
    """breaks a string up into a list of words.

    >>> unwords(['fun', 'on', 'functions'])
    'fun on functions'
    >>> ['fun', 'on', 'functions'] | unwords
    'fun on functions'
    """
    return " ".join(x)


@fx
def lines(x):
    """splits a string into a list of lines using the delimeter '\\n'.

    >>> lines("fun\\non\\nfunctions")
    ['fun', 'on', 'functions']
    >>> "fun\\non\\nfunctions" | lines
    ['fun', 'on', 'functions']
    """
    return x.splitlines()


@fx
def unlines(x):
    """joins a list of lines with the newline character, '\\n'.

    >>> unlines(['fun', 'on', 'functions'])
    'fun\\non\\nfunctions'
    >>> ['fun', 'on', 'functions'] | unlines
    'fun\\non\\nfunctions'
    """
    return "\n".join(x)


@fx
def elem(x, xs):
    """check if the element exists in the given iterable.

    >>> elem("fun", "functions")
    True
    >>> "functions" | elem("fun")
    True
    """
    return x in xs


@fx
def not_elem(x, xs):
    """the negation of 'elem'.

    >>> not_elem("fun", "functions")
    False
    >>> "functions" | not_elem("fun")
    False
    """
    return x not in xs


@fx
def nub(x):
    """removes duplicate elements from a given iterable.

    >>> nub("3333-13-1111111")
    ['3', '-', '1']
    >>> "3333-13-1111111" | nub
    ['3', '-', '1']
    """
    return dict.fromkeys(x) | unpack


@fx
def repeat(x, nf=True):
    """create an infinite list with x value of every element:
    if nf (NF, Normal Form) is set, callable objects will be evaluated.

    >>> take(3, repeat(5))
    [5, 5, 5]
    >>> repeat(5) | take(3)
    [5, 5, 5]
    """
    return (x() if nf and callable(x) else x for _ in count())


@fx
def replicate(n, x, nf=True):
    """get a list of length `n` from an infinite list with `x` values.

    >>> replicate(3, 5)
    [5, 5, 5]
    >>> 5 | replicate(3)
    [5, 5, 5]
    """
    return take(n, repeat(x, nf=nf))


@fx
def product(x):
    """product of the elements for the given iterable.

    >>> product(range(1, 11))
    3628800
    >>> range(1, 11) | product
    3628800
    """
    return foldl1(op.mul, x)


def flip(f):
    """flip(f) takes its arguments in the reverse order of f:
    `f :: a -> b -> ... -> c -> d -> o`
    `flip(f) :: d -> c -> ... -> b -> a -> o`

    >>> flip(pow)(7, 3)
    2187
    >>> (7, 3) | uncurry(flip("-"))
    -4
    """
    f = sym(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args[::-1], **kwargs)

    return wrapper


def f_(f, *args, **kwargs):
    """build left-associative partial application,
    where the given function's arguments partially evaluation from the left.

    >>> f_("-", 5)(2)
    3
    >>> mapl(f_("**", 2), range(3, 7))
    [8, 16, 32, 64]
    """
    return partial(sym(f), *args, **kwargs)


def ff_(f, *args, **kwargs):
    """build left-associative partial application,
    where the given function's arguments partially evaluation from the right.

    >>> ff_("-", 5)(2)
    -3
    >>> mapl(ff_("**", 2), range(3, 7))
    [9, 16, 25, 36]
    """
    return f_(flip(sym(f)), *args, **kwargs)


def curry(f, *, _n=None):
    """build curried function that takes the arguments from the left.
    The currying result is simply a nested unary function.

    This function takes positional arguments only when currying.
    Use partial application `f_` before currying if you need to change kwargs

    >>> curry("-")(5)(2)
    3
    >>> curry(foldl)("+")(0)(range(1, 11))
    55
    """
    f = sym(f)
    _n = nfpos(f) if _n is None else _n

    @wraps(f)
    def wrapper(x):
        return f(x) if _n <= 1 else curry(f_(f, x), _n=pred(_n))

    return wrapper


# for the sake of brevity
c_ = curry


def cc_(f):
    """build curried function that takes the arguments from the right.

    >>> cc_("-")(5)(2)
    -3
    >>> cc_(foldl)(range(1, 11))(0)("+")
    55
    """
    return c_(flip(sym(f)))


def uncurry(f):
    """convert a uncurried normal function to a unary function of a tuple args.
    This is not exact reverse operation of `curry`. Here `uncurry` simply does:
    `uncurry :: (a -> ... -> b -> o) -> (a, ..., b) -> o`

    >>> uncurry(pow)((2, 10))
    1024
    >>> (2, 3) | uncurry("+")
    5
    >>> ([1, 3], [2, 4]) | uncurry(zip) | unpack
    [(1, 2), (3, 4)]
    """
    f = sym(f)

    @fx
    def wrapper(x):
        return f(*x)

    return wrapper


# for the sake of brevity
u_ = uncurry


def cf_(*fs, rep=None):
    """compose a given list of functions then return the composed function.

    >>> cf_(f_("*", 7), f_("+", 3))(5)
    56
    >>> cf_(ff_("[]", "sofia"), dict)([("sofia", "piano"), ("maria", "violin")])
    'piano'
    """

    def compose(f, g):
        return lambda *args, **kwargs: f(g(*args, **kwargs))

    return reduce(compose, fs * rep if rep else fs)


def cfd(*fs, rep=None):
    """decorator using the composition of functions:
    decorate a function using the composition of the given functions.

    >>> cfd(set, list, tuple)(range)(5)
    {0, 1, 2, 3, 4}
    >>> cfd(ff_("[]", "maria"))(dict)([("sofia", "piano"), ("maria", "violin")])
    'violin'
    """

    def cfdeco(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return cf_(*fs, rep=rep)(f(*args, *kwargs))

        return wrapper

    return cfdeco


def map(f, *xs):
    """symbol-composable 'map', which seamlessly extends 'builtins.map'

    >>> (unpack . map(abs))(range(-2, 3)) | unpack
    [2, 1, 0, 1, 2]
    >>> map(abs)(range(-2, 3)) | unpack
    [2, 1, 0, 1, 2]

    >>> (unpack . map(lambda x: x*8))(range(1, 6))
    [8, 16, 24, 32, 40]
    >>> range(1, 6) | map(lambda x: x*8) | unpack
    [8, 16, 24, 32, 40]

    >>> (unpack . map("*", [1, 2, 3]))([4, 5, 6])
    [4, 10, 18]
    >>> [4, 5, 6] | map("*", [1, 2, 3]) | unpack
    [4, 10, 18]
    """
    f = sym(f)
    if not xs or len(xs) < nfpos(f):
        return fx(f_(builtins.map, f, *xs))
    else:
        return builtins.map(f, *xs)


def mapl(f, *xs):
    """the same as 'map', but returns in 'list'.

    >>> mapl(abs)(range(-2, 3))
    [2, 1, 0, 1, 2]
    >>> mapl(f_("*", 8), range(1, 6))
    [8, 16, 24, 32, 40]
    >>> mapl("*", [1, 2, 3], [4, 5, 6])
    [4, 10, 18]
    >>> [4, 5, 6] | mapl("*", [1, 2, 3])
    [4, 10, 18]
    """
    f = sym(f)
    if not xs or len(xs) < nfpos(f):
        return fx(cfd(list)(f_(builtins.map, f, *xs)))
    else:
        return cfd(list)(builtins.map)(f, *xs)


@fx
def filter(p, xs):
    """the same as 'builtins.filter', but a composable.

    >>> (unpack . filter(f_("==", "f")))("fun-on-functions")
    ['f', 'f']
    >>> filter(f_("==", "f"))("fun-on-functions") | unpack
    ['f', 'f']

    >>> primes = [2, 3, 5, 7, 11, 13, 17, 19]
    >>> (unpack . filter(lambda x: x % 3 == 2))(primes)
    [2, 5, 11, 17]
    >>> primes | filter(cf_(ff_("==", 2), ff_("%", 3))) | unpack
    [2, 5, 11, 17]
    """
    return builtins.filter(p, xs)


@fx
def filterl(p, xs):
    """the same as 'filter', but returns in 'list'.

    >>> filterl(f_("==", "f"))("fun-on-functions")
    ['f', 'f']
    >>> primes = [2, 3, 5, 7, 11, 13, 17, 19]
    >>> primes | filterl(lambda x: x % 3 == 2)
    [2, 5, 11, 17]
    """
    return filter(p, xs) | unpack


@fx
def zip(*xs, strict=False):
    """the same as 'builtins.zip', but a composable.

    >>> (unpack . f_(zip, "LOVE") . range)(3)
    [('L', 0), ('O', 1), ('V', 2)]
    >>> zip("LOVE", range(3)) | unpack
    [('L', 0), ('O', 1), ('V', 2)]

    >>> (unpack . uncurry(zip))(("LOVE", range(3),))
    [('L', 0), ('O', 1), ('V', 2)]
    >>> ("LOVE", range(3)) | uncurry(zip) | unpack
    [('L', 0), ('O', 1), ('V', 2)]
    """
    return builtins.zip(*xs, strict=strict)


@fx
def zipl(*xs, strict=False):
    """the same as 'zip', but returns in 'list'.

    >>> zipl("LOVE", range(3))
    [('L', 0), ('O', 1), ('V', 2)]
    >>> ("LOVE", range(3)) | uncurry(zipl)
    [('L', 0), ('O', 1), ('V', 2)]
    """
    return zip(*xs, strict=strict) | unpack


@fx
def unzip(xs):
    """reverse operation of 'zip' function.

    >>> (unpack . unzip . zip)("LOVE", range(3))
    [('L', 'O', 'V'), (0, 1, 2)]
    >>> unzip(zip("LOVE", range(3))) | unpack
    [('L', 'O', 'V'), (0, 1, 2)]
    >>> zip("LOVE", range(3)) | unzip | unpack
    [('L', 'O', 'V'), (0, 1, 2)]
    """
    return zip(*xs)


@fx
def unzipl(x):
    """the same as 'unzip', but returns in 'list'.

    >>> unzipl(zip("LOVE", range(3)))
    [('L', 'O', 'V'), (0, 1, 2)]
    >>> zip("LOVE", range(3)) | unzipl
    [('L', 'O', 'V'), (0, 1, 2)]
    """
    return unzip(x) | unpack


def rangel(*args, **kwargs):
    """the same as 'range', but returns in 'list'.

    >>> rangel(10) == range(10) | unpack
    True
    """
    return range(*args, **kwargs) | unpack


def enumeratel(*args, **kwargs):
    """the same as 'enumerate', but returns in 'list'.

    >>> enumeratel(range(10)) == enumerate(range(10)) | unpack
    True
    """
    return enumerate(*args, **kwargs) | unpack


@fx
def rev(x, *xs):
    """return reversed sequence: this exends the 'reverse' function.
    for given multiple args, it returns a reversed tuple of the arguments
    otherwise, it returns only the reversed sequence like 'reverse'.

    >>> rev(range(5))
    [4, 3, 2, 1, 0]
    >>> rev(range(5)) == range(5) | rev
    True
    >>> rev("dog")
    ['g', 'o', 'd']
    """
    if xs:
        return (x, *xs)[::-1]
    else:
        return list(x)[::-1]


def takewhilel(*args, **kwargs):
    """the same as 'takewhile', but returns in 'list'.

    >>> takewhilel(even, [2, 4, 6, 1, 3, 5])
    [2, 4, 6]
    """
    return takewhile(*args, **kwargs) | unpack


def dropwhilel(*args, **kwargs):
    """the same as 'dropwhile', but returns in 'list'.

    >>> dropwhilel(even, [2, 4, 6, 1, 3, 5])
    [1, 3, 5]
    """
    return dropwhile(*args, **kwargs) | unpack


@fx
def _not(x):
    """`not` as a function.

    >>> _not(False)
    True
    >>> False | _not
    True
    """
    return not x


@fx
def _and(a, b):
    """`and` as a function.

    >>> _and(True, False)
    False
    >>> False | _and(True)
    False
    """
    return a and b


@fx
def _or(a, b):
    """`and` as a function.

    >>> _or(True, False)
    True
    >>> False | _or(True)
    True
    """
    return a or b


@fx
def _in(a, b):
    """`in` as a function.

    >>> _in("fun", "function")
    True
    >>> 'function' | _in("fun")
    True
    """
    return flip(op.contains)(a, b)


@fx
def _is(a, b):
    """`is` as a function.

    >>> _is("war", "LOVE")
    False
    >>> "LOVE" | _is("war")
    False
    """
    return op.is_(a, b)


@fx
def _isnt(a, b):
    """`is not` as a function.

    >>> _isnt("war", "LOVE")
    True
    >>> "LOVE" | _isnt("war")
    True
    """
    return op.is_not(a, b)


@fx
def bimap(f, g, x):
    """map over both 'first' and 'second' arguments at the same time.
    bimap(f, g) == first(f) . second(g)

    >>> bimap(f_("+", 3), f_("*", 7), (5, 7))
    (8, 49)
    >>> (5, 7) | bimap(f_("+", 3), f_("*", 7))
    (8, 49)
    """
    return f(fst(x)), g(snd(x))


@fx
def first(f, x):
    """map covariantly over the 'first' argument.

    >>> first(f_("+", 3), (5, 7))
    (8, 7)
    >>> (5, 7) | first(f_("+", 3))
    (8, 7)
    """
    return bimap(f, id, x)


@fx
def second(g, x):
    """map covariantly over the 'second' argument.

    >>> second(f_("*", 7), (5, 7))
    (5, 49)
    >>> (5, 7) | second(f_("*", 7))
    (5, 49)
    """
    return bimap(id, g, x)


@fx
def until(p, f, x):
    """return the result of applying the given 'f' until the given 'p' holds

    >>> until(ff_(">", 1024), f_("*", 3), 5)
    1215
    >>> until(cf_(f_("==", 3), ff_("%", 5)), f_("*", 3), 119)
    3213
    """
    while not p(x):
        x = f(x)
    return x


@fx
def iterate(f, x):
    """return an infinite list of repeated applications of 'f' to 'x'.

    >>> take(5, iterate(ff_("**", 2), 2))
    [2, 4, 16, 256, 65536]
    >>> 2 | iterate(ff_("**", 2)) | take(5)
    [2, 4, 16, 256, 65536]
    """
    while True:
        yield x
        x = f(x)


def apply(f, *args, **kwargs):
    """call a given function with the given arguments.

    >>> apply(str.split, "go get some coffee")
    ['go', 'get', 'some', 'coffee']
    >>> apply(mapl, even, range(4))
    [True, False, True, False]
    """
    return sym(f)(*args, **kwargs)


@fx
def foldl(f, initial, xs):
    """left-associative fold of an iterable.

    >>> foldl("-", 10, range(1, 5))
    0
    >>> range(1, 5) | foldl("-", 10)
    0
    """
    return reduce(sym(f), xs, initial)


@fx
def foldl1(f, xs):
    """`foldl` without initial value.

    >>> foldl1("-", range(1, 5))
    -8
    >>> range(1, 5) | foldl1("-")
    -8
    """
    return reduce(sym(f), xs)


@fx
def foldr(f, inital, xs):
    """right-associative fold of an iterable.

    >>> foldr("-", 10, range(1, 5))
    8
    >>> range(1, 5) | foldr("-", 10)
    8
    """
    return reduce(flip(sym(f)), xs[::-1], inital)


@fx
def foldr1(f, xs):
    """`foldr` without initial value.

    >>> foldr1("-", range(1, 5))
    -2
    >>> range(1, 5) | foldr1("-")
    -2
    """
    return reduce(flip(sym(f)), xs[::-1])


@fx
def scanl(f, initial, xs):
    """return a list of successive reduced values from the left.

    >>> scanl("-", 10, range(1, 5))
    [10, 9, 7, 4, 0]
    >>> range(1, 5) | scanl("-", 10)
    [10, 9, 7, 4, 0]
    """
    return accumulate(xs, sym(f), initial=initial) | unpack


@fx
def scanl1(f, xs):
    """`scanl` without starting value.

    >>> scanl1("-", range(1, 5))
    [1, -1, -4, -8]
    >>> range(1, 5) | scanl1("-")
    [1, -1, -4, -8]
    """
    return accumulate(xs, sym(f)) | unpack


@fx
def scanr(f, initial, xs):
    """return a list of successive reduced values from the right.

    >>> scanr("-", 10, range(1, 5))
    [8, -7, 9, -6, 10]
    >>> range(1, 5) | scanr("-", 10)
    [8, -7, 9, -6, 10]
    """
    return accumulate(xs[::-1], flip(sym(f)), initial=initial) | rev


@fx
def scanr1(f, xs):
    """`scanr` without starting value.

    >>> scanr1("-", range(1, 5))
    [-2, 3, -1, 4]
    >>> range(1, 5) | scanr1("-")
    [-2, 3, -1, 4]
    """
    return accumulate(xs[::-1], flip(sym(f))) | rev


@fx
def capture(p, string):
    """capture"""
    x = captures(p, string)
    if x:
        return fst(x)


@fx
def captures(p, string):
    """captures"""
    return re.compile(p).findall(string)


def sym(f=None):
    """get binary functions from the symbolic operators.

    >>> sym("*")(5, 5) - sym("**")(5, 2)
    0
    >>> sym(".")(dmap(sofia="maria"), "sofia")
    'maria'
    """
    ops = {
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
        ",": pair,
        ".": getattr,
        "..": range,
        "~": capture,
        "~~": captures,
    }
    return (
        ops.get(f, f)
        if f is not None
        else nprint({k: v.__name__ for k, v in ops.items()}, _cols=14, _repr=False)
    )


@fx
def permutation(x, r, rep=False):
    """return all permutations in a list form

    >>> permutation("abc", 2) | unpack
    [('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]
    """
    return cprod(x, repeat=r) if rep else it.permutations(x, r)


@fx
def combination(x, r, rep=False):
    """return all combinations in a list form
    >>> combination("abc", 2) | unpack
    [('a', 'b'), ('a', 'c'), ('b', 'c')]

    """
    return it.combinations_with_replacement(x, r) if rep else it.combinations(x, r)


@fx
def cartprod(*args, **kwargs):
    """return Cartesian product.

    >>> cartprod("↑↓", "↑↓") | unpack
    [('↑', '↑'), ('↑', '↓'), ('↓', '↑'), ('↓', '↓')]
    >>> ("↑↓", "↑↓") | uncurry(cartprod) | unpack
    [('↑', '↑'), ('↑', '↓'), ('↓', '↑'), ('↓', '↓')]
    """
    return f_(cprod, repeat=1)(*args, **kwargs)


@fx
def cartprodl(*args, **kwargs):
    """the same as 'cartprod', but returns in 'list'.

    >>> cartprodl("↑↓", "↑↓")
    [('↑', '↑'), ('↑', '↓'), ('↓', '↑'), ('↓', '↓')]
    >>> ("↑↓", "↑↓") | uncurry(cartprodl)
    [('↑', '↑'), ('↑', '↓'), ('↓', '↑'), ('↓', '↓')]
    """
    return cartprod(*args, **kwargs) | unpack


@fx
def concat(iterable):
    """concatenates all elements of iterables.

    >>> concat(["so", "fia"]) | unpack
    ['s', 'o', 'f', 'i', 'a']
    >>> ["so", "fia"] | concat | unpack
    ['s', 'o', 'f', 'i', 'a']
    """
    return it.chain.from_iterable(iterable)


@fx
def concatl(iterable):
    """the same as 'concat', but returns in 'list'.

    >>> concatl(["so", "fia"])
    ['s', 'o', 'f', 'i', 'a']
    >>> ["so", "fia"] | concatl
    ['s', 'o', 'f', 'i', 'a']
    """
    return concat(iterable) | unpack


@fx
def concatmap(f, x, *xs):
    """map a function over the given iterable then concat it.

    >>> concatmap(str.upper, ["mar", "ia"]) | unpack
    ['M', 'A', 'R', 'I', 'A']
    >>> ["mar", "ia"] | concatmap(str.upper) | unpack
    ['M', 'A', 'R', 'I', 'A']
    """
    return map(f, x, *xs) | concat


@fx
def concatmapl(f, x, *xs):
    """the same as 'concatmap', but returns in 'list'.

    >>> concatmapl(str.upper, ["mar", "ia"])
    ['M', 'A', 'R', 'I', 'A']
    >>> ["mar", "ia"] | concatmapl(str.upper)
    ['M', 'A', 'R', 'I', 'A']
    """
    return map(f, x, *xs) | concat | unpack


@fx
def intersperse(sep, x):
    """intersperse the given element between the elements of the list.

    >>> intersperse("\u2764", ["francis", "claire"])
    ['francis', '❤', 'claire']
    >>> ["francis", "claire"] | intersperse("\u2764")
    ['francis', '❤', 'claire']
    """
    return concatl(zip(repeat(sep), x))[1:]


@fx
def intercalate(sep, x):
    """inserts the given list between the lists then concat it.

    >>> intercalate("\u2764", [["francis"], ["claire"]])
    ['francis', '❤', 'claire']
    >>> [["francis"], ["claire"]] | intercalate("\u2764")
    ['francis', '❤', 'claire']
    """
    return intersperse(sep, x) | concatl


def flatten(*args):
    """flatten all kinds of iterables except for string-like objects."""

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


@fx
def flat(*args):
    """the same as 'flatten', but a 'composable'.

    >>> flat([1, [(2,), [[{3}, (x for x in range(3))]]]]) | unpack
    [1, 2, 3, 0, 1, 2]
    >>> [1, [(2,), [[{3}, (x for x in range(3))]]]] | flat | unpack
    [1, 2, 3, 0, 1, 2]

    """
    return flatten(*args)


@fx
def flatl(*args):
    """the same as 'flat', but returns in 'list'.

    >>> [1, [(2,), [[{3}, (x for x in range(3))]]]] | flatl
    [1, 2, 3, 0, 1, 2]
    >>> flatl([1, [(2,), [[{3}, (x for x in range(3))]]]])
    [1, 2, 3, 0, 1, 2]
    """
    return flatten(*args) | unpack


# easy-to-use alias for lazy operation
lazy = f_


@fx
def force(expr):
    """forces the delayed-expression to be fully evaluated."""
    return expr() if callable(expr) else expr


@fx
def mforce(iterables):
    """map 'force' over iterables of delayed-evaluation."""
    return mapl(force)(iterables)


def reader(f=None, mode="r", zipf=False):
    """get ready to read stream from a file or stdin, then returns the handle."""
    if f is not None:
        guard(exists(f, "f"), f"reader, not found such a file: {f}")
    return (
        sys.stdin
        if f is None
        else zipfile.ZipFile(normpath(f), mode)
        if zipf
        else open(normpath(f), mode)
    )


def writer(f=None, mode="w", zipf=False):
    """get ready to write stream to a file or stout, then returns the handle."""
    return (
        sys.stdout
        if f is None
        else zipfile.ZipFile(normpath(f), mode)
        if zipf
        else open(normpath(f), mode)
    )


def split_at(ix, x):
    """split the given iterable 'x' at the given splitting-indices 'ix'."""
    s = flatl(0, ix, None)
    return ([*it.islice(x, begin, end)] for begin, end in zip(s, s[1:]))


def chunks_of(n, x, fillvalue=None, fill=True):
    """split interables into the given `n'-length pieces"""
    if not fill:
        x = list(x)
        x = x[: len(x) // n * n]
    return it.zip_longest(*(iter(x),) * n, fillvalue=fillvalue)


def guard(p, msg="guard", e=SystemExit):
    """'assert' as a function or expression."""
    if not p:
        error(msg=msg, e=e)


def guard_(f, msg="guard", e=SystemExit):
    """partial application builder for 'guard':
    the same as 'guard', but the positional predicate is given
    as a function rather than an boolean expression."""
    return lambda x: seq(guard(f(x), msg=msg, e=e))(x)


def error(msg="error", e=SystemExit):
    """'raise' an exception with a function or expression."""
    raise e(msg) from None


def HOME():
    """get the current user's home directory: the same as '$HOME'."""
    return os.getenv("HOME")


def cd(path=None):
    """change directories: similar to the shell-command 'cd'."""
    if path:
        os.chdir(normpath(path, abs=True))
    else:
        os.chdir(HOME())
    return pwd()


def pwd():
    """get the current directory: similar to the shell-command 'pwd'."""
    return os.getcwd()


def normpath(path, abs=False):
    """normalize the given filepath"""
    return cf_(
        os.path.abspath if abs else id,
        os.path.normpath,
        os.path.expanduser,
    )(path)


def exists(path, kind=None):
    """check if the given filepath (file or directory) is available."""
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


def ls(*paths, grep=None, i=False, r=False, f=False, d=False, g=False, _root=True):
    """list directory contents: just like 'ls -a1'.

    Note:
      - allowed glob patterns (*,?,[) in <path..>
      - given 'grep=<regex>', it behaves like 'ls -a1 <path..> | grep <regex>'
      - if i is set, it makes 'grep' case-insensitive (-i flag in grep)
      - if r is set, it behaves like 'find -s <path..>' (-R flag in ls)
      - if f is set, it lists only files like 'find <path..> -type f'
      - if d is set, it lists only directories like 'find <path..> -type d'
      - if g is set, it returns a generator instead of a sorted list
    """
    paths = paths or ["."]
    typef = f and f ^ d
    typed = d and f ^ d

    def fd(x):
        return (typef and exists(x, "f")) or (typed and exists(x, "d"))

    def root(xs):
        return flat(
            glob(normpath(x))
            if re.search(r"[\*\+\?\[]", x)
            else cf_(
                guard_(exists, f"ls, no such file or directory: {x}"),
                normpath,
            )(x)
            for x in xs
        )

    def rflag(xs):
        return flat(
            (x, ls(x, grep=grep, i=i, r=r, f=f, d=d, g=g, _root=False))
            if exists(x, "d")
            else x
            for x in xs
        )

    return cf_(
        id if g else sort,  # return generator or sort by filepath
        filter(fd) if typef ^ typed else id,  # filetype filter: -f or -d flag
        globals()["grep"](grep, i=i) if grep else id,  # grep -i flag
        rflag if r else id,  # recursively listing: -R flag
    )(
        flat(
            [normpath(f"{x}/{o}") for o in (os.listdir(x))] if exists(x, "d") else x
            for x in (root(paths) if _root else paths)
        )
    )


@safe
def grep(regex, *, i=False):
    """build a filter to select items matching 'regex' pattern from iterables

    >>> grep(r".json$", i=True)([".json", "Jason", ".JSON", "jsonl", "JsonL"])
    ['.json', '.JSON']
    """
    return fx(filterl(f_(re.search, regex, flags=re.IGNORECASE if i else 0)))


@fx
def echo(*xs, n=True):
    """echo: display a line of text

    >>> echo("sofia", "maria", "LOVE")
    'sofia maria LOVE\\n'
    >>> ("sofia", "maria", "LOVE") | xargs(echo)
    'sofia maria LOVE\\n'
    """
    return unwords([str(o) for o in xs]) + ("\n" if n else "")


@fx
def tee(f, s, a=False):
    def fwrite(f, x, a=a):
        writer(f, mode="a" if a else "w").write(x)
        return x

    return cf_(f_(fwrite, f, a=a), f_(echo, n=True))(s)


@fx
def split(f, /, nbytes, prefix):
    """split a file into multiple parts of specified byte-size like:
    $ split -b bytes f prefix_

    >>> split(FILE, 1024, "part-")  # doctest: +SKIP
    """
    guard(exists(f, "f"), f"split, not found file: {f}")
    fmt = f"0{len(str(os.stat(f).st_size // nbytes))}d"
    n = 0
    rf = reader(f, "rb")
    chunk = rf.read(nbytes)
    while chunk:
        writer(f"{dirname(f)}/{prefix}{n:{fmt}}", "wb").write(chunk)
        chunk = rf.read(nbytes)
        n += 1


def bytes_to_int(x, byteorder="big"):
    return int.from_bytes(x, byteorder=byteorder)


def int_to_bytes(x, size=None, byteorder="big"):
    if size is None:
        size = (x.bit_length() + 7) // 8
    return x.to_bytes(size, byteorder=byteorder)


def bytes_to_bin(x, sep=""):
    return sep.join(f"{b:08b}" for b in x)


def bin_to_bytes(x):
    return int_to_bytes(int(x, base=2))


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
        guard(low < high, f"randint, low({low}) must be less than high({high})")
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


@fx
def shuffle(x):
    """Fisher-Yates shuffle in a cryptographically secure way"""
    for i in range(len(x) - 1, 0, -1):
        j = randint(0, i)
        x[i], x[j] = x[j], x[i]
    return x


@fx
def choice(x, size=None, *, replace=False, p=None):
    """Generate a sample with/without replacement from a given iterable."""

    def fromp(x, probs, e=1e-6):
        guard(
            len(x) == len(probs),
            f"choice, not the same size: {len(x)}, {len(probs)}",
        )
        guard(
            1 - e < sum(probs) < 1 + e,
            f"choice, sum of probs({sum(probs)}) != 1",
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


@fx
class dmap(dict):
    """dot-accessible dict(map) using DWIM"""

    __slots__ = ()
    __dwim__ = "- "

    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, val in self.items():
            self[key] = self._g(val)

    def _g(self, val):
        if isinstance(val, dict):
            return dmap(val)
        elif isinstance(val, list):
            return [self._g(x) for x in val]
        return val

    def _k(self, key):
        if self.__class__.__dwim__:  # dmap using the DWIM key
            for s in chars(self.__class__.__dwim__):
                if (sub := re.sub("_", s, key)) in self:
                    return sub
        return key

    def __getattr__(self, key):
        if key.startswith("__"):  # disabled for stability
            return
        if key not in self and key != "_ipython_canary_method_should_not_exist_":
            if (sub := self._k(key)) in self:
                return self[sub]
            self[key] = dmap()
        return self[key]

    def __setattr__(self, key, val):
        self[self._k(key)] = self._g(val)

    def __delattr__(self, key):
        key = key if key in self else self._k(key)
        if key in self:
            del self[key]


def singleton(cls):
    """decorate a class and make it a singleton class."""
    _reg = {}

    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in _reg:
            _reg[cls] = cls(*args, **kwargs)
        return _reg[cls]

    return wrapper


def thread(daemon=False):
    """decorator factory that turns functions into threading.Thread.

    >>> mouse = thread()(mouse_listener)()  # doctest: +SKIP
    >>> mouse.start()                       # doctest: +SKIP
    >>> mouse.join()                        # doctest: +SKIP
    """

    def t(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return Thread(target=f, args=args, kwargs=kwargs, daemon=daemon)

        return wrapper

    return t


def proc(daemon=False):
    """decorator factory that turns functions into multiprocessing.Process.

    >>> ps = [proc(True)(bruteforce)(x) for x in xs]     # doctest: +SKIP
    >>> for p in ps: p.start()                           # doctest: +SKIP
    >>> for p in ps: p.join()                            # doctest: +SKIP
    """

    def p(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return Process(target=f, args=args, kwargs=kwargs, daemon=daemon)

        return wrapper

    return p


class polling:
    """repeatedly executes a provided function at fixed time intervals.

    >>> g = f_(cf_(print, force), lazy(randint, 100))  # doctest: +SKIP
    >>> p = polling(1, g)                              # doctest: +SKIP
    >>> p.start()                                      # doctest: +SKIP
    """

    def __init__(self, sec, f, *args, **kwargs):
        self.expr = lazy(f, *args, **kwargs)
        self.timer = f_(Timer, sec, self._g)
        self.on = False
        self.t = None

    def _g(self):
        if self.on:
            self.expr()
            self.t = self.timer()
            self.t.start()

    def start(self):
        if not self.on:
            self.on = True
            self._g()

    def stop(self):
        self.on = False
        if self.t:
            self.t.cancel()


@fx
def shell(cmd, sync=True, o=True, *, executable="/bin/bash"):
    """execute shell commands [sync|async]hronously and capture its outputs.

      --------------------------------------------------------------------
        o-value  |  return |  meaning
      --------------------------------------------------------------------
         o =  1  |  [str]  |  captures stdout/stderr (2>&1)
         o = -1  |  None   |  discard (&>/dev/null)
      otherwise  |  None   |  do nothing or redirection (2>&1 or &>FILE)
      --------------------------------------------------------------------

    >>> shell("ls -1 ~")                   # doctest: +SKIP
    >>> shell("find . | sort" o=-1)        # doctest: +SKIP
    >>> shell("cat *.md", o=writer(FILE))  # doctest: +SKIP
    """
    import shlex

    o = PIPE if o == 1 else DEVNULL if o == -1 else 0 if isinstance(o, int) else o
    sh = f_(
        Popen,
        cf_(unwords, mapl(normpath), shlex.split)(cmd),
        stdin=PIPE,
        stderr=STDOUT,
        shell=True,
        executable=executable,
    )
    if sync:
        if o == PIPE:
            proc = sh(stdout=o)
            out, _ = proc.communicate()
            return lines(out.decode())
        else:
            sh(stdout=o).communicate()
    else:
        sh(stdout=o)


@fx
def pbcopy(x):
    Popen("pbcopy", stdin=PIPE).communicate(x.encode())


def pbpaste():
    return Popen("pbpaste", stdout=PIPE).stdout.read().decode()


def timer(t, msg="", quiet=False):
    guard(isinstance(t, (int, float)), f"timer, not a number: {t}")
    guard(t > 0, "timer, must be given a positive number: {t}")
    t = int(t)
    fmt = f"{len(str(t))}d"
    while t >= 0:
        if not quiet:
            print(f"{msg}  {t:{fmt}}", end="\r")
        time.sleep(1)
        t -= 1
    writer().write("\033[K")


def neatly(x, _cols=None, _width=10000, _repr=True, _root=True, **kwargs):
    """create neatly formatted string for data structure of 'dict' and 'list'."""

    def indent(x, i):
        def u(c, j=0):
            return f"{c:3}{x[j:]}"

        return (
            (u("-", 3) if i else u("+", 3))
            if x and x[0] == "|"
            else f"{x}"
            if x and x[0] == ":"
            else ((u("-") if x[0] == "+" else u("")) if i else u("+"))
        )

    def bullet(o, s):
        return (
            (indent(x, i) for i, x in enumerate(s))
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
            for a, o in sort(d.items())
            for k, v in [
                ("", b) if i else (a, b)
                for i, b in enumerate(
                    bullet(o, lines(neatly(o, _repr=_repr, _root=False)))
                )
            ]
        )
    elif isinstance(x, list):
        if _root:
            return neatly({"'": x}, _repr=_repr, _root=False)
        return unlines(
            filine(v, _width, "", "   ")
            for o in x
            for v in bullet(o, lines(neatly(o, _repr=_repr, _root=False)))
        )
    else:
        return (repr if _repr else str)(x)


@fx
def nprint(x, *, _cols=None, _width=10000, _repr=True, **kwargs):
    """neatly print data structures of 'dict' and 'list' using `neatly`."""
    print(neatly(x, _cols=_cols, _width=_width, _repr=_repr, **kwargs))


def timestamp(*, origin=None, w=0, d=0, h=0, m=0, s=0, from_iso=None, to_iso=False):
    if from_iso:
        t = datetime.strptime(from_iso, "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
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

    return to_iso and f"{datetime.fromtimestamp(t).isoformat()[:26]}Z" or t


def taskbar(x=None, desc="working", *, start=0, total=None, barcolor="white", **kwargs):
    """flexible tqdm-like progress bar relying on 'pip' package only"""
    import pip._vendor.rich.progress as rp

    class SpeedColumn(rp.ProgressColumn):
        def render(self, task) -> rp.Text:
            if task.speed is None:
                return rp.Text("?", style="progress.data.speed")
            return rp.Text(f"{task.speed:2.2f} it/s", style="progress.data.speed")

    def track(tb, x, start, total):
        with tb:
            if total is None:
                total = len(x) if float(op.length_hint(x)) else None
            if start:
                guard(total is not None, f"taskbar, not subscriptable: {x}")
                start = total + start if start < 0 else start
                x = islice(x, start, None)
            task = tb.add_task(desc, completed=start, total=total)
            yield from tb.track(x, task_id=task, total=total, description=desc)
            if total:
                tb._tasks.get(task).completed = total

    tb = rp.Progress(
        "[progress.description]{task.description}",
        "",
        rp.TaskProgressColumn(),
        "",
        rp.BarColumn(complete_style=barcolor, finished_style=barcolor),
        "",
        rp.MofNCompleteColumn(),
        "",
        rp.TimeElapsedColumn(),
        "<",
        rp.TimeRemainingColumn(),
        "",
        SpeedColumn(),
        **kwargs,
    )
    return tb if x is None else track(tb, x, start, total)


@lru_cache
@trap(f_(const, 0))
def nfpos(f):
    """get the number of positional-only-arguments of a given function.

    >>> nfpos(bimap)
    3
    >>> nfpos(print)
    0
    """
    return builtins.sum(
        1
        for p in signature(f).parameters.values()
        if (
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.default is p.empty
        )
    )


def __sig__(xs):
    @trap(f_(const, " is a built-in: live-inspect not available"))
    def sig(o):
        return signature(o).__str__()

    return dmap({x: x + sig(eval(x)) for x in xs})


def catalog(*, fx=False, dict=False):
    """display/get the list of functions available."""
    o = __sig__(lsfx() if fx else __all__)
    if dict:
        return o
    else:
        nprint(o, _cols=14, _repr=False)


def lsfx():
    return [
        key
        for key, o in sort(sys.modules[__name__].__dict__.items())
        if callable(o) and isinstance(o, composable)
    ]


# -------------------------------
# aliases for convenience
# -------------------------------
uniq = nub
unpack = chars
xargs = uncurry
zipwith = mapl
sort = fx(sorted)
reverse = fx(reversed)
length = fx(len)
abs = fx(abs)
sum = fx(sum)
min = fx(min)
max = fx(max)
ord = fx(ord)
chr = fx(chr)
all = fx(all)
any = fx(any)


sys.setrecursionlimit(5000)
