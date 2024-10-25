import builtins
import itertools as it
import operator as op
import sys
from collections import deque
from collections.abc import Iterable, Iterator
from functools import lru_cache, partial, reduce, wraps
from inspect import Signature, signature
from itertools import accumulate, count, cycle, dropwhile, islice
from itertools import product as cprod
from itertools import takewhile

__version__ = "0.5.2"

__all__ = [
    "_",
    "abs",
    "all",
    "and_",
    "any",
    "apply",
    "bimap",
    "c_",
    "c__",
    "cartprod",
    "cartprodl",
    "catalog",
    "cf_",
    "cf__",
    "cfd_",
    "cfd__",
    "chars",
    "chr",
    "collect",
    "combination",
    "concat",
    "concatl",
    "concatmap",
    "concatmapl",
    "const",
    "count",
    "curry",
    "cycle",
    "deque",
    "divmod",
    "drop",
    "dropwhile",
    "dropwhilel",
    "elem",
    "enumeratel",
    "error",
    "even",
    "f_",
    "f__",
    "filter",
    "filterl",
    "first",
    "flip",
    "foldl",
    "foldl1",
    "foldr",
    "foldr1",
    "force",
    "fst",
    "fx",
    "g_",
    "guard",
    "guard_",
    "head",
    "id",
    "ilen",
    "in_",
    "init",
    "intercalate",
    "intersperse",
    "is_",
    "islice",
    "isnt_",
    "iterate",
    "last",
    "lazy",
    "length",
    "lines",
    "map",
    "mapl",
    "max",
    "min",
    "not_",
    "not_elem",
    "nth",
    "nub",
    "null",
    "odd",
    "op",
    "or_",
    "ord",
    "pair",
    "permutation",
    "pred",
    "product",
    "rangel",
    "repeat",
    "replicate",
    "rev",
    "rg",
    "safe",
    "scanl",
    "scanl1",
    "scanr",
    "scanr1",
    "second",
    "seq",
    "snd",
    "sort",
    "succ",
    "sum",
    "tail",
    "take",
    "takewhile",
    "takewhilel",
    "trap",
    "u_",
    "unchars",
    "uncurry",
    "unfx",
    "uniq",
    "unlines",
    "until",
    "unwords",
    "void",
    "words",
    "xlambda",
    "zip",
    "zipl",
    "zipwith",
]


class fx:
    """Create a function to make a given function composable using symbols.
    The ``fx`` generates symbol-composable functions on the spot.

    >>> 7 | fx(lambda x: x * 6)
    42
    >>> @fx
    ... def fn(x, y):
    ...     ...

    ``fx`` stands for 'Function eXtension' which can lift all kinds of functions.
    ``fx`` allows functions to be composed in two intuitive ways with symbols.

    +---------------+-----------------------------------------+---------------+
    |     symbol    |               description               |  eval-order   |
    +---------------+-----------------------------------------+---------------+
    | ``^`` (caret) | same as dot(``.``) mathematical symbol  | right-to-left |
    +---------------+-----------------------------------------+---------------+
    | ``|`` (pipe)  | in Unix pipeline (``|``) manner         | left-to-right |
    +---------------+-----------------------------------------+---------------+
    > If you don't like function composition using symbols, use ``cf_``.
    > It's the most reliable and safe way to use it for all functions.

    >>> ((_ * 6) ^ (_ + 4))(3)
    42
    >>> 3 | (_ + 4) | (_ * 6)
    42
    >>> cf_(_ * 6, _ + 4)(3)
    42

    >>> (length ^ range)(10)
    10
    >>> range(10) | length
    10
    >>> cf_(length, range)(10)
    10

    >>> (collect ^ filter(even) ^ range)(10)
    [0, 2, 4, 6, 8]
    >>> range(10) | filter(even) | collect
    [0, 2, 4, 6, 8]
    >>> cf_(collect, filter(even), range)(10)
    [0, 2, 4, 6, 8]

    >>> (collect ^ map(pred ^ succ) ^ range)(5)
    [0, 1, 2, 3, 4]
    >>> (sum ^ map(_ + 5) ^ range)(10)
    95
    >>> range(10) | map(_ + 5) | sum
    95

    >>> ((_ * 5) ^ nth(3) ^ range)(5)
    10
    >>> 5 | fx(range) | nth(3) | (_ * 5)
    10

    >>> (unchars ^ map(chr))(range(73, 82))
    'IJKLMNOPQ'
    >>> range(73, 82) | map(chr) | unchars
    'IJKLMNOPQ'
    """

    def __new__(cls, f):
        if type(f) is fx:  # join :: m (m a) -> m a
            return f
        obj = super().__new__(cls)
        obj.f = f
        wraps(f)(obj)
        return obj

    def __xor__(self, o):
        """dot composition operator (``^``)"""
        return cf_(self, o)

    def __or__(self, o):
        """pipe composition operator (``|``)"""
        return cf_(o, self)

    def __ror__(self, o):
        """pipe composition operator (``|``)"""
        return self.f(o)

    def __call__(self, *args, **kwargs):
        arity = farity(self.f)
        if args and len(args) >= arity or (not args and arity < 1):
            return self.f(*args, **kwargs)
        else:
            return f_(self.f, *args, **kwargs)

    def __repr__(self):
        return show(self)


@lru_cache
def unfx(f):
    """Unlift a given ``fx``-lifted function."""
    return f.f if type(f) is fx else f


@fx
def id(x):
    """Identity function

    >>> id("francis")
    'francis'
    >>> id("francis") == "francis" | id
    True
    """
    return x


@fx
def const(x, _):
    """Build an id function that returns a given ``x``.

    >>> const(5, "no-matther-what-comes-here")
    5
    >>> 'whatever' | const(5)
    5
    """
    return x


@fx
def seq(_, x):
    """Return the id function after consuming the given argument.

    >>> seq("only-returns-the-following-arg", 5)
    5
    >>> 5 | seq('whatever')
    5
    """
    return x


@fx
def void(_):
    """Return 'None' after consuming the given argument.

    >>> void("no-matther-what-comes-here")
    >>> "no-matther-what-comes-here"| void
    """
    return


@fx
def fst(x):
    """get the first component of a given iterable.

    >>> fst(["sofia", "maria", "claire"])
    'sofia'
    >>> ["sofia", "maria", "claire"] | fst
    'sofia'
    """
    return safe(nth)(1, x)


@fx
def snd(x):
    """Get the second component of a given iterable.

    >>> snd(("sofia", "maria", "claire"))
    'maria'
    >>> ("sofia", "maria", "claire") | snd
    'maria'
    """
    return safe(nth)(2, x)


@fx
def nth(n, x):
    """Get the ``n``-th component of a given iterable ``x``.

    >>> nth(3, ["sofia", "maria", "claire"])
    'claire'
    >>> ["sofia", "maria", "claire"] | nth(3)
    'claire'
    """
    return x[n - 1] if hasattr(x, "__getitem__") else next(islice(x, n - 1, None))


@fx
def take(n, x):
    """Take ``n`` items from a given iterable ``x``.

    >>> take(3, range(5, 10))
    [5, 6, 7]
    >>> range(5, 10) | take(3) | collect
    [5, 6, 7]
    """
    return islice(x, n) | collect


@fx
def drop(n, x):
    """Return items of the iterable ``x`` after skipping ``n`` items.

    >>> list(drop(3, 'github'))
    ['h', 'u', 'b']
    >>> 'github' | drop(3) | collect
    ['h', 'u', 'b']
    """
    return islice(x, n, None)


@fx
def head(x):
    """Extract the first element of a given iterable: the same as ``fst``.

    >>> head(range(1, 5))
    1
    >>> range(1, 5) | head
    1
    """
    return fst(x)


@fx
def tail(x):
    """Extract the elements after the ``head`` of a given iterable.

    >>> list(tail(range(1, 5)))
    [2, 3, 4]
    >>> range(1, 5) | tail | collect
    [2, 3, 4]
    """
    return safe(drop)(1, x)


@fx
def init(x):
    """Return all the elements of an iterable except the ``last`` one.

    >>> list(init(range(1, 5)))
    [1, 2, 3]
    >>> range(1, 5) | init | collect
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
def last(x):
    """Extract the last element of a given iterable.

    >>> last(range(1, 5))
    4
    >>> range(1, 5) | last
    4
    """
    return fst(deque(x, maxlen=1))


@fx
def ilen(x):
    """Get the length of a given iterator.

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
    """Make the given two arguments a tuple pair.

    >>> pair("sofia", "maria")
    ('sofia', 'maria')
    >>> "maria" | pair("sofia")
    ('sofia', 'maria')
    """
    return (a, b)


@fx
def pred(x):
    """Return the predecessor of a given value: it substracts 1.

    >>> pred(3)
    2
    >>> 3 | pred
    2
    """
    return x - 1


@fx
def succ(x):
    """Return the successor of a given value: it adds 1.

    >>> succ(3)
    4
    >>> 3 | succ
    4
    """
    return x + 1


@fx
def odd(x):
    """Check if the given number is odd.

    >>> odd(3)
    True
    >>> 3 | odd
    True
    """
    return x % 2 == 1


@fx
def even(x):
    """Check if the given number is even.

    >>> even(3)
    False
    >>> 3 | even
    False
    """
    return x % 2 == 0


@fx
def null(x):
    """Check if a given collection is empty.

    >>> null([]) == null(()) == null({}) == null('')
    True
    >>> [] | null == () | null == {} | null == '' | null
    True
    """
    return len(x) == 0


@fx
def chars(x):
    """Split string ``x`` into `chars`: the same as (:[]) <$> x.

    >>> chars("sofimarie")
    ['s', 'o', 'f', 'i', 'm', 'a', 'r', 'i', 'e']
    >>> chars("sofimarie") == "sofimarie" | chars
    True
    """
    return list(x)


@fx
def unchars(x):
    """Inverse operation of ``chars``: the same as ``concat``.

    >>> unchars(['s', 'o', 'f', 'i', 'm', 'a', 'r', 'i', 'e'])
    'sofimarie'
    >>> ['s', 'o', 'f', 'i', 'm', 'a', 'r', 'i', 'e'] | unchars
    'sofimarie'
    """
    return "".join(x)


@fx
def words(x):
    """Joins a list of words with the blank character.

    >>> words("fun on functions")
    ['fun', 'on', 'functions']
    >>> 'fun on functions' | words
    ['fun', 'on', 'functions']
    """
    return x.split()


@fx
def unwords(x):
    """Breaks a string up into a list of words.

    >>> unwords(['fun', 'on', 'functions'])
    'fun on functions'
    >>> ['fun', 'on', 'functions'] | unwords
    'fun on functions'
    """
    return " ".join(x)


@fx
def lines(x):
    """Splits a string into a list of lines using the delimeter ``\`n`.

    >>> lines("fun\\non\\nfunctions")
    ['fun', 'on', 'functions']
    >>> "fun\\non\\nfunctions" | lines
    ['fun', 'on', 'functions']
    """
    return x.splitlines()


@fx
def unlines(x):
    """Joins a list of lines with the newline character, ``\n``.

    >>> unlines(['fun', 'on', 'functions'])
    'fun\\non\\nfunctions'
    >>> ['fun', 'on', 'functions'] | unlines
    'fun\\non\\nfunctions'
    """
    return "\n".join(x)


@fx
def elem(x, xs):
    """Check if the element exists in the given iterable.

    >>> elem("fun", "functions")
    True
    >>> "functions" | elem("fun")
    True
    """
    return x in xs


@fx
def not_elem(x, xs):
    """Negation of ``elem``. The same as `not_ . elem`.

    >>> not_elem("fun", "functions")
    False
    >>> "functions" | not_elem("fun")
    False
    """
    return x not in xs


@fx
def nub(x):
    """Removes duplicate elements from a given iterable.

    >>> nub("3333-13-1111111")
    ['3', '-', '1']
    >>> "3333-13-1111111" | nub
    ['3', '-', '1']
    """
    return list(dict.fromkeys(x))


@fx
def repeat(x):
    """Create an infinite list with the argument ``x``.

    >>> take(3, repeat(5))
    [5, 5, 5]
    >>> repeat(5) | take(3)
    [5, 5, 5]
    """
    return (x for _ in count())


@fx
def replicate(n, x):
    """Get a list of length `n` from an infinite list with `x` values.

    >>> replicate(3, 5)
    [5, 5, 5]
    >>> 5 | replicate(3)
    [5, 5, 5]
    """
    return take(n, repeat(x))


@fx
def product(x):
    """Product of the elements for the given iterable.

    >>> product(range(1, 11))
    3628800
    >>> range(1, 11) | product
    3628800
    """
    return foldl1(op.mul, x)


def flip(f):
    """Reverses the order of arguments of a given function ``f``.

    >>> flip(pow)(7, 3)
    2187
    >>> (7, 3) | u_(flip((op.sub)))
    -4
    """

    @wraps(f)
    def go(*args, **kwargs):
        return f(*reversed(args), **kwargs)

    go.__signature__ = sig(f, reverse=True)
    return go


def f_(f, *args, **kwargs):
    """Build left-associative partial application,
    where the given function's arguments partially evaluation from the left.

    >>> f_(op.sub, 7)(5)
    2
    >>> 5 | f_(op.sub, 7)
    2
    """
    return fx(partial(unfx(f), *args, **kwargs))


def f__(f, *args, **kwargs):
    """Build right-associative partial application,
    where the given function's arguments partially evaluation from the right.

    >>> f__(op.sub, 7)(5)
    -2
    >>> 5 | f__(op.sub, 7)
    -2
    """
    return fx(partial(flip(f), *args, **kwargs))


def curry(f, *, a=None):
    """Build left-associative curried function.

    This function takes positional arguments only when currying.
    Use partial application `f_` if you need to change keyword arguments.

    >>> curry(op.sub)(5)(2)
    3
    >>> curry(foldl)(op.add)(0)(range(1, 101))
    5050
    """
    f = unfx(f)
    a = farity(f) if a is None else a

    @wraps(f)
    def go(x):
        return f(x) if a <= 1 else curry(partial(f, x), a=a - 1)

    return go


def c__(f):
    """Build right-associative curried function.

    >>> c__(op.sub)(5)(2)
    -3
    >>> c__(foldl)(range(1, 11))(0)(op.add)
    55
    """
    return curry(flip(f))


def uncurry(f):
    """Convert a function to a unary function that takes a tuple of arguments.
    This is identical to the behavior of the `uncurry` in Haskell.
    ``uncurry :: (a -> b -> ... -> x -> o) -> (a, b, ..., x) -> o``

    ``uncurry`` is useful for controlling arguments when composing functions.

    Python functions do not automatically curry like Haskell, which means
    this function cannot restore a curried function to its original form.
    In that case, use ``f.__wrapped__`` to get the original uncurried function.

    >>> u_ = uncurry
    >>> u_(pow)((2, 10))
    1024
    >>> (2, 3) | u_(op.add)
    5
    >>> ([1, 3], [2, 4]) | u_(zip) | collect
    [(1, 2), (3, 4)]
    """
    return fx(lambda x: f(*x))


def cf_(*fs, rep=None):
    """Create composite functions using functions provided in arguments.

    >>> cf_(_ * 7, _ + 3)(5)
    56
    >>> cf_(_["sofia"], dict)([("sofia", "piano"), ("maria", "violin")])
    'piano'
    """

    def g_f(g, f):
        return lambda *args, **kwargs: g(f(*args, **kwargs))

    return fx(reduce(g_f, [unfx(f) for f in fs] * (rep or 1)))


def cf__(*fs, rep=None):
    """Composes functions in the same way as ``cf_``, but uses arguments in order.

    >>> cf__(_ + 3, _ * 7)(5)
    56
    >>> cf__(dict, _["sofia"])([("sofia", "piano"), ("maria", "violin")])
    'piano'
    """
    return cf_(*reversed(fs), rep=rep)


def cfd_(*fs, rep=None):
    """Decorate functions using function composition of given arguments.

    >>> cfd_(set, list, tuple)(range)(5)
    {0, 1, 2, 3, 4}
    >>> cfd_(chars, _["maria"])(dict)([("sofia", "piano"), ("maria", "violin")])
    ['v', 'i', 'o', 'l', 'i', 'n']
    """

    def go(f):
        return fx(lambda *args, **kwargs: cf_(*fs, rep=rep)(f(*args, *kwargs)))

    return go


def cfd__(*fs, rep=None):
    """Decorate functions in the same way as ``cfd_``, but uses arguments in order.

    >>> cfd__(set, list, tuple)(range)(5)
    (0, 1, 2, 3, 4)
    >>> cfd__(_["maria"], chars)(dict)([("sofia", "piano"), ("maria", "violin")])
    ['v', 'i', 'o', 'l', 'i', 'n']
    """
    return cfd_(*reversed(fs), rep=rep)


def g_(f):
    """Build partially-applied methods using attribute getter ``f``.
    This is useful for expressing method chaining (fluent pattern)
    as function composition.

    >>> g_(_.join)(["sofia", "maria"])(", ")
    'sofia, maria'
    """
    guard(
        type(f) is fx and isinstance(f.f, op.attrgetter),
        msg=f"error, no attribute getter found: {show(f)}",
    )

    def go(*args, **kwargs):
        return fx(lambda o: f(o)(*args, **kwargs))

    return go


def map(f, *xs):
    """fx-lifted ``map``, which seamlessly extends ``builtins.map``.

    >>> (collect ^ map(abs))(range(-2, 3)) | collect
    [2, 1, 0, 1, 2]
    >>> map(abs)(range(-2, 3)) | collect
    [2, 1, 0, 1, 2]

    >>> (collect ^ map(_*8))(range(1, 6))
    [8, 16, 24, 32, 40]
    >>> range(1, 6) | map(_*8) | collect
    [8, 16, 24, 32, 40]

    >>> (collect ^ map(op.mul, [1, 2, 3]))([4, 5, 6])
    [4, 10, 18]
    >>> [4, 5, 6] | map(op.mul, [1, 2, 3]) | collect
    [4, 10, 18]
    """
    if xs and len(xs) >= farity(f):  # when populated args
        return builtins.map(f, *xs)
    else:
        return f_(builtins.map, f, *xs)


def mapl(f, *xs):
    """The same as ``map``, but returns in ``list``.

    >>> mapl(abs)(range(-2, 3))
    [2, 1, 0, 1, 2]
    >>> range(-2, 3) | mapl(abs)
    [2, 1, 0, 1, 2]
    >>> mapl(op.mul, [1, 2, 3], [4, 5, 6])
    [4, 10, 18]
    >>> [4, 5, 6] | mapl(op.mul, [1, 2, 3])
    [4, 10, 18]
    """
    if xs and len(xs) >= farity(f):  # when populated args
        return builtins.map(f, *xs) | collect
    else:
        return cfd_(list)(f_(builtins.map, f, *xs))


@fx
def filter(p, xs):
    """The same as ``builtins.filter``, but lifted by ``fx``.

    >>> (collect ^ filter(_ == "f"))("fun-on-functions")
    ['f', 'f']
    >>> "fun-on-functions" | filter(_ == "f") | collect
    ['f', 'f']

    >>> primes = [2, 3, 5, 7, 11, 13, 17, 19]
    >>> (collect ^ filter((_ == 2) ^ (_ % 3)))(primes)
    [2, 5, 11, 17]
    >>> (filter((_ == 2) ^ (_ % 3)) | collect)(primes)
    [2, 5, 11, 17]
    >>> primes | filter((_ == 2) ^ (_ % 3)) | collect
    [2, 5, 11, 17]
    >>> primes | filter((_ % 3) | (_ == 2)) | collect
    [2, 5, 11, 17]
    """
    return builtins.filter(p, xs)


@fx
def filterl(p, xs):
    """The same as ``filter``, but returns in ``list``.

    >>> (collect ^ filter(_ == "f"))("fun-on-functions")
    ['f', 'f']
    >>> "fun-on-functions" | filter(_ == "f") | collect
    ['f', 'f']

    >>> primes = [2, 3, 5, 7, 11, 13, 17, 19]
    >>> filterl((_ == 2) ^ (_ % 3))(primes)
    [2, 5, 11, 17]
    >>> filterl((_ % 3) | (_ == 2))(primes)
    [2, 5, 11, 17]
    >>> primes | filterl((_ == 2) ^ (_ % 3))
    [2, 5, 11, 17]
    >>> primes | filterl((_ % 3) | (_ == 2))
    [2, 5, 11, 17]
    """
    return filter(p, xs) | collect


@fx
def zip(*xs, strict=False):
    """The same as ``builtins.zip``, but lifted by ``fx``.

    >>> zip("LOVE", range(3)) | collect
    [('L', 0), ('O', 1), ('V', 2)]
    >>> (f_(zip, "LOVE") ^ range)(3) | collect
    [('L', 0), ('O', 1), ('V', 2)]

    Note that ``u_(zip)(x) equals zip(*x)`` for an iterable ``x``.

    >>> (collect ^ u_(zip))(["LOVE", range(3)])
    [('L', 0), ('O', 1), ('V', 2)]
    >>> ["LOVE", range(3)] | u_(zip) | collect
    [('L', 0), ('O', 1), ('V', 2)]
    """
    return builtins.zip(*xs, strict=strict)


@fx
def zipl(*xs, strict=False):
    """The same as ``zip``, but returns in ``list``.

    >>> zipl("LOVE", range(3))
    [('L', 0), ('O', 1), ('V', 2)]
    >>> (f_(zipl, "LOVE") ^ range)(3)
    [('L', 0), ('O', 1), ('V', 2)]

    Note that ``u_(zipl)(x) equals zipl(*x)`` for an iterable ``x``.

    >>> u_(zipl)(["LOVE", range(3)])
    [('L', 0), ('O', 1), ('V', 2)]
    >>> ["LOVE", range(3)] | u_(zipl)
    [('L', 0), ('O', 1), ('V', 2)]
    """
    return zip(*xs, strict=strict) | collect


def rangel(*args, **kwargs):
    """The same as ``range``, but returns in ``list``.

    >>> rangel(10) == (range(10) | collect)
    True
    """
    return range(*args, **kwargs) | collect


def enumeratel(xs, start=0):
    """The same as ``enumerate``, but returns in ``list``.

    >>> enumeratel(range(10)) == (enumerate(range(10)) | collect)
    True
    """
    return enumerate(xs, start=start) | collect


@fx
def takewhilel(p, xs):
    """The same as ``takewhile``, but returns in ``list``.

    >>> takewhilel(even, [2, 4, 6, 1, 3, 5])
    [2, 4, 6]
    """
    return takewhile(p, xs) | collect


@fx
def dropwhilel(p, xs):
    """The same as ``dropwhile``, but returns in ``list``.

    >>> dropwhilel(even, [2, 4, 6, 1, 3, 5])
    [1, 3, 5]
    """
    return dropwhile(p, xs) | collect


@fx
def not_(x):
    """``not`` as a function.

    >>> not_(False)
    True
    >>> False | not_
    True
    """
    return not x


@fx
def and_(a, b):
    """``and`` as a function.

    >>> and_(True, False)
    False
    >>> False | and_(True)
    False
    """
    return a and b


@fx
def or_(a, b):
    """``or`` as a function.

    >>> or_(True, False)
    True
    >>> False | or_(True)
    True
    """
    return a or b


@fx
def in_(a, b):
    """``in`` as a function.

    >>> in_("fun", "function")
    True
    >>> 'function' | in_("fun")
    True
    """
    return op.contains(b, a)


@fx
def is_(a, b):
    """``is`` as a function.

    >>> is_("war", "LOVE")
    False
    >>> "LOVE" | is_("war")
    False
    """
    return op.is_(a, b)


@fx
def isnt_(a, b):
    """``is not`` as a function.

    >>> isnt_("war", "LOVE")
    True
    >>> "LOVE" | isnt_("war")
    True
    """
    return op.is_not(a, b)


@fx
def bimap(f, g, x):
    """Map over both ``first`` and ``second`` arguments at the same time.
    bimap(f, g) == first(f) . second(g)

    >>> bimap(_ + 3, _ * 7, (5, 7))
    (8, 49)
    >>> (5, 7) | bimap(_ + 3, _ * 7)
    (8, 49)
    """
    return f(fst(x)), g(snd(x))


@fx
def first(f, x):
    """Map only the first argument and leave the rest unchanged.

    >>> first(_ + 3, (5, 7))
    (8, 7)
    >>> (5, 7) | first(_ +  3)
    (8, 7)
    """
    return bimap(f, id, x)


@fx
def second(g, x):
    """Map only the second argument and leave the rest unchanged.

    >>> second(_ * 7, (5, 7))
    (5, 49)
    >>> (5, 7) | second(_ * 7)
    (5, 49)
    """
    return bimap(id, g, x)


@fx
def until(p, f, x):
    """Return the result of applying the given ``f`` until the given ``p`` holds

    >>> until(_ > 1024, _ * 3, 5)
    1215
    >>> until(cf_(_ == 3, _ % 5), _ * 3, 119)
    3213
    """
    while not p(x):
        x = f(x)
    return x


@fx
def iterate(f, x):
    """Return an infinite list of repeated applications of ``f`` to ``x``.

    >>> take(5, iterate(_ ** 2, 2))
    [2, 4, 16, 256, 65536]
    >>> 2 | iterate(_ ** 2) | take(5)
    [2, 4, 16, 256, 65536]
    """
    while True:
        yield x
        x = f(x)


def apply(f, *args, **kwargs):
    """Call a given function with the given arguments.

    >>> apply(str.split, "go get some coffee")
    ['go', 'get', 'some', 'coffee']
    >>> apply(mapl, even, range(4))
    [True, False, True, False]
    """
    return fx(f)(*args, **kwargs)


@fx
def foldl(f, initial, xs):
    """Left-associative fold of an iterable.

    >>> foldl(op.sub, 10, range(1, 5))
    0
    >>> range(1, 5) | foldl(op.sub, 10)
    0
    """
    return reduce(f, xs, initial)


@fx
def foldl1(f, xs):
    """`foldl` without initial value.

    >>> foldl1(op.sub, range(1, 5))
    -8
    >>> range(1, 5) | foldl1(op.sub)
    -8
    """
    return reduce(f, xs)


@fx
def foldr(f, inital, xs):
    """Right-associative fold of an iterable.

    >>> foldr(op.sub, 10, range(1, 5))
    8
    >>> range(1, 5) | foldr(op.sub, 10)
    8
    """
    return reduce(flip(f), reversed(xs), inital)


@fx
def foldr1(f, xs):
    """`foldr` without initial value.

    >>> foldr1(op.sub, range(1, 5))
    -2
    >>> range(1, 5) | foldr1(op.sub)
    -2
    """
    return reduce(flip(f), reversed(xs))


@fx
def scanl(f, initial, xs):
    """Return a list of successive reduced values from the left.

    >>> scanl(op.sub, 10, range(1, 5))
    [10, 9, 7, 4, 0]
    >>> range(1, 5) | scanl(op.sub, 10)
    [10, 9, 7, 4, 0]
    """
    return accumulate(xs, f, initial=initial) | collect


@fx
def scanl1(f, xs):
    """`scanl` without starting value.

    >>> scanl1(op.sub, range(1, 5))
    [1, -1, -4, -8]
    >>> range(1, 5) | scanl1(op.sub)
    [1, -1, -4, -8]
    """
    return accumulate(xs, f) | collect


@fx
def scanr(f, initial, xs):
    """Return a list of successive reduced values from the right.

    >>> scanr(op.sub, 10, range(1, 5))
    [8, -7, 9, -6, 10]
    >>> range(1, 5) | scanr(op.sub, 10)
    [8, -7, 9, -6, 10]
    """
    return accumulate(reversed(xs), flip(f), initial=initial) | rev


@fx
def scanr1(f, xs):
    """`scanr` without starting value.

    >>> scanr1(op.sub, range(1, 5))
    [-2, 3, -1, 4]
    >>> range(1, 5) | scanr1(op.sub)
    [-2, 3, -1, 4]
    """
    return accumulate(reversed(xs), flip(f)) | rev


@fx
def permutation(x, r, rep=False):
    """Return all permutations in a list form

    >>> permutation("abc", 2) | collect
    [('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]
    """
    return cprod(x, repeat=r) if rep else it.permutations(x, r)


@fx
def combination(x, r, rep=False):
    """Return all combinations in a list form
    >>> combination("abc", 2) | collect
    [('a', 'b'), ('a', 'c'), ('b', 'c')]

    """
    return it.combinations_with_replacement(x, r) if rep else it.combinations(x, r)


@fx
def cartprod(*xs):
    """Return Cartesian product.

    >>> cartprod("↑↓", "↑↓") | collect
    [('↑', '↑'), ('↑', '↓'), ('↓', '↑'), ('↓', '↓')]
    >>> ("↑↓", "↑↓") | u_(cartprod) | collect
    [('↑', '↑'), ('↑', '↓'), ('↓', '↑'), ('↓', '↓')]
    """
    return cprod(*xs, repeat=1)


@fx
def cartprodl(*xs):
    """The same as ``cartprod``, but returns in ``list``.

    >>> cartprodl("↑↓", "↑↓")
    [('↑', '↑'), ('↑', '↓'), ('↓', '↑'), ('↓', '↓')]
    >>> ("↑↓", "↑↓") | u_(cartprodl)
    [('↑', '↑'), ('↑', '↓'), ('↓', '↑'), ('↓', '↓')]
    """
    return cartprod(*xs) | collect


@fx
def concat(xs):
    """Concatenates all elements of iterables.

    >>> concat(["so", "fia"]) | collect
    ['s', 'o', 'f', 'i', 'a']
    >>> ["so", "fia"] | concat | collect
    ['s', 'o', 'f', 'i', 'a']
    """
    return it.chain.from_iterable(xs)


@fx
def concatl(xs):
    """The same as ``concat``, but returns in ``list``.

    >>> concatl(["so", "fia"])
    ['s', 'o', 'f', 'i', 'a']
    >>> ["so", "fia"] | concatl
    ['s', 'o', 'f', 'i', 'a']
    """
    return concat(xs) | collect


@fx
def concatmap(f, x, *xs):
    """Map a function over the given iterable then concat it.

    >>> concatmap(str.upper, ["mar", "ia"]) | collect
    ['M', 'A', 'R', 'I', 'A']
    >>> ["mar", "ia"] | concatmap(str.upper) | collect
    ['M', 'A', 'R', 'I', 'A']
    """
    return map(f, x, *xs) | concat


@fx
def concatmapl(f, x, *xs):
    """The same as ``concatmap``, but returns in ``list``.

    >>> concatmapl(str.upper, ["mar", "ia"])
    ['M', 'A', 'R', 'I', 'A']
    >>> ["mar", "ia"] | concatmapl(str.upper)
    ['M', 'A', 'R', 'I', 'A']
    """
    return map(f, x, *xs) | concat | collect


@fx
def intersperse(sep, x):
    """Intersperse the given element between the elements of the list.

    >>> intersperse("\u2764", ["francis", "claire"])
    ['francis', '❤', 'claire']
    >>> ["francis", "claire"] | intersperse("\u2764")
    ['francis', '❤', 'claire']
    """
    return concatl(zip(repeat(sep), x))[1:]


@fx
def intercalate(sep, x):
    """Inserts the given list between the lists then concat it.

    >>> intercalate("\u2764", [["francis"], ["claire"]])
    ['francis', '❤', 'claire']
    >>> [["francis"], ["claire"]] | intercalate("\u2764")
    ['francis', '❤', 'claire']
    """
    return intersperse(sep, x) | concatl


@fx
def collect(x):
    """Unpack lazy-iterables in lists and leave other iterables unchanged

    >>> collect((1,2,3,4,5))
    (1, 2, 3, 4, 5)
    >>> collect(range(5))
    [0, 1, 2, 3, 4]
    >>> (x for x in range(5)) | collect
    [0, 1, 2, 3, 4]
    """
    if isinstance(x, Iterable):
        return (
            list(x)
            if (
                isinstance(x, Iterator)
                or (isinstance(x, Iterable) and not hasattr(x, "__getitem__"))
                or isinstance(x, range)
            )
            else x
        )
    else:
        error(f"collect, no iterables given, got {type(x)}")


@fx
def rev(x):
    """Take an iterable then return reversed strict sequence.

    >>> rev(range(5))
    [4, 3, 2, 1, 0]
    >>> rev(range(5)) == range(5) | rev
    True
    >>> rev("dog")
    'god'
    """
    return x[::-1] if isinstance(x, (str, bytes, bytearray)) else list(x)[::-1]


def rg(i, j, k=None, l=None, /):
    """Introduce intuitive Haskell-like lazy range notation.

    +----------+---------------+----------------------------------------+
    | Haskell  | rg function   | definition                             |
    +----------+---------------+----------------------------------------+
    | [a..]    | rg(a,...)     | [a, a+1, ..)                           |
    +----------+---------------+----------------------------------------+
    | [a..b]   | rg(a,...,b)   | [a, a+1, ..), till (a+n) <= b          |
    +----------+---------------+----------------------------------------+
    | [a,b..]  | rg(a,b,...)   | [a, a+(b-a), ..)                       |
    +----------+---------------+----------------------------------------+
    | [a,b..c] | rg(a,b,...,c) | [a, a+(b-a), ..), till (a+n(b-a)) <= c |
    +----------+---------------+----------------------------------------+

    >>> rg(3,...) | take(5)
    [3, 4, 5, 6, 7]
    >>> rg(3,...,5) | take(5)
    [3, 4, 5]
    >>> rg(3,7,...) | take(5)
    [3, 7, 11, 15, 19]
    >>> rg(3,7,...,17) | take(5)
    [3, 7, 11, 15]
    """

    def ensure_dots(x):
        guard(x == ..., f"invalid syntax, {x} must be '...'.")

    if k is None and l is None:
        ensure_dots(j)
        return count(i)  # rg(a,...) == [a..]
    elif k is None:
        error(f"invalid syntax, {l} should not be given.")
    elif l is None:
        if j == ...:
            return takewhile(lambda x: x <= k, count(i))  # rg(a,...,b) == [a..b]
        else:
            ensure_dots(k)
            return count(i, j - i)  # rg(a,b,...) == [a,b..]
    else:
        ensure_dots(k)  # rg(a,b,...,c) == [a,b..c]
        return takewhile(lambda x: x <= l, count(i, j - i))


@fx
def force(expr):
    """Forces the delayed-expression to be fully evaluated."""
    return expr() if callable(expr) else expr


def error(msg="error", e=SystemExit):
    """``raise`` an exception with a function or expression."""
    raise e(msg) from None


def safe(f):
    """Make a given function return ``None`` instead of raising an exception.

    >>> safe(error)("never-throw-errors")
    """
    return trap(callback=void, e=None)(f)


def trap(callback, e=None):
    """Decorator factory that creates exception catchers."""

    def catcher(f):
        @wraps(f)
        def go(*args, **kwargs):
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

        return go

    return catcher


def guard(p, msg="guard", e=SystemExit):
    """``assert`` as a function or expression."""
    if not p:
        error(msg=msg, e=e)


def guard_(f, msg="guard", e=SystemExit):
    """Partial application builder for ``guard``:
    the same as ``guard``, but the positional predicate is given
    as a function rather than an boolean expression.
    """
    return lambda x: seq(guard(f(x), msg=msg, e=e))(x)


@lru_cache
def farity(f):
    """Get the number of positional arguments of a given function.

    >>> farity(bimap)
    3
    >>> farity(op.mul)
    2
    >>> farity(print)
    -1
    """
    try:
        return builtins.sum(
            1
            for p in signature(f).parameters.values()
            if (
                p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty
            )
        )
    except:
        return -1  # uninspectable functions


@lru_cache
def sig(f, reverse=False):
    """Inspect the signature of a given function ``f`` using ``inspect``"""
    try:
        return (
            Signature(list(signature(f).parameters.values())[::-1])
            if reverse
            else signature(f)
        )
        return signature(f)
    except:
        return None


def catalog():
    """Display/get the list of functions available."""
    d = {}
    for fn in sort(__all__):
        f = eval(fn)
        if not callable(f) or type(f) is xlambda:
            continue
        s = sig(eval(fn))
        d[fn] = s if s else "inspection not available"
    return d


@lru_cache
def show(f):
    """Stringify functions into human-readable form."""
    if type(f) is fx:
        return f"fx({show(f.f)})"
    elif isinstance(f, partial):
        args = f", {', '.join(f'{x}' for x in f.args)}" if f.args else ""
        kwargs = (
            f", {', '.join(f'{k}={v}' for k, v in f.keywords.items())}"
            if f.keywords
            else ""
        )
        return f"f_({show(f.func)}{args}{kwargs})"
    elif sig(f):
        return f"{f.__name__}{sig(f)}"
    elif hasattr(f, "__name__"):
        return f.__name__
    else:
        return repr(f)


class xlambda:
    """Scala-style lambda expression (unary or arity-1) builder,
    inspired by Scala's placeholder syntax for unary lambda expressions.

    Using ``_`` placeholder makes it easy to write lambda expressions.
    Just remove the ``lambda x:`` and to replace ``x`` with placeholder, ``_``

    >>> (lambda x: x + 7)(3) == (_ + 7)(3)
    True
    >>> 7 | (_ * 6) | (45 - _) | (_ // 3)
    1
    >>> cf_(sum, map(_ + 1), range)(10)
    55

    Partial application using placeholders is also possible when accessing items
    in ``dict``, ``object`` or ``iterable``, or even calling functions.

    +----------------+-------------------------+
    |    Operator    |   Equiv-Function        |
    +----------------+-------------------------+
    |  ``_[_]``      | ``op.getitem``          |
    +----------------+-------------------------+
    |  ``_[item]``   | ``op.itemgetter(item)`` |
    +----------------+-------------------------+
    |  ``_._``       | ``getattr``             |
    +----------------+-------------------------+
    |  ``_.attr``    | ``op.attrgetter(attr)`` |
    +----------------+-------------------------+
    |  ``_(_)``      | ``apply``               |
    +----------------+-------------------------+
    |  ``_(*a,**k)`` | ``lambda f: f(*a,**k)`` |
    +----------------+-------------------------+

    >>> d = dict(one=1, two=2, three="three")
    >>> o = type('', (), {"one": 1, "two": 2, "three": "three"})()
    >>> r = range(5)
    >>> (_[_])(d)("two") == (_._)(o)("two")
    True
    >>> (_["one"])(d) == (_.one)(o)
    True
    >>> cf_(_[2:4], _["three"])(d)
    're'
    >>> o | _.three | _[2:4]
    're'
    >>> _(1, 2)(op.sub)
    -1
    >>> _(3 * _)(mapl)(range(1, 5))
    [3, 6, 9, 12]
    """

    __slots__ = ()

    def __neg__(self):
        """
        >>> (-_)(3) == (lambda x: -x)(3)
        True
        """
        return fx(op.neg)

    def __pos__(self):
        """
        >>> (+_)(-3) == (lambda x: +x)(-3)
        True
        """
        return fx(op.pos)

    def __abs__(self):
        """
        >>> abs(_)(-3) == (lambda x: abs(x))(-3)
        True
        """
        return fx(abs)

    def __invert__(self):
        """
        >>> (~_)(3) == (lambda x: ~x)(3)
        True
        """
        return fx(op.invert)

    def __add__(self, o):
        """
        >>> (_ + 2)(3) == (lambda x: x + 2)(3)
        True
        """
        return f__(op.add, o)

    def __radd__(self, o):
        """
        >>> (2 + _)(3) == (lambda x: 2 + x)(3)
        True
        """
        return f_(op.add, o)

    def __sub__(self, o):
        """
        >>> (_ - 2)(3) == (lambda x: x - 2)(3)
        True
        """
        return f__(op.sub, o)

    def __rsub__(self, o):
        """
        >>> (2 - _)(3) == (lambda x: 2 - x)(3)
        True
        """
        return f_(op.sub, o)

    def __mul__(self, o):
        """
        >>> (_ * 2)(3) == (lambda x: x * 2)(3)
        True
        """
        return f__(op.mul, o)

    def __rmul__(self, o):
        """
        >>> (2 * _)(3) == (lambda x: 2 * x)(3)
        True
        """
        return f_(op.mul, o)

    def __matmul__(self, o):
        """
        >>> (_ @ A)(B) == (lambda x: A @ x)(B)  # doctest: +SKIP
        """
        return f__(op.matmul, o)

    def __rmatmul__(self, o):
        """
        >>> (A @ _)(B) == (lambda x: A @ x)(B)  # doctest: +SKIP
        """
        return f_(op.matmul, o)

    def __truediv__(self, o):
        """
        >>> (_ / 2)(3) == (lambda x: x / 2)(3)
        True
        """
        return f__(op.truediv, o)

    def __rtruediv__(self, o):
        """
        >>> (2 / _)(3) == (lambda x: 2 / x)(3)
        True
        """
        return f_(op.truediv, o)

    def __floordiv__(self, o):
        """
        >>> (_ // 2)(3) == (lambda x: x // 2)(3)
        True
        """
        return f__(op.floordiv, o)

    def __rfloordiv__(self, o):
        """
        >>> (2 // _)(3) == (lambda x: 2 // x)(3)
        True
        """
        return f_(op.floordiv, o)

    def __mod__(self, o):
        """
        >>> (_ % 2)(3) == (lambda x: x % 2)(3)
        True
        """
        return f__(op.mod, o)

    def __rmod__(self, o):
        """
        >>> (2 % _)(3) == (lambda x: 2 % x)(3)
        True
        """
        return f_(op.mod, o)

    def __pow__(self, o):
        """
        >>> (_ ** 2)(3) == (lambda x: x ** 2)(3)
        True
        """
        return f__(op.pow, o)

    def __rpow__(self, o):
        """
        >>> (2 ** _)(3) == (lambda x: 2 ** x)(3)
        True
        """
        return f_(op.pow, o)

    def __lt__(self, o):
        """
        >>> (_ < 2)(3) == (lambda x: x < 2)(3)
        True
        """
        return f__(op.lt, o)

    def __le__(self, o):
        """
        >>> (_ <= 2)(3) == (lambda x: x <= 2)(3)
        True
        """
        return f__(op.le, o)

    def __gt__(self, o):
        """
        >>> (_ > 2)(3) == (lambda x: x > 2)(3)
        True
        """
        return f__(op.gt, o)

    def __ge__(self, o):
        """
        >>> (_ >= 2)(3) == (lambda x: x >= 2)(3)
        True
        """
        return f__(op.ge, o)

    def __eq__(self, o):
        """
        >>> (_ == 2)(3) == (lambda x: x == 2)(3)
        True
        """
        return f__(op.eq, o)

    def __ne__(self, o):
        """
        >>> (_ != 2)(3) == (lambda x: x != 2)(3)
        True
        """
        return f__(op.ne, o)

    def __and__(self, o):
        """
        >>> (_ & 2)(3) == (lambda x: x & 2)(3)
        True
        """
        return f__(op.and_, o)

    def __rand__(self, o):
        """
        >>> (2 & _)(3) == (lambda x: 2 & x)(3)
        True
        """
        return f_(op.and_, o)

    def __or__(self, o):
        """
        >>> (_ | 2)(3) == (lambda x: x | 2)(3)
        True
        """
        return f__(op.or_, o)

    def __ror__(self, o):
        """
        >>> (2 | _)(3) == (lambda x: 2 | x)(3)
        True
        """
        return f_(op.or_, o)

    def __xor__(self, o):
        """
        >>> (_ ^ 2)(3) == (lambda x: x ^ 2)(3)
        True
        """
        return f__(op.xor, o)

    def __rxor__(self, o):
        """
        >>> (2 ^ _)(3) == (lambda x: 2 ^ x)(3)
        True
        """
        return f_(op.xor, o)

    def __lshift__(self, o):
        """
        >>> (_ << 2)(3) == (lambda x: x << 2)(3)
        True
        """
        return f__(op.lshift, o)

    def __rlshift__(self, o):
        """
        >>> (2 << _)(3) == (lambda x: 2 << x)(3)
        True
        """
        return f_(op.lshift, o)

    def __rshift__(self, o):
        """
        >>> (_ >> 2)(3) == (lambda x: x >> 2)(3)
        True
        """
        return f__(op.rshift, o)

    def __rrshift__(self, o):
        """
        >>> (2 >> _)(3) == (lambda x: 2 >> x)(3)
        True
        """
        return f_(op.rshift, o)

    def __getitem__(self, o):
        """
        >>> d = dict(one=1, two=2)
        >>> (_[_])(d)("two") == curry(lambda a, b: a[b])(d)("two")
        True
        >>> (_["one"])(d) == (lambda x: x["one"])(d)
        True

        >>> r = range(5)
        >>> (_[_])(r)(3) == curry(lambda a, b: a[b])(r)(3)
        True
        >>> (_[3])(r) == (lambda x: x[3])(r)
        True
        """
        if type(o) is xlambda:
            return f_(op.getitem)
        return f__(op.getitem, o)

    def __getattr__(self, o):
        """
        >>> o = type('', (), {"one": 1, "two": 2})()
        >>> (_._)(o)("two") == curry(lambda a, b: getattr(a, b))(o)("two")
        True
        >>> (_.one)(o) == (lambda x: x.one)(o)
        True
        """
        if o == "_":
            return fx(lambda a, b: getattr(a, b))
        return fx(op.attrgetter(o))

    def __call__(self, *args, **kwargs):
        """
        >>> _(1, 2)(op.sub)
        -1
        >>> _(_)(foldl)(op.add)(0)(range(5))
        10
        >>> _(7 * _)(mapl)(range(1, 10))
        [7, 14, 21, 28, 35, 42, 49, 56, 63]
        """
        if len(args) == 1 and not kwargs and type(args[0]) is xlambda:
            return f_(apply)
        else:
            return fx(lambda f: f(*args, **kwargs))

    def __repr__(self):
        return "xlambda()"


# -------------------------------
# aliases for convenience
# -------------------------------
uniq = nub
zipwith = mapl
sort = fx(sorted)
length = fx(len)
abs = fx(abs)
sum = fx(sum)
min = fx(min)
max = fx(max)
ord = fx(ord)
chr = fx(chr)
all = fx(all)
any = fx(any)
divmod = fx(divmod)
takewhile = fx(takewhile)
dropwhile = fx(dropwhile)
lazy = f_
c_ = curry
u_ = uncurry
_ = xlambda()

sys.setrecursionlimit(5000)
