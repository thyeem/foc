import operator as op

import pytest
from foc import *


@pytest.fixture
def d():
    return [
        1,
        2,
        [3, 4, [[[5], 6], 7, {8}, ((9), 10)], range(11, 13)],
        (x for x in [13, 14, 15]),
    ]


def fn(a, b):
    return f"{a}-{b}"


def test_safe():
    @safe
    def f():
        raise Exception

    assert f() is None


def test_id():
    assert id("francis") == "francis"
    assert id([1, 2]) == [1, 2]


def test_fst():
    assert fst(["sofia", "maria", "claire"]) == "sofia"


def test_snd():
    assert snd(["sofia", "maria", "claire"]) == "maria"


def test_nth():
    assert nth(3, ["sofia", "maria", "claire"]) == "claire"


def test_take():
    assert take(3, range(5, 10)) | collect == [5, 6, 7]
    assert take(0, range(5, 10)) | collect == []


def test_drop():
    assert list(drop(3, "github")) == ["h", "u", "b"]
    assert list(drop(3, [])) == []


def test_head():
    assert head(range(1, 5)) == 1
    assert head("maria") == "m"
    assert head([]) is None


def test_tail():
    assert list(tail(range(1, 5))) == [2, 3, 4]
    assert list(tail("maria")) == ["a", "r", "i", "a"]


def test_last():
    assert last(range(1, 5)) == 4
    assert last("maria") == "a"
    assert last([]) is None


def test_init():
    assert list(init(range(1, 5))) == [1, 2, 3]
    assert list(init("maria")) == ["m", "a", "r", "i"]


def test_pred():
    assert pred(3) == 2


def test_succ():
    assert succ(3) == 4


def test_odd():
    assert odd(3)
    assert not odd(4)


def test_even():
    assert not even(3)
    assert even(4)


def test_null():
    assert null([])
    assert null(())
    assert null({})
    assert null("")
    assert null(deque([]))
    assert null(range(3, 3))
    assert not null(range(1, 5))


def test_elem():
    assert elem("f", "francis")
    assert elem(5, range(10))
    assert elem("sofia", dict(sofia="painter", maria="ballerina"))


def test_words():
    assert words("fun on functions") == ["fun", "on", "functions"]


def test_unwords():
    assert unwords(["fun", "on", "functions"]) == "fun on functions"


def test_lines():
    assert lines("fun\non\nfunctions") == ["fun", "on", "functions"]


def test_unlines():
    assert unlines(["fun", "on", "functions"]) == "fun\non\nfunctions"


def test_replicate():
    assert replicate(3, 5) | collect == [5, 5, 5]


def test_product():
    assert product([1, 2, 3, 4, 5]) == 120


def test_flip():
    assert fn(2, 1) == flip(fn)(1, 2)


def test_f_f__():
    assert f_(op.add, 5)(2) == 7
    assert f__(op.add, 5)(2) == 7
    assert f_(op.sub, 5)(2) == 3
    assert f__(op.sub, 5)(2) == -3
    assert f_(fn, 1)(2) == "1-2"
    assert f_(fn)(1, 2) == "1-2"
    assert f__(fn, 1)(2) == "2-1"
    assert f__(fn)(1, 2) == "2-1"


def test_curry():
    assert c_(op.add)(5)(2) == 7
    assert c_(op.add)(2)(5) == 7
    assert c_(op.sub)(5)(2) == 3
    assert c_(op.sub)(2)(5) == -3
    assert c_(op.sub)(5)(2) == c__(op.sub)(2)(5)
    assert c_(fn)(1)(2) == "1-2"
    assert c__(fn)(1)(2) == "2-1"
    assert c_(fn)(1)(2) == c__(fn)(2)(1)


def test_cf_():
    assert cf_(_ * 7, _ + 5, 3**_)(2) == 98


def test_mapl():
    fn = _ * 8
    assert mapl(fn, range(1, 6)) == [8, 16, 24, 32, 40]
    assert list(map(fn, range(1, 6))) == mapl(fn, range(1, 6))


def test_filterl():
    assert filterl(even, range(10)) == [0, 2, 4, 6, 8]
    assert list(filter(even, range(10))) == filterl(even, range(10))


def test_rev():
    assert rev((1, 2, 3, 4)) == [4, 3, 2, 1]
    assert rev(range(1, 5)) == [4, 3, 2, 1]
    assert rev("sofia") == "aifos"
    assert rev([]) == []


def test_takewhilel():
    assert takewhilel(even, [2, 4, 6, 1, 3, 5]) == [2, 4, 6]


def test_dropwhilel():
    assert dropwhilel(even, [2, 4, 6, 1, 3, 5]) == [1, 3, 5]


def test_bimap():
    assert bimap(_ + 3, _ * 7, (5, 7)) == (8, 49)


def test_first():
    assert first(_ + 3, (5, 7)) == (8, 7)


def test_second():
    assert second(_ * 7, (5, 7)) == (5, 49)


def test_until():
    assert until(100 < _, _ * 2, 2) == 128


def test_iterate():
    assert take(5, iterate(_**2, 2)) | collect == [2, 4, 16, 256, 65536]


def test_apply():
    assert apply(fn, 1, 2) == "1-2"


def test_foldl():
    assert foldl(op.sub, 10, range(1, 5)) == 0


def test_foldr():
    assert foldr(op.sub, 10, range(1, 5)) == 8


def test_foldl1():
    assert foldl1(op.sub, range(1, 5)) == -8


def test_foldr1():
    assert foldr1(op.sub, range(1, 5)) == -2


def test_scanl():
    assert scanl(op.sub, 10, range(1, 5)) == [10, 9, 7, 4, 0]


def test_scanr():
    assert scanr(op.sub, 10, range(1, 5)) == [8, -7, 9, -6, 10]


def test_scanl1():
    assert scanl1(op.sub, range(1, 5)) == [1, -1, -4, -8]


def test_scanr1():
    assert scanr1(op.sub, range(1, 5)) == [-2, 3, -1, 4]


def test_concatl():
    assert concatl(["sofia", "maria"]) == [
        "s",
        "o",
        "f",
        "i",
        "a",
        "m",
        "a",
        "r",
        "i",
        "a",
    ]


def test_concatmapl():
    assert concatmapl(str.upper, ["sofia", "maria"]) == [
        "S",
        "O",
        "F",
        "I",
        "A",
        "M",
        "A",
        "R",
        "I",
        "A",
    ]


def test_cartprodl():
    assert cartprodl([1, 2], [3, 4], [5, 6]) == [
        (1, 3, 5),
        (1, 3, 6),
        (1, 4, 5),
        (1, 4, 6),
        (2, 3, 5),
        (2, 3, 6),
        (2, 4, 5),
        (2, 4, 6),
    ]


def test_permutations():
    assert list(permutation([1, 2, 3], 2)) == [
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 3),
        (3, 1),
        (3, 2),
    ]

    assert list(permutation([1, 2, 3], 2, rep=True)) == [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
    ]


def test_combinations():
    assert list(combination([1, 2, 3], 2)) == [
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    assert list(combination([1, 2, 3], 2, rep=True)) == [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 2),
        (2, 3),
        (3, 3),
    ]


def test_intersperse():
    assert (
        "".join(
            intersperse("-", words("sofia maria claire")),
        )
        == "sofia-maria-claire"
    )


def test_intercalate():
    assert intercalate(
        [-55],
        [[1, 2], [3, 4], [5, 6]],
    ) == [1, 2, -55, 3, 4, -55, 5, 6]


def test_not_():
    assert not_(False)
    assert not (not_(True))


def test_and_():
    assert and_(1, True)
    assert not (and_("foc", False))
    assert not (and_([], False))


def test_or_():
    assert or_(1, True)
    assert or_("foc", False)
    assert not (or_([], False))


def test_in_():
    assert in_("s", "sofia")
    assert in_(3, range(5))
    assert not in_("qux", "maria")


def test_is_():
    assert is_("sofia", "sofia")
    assert is_((0, 1, 2), (0, 1, 2))
    assert not is_([0, 1, 2], rangel(3))


def test_isnt():
    assert not isnt_("sofia", "sofia")
    assert not isnt_((0, 1, 2), (0, 1, 2))
    assert isnt_([0, 1, 2], rangel(3))


def test_lazy():
    assert not lazy(pow, 2, 10) == 1024


def test_force():
    assert force(lazy(pow, 2, 10)) == 1024
