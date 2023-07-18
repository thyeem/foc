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


def fn(a, b, c, d):
    return f"{a}-{b}-{c}-{d}"


def test_safe():
    @safe
    def f():
        raise Exception

    assert f() is None


def test_not_():
    assert not_(False)
    assert not (not_(True))


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
    assert take(3, range(5, 10)) == [5, 6, 7]
    assert take(0, range(5, 10)) == []


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
    assert replicate(3, 5) == [5, 5, 5]


def test_product():
    assert product([1, 2, 3, 4, 5]) == 120


def test_flip():
    assert fn(4, 3, 2, 1) == flip(fn)(1, 2, 3, 4)


def test_ff_():
    assert f_("+", 5)(2) == 7
    assert ff_("+", 5)(2) == 7
    assert f_("-", 5)(2) == 3
    assert ff_("-", 5)(2) == -3
    assert f_(fn, 1, 2)(3, 4) == "1-2-3-4"
    assert f_(fn, 1, 2, 3)(4) == "1-2-3-4"
    assert ff_(fn, 1, 2)(3, 4) == "3-4-1-2"
    assert ff_(fn, 1, 2, sgra=True)(3, 4) == "4-3-2-1"


def test_curry():
    assert c_("+")(5)(2) == 7
    assert c_("+")(2)(5) == 7
    assert c_("-")(5)(2) == 3
    assert c_("-")(2)(5) == -3
    assert c_("-")(5)(2) == cc_("-")(2)(5)
    assert c_(fn)(1)(2)(3)(4) == "1-2-3-4"
    assert c_(fn)(4)(3)(2)(1) == "4-3-2-1"
    assert c_(fn)(4)(3)(2)(1) == cc_(fn)(1)(2)(3)(4)


def test_cf_():
    assert cf_(f_("*", 7), f_("+", 5), f_("**", 3))(2) == 98


def test_cfd():
    @cfd(f_("*", 7), f_("+", 5), ff_("**", 3))
    def f(x, y):
        return x**2 + y**2

    assert f(3, 4) == 109410


def test_mapl():
    fn = f_("*", 8)
    assert mapl(fn, range(1, 6)) == [8, 16, 24, 32, 40]
    assert list(map(fn, range(1, 6))) == mapl(fn, range(1, 6))


def test_ml():
    fn = f_("*", 8)
    assert ml_(fn)(range(1, 6)) == [8, 16, 24, 32, 40]
    assert list(m_(fn)(range(1, 6))) == ml_(fn)(range(1, 6))


def test_mml():
    fn = f_("*", 8)
    assert mml_(range(1, 6))(fn) == [8, 16, 24, 32, 40]
    assert list(mm_(range(1, 6))(fn)) == mml_(range(1, 6))(fn)


def test_filterl():
    assert filterl(even, range(10)) == [0, 2, 4, 6, 8]
    assert list(filter(even, range(10))) == filterl(even, range(10))


def test_vl():
    assert vl_(even)(range(10)) == [0, 2, 4, 6, 8]
    assert list(v_(even)(range(10))) == vl_(even)(range(10))


def test_vvl():
    assert vvl_(range(10))(even) == [0, 2, 4, 6, 8]
    assert list(vv_(range(10))(even)) == vvl_(range(10))(even)


def test_reverse():
    assert reverse((1, 2, 3, 4)) == [4, 3, 2, 1]
    assert reverse(range(1, 5)) == [4, 3, 2, 1]
    assert reverse("sofia") == ["a", "i", "f", "o", "s"]
    assert reverse([]) == []


def test_takewhilel():
    assert takewhilel(even, [2, 4, 6, 1, 3, 5]) == [2, 4, 6]


def test_dropwhilel():
    assert dropwhilel(even, [2, 4, 6, 1, 3, 5]) == [1, 3, 5]


def test_bimap():
    assert bimap(f_("+", 3), f_("*", 7), (5, 7)) == (8, 49)


def test_first():
    assert first(f_("+", 3), (5, 7)) == (8, 7)


def test_second():
    assert second(f_("*", 7), (5, 7)) == (5, 49)


def test_iterate():
    assert take(5, iterate(lambda x: x**2, 2)) == [2, 4, 16, 256, 65536]


def test_foldl():
    assert foldl("-", 10, range(1, 5)) == 0


def test_foldr():
    assert foldr("-", 10, range(1, 5)) == 8


def test_foldl1():
    assert foldl1("-", range(1, 5)) == -8


def test_foldr1():
    assert foldr1("-", range(1, 5)) == -2


def test_scanl():
    assert scanl("-", 10, range(1, 5)) == [10, 9, 7, 4, 0]


def test_scanr():
    assert scanr("-", 10, range(1, 5)) == [8, -7, 9, -6, 10]


def test_scanl1():
    assert scanl1("-", range(1, 5)) == [1, -1, -4, -8]


def test_scanr1():
    assert scanr1("-", range(1, 5)) == [-2, 3, -1, 4]


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


def test_lazy():
    assert not lazy(pow, 2, 10) == 1024


def test_force():
    assert force(lazy(pow, 2, 10)) == 1024


def test_flatl(d):
    assert flatl(d) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def test_flatt(d):
    assert flatt(d) == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)


def test_flatd(d):
    assert flatd(d) == deque([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])


def test_flats(d):
    assert flats(d) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}


def test_capture():
    assert capture(r"\d+", "2023Year-05Month-24Day") == "2023"


def test_captures():
    assert captures(r"\d+", "2023Year-05Month-24Day") == ["2023", "05", "24"]


def test_split_at():
    assert list(split_at((3, 5), range(1, 11))) == [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]


def test_chunks_of():
    assert list(chunks_of(3, range(1, 11))) == [
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 9),
        (10, None, None),
    ]


def test_bytes_to_int():
    assert bytes_to_int(b"francis") == 28836210413889907


def test_int_to_bytes():
    assert int_to_bytes(28836210413889907) == b"francis"


def test_fn_args():
    assert fn_args(fn) == ["a", "b", "c", "d"]
