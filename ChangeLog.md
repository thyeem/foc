# 0.1.1
- First public release

# 0.1.2
- Modified folding functions: `foldl`, `foldr`, `foldl1`, `foldr1`, `scanl`, `scanr`, `scanl1`, and `scanr1`
- Added pytest

# 0.1.3
- Added symbolic operators (the same notations as the python standard library)
- Fix neatly formatter
- Added a few of basic functions

# 0.1.4
- Added `intersperse` and `intercalate`
- Some modifications on functions related to path

# 0.1.5
- Added `grep`
- Fixed `ls` and `split_at`

# 0.1.6
- Remove the unnecessary caching (possible memory leak)
- Added `shuffle`
- Fix `random_int`

# 0.1.7
- Added `const`, `until`, and `apply`

# 0.1.8
- Added `choice`

# 0.1.9
- Fixed `fread`. Keep whitespaces at the end of lines.
- Fixed `chunks_of`. Filling values is optional.
- Fixed `lines`. No splits on trailing newlines.

# 0.1.10
- Removed unnecessary `lazy` and `tup`
- Fixed `shuffle` and `choice` not to waste memory
- Added `ilen`
- Fixed `apply`, `_in`, and `neatly`
- Renamed `_not`, `_and`, `_or` and `_in`

# 0.1.11
- Hotfix

# 0.1.12
- Fix `ls` by adding recursive-search and regex filter
- Added `reads` and `writes`
- Removed `fread` and `fwrite`

# 0.2.0
- Added `uncurry`
- Simplified lazy operation: `lazy` and `force`.
- Fix `repeat` and `replicate` and they support callable object
- Imporved: `random_int`, `ls`, and `apply`

# 0.2.1
- Fix `ff_`
- Added `rand`, `randn`, renamed `randint`
- Added a few of symbolic operators
- Added functional contructors: `_t`, `_l`, `_s` and `_r`

# 0.2.2
- Fixed `choice`: support sampling from probability lists

# 0.2.3
- Fixed `reads` and `writes` -> `reader` and `writer` (renamed)
- Added `chars` and `unchars`
- Added `_d`, functional form of deque constructor
- Added `guard`, assertion as an expression

# 0.2.3.1
- Hotfix `reader`

# 0.2.4
- Added `seq`, `void`, and `guard_`
- Fixed `ls` and `bimap`

# 0.2.5
- Imporved `ls` and `grep`
- Added `nub`

# 0.2.6
- Imporved `dmap` and `nprint`
- Added `pbcopy` and `pbpaste`

# 0.2.7
- Refactored to explicit form of functions
- Fix `neatly` and `nprint`

# 0.2.8
- Added `timer`

# 0.2.9
- Added `taskbar` and `docfrom`
- Fixed `neatly`

# 0.2.10
- Fixed `taskbar`: support generators and fix bugs

# 0.2.11
- Leap version

# 0.2.12
- Fix `ls`

# 0.2.13
- Add `thread` and `shell`
- Fix `sym` and `polling`
- Add doctest

# 0.3.0
- Leap version

# 0.3.1
- A huge update: introduced `piper` and removed all the unnecessary
- Add `zipwith`, `proc`, `trap` and other utilities

# 0.4.0
- Replace `piper` with `composable`: composition of functions by symbol
- Fix: Quick start
- Add more documentation on functions

# 0.4.1
- Fix `neatly` and `nprint`: bugs in the defaults

# 0.4.2
- Add `flatten` and `collect`
- Remove `unpack`

# 0.4.3
- Fix `shuffle` (index bug fix)

# 0.4.4
- Fix `flip` 

# 0.4.5
- Fix `choice` and `chunks_of`

# 0.4.6
- Add `chunks_from`
- Fix `neatly` and `nprint` (add sorting flags)

# 0.4.7
- Fix `ls`: don't list hidden files(dotfile) by default. add '-a' flag.

# 0.4.8
- Add `parmap`

# 0.5.0
- A huge update across the project
- **Separated lots of utility functions into `ouch` project.**
- Introduced place holder lambda expression.
- Add `g_`, object method partial application builder

# 0.5.2
- Add `rg`, _Haskell-like_ range notation 
- Add `unfx`, `_(_)` operator
- Fix `rev`, `uncurry`, `repeat`, and `replicate`
- Remove `xargs`, `reverse`, `unzip` and `unzipl`
- Remove `composable`!

# 0.5.3
- Add `tap` (`IO` action)
- Remove `seq` (unnecessary in `python`)
- Rename `rg` to `seq` (flexible sequence generation)
- Bug fixes

# 0.5.4
- Revert `take` to a lazy function form.
- Add `takel`, `dropl`, and `replicatel`

# 0.6.0
- Return to function composition using `.` _dot notation_.
- Remove redundant `cfd_`, `cfd__` and `pair`.
- Add `on` and `pack`.
- Add `__signature__` to make the function easier to trace.

# 0.6.1
- Rename `g_` to `ob` for ease of use.

# 0.6.2
- Add `cons` and `snoc`
- Fix `xlambda.__getitem__`: getting multiple items instead of giving up tuple key.
