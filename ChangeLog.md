# 0.1.1
- First public release

# 0.1.2
- Modified folding functions: `foldl`, `foldr`, `foldl1`, `foldr1`, `scanl`, `scanr`, `scanl1`, and `scanr1`
- Added pytest

# 0.1.3
- Added symbolic operators (the same notations as the python standard library)
- Improved neatly formatter
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
- Improved `random_int`

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
- Improved `ls` by adding recursive-search and regex filter
- Added `reads` and `writes`
- Removed `fread` and `fwrite`

# 0.2.0
- Added `uncurry`
- Simplified lazy operation: `lazy` and `force`.
- Improved `repeat` and `replicate` and they support callable object
- Imporved: `random_int`, `ls`, and `apply`

# 0.2.1
- Improved `ff_`
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
