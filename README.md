--------------------------------------------------------------------------------
# GStamps: C++ routines for the Local & the Global Postage Stamp Problem
--------------------------------------------------------------------------------

The postage stamp problem states that an envelope may be franked with
a total of at most s stamps while one has available an (additive)
basis of k stamp denominations (integers: a1 < a2 < ... < ak).

- The local postage stamp problem (LPSP) is the determination of the
smallest integer not attainable by a given basis with at most s stamps
- The global postage stamp problem (GPSP) is the determination of a
basis with the largest LPSP for given parameters k and s.

**Authors**: 
Léo Colisson Palais,
Jean-Guillaume Dumas,
Alexis Galan,
Bruno Grenet,
Aude Maignan,
Daniel S. Roche
[ Algorithms for the local and the global postage stamp problem. https://hal.science/hal-05479676](https://hal.science/hal-05479676)

**Requirements**:
- C++
- Optional: [Givaro](https://github.com/linbox-team/givaro), dev: headers & library (version ≥ 4.2.0)


**Installation**:
- Requires some distribution packages like:
           `sudo apt install git make g++ libboost-dev`
- If arbitrary precision is needed, add:
           `sudo apt install pkg-config libgmp-dev libgivaro-dev`
- Then just run `make`, in order to produce the following executable programs
- See also [`bin/auto-docker.run`](https://github.com/jgdumas/gstamps/blob/main/bin/auto-docker.run)


**Programs**:
|  |  |
| :--------- | :------ |
|`bin/range`| LPSP: Computes the range of a basis with s stamps |
|`bin/basis`| GPSP: Divide & Conquer basis polynomial-time computation of k denominations and s stamps |
|`bin/dynprg`| GPSP: Dynamic programming search of best Divide & Conquer cut |
|  |  |


**Search Tools**:
|  |  |
| :--------- | :------ |
|`bin/search`| Smallest basis reaching N with s stamps |
|`bin/incremental`| Exact exhaustive search with incremental prefix reachability |
|`bin/incremental_parallel`| Exact incremental search with bounded OpenMP prefix splitting |
|`bin/incremental_tbb`| Optional exact incremental search with dynamic oneTBB frontier |
|`bin/complement`| Exhausts all additional denominations in parallel |
|`bin/supplement`| Exhausts additional denominations several values at a time |
|`bin/brute`| Exhaustive search of an extremal basis |
|  |  |

**Benchmarking other basis**:
|  |  |
| :--------- | :------ |
|`bin/fibo`| Fibonacci basis |
|`bin/geom`| Geometric progression basis |
|`bin/alba`| Alter & Barnett basis |
|`bin/bala`| Improved Alter & Barnett with balanced basis |
|  |  |

**Benchmarking other range determination**:
|  |  |
| :--------- | :------ |
|`bin/krange`| Sliding window denomination-range |
|`bin/reach`| Full table denomonation-range |
|`bin/srange`| Mossige stamp-range |
|`bin/depthrange`| counts reached integers per additive depth |
|`bin/decompositions`| shows all additive chains up to range |
|  |  |

**Usage**:
- #k: k denominations in the basis
- #s: the basis is for s stamps
- #v: verbosisty level
- #r: after r recursive levels (rlevel), stops searching for the best cut, just use the midpoint
- #a: if true provides only a lower bound on the range of the basis (approximate), otherwise computes the range exactly

`bin/incremental #k #h` automatically selects the validated exact search
policy for the requested parameters. For h=4 this enables the completion
bound, descending candidates, an approximate lower-bound seed, exact final
target filtering, final-row evaluation, four consecutive target checks, and
the exact two-final-denomination feasibility filter for k>=6. Validated h=4
searches with k>=8 also use bounded OpenMP prefix splitting when more than one
OpenMP thread is available. Set `OMP_NUM_THREADS=1` or use `serial` to force
the lower-overhead serial execution.
The optional numeric arguments remain available for ablation and reproducible
comparisons:

```
bin/incremental #k #h [bound] [descending] [seed] [target-filter]
                     [final-fast] [target-count] [pair-filter]
```

The validated automatic policy can be selected explicitly with `auto`, forced
to serial execution with `serial`, or forced to bounded parallel execution with
`parallel`. Numeric arguments retain the serial manual/ablation interface.

For exact parallel execution, use the separate tool:

```text
bin/incremental_parallel #k #h [threads] [split-depth]
```

The parallel tool uses bounded shallow prefix splitting and seeded exact
subsearches. It exposes explicit thread and split-depth controls; the regular
`bin/incremental` command uses the same runner automatically for validated
large h=4 searches.

If oneTBB development headers and libraries are installed, `make tbb` builds
`bin/incremental_tbb`. It uses a dynamic `parallel_for_each` feeder so active
subtrees can add finer-grained work instead of materializing one fixed
frontier. Its optional arguments are the thread count and maximum split depth:

```text
bin/incremental_tbb #k #h [threads] [split-depth]
```

**Examples**:
- `./bin/basis 4 2`: produces a basis of 4 denominations for 2 stamps (1 3 5 6, attaining all integers 1..12)
- `echo '1 3 5 6' | ./bin/range 2`: the basis can range (all integers up to 12) with 2 stamps
- `./bin/search 4 1024 1`: produces a basis (with 14 denominations), that can range at least all integers up to 1024 with 4 stamps
- `./bin/basis 3 4 | ./bin/decompositions 4 1`: shows the additive chains from that basis with 4 stamps

**Nix support**:

If you have nix installed, you can directly run these programs as follows (you can omit `github:jgdumas/GStamps` if you run `nix shell` from a local clone):
```
$ nix shell github:jgdumas/GStamps
## Now you are in a shell with all above binaries installed, like:
$ basis 3 3
```

If you want to develop this library, you can simply run in a clone of this library:
```
$ nix develop
## Now you are in a shell with all dependencies installed, so you can develop as usual:
$ make
$ ./bin/basis
```

If you have [direnv](https://direnv.net/) and [nix-direnv](https://github.com/nix-community/nix-direnv) installed, you don't even need to specify `nix develop` each time you want to start developing. Just run once `direnv allow .`, and next time you enter this folder it will automatically load the requested binaries.
