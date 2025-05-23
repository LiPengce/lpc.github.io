# HyperPianist
Optimized, fully distributed version of [HyperPlonk](https://github.com/EspressoSystems/hyperplonk).

## Building
Requires Rust Nightly. For optimal performance, please build as `--release`, with
```
RUSTFLAGS='-C target-cpu=native -C target-feature=+bmi2,+adx'
```

## Tests
Most of our code has distributed and non-distributed versions. There are a number of non-distributed tests that can be run as usual. For distributed tests (in the `dTests` folder of each project), you may run them locally with the `run.sh` script. (This will spawn 4 sub-provers on the same machine.) Example (in `subroutines`):
```
./dTests/run.sh dSumcheck
```

## Benchmarks
The main benchmark is `hyperpianist/dTests/bench.rs`.
You need to have a list of IPs of sub-provers, on each sub-prover.
Each sub-prover should have the same list. Please make sure to list LAN IPs in the files.
Then, using a script or whatever else, arrange for all sub-provers to run, simultaneously
```
<executable> <path to IP list> <index of self> <nv> [--dory] [--jellyfish]
```
`<index of self>` must correspond to the position of the sub-prover in the IP list. `<nv>` is number of variables to test (e.g. 22 corresponds to a circuit of `2^22` constraints).
Flags `--dory` and `--jellyfish` change the PCS and gate type, respectively. (When not provided, defaults to Mkzg, and vanilla gates.)

