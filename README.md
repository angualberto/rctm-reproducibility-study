# Independent Reproducibility Study of RCTM-based CSPRNG

This repository aims to promote independent reproducibility and transparent evaluation of chaotic PRNGs, focusing on the Robust Chaotic Tent Map (RCTM) as a candidate CSPRNG.

## Objective

To verify reproducibility and evaluate statistical robustness under extended test batteries.

## Implementation Details

- Language: C (GCC 13)
- Precision: IEEE-754 double precision
- OS: Linux x86_64
- Tests:
  - Dieharder (-a)
  - TestU01 SmallCrush
  - ENT

## Reproducibility

To compile:

```
gcc -O3 src/rctm_uint32.c -lm -o build/rctm
```

To generate sequence:

```
./build/rctm 0.23 61.81 10000000 data/rctm_uint32.bin
```

To run TestU01:

```
./build/testu01_wrapper
```

Full logs are available in the results folder.

## Transparency

All p-values, failed tests, and full logs are included for maximum transparency and scientific credibility.

## Citation

If you use this repository, please cite the Zenodo DOI: `doi:10.5281/zenodo.xxxxxxx`

---

## Reproducibility Statement

This repository contains all code, data, and logs required to independently reproduce the main results and statistical findings of the RCTM-based CSPRNG evaluation. No code or data was omitted. All parameters and algorithms are faithful to the original publication.

---

## Contact

For questions or collaboration, open an issue or contact the maintainer.
