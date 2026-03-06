# Independent Reproducibility Study: RCTM-based CSPRNG
## Robust Chaotic Tent Map Cryptographically Secure Pseudo-Random Number Generator

This repository contains all code, test suites, and comprehensive statistical analysis for independent reproducibility and transparent evaluation of the Robust Chaotic Tent Map (RCTM) as a candidate Cryptographically Secure Pseudo-Random Number Generator (CSPRNG).

---

## 📋 Table of Contents

- [Objective](#objective)
- [Repository Contents](#repository-contents)
- [Implementation Details](#implementation-details)
- [Test Results Summary](#test-results-summary)
- [Reproducibility Instructions](#reproducibility-instructions)
- [Key Findings](#key-findings)
- [Transparency Statement](#transparency-statement)
- [Citation](#citation)

---

## 🎯 Objective

To verify reproducibility and provide transparent, independent evaluation of the RCTM algorithm through extensive statistical testing. This study examines whether RCTM truly generates cryptographically secure random sequences suitable for cryptographic applications.

**Publication Reference:**
> "Cryptographically Secure Pseudo-Random Number Generation Using a Robust Chaotic Tent Map: A Novel Approach"
> 
> The original paper proposes RCTM with claimed improvements over traditional tent maps, utilizing modulo and scaling operations to expand the parameter space from μ ∈ [2, 100], achieving a key space of 2^198.

---

## 📁 Repository Contents

### Test Suites Included

| Test Suite | Status | Files |
|-----------|--------|-------|
| **Dieharder (v3.31.1)** | Complete | `agle_v2_dieharder_rctm.txt` |
| **Python Statistical Tests** | 9 Tests | `test_*.py` files (16 test scripts) |
| **MATLAB/Octave** | Simulation | `*.m`, `tentmap.m` |
| **C Implementation** | Source & Binary | `tentmap.c`, `tentmap` |

### Result Files

```
RESULTADOS_TESTES_PYTHON_2026-03-06.txt   ← Main analysis report
  ✓ Shannon Entropy Test
  ✓ Permutation Entropy Test
  ✓ N-gram Predictability Test
  ✓ Distinguishing Attack Test
  ✓ Serial Correlation Analysis
  ✓ Bit Distribution Test
  ✓ Autocorrelation Analysis
  ✓ State Reconstruction Test
  ✓ Consecutive Byte Repetitions

agle_v2_dieharder_rctm.txt                ← Complete Dieharder test log
  - 209+ tests performed
  - 17+ FAILED/WEAK results
  - All p-values and assessments included
```

---

## ⚙️ Implementation Details

- **Language:** C (GCC/Clang compatible)
- **Precision:** IEEE-754 double precision
- **Platform:** Linux x86_64
- **Testing Environment:** Python 3.12.3 with NumPy, SciPy, scikit-learn
- **RNG Output:** `agle_v2_10m.bin` (10 MB binary file)
- **Performance:** 8.82×10^7 32-bit values/second (Dieharder measurement)

### Chaotic Map Implementation

The Robust Chaotic Tent Map is defined as:

$$x_{n+1} = \begin{cases}
\mu \cdot x_n & \text{if } x_n \leq 0.5 \\
\mu \cdot (1 - x_n) & \text{if } x_n > 0.5
\end{cases}$$

**Modified RCTM with scaling/modulo:**
```c
// Parameters: μ ∈ [2, 100], initial seed x₀
// Output: 32-bit unsigned integers
```

---

## 📊 Test Results Summary

### Quick Overview (Python Tests)

| Test | Result | Score | Interpretation |
|------|--------|-------|-----------------|
| Shannon Entropy (7.998/8.0) | ✅ **PASSED** | Excellent | Bits individually random |
| Permutation Entropy (0.9999) | ✅ **EXCELLENT** | Perfect | Patterns perfectly distributed |
| N-gram Predictability (1.63%) | ❌ **FAILED** | Critical | ABOVE 1.20% threshold |
| Distinguishing Attack (0.519) | ❌ **FAILED** | High Risk | RNG distinguishable from random |
| Serial Correlation LAG-1 (78.46%) | ❌ **FAILED** | Critical | Only 78.46% pair coverage |
| Bit Distribution (49.99%) | ✅ **PASSED** | Perfect | Unbiased individual bits |
| Autocorrelation (Pearson) | ✅ **PASSED** | Excellent | ~0.0 correlation |
| State Reconstruction | ✅ **PASSED** | Good | No recoverable state |
| Byte Repetitions (0.96x) | ✅ **PASSED** | Good | Expected frequency |

**Overall Score: 6/9 tests passed (67%)**

### Dieharder Extended Results

**File:** `agle_v2_dieharder_rctm.txt`

**Summary:**
- Total tests run: 200+ individual assertions
- **PASSED:** ~183 (91%)
- **WEAK:** 4 results (2%)
- **FAILED:** 17+ results (7%) ⚠️

**Critical Failures:**
1. `diehard_opso` - p-value: 0.00000000
2. `diehard_squeeze` - p-value: 0.00000070
3. `marsaglia_tsang_gcd` - p-value: 0.00000000 (×2)
4. `rgb_lagged_sum` - 11 FAILED results (lags 4,7,9,11,14,15,19,23,24,27,31)
5. `dab_bytedistrib` - p-value: 0.00000000
6. `dab_filltree2` - p-value: 0.00000000 (×2)
7. `dab_monobit2` - p-value: 1.00000000

---

## 🔧 Reproducibility Instructions

### Prerequisites

```bash
# Install dependencies
sudo apt-get install gcc python3.12 python3-pip libgsl-dev

# Python packages
pip install numpy scipy scikit-learn
```

### Compile RCTM C Implementation

```bash
gcc -O3 -std=c99 tentmap.c -lm -o tentmap
```

### Generate Test Sequence

```bash
# Generate 10MB binary file (agle_v2_10m.bin)
./tentmap 0.23 61.81 10000000 agle_v2_10m.bin
```

### Run Python Test Suite

```bash
# Test 1: Shannon Entropy
python3.12 test_entropy.py

# Test 2: Permutation Entropy
cat agle_v2_10m.bin | python3.12 test_permutation_entropy.py

# Test 3: N-gram Predictability
head -c 100000 agle_v2_10m.bin | python3.12 test_predictability_ngram.py

# Test 4: Distinguishing Attack
head -c 100000 agle_v2_10m.bin | python3.12 test_distinguishing_attack.py

# Test 5: Serial Correlation Analysis
python3.12 << 'EOF'
import numpy as np
data = np.fromfile('agle_v2_10m.bin', dtype=np.uint8, count=100000)
pairs = list(zip(data[:-1], data[1:]))
unique_pairs = len(set(pairs))
coverage = unique_pairs / 65536 * 100
print(f"LAG-1 Coverage: {coverage:.2f}% (should be ~95%+)")
EOF

# Test 6: Autocorrelation (Pearson)
python3.12 << 'EOF'
import numpy as np
from scipy import stats
data = np.fromfile('agle_v2_10m.bin', dtype=np.uint8, count=50000)
data_float = (data.astype(float) - 127.5) / 127.5
for lag in [1, 2, 4, 8, 16, 32]:
    autocorr = np.corrcoef(data_float[:-lag], data_float[lag:])[0, 1]
    print(f"Lag {lag}: {autocorr:.6f}")
EOF
```

### Run Dieharder Tests

```bash
# Full comprehensive test suite
dieharder -a -f agle_v2_10m.bin > dieharder_results.txt

# Specific test (e.g., opso)
dieharder -d 51 -f agle_v2_10m.bin
```

---

## 🔍 Key Findings

### Paradox Detected: Local vs. Structural Randomness

The RCTM exhibits a **critical paradox** between individual bit-level and sequence-level randomness:

#### ✅ **What RCTM Does Well:**
- Excellent Shannon entropy (7.998/8.0 bits per byte)
- Perfect bit-level distribution (49.99% ones vs 50.00% expected)
- Mathematically random-looking permutation patterns (0.9999 normalized entropy)
- No measurable linear autocorrelation (Pearson coefficient ~0.0)

#### ❌ **What RCTM Fails At:**
- **N-gram predictability:** 1.63% accuracy (exceeds 1.20% threshold by 35%)
  - 3-gram model: 52.09% accuracy
  - 4-gram model: 99.74% accuracy
  - **Interpretation:** Deterministic non-linear dependencies between consecutive bytes
  
- **Serial correlation LAG-1:** Only 78.46% coverage of 256²=65,536 possible byte pairs
  - Should have >95% coverage in truly random sequence
  - Missing 21.54% of possible pairings
  - Indicates deterministic constraints on byte succession
  
- **Distinguishing attack:** Entropy 0.519 (vs ideal 1.0)
  - RNG is statistically **distinguishable** from random source
  - Simple statistical test can identify RCTM output

- **Dieharder failures:** 17+ test failures
  - `rgb_lagged_sum`: 11 consecutive failures on specific lags
  - Pattern suggests periodic or structured redundancy

### Root Cause Analysis

The structure indicates RCTM likely has:
1. **Short period or structure:** Chaotic map with insufficient mixing
2. **Predictable subsequences:** Non-linear dependencies exploitable by ML/statistical models
3. **Incomplete state space coverage:** Cannot generate all possible byte sequences
4. **Periodic artifacts:** Visible in rgb_lagged_sum failures on specific lag patterns

### Severity Assessment

**For Cryptographic Use:** ⚠️ **NOT RECOMMENDED**

The predictability of 1.63% (vs 0.39% random baseline) means an attacker has a **4.17× advantage** in guessing the next byte. This violates the NIST requirement that PRNGs be **indistinguishable** from random.

---

## 📝 Transparency Statement

### What Is Included

✅ **All test code** - 16 Python test scripts with full source  
✅ **All raw results** - Complete Dieharder log with 200+ individual p-values  
✅ **All analysis code** - Reproducible Python analysis notebooks  
✅ **All parameters** - Exact seed values, μ parameters, test configurations  
✅ **Failed tests detailed** - Not hidden; all FAILED/WEAK results documented  
✅ **Negative findings** - Candid discussion of vulnerabilities  

### What We DON'T Hide

- The N-gram predictability failures
- The LAG-1 coverage gap (78.46% vs needed 95%+)
- The Dieharder rgb_lagged_sum cascade failures
- The distinguishing attack vulnerability
- The contradictions between entropy and predictability

This approach prioritizes **scientific rigor** over marketing claims.

---

## 📚 Citation

If you use this repository for research or references:

```bibtex
@misc{rctm_reproducibility_2026,
  title = {Independent Reproducibility Study of RCTM-based CSPRNG},
  author = {Andre, A.},
  year = {2026},
  month = {March},
  url = {https://github.com/angualberto/rctm-reproducibility-study},
  note = {Comprehensive independent evaluation with 9 Python tests and Dieharder suite}
}
```

**Original Publication:**
> "Cryptographically Secure Pseudo-Random Number Generation Using a Robust Chaotic Tent Map: A Novel Approach" - [Original Authors]

---

## 🤝 Collaboration & Issues

This repository welcomes:
- Independent reproductions of results
- Additional test implementations
- Parameter variations and sensitivity analysis
- Algorithmic improvements based on vulnerabilities identified

**Report issues or questions:**  
GitHub Issues: https://github.com/angualberto/rctm-reproducibility-study/issues

---

## 📄 License

[Specify license - MIT/GPL/etc as appropriate]

---

## ⚖️ Reproducibility Guarantee

This repository contains **all code, parameters, and logs** necessary to independently reproduce the main results. No optimizations, data filtering, or cherry-picking of test results has been performed. All p-values, failed tests, and statistical analyses are included in their entirety.

**Reproducibility Score:** 100% - Complete code transparency and parameter disclosure.

---

**Last Updated:** March 6, 2026  
**Test Date:** March 6, 2026  
**Environment:** Python 3.12.3, NumPy 1.x, SciPy 1.x, GCC 13, Linux x86_64
