# Quick Start Guide: RCTM Reproducibility Study

## 📌 What This Repository Contains

A **complete, transparent independent reproducibility study** of the Robust Chaotic Tent Map (RCTM) CSPRNG proposed in the original publication.

## ⚡ 30-Second Summary

| Finding | Result |
|---------|--------|
| **Verdict** | ❌ **NOT suitable for cryptography** |
| **Reason** | N-gram predictability: 1.63% (4.17× above random) |
| **Score** | 6/9 Python tests passed (67%) |
| **Dieharder** | 17+ critical failures out of 200+ tests |
| **Root Cause** | Non-linear dependencies + incomplete pair coverage |

## 🚀 Run Tests (5 minutes)

### 1. Generate Sequence
```bash
gcc -O3 tentmap.c -lm -o tentmap
./tentmap 0.23 61.81 1000000 test.bin
```

### 2. Run Python Tests
```bash
# Install deps
pip install numpy scipy scikit-learn

# Run all tests
cat test.bin | python test_permutation_entropy.py
head -c 100000 test.bin | python test_predictability_ngram.py
head -c 100000 test.bin | python test_distinguishing_attack.py
```

### 3. View Full Results
```
See: RESULTADOS_TESTES_PYTHON_2026-03-06.txt
     RESEARCH_FINDINGS.md
     README.md
```

## 📊 Key Results

### ✅ What RCTM Does Well
- Shannon Entropy: **7.998/8.0 bits** (EXCELLENT)
- Bit distribution: **49.99%** (PERFECT)
- Permutation entropy: **0.9999** (PERFECT)

### ❌ What RCTM Fails At
- **N-gram predictability: 1.63%** (FAILED - threshold 0.60%)
  - 2-gram: 4.17× advantage over random
  - 3-gram: 52.1% accuracy
  - 4-gram: 99.7% accuracy
  
- **LAG-1 coverage: 78.46%** (FAILED - should be >95%)
  - Missing 21.54% of possible byte pairs
  
- **Distinguishing attack: p<0.001** (FAILED)
  - RNG is statistically distinguishable

- **Dieharder: 17+ failures** (FAILED)
  - rgb_lagged_sum: 11 cascade failures
  - diehard_opso, diehard_squeeze, marsaglia_tsang_gcd

## 🔍 The Paradox

RCTM is "**locally random**" but "**globally structured**":

```
Individual bits:     ✅ Random-looking
Byte sequences:      ❌ Predictable patterns
N-gram ML model:     ❌ 99.7% prediction rate with 3 bytes
Dieharder metrics:   ❌ Systematic failures
```

## 💡 Bottom Line

**For Cryptography:** ❌ **DO NOT USE**
- Fails NIST indistinguishability requirement
- Vulnerable to simple statistical attacks
- 1.63% predictability vs 0.39% random baseline

**For Simulation:** ✅ **May be acceptable**
- Good statistical properties at bit level
- Fast computation
- Not suitable as TRNG or CSPRNG

## 📚 Read More

| Document | Purpose |
|----------|---------|
| **README.md** | Complete overview + reproducibility instructions |
| **RESEARCH_FINDINGS.md** | Detailed scientific analysis and root causes |
| **RESULTADOS_TESTES_PYTHON_2026-03-06.txt** | Full test results with statistics |
| **agle_v2_dieharder_rctm.txt** | Complete Dieharder test log (200+ tests) |

## 🔬 How to Verify

**Any researcher can reproduce all findings in <1 hour:**

1. Clone repo: `git clone https://github.com/angualberto/rctm-reproducibility-study.git`
2. Compile: `gcc -O3 tentmap.c -lm -o tentmap`
3. Generate: `./tentmap 0.23 61.81 10000000 test.bin`
4. Run tests: `python test_predictability_ngram.py < test.bin`
5. Compare output with included results

## ⚖️ Transparency Commitment

✅ **Included:**
- All source code
- All test scripts
- All raw results
- All failed tests (NOT hidden)
- All analysis code
- All parameter values

❌ **NOT included:**
- Cherry-picked favorable results
- Hidden failures
- Modifications to original algorithm
- Custom tests designed to pass

## 🤝 Contributing

Found something different? Want to reproduce independently?

-> **Open an issue or submit results** -> We'll include your findings!

## 📖 Citation

```bibtex
@misc{rctm_reproducibility_2026,
  title = {Independent Reproducibility Study of RCTM-based CSPRNG},
  author = {Andre, A.},
  year = {2026},
  url = {https://github.com/angualberto/rctm-reproducibility-study}
}
```

## ⏱️ Timeline

| Date | Event |
|------|-------|
| 2026-03-06 | Initial test runs (9 Python tests) |
| 2026-03-06 | Complete analysis and documentation |
| 2026-03-06 | Repository published |

## 🎯 Next Steps

1. **Share findings** with academic community
2. **Request errata** from original authors
3. **Propose improvements** to RCTM algorithm
4. **Benchmark alternatives** (ChaCha20, AES-CTR, etc.)

---

**Status:** ✅ Reproducible | ✅ Transparent | ✅ Research Grade

**Questions?** → Open an issue on GitHub

---

*Last Updated: March 6, 2026*
