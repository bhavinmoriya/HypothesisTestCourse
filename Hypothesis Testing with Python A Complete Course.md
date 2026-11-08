# Hypothesis Testing with Python: A Complete Course

A hands-on course in 10 modules. Every concept comes with runnable code, a simulation that builds intuition, and exercises. Expected time: 15 to 20 hours. Prerequisites: basic Python, NumPy, and introductory probability.

**Learning outcomes.** By the end you will be able to: frame a question as a testable hypothesis, choose the right test, run and interpret it in Python, check assumptions, compute power and sample size, correct for multiple comparisons, use resampling methods, and report results honestly.

## Module 0: Setup

Use `uv` for the environment.

```bash
uv init ht-course && cd ht-course
uv add numpy scipy pandas statsmodels matplotlib seaborn jupyterlab
uv run jupyter lab
```

Standard header for every notebook:

```python
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

rng = np.random.default_rng(42)  # reproducibility
```

## Module 1: The Logic of Hypothesis Testing

### Concepts

- **Null hypothesis H0**: the default claim (no effect, no difference).
- **Alternative H1**: what you suspect instead. Two-sided (≠), or one-sided (>, <).
- **Test statistic**: a number summarizing how far the data are from H0.
- **p-value**: the probability, assuming H0 is true, of a test statistic at least as extreme as the one observed. It is **not** the probability that H0 is true.
- **Significance level α**: the Type I error rate you accept (commonly 0.05).
- **Type I error**: rejecting a true H0 (false positive), probability α.
- **Type II error**: failing to reject a false H0 (false negative), probability β. **Power** = 1 − β.

### Five-step recipe

1. State H0 and H1. 2. Choose α and the test *before* looking at data. 3. Check assumptions. 4. Compute the statistic and p-value. 5. Report the effect size, a confidence interval, and the decision.

### Lab 1.1: p-values are uniform under H0

If H0 is true, p-values are uniformly distributed, so you reject exactly α of the time.

```python
n_sims, n, alpha = 10_000, 30, 0.05
pvals = np.array([
    stats.ttest_1samp(rng.normal(0, 1, n), popmean=0).pvalue
    for _ in range(n_sims)
])
print("Type I error rate:", (pvals < alpha).mean())   # ≈ 0.05

plt.hist(pvals, bins=20, edgecolor="k")
plt.title("p-values when H0 is true"); plt.xlabel("p"); plt.show()
```

### Lab 1.2: p-values when H1 is true

```python
pvals_alt = np.array([
    stats.ttest_1samp(rng.normal(0.5, 1, n), popmean=0).pvalue
    for _ in range(n_sims)
])
print("Power at effect 0.5, n=30:", (pvals_alt < alpha).mean())
plt.hist(pvals_alt, bins=20, edgecolor="k"); plt.show()
```

The histogram piles up near zero. The fraction below α is the power.

### Exercises

1. Change `n` to 10 and to 200. How does the Type I error rate change? Why not at all?
2. Change the true mean to 0.2. Plot power as a function of `n` for n = 10, 20, ..., 200.

## Module 2: The z-test and t-tests

### One-sample t-test

H0: μ = μ0. Statistic: t = (x̄ − μ0) / (s/√n), with n − 1 degrees of freedom.

Build it from scratch, then compare with SciPy:

```python
def one_sample_t(x, mu0):
    x = np.asarray(x); n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    t = (x.mean() - mu0) / se
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    ci = stats.t.interval(0.95, df=n - 1, loc=x.mean(), scale=se)
    return t, p, ci

x = rng.normal(loc=52, scale=8, size=40)   # e.g., exam scores
print(one_sample_t(x, 50))
print(stats.ttest_1samp(x, 50))            # should match
```

One-sided: `stats.ttest_1samp(x, 50, alternative="greater")`.

### Two-sample t-test (independent groups)

Default to **Welch's** test (`equal_var=False`). It does not assume equal variances and loses almost nothing when they are equal.

```python
a = rng.normal(100, 15, 60)
b = rng.normal(108, 20, 55)

res = stats.ttest_ind(a, b, equal_var=False)
print(res)
print(res.confidence_interval())   # CI for mean(a) - mean(b)
```

### Paired t-test

Use when the same units are measured twice (before and after). It is a one-sample t-test on the differences.

```python
before = rng.normal(80, 10, 30)
after = before + rng.normal(3, 5, 30)
print(stats.ttest_rel(after, before))
print(stats.ttest_1samp(after - before, 0))   # identical
```

### Effect size: Cohen's d

A p-value says whether an effect is detectable. The effect size says whether it matters.

```python
def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return (a.mean() - b.mean()) / sp

print(cohens_d(a, b))  # rough guide: 0.2 small, 0.5 medium, 0.8 large
```

### The z-test

When σ is known (rare) or n is very large:

```python
from statsmodels.stats.weightstats import ztest
z, p = ztest(x, value=50)
```

### Exercises

1. Simulate two groups with equal means but unequal variances and unequal sizes (n = 10 and 100). Compare the Type I error of Student's (`equal_var=True`) and Welch's tests over 10,000 runs.
2. Why does pairing usually give more power than treating the samples as independent? Verify by simulation with correlated measurements.

## Module 3: Proportions and Categorical Data

### One- and two-proportion z-tests

```python
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# Did 56 of 100 customers convert, vs. H0: p = 0.5?
print(proportions_ztest(count=56, nobs=100, value=0.5))
print(proportion_confint(56, 100, method="wilson"))

# Two groups: 120/1000 vs 150/1000
print(proportions_ztest(count=[120, 150], nobs=[1000, 1000]))
```

### Exact binomial test

Prefer this for small samples.

```python
print(stats.binomtest(k=9, n=12, p=0.5, alternative="two-sided"))
```

### Chi-square test of independence

H0: two categorical variables are independent.

```python
table = pd.DataFrame(
    [[30, 10, 20], [20, 25, 15]],
    index=["Treatment", "Control"], columns=["Improved", "Same", "Worse"],
)
chi2, p, dof, expected = stats.chi2_contingency(table)
print(chi2, p, dof)
print(pd.DataFrame(expected, index=table.index, columns=table.columns))
```

Rule of thumb: all expected counts should be at least 5. Otherwise use Fisher's exact test for 2×2 tables:

```python
print(stats.fisher_exact([[8, 2], [1, 5]]))
```

### Chi-square goodness of fit

Is a die fair?

```python
observed = np.array([18, 22, 16, 25, 20, 19])
print(stats.chisquare(observed))   # H0: uniform
```

### Exercises

1. Simulate 1,000 fair-die experiments of 120 rolls. Check the goodness-of-fit Type I error rate.
2. Make a 2×2 table with small counts and compare the chi-square and Fisher p-values.

## Module 4: Power and Sample Size

Power depends on four linked quantities: effect size, sample size, α, and variability. Fix three, solve for the fourth.

### Analytical power with statsmodels

```python
from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()

# Sample size per group for d=0.5, power 0.8, alpha 0.05
n_needed = analysis.solve_power(effect_size=0.5, power=0.8, alpha=0.05)
print(round(n_needed))   # ≈ 64

# Power for n=30 per group
print(analysis.power(effect_size=0.5, nobs1=30, alpha=0.05))

analysis.plot_power(dep_var="nobs", nobs=np.arange(5, 150),
                    effect_size=[0.2, 0.5, 0.8], alpha=0.05)
plt.show()
```

### Power for proportions

```python
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

es = proportion_effectsize(0.12, 0.15)   # detect 12% -> 15%
print(NormalIndPower().solve_power(es, power=0.8, alpha=0.05))
```

### Power by simulation (works for any test)

```python
def simulate_power(sampler_a, sampler_b, test, n_sims=5000, alpha=0.05):
    hits = 0
    for _ in range(n_sims):
        hits += test(sampler_a(), sampler_b()).pvalue < alpha
    return hits / n_sims

# Skewed data: Welch t-test vs Mann-Whitney
sa = lambda: rng.lognormal(0.0, 1.0, 40)
sb = lambda: rng.lognormal(0.5, 1.0, 40)
print("Welch:", simulate_power(sa, sb, lambda x, y: stats.ttest_ind(x, y, equal_var=False)))
print("MWU:  ", simulate_power(sa, sb, stats.mannwhitneyu))
```

### Exercises

1. How many samples per group do you need for d = 0.2 at 90% power?
2. A study of n = 20 per group found p = 0.08. Compute its power for d = 0.5. What does a non-significant result mean here?

## Module 5: Nonparametric, Permutation, and Bootstrap Methods

Use these when normality is doubtful, samples are small, or the statistic is unusual.

### Rank-based tests

| Parametric | Nonparametric alternative |
| --- | --- |
| One-sample / paired t-test | Wilcoxon signed-rank |
| Two-sample t-test | Mann-Whitney U |
| One-way ANOVA | Kruskal-Wallis |

```python
print(stats.wilcoxon(after - before))
print(stats.mannwhitneyu(a, b, alternative="two-sided"))
print(stats.kstest(x, "norm", args=(x.mean(), x.std(ddof=1))))  # distribution check
print(stats.ks_2samp(a, b))
```

### Permutation test from scratch

Idea: if H0 holds, group labels are exchangeable. Shuffle them and see how unusual the observed statistic is.

```python
def permutation_test(a, b, stat=lambda x, y: x.mean() - y.mean(),
                     n_perm=20_000, rng=rng):
    obs = stat(a, b)
    pooled = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        count += abs(stat(pooled[:na], pooled[na:])) >= abs(obs)
    p = (count + 1) / (n_perm + 1)    # +1 avoids p = 0
    return obs, p

print(permutation_test(a, b))
# Or use scipy:
print(stats.permutation_test((a, b), lambda x, y: x.mean() - y.mean(),
                             n_resamples=9999, random_state=1))
```

Swap in any statistic: difference of medians, ratio of variances, a correlation.

### Bootstrap confidence interval

```python
def bootstrap_ci(x, stat=np.median, n_boot=10_000, level=0.95, rng=rng):
    boots = np.array([stat(rng.choice(x, size=len(x), replace=True))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(1-level)/2*100, (1+level)/2*100])
    return lo, hi

print(bootstrap_ci(x))
print(stats.bootstrap((x,), np.median, confidence_level=0.95,
                      method="BCa", random_state=1).confidence_interval)
```

Relation to testing: a 95% CI that excludes the null value corresponds to rejecting H0 at α = 0.05.

### Exercises

1. Test whether two groups have different *medians* using a permutation test.
2. Compare the bootstrap CI for the mean to the t-based CI on normal data and on heavily skewed data.

## Module 6: ANOVA and Post-hoc Tests

Comparing more than two means with repeated t-tests inflates Type I error. ANOVA tests all at once: H0: all group means are equal.

### One-way ANOVA

```python
g1 = rng.normal(10, 2, 30)
g2 = rng.normal(11, 2, 30)
g3 = rng.normal(13, 2, 30)

F, p = stats.f_oneway(g1, g2, g3)
print(F, p)

# Non-parametric version
print(stats.kruskal(g1, g2, g3))
```

### Table form with statsmodels

```python
import statsmodels.formula.api as smf

df = pd.DataFrame({
    "y": np.concatenate([g1, g2, g3]),
    "group": np.repeat(["A", "B", "C"], 30),
})
model = smf.ols("y ~ C(group)", data=df).fit()
print(sm.stats.anova_lm(model, typ=2))
```

### Post-hoc: which groups differ?

Run only after a significant ANOVA.

```python
print(stats.tukey_hsd(g1, g2, g3))

# Alternative with a tidy table
from statsmodels.stats.multicomp import pairwise_tukeyhsd
print(pairwise_tukeyhsd(df["y"], df["group"]))
```

### Effect size: eta squared

```python
anova = sm.stats.anova_lm(model, typ=2)
eta2 = anova["sum_sq"]["C(group)"] / anova["sum_sq"].sum()
print(eta2)
```

### Two-way ANOVA (interaction)

```python
df2 = pd.DataFrame({
    "y": rng.normal(0, 1, 120),
    "drug": np.tile(["X", "Y"], 60),
    "dose": np.repeat(["low", "mid", "high"], 40),
})
m2 = smf.ols("y ~ C(drug) * C(dose)", data=df2).fit()
print(sm.stats.anova_lm(m2, typ=2))
```

### Exercises

1. Simulate 5 groups with identical means. Over 5,000 runs, compare the Type I error of ANOVA with the family-wise error of all 10 pairwise t-tests.
2. Create data where the interaction is significant but main effects are not.

## Module 7: Multiple Testing

Test 20 true nulls at α = 0.05 and the chance of at least one false positive is 1 − 0.95²⁰ ≈ 64%.

- **FWER** (family-wise error rate): P(at least one false positive). Controlled by Bonferroni and Holm.
- **FDR** (false discovery rate): expected fraction of false positives among rejections. Controlled by Benjamini-Hochberg. Preferred in exploratory work with many tests.

### Lab 7.1: See the problem

```python
m, n_sims = 20, 5000
any_hit = 0
for _ in range(n_sims):
    p = [stats.ttest_1samp(rng.normal(0, 1, 30), 0).pvalue for _ in range(m)]
    any_hit += min(p) < 0.05
print("FWER uncorrected:", any_hit / n_sims)   # ≈ 0.64
```

### Lab 7.2: Corrections

```python
from statsmodels.stats.multitest import multipletests

# 1000 tests: 900 true nulls, 100 real effects
m0, m1 = 900, 100
z = np.concatenate([rng.normal(0, 1, m0), rng.normal(3, 1, m1)])
pvals = 2 * stats.norm.sf(np.abs(z))
truth = np.array([False]*m0 + [True]*m1)

for method in ["bonferroni", "holm", "fdr_bh"]:
    reject, p_adj, *_ = multipletests(pvals, alpha=0.05, method=method)
    fp = (reject & ~truth).sum()
    tp = (reject & truth).sum()
    print(f"{method:11s} discoveries={reject.sum():3d}  true={tp:3d}  false={fp:2d}")
```

Bonferroni and Holm are conservative (few false positives, fewer discoveries). BH finds more true effects while keeping the false discovery proportion near 5%.

### Exercises

1. Vary the effect size (3 → 2) and the share of true effects. When does BH beat Holm most?
2. Implement Holm and Benjamini-Hochberg by hand and check against `multipletests`.

## Module 8: Checking Assumptions

| Test | Assumptions |
| --- | --- |
| t-test | Independent observations; roughly normal data (or normal sampling distribution via large n) |
| Welch | Same, but equal variances not needed |
| ANOVA | Independence, normal residuals, equal variances |
| Chi-square | Independence, expected counts ≥ 5 |

### Normality

```python
print(stats.shapiro(x))           # H0: data are normal
sm.qqplot(x, line="s"); plt.show()
plt.hist(x, bins=15); plt.show()
```

### Equal variances

```python
print(stats.levene(g1, g2, g3))   # robust to non-normality
print(stats.bartlett(g1, g2, g3)) # more powerful but sensitive to non-normality
```

### Practical advice

- Formal tests of assumptions are **underpowered for small n** and **oversensitive for large n**. Always look at plots.
- The t-test is robust to mild non-normality when n is moderate. Skewness and outliers are the real concern.
- Pre-testing normality and then choosing the test changes the overall error rates. Prefer a method that is robust by default (Welch, permutation).
- Independence violations (clustered data, time series) are much more serious than non-normality.

### Lab 8.1: How robust is the t-test?

```python
def type1_rate(sampler, n, n_sims=10_000):
    return np.mean([stats.ttest_1samp(sampler(n), sampler.true_mean).pvalue < 0.05
                    for _ in range(n_sims)])

def make(sampler_fn, mean):
    f = lambda n: sampler_fn(n); f.true_mean = mean; return f

exp = make(lambda n: rng.exponential(1.0, n), 1.0)   # heavily skewed
for n in [5, 15, 30, 100]:
    print(n, type1_rate(exp, n))
```

Watch the error rate approach 0.05 as the central limit theorem kicks in.

## Module 9: Applied Case Study, A/B Testing

Scenario: a website tests a new checkout page. Control converts at roughly 10%. We want to detect a lift to 12%.

### Step 1: Design

```python
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

es = proportion_effectsize(0.12, 0.10)
n_per_arm = int(np.ceil(NormalIndPower().solve_power(es, power=0.8, alpha=0.05)))
print("Visitors per arm:", n_per_arm)
```

### Step 2: Simulate the experiment and analyze

```python
n = n_per_arm
conv_a = rng.binomial(n, 0.10)
conv_b = rng.binomial(n, 0.12)

z, p = proportions_ztest([conv_b, conv_a], [n, n], alternative="larger")
lift = conv_b / n - conv_a / n
print(f"A={conv_a/n:.3%}  B={conv_b/n:.3%}  lift={lift:.3%}  z={z:.2f}  p={p:.4f}")

from statsmodels.stats.proportion import confint_proportions_2indep
print(confint_proportions_2indep(conv_b, n, conv_a, n, method="wald"))
```

### Step 3: The peeking problem

Checking the p-value repeatedly and stopping at the first p < 0.05 inflates the false positive rate, even when there is no true difference.

```python
def peeking_experiment(n_total=4000, looks=20, p_true=0.10, alpha=0.05):
    a = rng.binomial(1, p_true, n_total)
    b = rng.binomial(1, p_true, n_total)
    for k in np.linspace(n_total / looks, n_total, looks, dtype=int):
        _, p = proportions_ztest([b[:k].sum(), a[:k].sum()], [k, k])
        if p < alpha:
            return True    # declared "significant"
    return False

rate = np.mean([peeking_experiment() for _ in range(2000)])
print("False positive rate with 20 peeks:", rate)   # far above 0.05
```

Fixes: fix the sample size in advance, or use sequential methods designed for continuous monitoring (alpha-spending, group-sequential tests, or Bayesian decision rules).

### Step 4: Report

A good report states: the hypothesis and primary metric (declared in advance), sample size and power calculation, observed effect with a confidence interval, the p-value, practical significance (revenue impact), and every other test you ran.

## Module 10: Capstone Project

Choose one, or use your own data. Deliver a notebook that follows the five-step recipe.

**Option A: Clinical trial.** Simulate a randomized trial with 3 arms and a skewed outcome. Test with ANOVA and Kruskal-Wallis, run post-hoc comparisons, compute power, and compare to a permutation test.

**Option B: Screening many features.** Take a dataset with 500 candidate features (simulate or use a public one). Test each against a binary outcome, then compare discoveries under no correction, Holm, and BH. Because you know the truth in a simulation, compute the realized FWER and FDR.

**Option C: Financial returns.** Using daily returns of an asset, test whether the mean return differs from zero, whether two assets have equal variance, and whether returns are normal (Jarque-Bera: `stats.jarque_bera`). Discuss why autocorrelation and fat tails threaten the independence and normality assumptions, and use a block bootstrap as a more robust alternative.

**Grading rubric.** Clear hypotheses (15%). Appropriate test choice with justification (20%). Assumption checks (15%). Correct computation and interpretation (20%). Effect sizes and CIs (15%). Honest reporting of limitations and multiplicity (15%).

## Appendix A: Choosing a Test

| Question | Data | Test |
| --- | --- | --- |
| One mean vs. a value | Numeric, roughly normal | One-sample t |
| Two independent means | Numeric | Welch t (or Mann-Whitney if skewed or ordinal) |
| Two paired measurements | Numeric | Paired t (or Wilcoxon signed-rank) |
| 3+ means | Numeric | ANOVA, then Tukey (or Kruskal-Wallis) |
| One proportion | Binary | Binomial exact / z-test |
| Two proportions | Binary | Two-proportion z, Fisher if small |
| Association of two categoricals | Counts | Chi-square independence |
| Fit to a distribution | Counts or continuous | Chi-square GOF, KS, Shapiro |
| Equal variances | Numeric | Levene |
| Correlation | Two numeric | `stats.pearsonr`, `stats.spearmanr` |
| Anything unusual | Any | Permutation test or bootstrap |

## Appendix B: Common Mistakes

1. Reading the p-value as the probability that H0 is true.
2. Treating "not significant" as "no effect". It may just mean low power.
3. Choosing the test or the one-sided alternative after seeing the data.
4. Running many tests and reporting only the significant ones.
5. Ignoring effect size: with huge n, trivial effects become significant.
6. Using independent-samples tests on paired or clustered data.
7. Stopping an experiment as soon as p < 0.05.
8. Reporting p-values without confidence intervals.

## Appendix C: Self-Test Quiz

1. If H0 is true and α = 0.05, what is the distribution of the p-value? *(Uniform on \[0, 1\].)*
2. What happens to power if you halve the standard deviation? *(It increases, because the effect size d doubles.)*
3. Why default to Welch's t-test? *(It keeps correct Type I error under unequal variances and costs almost no power otherwise.)*
4. You run 40 independent tests at α = 0.05 with all nulls true. Expected number of false positives? *(2.)*
5. When prefer BH over Bonferroni? *(When you accept some false discoveries and want more power, as in exploratory screening.)*
6. A 95% CI for a mean difference is (−0.3, 4.1). What do you conclude at α = 0.05? *(Fail to reject H0 of zero difference, but the data are also compatible with a meaningful positive effect.)*
7. Why is the permutation test valid without normality? *(It relies only on exchangeability of labels under H0.)*

## Appendix D: Suggested Further Reading

- *Statistical Inference* by Casella and Berger (theory).
- *All of Statistics* by Larry Wasserman (concise, mathematical).
- *Statistical Rules of Thumb* by Gerald van Belle (practice).
- *Trustworthy Online Controlled Experiments* by Kohavi, Tang, and Xu (A/B testing).
- The American Statistical Association's 2016 statement on p-values.

## Suggested Schedule

| Week | Modules |
| --- | --- |
| 1 | 0, 1, 2 |
| 2 | 3, 4 |
| 3 | 5, 6 |
| 4 | 7, 8, 9 |
| 5 | 10 (capstone) |
