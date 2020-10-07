import numpy as np
import scipy.stats.distributions as distributions
from collections import namedtuple
import scipy.stats as stats

Ttest_indResult = namedtuple('Ttest_indResult',
                             ('statistic', 'pvalue', 'dof',
                              'mean1', 'mean2', 'var1', 'var2',
                              'tail',
                              'cohen_d'))

Ttest_relResult = namedtuple('Ttest_relResult',
                             ('statistic', 'pvalue', 'dof',
                              'mean', 'var', 'tail',
                              'cohen_d'))


def ttest_rel(x, y, equal_var=True, two_tailed=True, m0=0):

    if len(x) != len(y):
        raise ValueError('unequal length arrays')

    n = float(len(x))
    diff = x - y

    # Compute mean
    mean = np.mean(diff)

    # compute unbiased standard deviation
    std = np.std(diff, ddof=1)

    # degrees of freedom
    dof = n - 1

    # Compute t-statistic
    t = np.sqrt(n) * (mean - m0) / std

    # Compute pvalue
    p = distributions.t.sf(abs(t), dof)

    if two_tailed:
        p *= 2.0
        tail = 'two-tailed'

    else:
        tail = 'one-tailed'

    if equal_var:
        std_pooled = np.sqrt(0.5 * (np.var(x, ddof=1) + np.var(y, ddof=1)))
    else:
        std_pooled = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / n)
    cohen_d = abs((np.mean(x) - np.mean(y)) / std_pooled)

    return Ttest_relResult(t, p, dof, mean, std ** 2, tail, cohen_d)

def ttest_ind_from_stats(x, m2, two_tailed=True):

    m1 = np.mean(x)
    v1 = np.var(x, ddof=1)
    n1 = float(len(x))
    
    dof = n1 - 1
    t = np.sqrt(n1) * (m1 - m2) / v1

    cohen_d = abs(m1 - m2) / v1

    p = distributions.t.sf(t, dof)

    if two_tailed:
        p *= 2.0
        tail = 'two-tailed'
    else:
        tail = 'one-tailed'

    return Ttest_indResult(t, p, dof, m1, m2, v1, v1, tail, cohen_d)
    

def _ttest_ind_from_stats(m1, m2, v1, v2, sp, norm, dof, two_tailed):

    # Compute t statistic
    t = norm * (m1 - m2) / sp

    # Compute pvalue
    p = distributions.t.sf(abs(t), dof)

    # Cohen's d (effect size)
    cohen_d = abs(m1 - m2) / sp

    # Compute one-tailed or two tailed test
    if two_tailed:
        p *= 2.0
        tail = 'two-tailed'
    else:
        tail = 'one-tailed'
    return Ttest_indResult(t, p, dof, m1, m2, v1, v2, tail, cohen_d)


def _ttest_unequal_size_equal_var(m1, m2, v1, v2, n1, n2, two_tailed):

    # Standard deviation term
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    # Normalization term
    norm = np.sqrt(n1 * n2 / (n1 + n2))
    # degrees of freedom
    dof = n1 + n2 - 2

    return _ttest_ind_from_stats(m1, m2, v1, v2, sp, norm, dof, two_tailed)


def _ttest_unequal_size_unequal_var(m1, m2, v1, v2, n1, n2, two_tailed):

    v1n = v1 / n1
    v2n = v2 / n2
    sp = np.sqrt(v1n + v2n)
    norm = 1.0

    # degrees of freedom (Welch-Satterthwaite equation)
    dof = ((v1n + v2n) ** 2 / ((v1n ** 2) / (n1 - 1) + (v2n ** 2) / (n2 - 1)))

    return _ttest_ind_from_stats(m1, m2, v1, v2, sp, norm, dof, two_tailed)


def ttest_ind(x, y, equal_var=True, two_tailed=True):

    m1 = np.mean(x)
    m2 = np.mean(y)

    v1 = np.var(x, ddof=1)
    v2 = np.var(y, ddof=1)

    n1 = float(len(x))
    n2 = float(len(y))

    if equal_var:
        return _ttest_unequal_size_equal_var(m1, m2, v1, v2, n1, n2, two_tailed)
    else:
        return _ttest_unequal_size_unequal_var(m1, m2, v1, v2, n1, n2, two_tailed)


def pretty_print_results(results, cond1_name='A', cond2_name='B', alpha=0.01):

    # Check for significance
    if results.pvalue < alpha:
        sig = 'a significant'
    else:
        sig = 'no significant'

    sig += ' difference at the $\\alpha = {0:.2f}$ level'.format(alpha)

    if results.dof.is_integer():
        dof_s = '{0}'.format(int(results.dof))
    else:
        dof_s = '{0:.2f}'.format(results.dof)

    t_res = "$\\text{{t}}({0})={1:.2f}, p={2:.2f}, \\text{{Cohen's}}\\ d={3:.2f}$".format(
        dof_s, results.statistic, results.pvalue, results.cohen_d)

    if isinstance(results, Ttest_indResult):
        t_type = 'An independent-samples'

        g1_stats = '{2} $(\\text{{mean}}={0:.2f}, \\text{{std}}={1:.2f})$'.format(
            results.mean1,
            np.sqrt(results.var1),
            cond1_name)

        g2_stats = '{2} $(\\text{{mean}}={0:.2f}, \\text{{std}}={1:.2f})$'.format(
            results.mean2,
            np.sqrt(results.var2),
            cond2_name)

        g_stats = 'for ' + g1_stats + ' and ' + g2_stats

    elif isinstance(results, Ttest_relResult):
        t_type = 'A paired samples'

        g_stats = '$(\\text{{mean}}={0:.2f}, \\text{{std}}={1:.2f})$'.format(
            results.mean,
            np.sqrt(results.var))


    out_str = ('{0} {1} t-test was conducted to compare {2} and {3}. '
               'There was {4} in the scores {5}; {6}').format(
                   t_type, results.tail, cond1_name, cond2_name,
                   sig, g_stats, t_res)

    print(out_str)
    return out_str, t_res, g_stats


if __name__ == '__main__':

    # Compare to scipy.stats example
    rvs1 = stats.norm.rvs(loc=5, scale=10, size=500)
    rvs2 = stats.norm.rvs(loc=5, scale=10, size=500)
    results_scipy = stats.ttest_ind(rvs1, rvs2)
    results_scipy2 = stats.ttest_ind(rvs1, rvs2, equal_var=False)
    results = ttest_ind(rvs1, rvs2)
    results_2 = ttest_ind(rvs1, rvs2, equal_var=False)

    results_rel = stats.ttest_rel(rvs1, rvs2)
    results_rel2 = ttest_rel(rvs1, rvs2)

    pretty_print_results(results_2)

    pretty_print_results(results_rel2)
    # rvs3 = stats.norm.rvs(loc=5, scale=20, size=500)
    # results_scipy = stats.ttest_ind(rvs1, rvs3)

    # results_scipy2 = stats.ttest_ind(rvs1, rvs3, equal_var=False)

    # rvs4 = stats.norm.rvs(loc=5, scale=20, size=100)
    # results_scipy = stats.ttest_ind(rvs1, rvs4)

    # results_scipy2 = stats.ttest_ind(rvs1, rvs4, equal_var=False)

    # rvs5 = stats.norm.rvs(loc=8, scale=20, size=100)
    # results_scipy = stats.ttest_ind(rvs1, rvs5)

    # results_scipy2 = stats.ttest_ind(rvs1, rvs5, equal_var=False)
