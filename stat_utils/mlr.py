"""
A collection of models for performing statistical tests.
"""
import copy

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import lsmr
import scipy.stats as stats
import scipy.special as special

try:
    from ttest import ttest_ind_from_stats
except:
    from .ttest import ttest_ind_from_stats

class StatistictestResult(object):
    """
    Baseclass to hold and display the results of a statistical test.
    
    Parameters
    ----------
    statistic : float
        Value of the test statistic.
    pvalue : float
        p-value.
    significance_level : float
        Level at which the test is considered to be statistically significant.
    hypothesis_type : ('equal', 'lt', 'gt')
        Type of hypothesis. If 'equal', the test is a two-sided test, if 'lt' or 'gt', 
        the test is one sided (see `test_type` property).
    parameter_name : str
        Name of the parameter to be tested.
    """
    def __init__(self, statistic, pvalue, significance_level, hypothesis_type='equal', parameter_name=None):
        self.statistic = statistic
        self.pvalue = pvalue
        self.significance_level = significance_level
        self.hypothesis_type = hypothesis_type
        self.parameter_name = parameter_name

    @property
    def test_type(self):
        """
        Whether the test is one- or two-sided
        """
        if self.hypothesis_type == 'equal':
            return 'two_sided'
        else:
            return 'one_sided'

    @property
    def reject_h0(self):
        """
        A boolean that specifies if the null hypothesis should be rejected at the 
        current significance_level and hypothesis type.
        """
        reject = self.pvalue < self.significance_level
        direction = True
        if reject:
            if self.hypothesis_type == 'lt':
                direction = self.statistic < 0
            elif self.hypothesis_type == 'gt':
                direction = self.statistic > 0
            return reject and direction
        else:
            return reject

    def __str__(self):
        test_name = type(self).__name__

        kwargs = []
        for attr in self.__dict__.keys():
            if isinstance(self.__dict__[attr], (float)):
                kwargs.append('{name}={val:.3f}'.format(name=attr,
                                                        val=self.__dict__[attr]))
            elif isinstance(self.__dict__[attr], str):
                kwargs.append('{name}="{val}"'.format(name=attr,
                                                      val=self.__dict__[attr]))
            else:
                kwargs.append('{name}={val}'.format(name=attr,
                                                    val=self.__dict__[attr]))

        out_str = '{test_name}({kwargs})'.format(test_name=test_name,
                                                 kwargs=','.join(kwargs))
        
        return out_str


class WaldtestResult(StatistictestResult):
    """
    Object to hold and display the results of a Wald test

    Parameters
    ----------
    statistic : float
        Value of the Wald statistic.
    pvalue : float
        p-value.
    significance_level : float
        Level at which the test is considered to be statistically significant.
    hypothesis_type : ('equal', 'lt', 'gt')
        Type of hypothesis. If 'equal', the test is a two-sided test, if 'lt' or 'gt', 
        the test is one sided (see `test_type` property).
    parameter_name : str
        Name of the parameter to be tested.

    """
    def __init__(self, statistic, pvalue, significance_level, hypothesis_type='equal', parameter_name=None):
        super().__init__(statistic=statistic,
                         pvalue=pvalue,
                         significance_level=significance_level,
                         hypothesis_type=hypothesis_type,
                         parameter_name=parameter_name)

class TtestResult(StatistictestResult):
    """
    Object to hold and display the results of a T-test

    Parameters
    ----------
    statistic : float
        Value of the T statistic.
    pvalue : float
        p-value.
    df : int
        Degrees of freedom
    significance_level : float
        Level at which the test is considered to be statistically significant.
    hypothesis_type : ('equal', 'lt', 'gt')
        Type of hypothesis. If 'equal', the test is a two-sided test, if 'lt' or 'gt', 
        the test is one sided (see `test_type` property).
    parameter_name : str
        Name of the parameter to be tested.
    """
    def __init__(self, statistic, pvalue, df, significance_level, hypothesis_type='equal', parameter_name=None):
        super().__init__(statistic=statistic,
                         pvalue=pvalue,
                         significance_level=significance_level,
                         hypothesis_type=hypothesis_type,
                         parameter_name=parameter_name)
        self.df = df


class F_onewayResult(StatistictestResult):
    """
    Object to hold and display the results of a One-way ANOVA

    Parameters
    ----------
    statistic : float
        Value of the T statistic.
    pvalue : float
        p-value.
    df_btwn : int
        Degrees of freedom between groups
    df_within : int
        Degrees of freedom within groups
    etasquared : float
        Effect Size
    significance_level : float
        Level at which the test is considered to be statistically significant.
    hypothesis_type : ('equal', 'lt', 'gt')
        Type of hypothesis. If 'equal', the test is a two-sided test, if 'lt' or 'gt', 
        the test is one sided (see `test_type` property).
    parameter_name : str
        Name of the parameter to be tested.
    """
    def __init__(self, statistic, pvalue, df_btwn, df_within, etasquared,
                 significance_level, hypothesis_type='equal', parameter_name=None):
        super().__init__(statistic=statistic,
                         pvalue=pvalue,
                         significance_level=significance_level,
                         hypothesis_type=hypothesis_type,
                         parameter_name=parameter_name)
        self.df_btwn = df_btwn
        self.df_within = df_within
        self.etasquared = etasquared


class MutipleLinearRegressionResult(object):
    def __init__(self, model_test,
                 predictor_tests,
                 model_params,
                 standard_error,
                 R2, R2_adj):

        self.model_test = model_test
        self.predictor_tests = predictor_tests
        self.model_params = model_params

        self.R2 = R2

    def pretty_print_results(self):

        model_fit = '{tail} t({dof}) = {statistic:.2f}, p={pvalue:.3f}, R^2={R2}, R^2_{{adj}}={R2adj}'.format(
            tail=self.model_test.tail,
            dof=int(self.model_test.dof),
            statistic=self.model_test.statistic,
            pvalue=self.model_test.pvalue,
            R2=self.R2,
            R2adj=self.R2_adj)

        
        header = '\t'.join(['Covariate', '$\beta_i$', 'se', '$t$', '$p$'])

        lines = [model_fit, 10 * '-', header, 10* '-']

        for st, beta, se in zip(self.predictor_tests, self.model_params, self.standard_error):
            # pvalue = 
            line = ','.join([st.parameter_name,
                             '{0:.2f}'.format(p),
                             '{0:.2f}'.format(se),
                             '{0:.2f}'.format(st.statistic),
                             '{0:.2f}'.format(st.pvalue),
                             st.test_type])

            lines.append(line)

        return('\n'.join(lines))
            

def ensure2d(x):
    """
    Ensure that the input object is a matrix.
    """
    if x.ndim == 1:
        return x.reshape(-1, 1)
    else:
        return x

class LinearModel(object):
    """
    Linear Model

    This model computes a linear regression using least squares

    Parameters
    ----------
    input_names : list
       Names of the input parameters (predictors)
    add_constant : bool (optional)
       Add intercept to the model (Default is True)
    """
    def __init__(self, input_names=None, add_constant=True):
        self.add_constant = add_constant
        self.params = None
        # copy input names, to avoid issues with indexing in pandas
        # for adding intercept
        self.input_names = copy.copy(input_names)

        # Add intercept to the input names
        if self.input_names is not None and self.add_constant:
            self.input_names.append('intercept')
        
    def fit(self, x, y):
        """
        Fit the model using LSMR, a computationally efficient LS algorithm.
        
        Parameters
        ----------
        x : np.ndarray
           Input (predictor) variables
        y : np.ndarray
           Target variables
        """
        # Compute parameters with LSMR
        self.params = lsmr(self._add_intercept(ensure2d(x)), y)[0]

        # Create dummy input variables
        if self.input_names is None:
            num_params = len(self.params) - 1 if self.add_constant else len(self.params)
            self.input_names = ['x{0}'.format(i) for i in range(num_params)]
            if self.add_constant:
                self.input_names.append('intercept')

    def fit_predict(self, x, y):
        """
        Fit and predict (sklearn API-like)
        """
        self.fit(x, y)
        return self.predict(x)

    def predict(self, x):
        """
        Predict the target variable using a fitted model.

        Parameters
        ----------
        x : np.ndarray
           Input (predictor) variables
        
        Returns
        -------
        np.ndarray
           Estimated (predicted) target values.
        """

        if self.params is None:
            raise ValueError('The model has not been fitted.')
        return np.dot(self._add_intercept(ensure2d(x)), self.params)        

    def _add_intercept(self, x):
        """
        Add an intercept to the input values as a column with a constant value
        if self.add_constant is True
        """
        return np.column_stack((x, np.ones(len(x)))) if self.add_constant else x
    
    def test(self, x, y, test_type='t',
             null_hypothesis=0.0,
             hypothesis_type='equal',
             significance_level=0.01, return_preds=False):
        """
        Test the parameters

        Parameters
        ----------
        x : np.ndarray
            Input (predictor/covariate) variables
        y : np.ndarray
            Target (predicted/response) variable
        test_type : ('t', 'wald')
            Type of test. 't' does a T-test and 'wald' does a Wald test.
            Use 't' tests for smaller sample sizes and 'wald' for larger samples.
        hypothesis_type : str or iterable
            Type of hypothesis for each input variable. If a single hypothesis type is given,
            it will be used for all variables.
        null_hypothesis : float or np.ndarray
            Value of the estimated parameters under the null hypothesis. Default is 0.
        significance_level : float
            Level at which the differences are considered to be significant.
        return_preds : bool (default: False)
            Return the predictions of the model under the current parameters.

        Returns
        statistics : list of instances of StatistictestResult
            Results of the statistical tests for each of the parameters of the model.
        y_hat : np.ndarray
            Model predictions (only returned if return_preds is True)
        """
        # Fit model if not trained
        if self.params is None:
            self.fit(x, y)

        if not isinstance(hypothesis_type, (list, tuple, np.ndarray)):
            hypothesis_type = len(self.params) * [hypothesis_type]

        for ht in hypothesis_type:
            if ht not in ('lt', 'gt', 'equal'):
                raise ValueError('hypothesis typ should be "lt", "gt" or "equal", but given {0}'.format(ht))

        if not (isinstance(null_hypothesis, (float, int)) or isinstance(null_hypothesis, np.ndarray)):
            raise ValueError('the value of the null hypothesis should be')

        if isinstance(null_hypothesis, np.ndarray):
            if len(null_hypothesis) != len(self.params):
                raise ValueError('The lenght of the null hypothesis must be the same as the parameters')

        # compute model predictions
        y_hat = self.predict(x)

        # Compute standard error
        se, _ = _compute_se_variance_linear_model(x=self._add_intercept(ensure2d(x)),
                                                  y=y,
                                                  y_hat=y_hat)
        # Compute test statistic
        statistic = self.params / se

        if test_type == 'wald':
            sf = stats.norm.sf
            kwargs = {}
            output_kwargs = {}
            outres = WaldtestResult
            p_scale = 1
        elif test_type == 't':
            sf = stats.t.sf
            kwargs = {'df': len(y) - 1}
            output_kwargs = {'df': len(y) - 1}
            outres = TtestResult
            p_scale = 2

        # Compute p-values
        pvalues = p_scale * sf(abs(statistic), **kwargs)

        ttest_approx = ttest_ind_from_stats(y - y_hat, 0)

        output = []
        for p, st, ht, pn in zip(pvalues, statistic, hypothesis_type, self.input_names):
            output.append(outres(statistic=st,
                                 pvalue=p,
                                 significance_level=significance_level,
                                 hypothesis_type=ht,
                                 parameter_name=pn,
                                 **output_kwargs))

        # output_test = MultipleLinearRegressionResult()

        if return_preds:
            return output, ttest_approx, y_hat
        else:
            return output, ttest_approx

def zheng_loh_model_selection(x, y, input_names=None, return_model_risks=False):
    """
    Zheng-Loh Model Selection method for linear models.
    """

    if input_names is None:
        input_names = ['x{0}'.format(i) for i in range(x.shape[1])]

    # number of datapoints
    n = len(y)

    # Model with all predictors
    lm = LinearModel(input_names=input_names,
                     add_constant=True)
    # fit model
    y_hat =  lm.fit_predict(x, y)
    # add intercept to the model
    x_intercept = lm._add_intercept(x)

    # Compute standard error and variance of the full model
    se, var = _compute_se_variance_linear_model(x=x_intercept,
                                                y=y,
                                                y_hat=y_hat)
    # compute Wald statistic (H0 : param == 0, H1 : param != 0)
    statistic = lm.params / se

    # Sort statistics descendingly by absolute value
    sort_idxs = np.argsort(abs(statistic))[::-1]

    # Initialize Zheng-Loh prediction risk
    risk = np.arange(1, len(sort_idxs) + 1) * var * np.log(n)
    zl_models = []
    p_indices = []

    def pn_idxs_itc(idxs):
        """
        Select subset of input names, indices and
        add_constant option
        """
        add_constant = False
        p_names = [lm.input_names[i] for i in idxs]

        p_idxs = idxs

        if 'intercept' in p_names:
            add_constant = True
            intercept_idx = p_names.index('intercept')
            del p_names[intercept_idx]
            p_idxs = np.delete(idxs, intercept_idx)

        return p_idxs, p_names, add_constant

    for j in range(len(sort_idxs)):
        # select subset of j largest predictors with the
        # largest absolute Wald statistics
        p_idxs, p_names, add_constant = pn_idxs_itc(sort_idxs[:j+1])
        p_indices.append(p_idxs)
        # Initialize linear model
        zl_lm = LinearModel(input_names=p_names,
                            add_constant=add_constant)

        zl_models.append(zl_lm)
        # Fit linear model
        zl_y_hat = zl_lm.fit_predict(x[:, p_idxs], y)
        # Fit linear model

        # Zheng-Loh prediction risk
        risk[j] += np.sum((y - zl_y_hat) ** 2)

    brix = np.argmin(risk)
    best_model = zl_models[brix]
    bp_idxs = p_indices[brix]
    zl_risk = risk[brix]

    if return_model_risks:
        risk_info = dict([(i, (r, zl.input_names))
                          for i, (r, zl) in enumerate(zip(risk, zl_models))])
        return best_model, bp_idxs, zl_risk, risk_info
    else:
        return best_model, bp_idxs

def _compute_se_variance_linear_model(x, y, y_hat):
    """
    x is suposed to be 2D and have an intercept if the model has an intercept
    """
    # Compute residuals
    residuals = y - y_hat
    # sum of square residuals
    rss = np.sum(residuals ** 2)
    n, k = x.shape
    # estimator variance
    var_est = (1  / (n - k) ) * rss
    # Compute covariance matrix
    xtx = np.dot(x.T, x)
    ixtx = np.linalg.pinv(xtx)
    # covariance matrix
    cov_est = var_est * ixtx
    # standard error
    se = np.sqrt(np.diag(cov_est))
    return se, var_est
    

def coefficient_of_determination(y, y_hat, p=None):
    """
    Coefficient of determination
    
    Parameters
    ----------
    y : np.ndarray
        Target variable
    y_hat : np.ndarray
        Predicted variable
    p : None or int (optional)
        number of explanatory terms (for adjusted R2). Default is None

    Returns
    -------
    R2 : float
        Coefficient of determination
    """
    y_bar = y.mean()
    sstot = np.sum((y - y_bar) ** 2)
    ssres = np.sum((y - y_hat) ** 2)
    R2 = 1 - ssres / sstot

    # Compute adjuted R2
    if p is not None:
        n = len(y)
        R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
    return R2

# Alias
cod = coefficient_of_determination

def cohens_f2_from_R2(R2):
    return R2 / (1 - R2)

def cohens_f2_from_data(y, y_hat, p=None):
    R2 = cod(y, y_hat, p=p)
    return cohens_f2_from_R2(R2)

def correlation(y, y_hat):
    return np.corrcoef(y, y_hat)[0, 1]

def standardize(x):
    return (x - x.mean(0)) / np.clip(x.std(0, keepdims=True), 1e-6, None)

def parameter_stats(x):
    mean = x.mean()
    std = x.std()
    kurtosis = stats.kurtosis(x)
    skewness = stats.skew(x)
    return mean, std, kurtosis, skewness


def example_LinearModel():
    x1 = np.linspace(0, 1)
    x2 = np.random.randn(len(x1))

    x = np.column_stack((x1, x2))

    # y does not depend on x2
    y = 3 * x1 + 0.1 * np.random.randn(len(x1))

    lm = LinearModel(input_names=['x1', 'x2'])

    statistics, y_hat = lm.test(x, y, hypothesis_type=['gt', 'equal'], return_preds=True)

    for s in statistics:
        print(s, s.reject_h0)

def example_zheng_loh():
    x1 = np.linspace(0, 1)
    x2 = np.random.randn(len(x1))
    x3 = np.random.randn(len(x1))
    x4 = np.random.randn(len(x3))
    x5 = np.sin(np.linspace(0, 2 * np.pi, num=len(x1)))
    x6 = np.cos(np.linspace(0, 2 * np.pi, num=len(x1)))

    x = np.column_stack((x1, x2, x3, x4, x5, x6))

    # y depends linearly on x1 and x5
    y = 3 * x1 + + 1.1 * x5 + 2 * x6 + 0.1 * np.random.randn(len(x1))

    # Model with all predictors
    lm = LinearModel(input_names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6'])
    # fit model
    y_hat =  lm.fit_predict(x, y)
    x_intercept = lm._add_intercept(ensure2d(x))
    params = lm.params
    n = len(y)
    se, var = _compute_se_variance_linear_model(x=x_intercept,
                                              y=y,
                                              y_hat=y_hat)
    # Compute Wald statistic (H0 : param == 0, H1 : param != 0)
    statistic = params / se

    # Sort statistics descendingly
    sort_idxs = np.argsort(abs(statistic))[::-1]
    # Initialize Zheng-Loh prediction risk
    risk = np.arange(1, len(sort_idxs) + 1) * var * np.log(n)
    for j in range(len(sort_idxs)):
        # select subset of j largest predictors with the
        # largest absolute Wald statistics
        predictors = x_intercept[:, sort_idxs[:j+1]]
        # Fit linear model
        zl_y_hat = LinearModel(add_constant=False).fit_predict(predictors, y)
        # Zheng-Loh prediction risk
        risk[j] += np.sum((y - zl_y_hat) ** 2)

    bp_idxs = sort_idxs[:np.argmin(risk) + 1]
    best_predictors = [lm.input_names[i] for i in bp_idxs]
    add_constant = False
    if 'intercept' in best_predictors:
        intercept_idx = best_predictors.index('intercept')
        del best_predictors[intercept_idx]
        add_constant = True
        bp_idxs = np.delete(bp_idxs, intercept_idx)
        
    best_model = LinearModel(input_names=best_predictors,
                             add_constant=add_constant)

    y_hat_best = best_model.fit_predict(x[:, bp_idxs], y)

    best_model2, bp_idxs2= zheng_loh_model_selection(x, y, input_names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6'])


def compute_F_statistic_and_pvalue(*args):
    """ 
    Return F statistic an p-value
    """
    # Compute degrees of freedom
    df_btwn, df_within = __degree_of_freedom_(*args)

    # Compute sums of squares
    mss_btwn = __ss_between_(*args) / float(df_btwn)
    mss_within = __ss_within_(*args) / float(df_within)

    # F statistic
    F = mss_btwn / mss_within
    
    pvalue = special.fdtrc(df_btwn, df_within, F)

    return (F, pvalue, df_btwn, df_within)


def compute_etasquared(*args):
    """ Return the eta squared as the effect size for ANOVA

    """
    return(float(__ss_between_(*args) / __ss_total_(*args)))


def __concentrate_(*args):
    """ Concentrate input list-like arrays

    """
    v = list(map(np.asarray, args))
    vec = np.hstack(np.concatenate(v))
    return(vec)


def __ss_total_(*args):
    """ Return total of sum of square

    """
    vec = __concentrate_(*args)
    ss_total = sum((vec - np.mean(vec)) ** 2)
    return(ss_total)


def __ss_between_(*args):
    """ Return between-subject sum of squares

    """
    # grand mean
    grand_mean = np.mean(__concentrate_(*args))

    ss_btwn = 0
    for a in args:
        ss_btwn += (len(a) * (np.mean(a) - grand_mean) ** 2)

    return(ss_btwn)


def __ss_within_(*args):
    """
    Return within-subject sum of squares

    """
    return(__ss_total_(*args) - __ss_between_(*args))


def __degree_of_freedom_(*args):
    """Return degree of freedom

       Output-
              Between-subject dof, within-subject dof
    """
    args = list(map(np.asarray, args))
    # number of groups minus 1
    df_btwn = len(args) - 1

    # total number of samples minus number of groups
    df_within = len(__concentrate_(*args)) - df_btwn - 1

    return(df_btwn, df_within)


def f_oneway(*args, **kwargs):
    """
    One-way ANOVA
    """
    statistic, pvalue, df_btwn, df_within = compute_F_statistic_and_pvalue(*args)
    etasquared = compute_etasquared(*args)

    significance_level = kwargs.pop('significance_level', 0.05)

    return F_onewayResult(statistic=statistic,
                          pvalue=pvalue,
                          etasquared=etasquared,
                          df_btwn=df_btwn,
                          df_within=df_within,
                          significance_level=significance_level,
                          **kwargs)


if __name__ == '__main__':


    # Example from scipy
    tillamook = [0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735,
                 0.0659, 0.0923, 0.0836]
    newport = [0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835,
               0.0725]
    petersburg = [0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]
    magadan = [0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764,
               0.0689]
    tvarminne = [0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]

    a = stats.f_oneway(tillamook, newport, petersburg, magadan, tvarminne)
    expected = (7.1210194716424473, 0.00028122423145345439)

    ap = compute_F_statistic_and_pvalue(tillamook, newport, petersburg, magadan, tvarminne)
    eta = compute_etasquared(tillamook, newport, petersburg, magadan, tvarminne)

    aap = f_oneway(tillamook, newport, petersburg, magadan, tvarminne)
    # print(np.isclose(a, ap[:1]))

        
        

    
