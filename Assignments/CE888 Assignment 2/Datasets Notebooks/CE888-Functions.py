import numpy as np

def abs_ate(effect_true, effect_pred):
    """
    Absolute error for the Average Treatment Effect (ATE)
    :param effect_true: true treatment effect value
    :param effect_pred: predicted treatment effect value
    :return: absolute error on ATE
    """
    return abs(np.mean(effect_pred) - np.mean(effect_true))


def pehe(effect_true, effect_pred):
    """
    Precision in Estimating the Heterogeneous Treatment Effect (PEHE)
    :param effect_true: true treatment effect value
    :param effect_pred: predicted treatment effect value
    :return: PEHE
    """
    return np.sqrt(np.mean(np.power((np.array(effect_true)-np.array(effect_pred)),2)))

def get_ps_weights(clf, x, t):
    ti = np.squeeze(t)
    clf.fit(x, ti)
    ptx = clf.predict_proba(x).T[1].T + 0.0001 # add a small value to avoid dividing by 0
    # Given ti and ptx values, compute the weights wi (see formula above):
    wi = (ti/ptx)+((1-ti)/(1-ptx))
    
    return wi

def mean_ci(data, ci=0.95):
    l_mean = np.mean(data)
    lower, upper = st.t.interval(ci, len(data)-1, loc=l_mean, scale=st.sem(data))
    return l_mean, lower, upper
