def unsupervised_distr(distr):
    variables = {k: k + '_u' for k in distr.var + distr.cond_var if k != 'z'}
    distr_unsupervised = distr.replace_var(**variables)
    return distr_unsupervised, variables


def unsupervised_distr_no_var(distr):
    variables = {k: k + '_u' for k in distr.var + distr.cond_var if k != 'z'}
    distr_unsupervised = distr.replace_var(**variables)
    return distr_unsupervised
