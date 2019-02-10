

def approximate_max_bivariate(one,other, bivar_ufunc,initial_base):
    initial_result = bivar_ufunc(one[::initial_base],other::initial_base)
