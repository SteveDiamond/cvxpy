def dgp_wrap_canon(expr, args):
    """DGP wrapped functions are 'unwrapped' by dgp2dcp.
    """
    func = expr.function
    del expr
    print(args[0].value)
    return func(*args), []
