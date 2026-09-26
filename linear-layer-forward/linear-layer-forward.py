def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    result = []
    samples = len(X)
    _in = len(X[0])
    _out = len(W[0])
    for n in range(samples):
        sum_list = []
        for d_out in range(_out):
            sum = 0
            for d_in in range(_in):
                sum += X[n][d_in] * W[d_in][d_out]
            sum += b[d_out]
            sum_list.append(sum)
        result.append(sum_list)
    print(result)
    return result
    pass