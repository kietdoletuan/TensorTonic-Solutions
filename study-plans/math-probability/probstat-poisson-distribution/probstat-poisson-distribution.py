from math import exp, factorial

def poisson_distribution(lam: float, max_k: int) -> list:
    """
    Returns the PMF list, cumulative probability, and zero-event probability.
    """
    PMF_0 = 0
    PMF_k = []
    for i in range(max_k+1):
        prob = (lam**i * exp(-lam)) / factorial(i)
        if i == 0:
            PMF_0 = prob
        PMF_k.append(prob)

    return [PMF_k, sum(PMF_k) ,PMF_0]
        
