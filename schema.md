# Plotfinder Regression Schema
### this is an example of how regression should be output:

``` Python
def create_regression_output(sin_terms, exp_terms, log_terms, poly_terms):
    """
    makes a structured output for a regression script of any function to return

    params:
        sin_terms (list of tuples): Each tuple contains (amplitude, frequency, phase)
        exp_terms (list of tuples): Each tuple contains (coefficient, base)
        log_terms (list of tuples): Each tuple contains (coefficient, base).

        poly_terms (dictionary): Keys are polynomial powers (int), values are coefficints (float)

    returns:
        dict: Structured regression output.

    """

    return {
        "sin_terms": sin_terms,             # list of (amplitude, frequency, phase)
        "exponential_terms": exp_terms,     # list of (coefficient, base)
        "logarithmic_terms": log_terms,     # list of (coefficient, base)
        "polynomial_terms": poly_terms      # dictionary {power: coefficient}
    }

sin_terms_example = [(A1, f1, p1), (A2, f2, p2)]
exp_terms_example = [(C1, B1), (C2, B2)]
log_terms_example = [(L1, logB1), (L2, logB2)]
poly_terms_example = {0: C0, 1: C1, 2: C2, 3: C3, n: Cn}

regression_output = create_regression_output(
    sin_terms_example,
    exp_terms_example,
    log_terms_example,
    poly_terms_example
)

for key, value in regression_output.items():
    print(f"{key}: {value}")
```
