# Crash Course on Regression Analysis

## Concept 1: Linear Regression Basics

Linear regression is a statistical algorithm we use to model the relationship between a dependent variable *y* and one or more independent variables *x*

### Singular Linear Regression
For a single independent variable, the model is:

\[y = β_0 + β_1x_1 + \epsilon\]

Where:
- \( β_0 \) = the intercept of the regression line.
- \( β_1 \) = the slope of the regression line.
- \( \epsilon \) = the error term capturing deviations from the model.

### Multiple Linear Regression
For multiple independent variables, the model generalizes to:

\[y = β_0 + β_1x_1 + β_2x_2 + \dots + β_nx_n + \epsilon\]

Here, \( x_1, x_2, \dots, x_n \) are the predictors, and \( β_1, β_2, \dots, β_n \) are their corresponding coefficients.

---

## Concept 2: Estimating Coefficients in Linear Regression

---

## Concept 3: Non Linear Functions

---

## Concept 4: The Error Term (\( \epsilon \))

### Error Expectations on Linear functions
In a typical linear regression on a linear function, the error term \( \epsilon \) is assumed to satisfy 2 conditions:

1. **Expected Value**: The expected value of \( \epsilon \) is zero (\( E[\epsilon] = 0 \)).
    -- This is because \( E[\epsilon] =  mean(\epsilon) = 0 \)
2. **Normality**: The error term is normally distributied (\( \epsilon \sim N(0, \sigma^2) \)).

### Error from Regression on Different Functions
**For \( y = x \)**:
   - The error \( \epsilon \) has \( E[\epsilon] = 0 \) since the model perfectly fits \( y = x \) (thus the mean evaluates to 0)

**However for \( y = x + \sin(x) \)**:
   - The expected value of the error term \( \epsilon \) is \( E[\epsilon] = \sin(x) \), reflecting the sinusoidal component not captured by the linear model.
   - How do we reduce the error further? By applying a regression on the error term

---



## Summary of Regression Analysis
