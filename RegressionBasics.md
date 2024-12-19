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


### Slope and Intercept
The Linear Regression Algorithm estimates coefficients using the **least squares method**, minimizing the sum of squared errors ( \( \epsilon \)^2 + \( \epsilon \)^2 + \( \epsilon \)^2 ) between real & predicted values. Since we only have a data set & not a formula, we cannot get the real value of the slope \( β_1 \) - but we can obtain the estimated value \( \hatβ_1 \) (pronounced beta hat 1)

#### Key Calculations

1. **Sum of Deviations (\( S_x \)):**
   \[
   S_x = \sum (x_i - \bar{x})
   \]
   - \( \bar{x} \) being the mean of \( x \).
   - & \( x_i \) being the value of the i'th index of x (remember x is a set of the x values of our dataset & not a continuous domain)


2. **Sum of Deviations Squared (\( S_{xx} \)):**
   \[
   S_{xx} = \sum (x_i - \bar{x})^2
   \]

3. **Estimated Slope (\( \hat{β}_1 \)):**
   \[
   \hat{β}_1 = \frac{S_{xy}}{S_{xx}}
   \]
   ( where \( S_{xy} = \sum (x_i - \bar{x})(y_i - \bar{y}) \) )

4. **Estimated Intercept (\( \hat{β}_0 \)):**
   \[
   \hat{β}_0 = \bar{y} - \hat{β}_1\bar{x}
   \]

5. **Overall (\( \hat{y}  =  \hat{β}_0  +  \hat{β}_1*x_1   \)):**
   \[
   \hat{y}_i  = b + mx_i =  \hat{β}_0  +  \hat{β}_1*x_i  =  \bar{y} - \hat{β}_1\bar{x} + \hat{β}_1*x_i
   \]

   \[
   = \bar{y} - \frac{S_{xy}}{S_{xx}}\bar{x} + \frac{S_{xy}}{S_{xx}}*x_i
   \]

   \[
   = \bar{y} - \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}\bar{x} + \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}*x_i
   \]

   \[
   = \hat{y}_i  = \bar{y} + \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}(x_i - \bar{x})
   \]
& this should be all you need to implement your very own linear regression

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
