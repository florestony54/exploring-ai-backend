---
layout: post
current: post
cover:  assets/images/a4.JPG
navigation: True
title: Multiple Linear Regression
date: 2022-02-20 15:00:00
tags: [Getting started]
class: post-template
subclass: 'post'
author: tony
---

The following is an excerpt from "An Introduction to Statistical Learning" by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (temporary filler content):

### 3.2 Multiple Linear Regression

Simple linear regression is a useful approach for predicting a response on the
basis of a single predictor variable. However, in practice we often have more
than one predictor. For example, in the Advertising data, we have examined
the relationship between sales and TV advertising. We also have data for
the amount of money spent advertising on the radio and in newspapers,
and we may want to know whether either of these two media is associated
with sales. How can we extend our analysis of the advertising data in order
to accommodate these two additional predictors?

One option is to run three separate simple linear regressions, each of
which uses a different advertising medium as a predictor. For instance,
we can fit a simple linear regression to predict sales on the basis of the
amount spent on radio advertisements. Results are shown in Table 3.3 (top
table). We find that a $1,000 increase in spending on radio advertising is
associated with an increase in sales by around 203 units. Table 3.3 (bottom
table) contains the least squares coefficients for a simple linear regression of
sales onto newspaper advertising budget. A $1,000 increase in newspaper
advertising budget is associated with an increase in sales by approximately
55 units.

However, the approach of fitting a separate simple linear regression model
for each predictor is not entirely satisfactory. First of all, it is unclear how to
make a single prediction of sales given levels of the three advertising media
budgets, since each of the budgets is associated with a separate regression
equation. Second, each of the three regression equations ignores the other
two media in forming estimates for the regression coefficients. We will see
shortly that if the media budgets are correlated with each other in the 200
markets that constitute our data set, then this can lead to very misleading
estimates of the individual media effects on sales.

Instead of fitting a separate simple linear regression model for each predictor, a better approach is to extend the simple linear regression model
(3.5) so that it can directly accommodate multiple predictors. We can do
this by giving each predictor a separate slope coefficient in a single model.
In general, suppose that we have p distinct predictors. Then the multiple
linear regression model takes the form:

>Y = β0 + β1X1 + β2X2 + ··· + βpXp + e, (3.19)

where Xj represents the jth predictor and βj quantifies the association
between that variable and the response. We interpret βj as the average
effect on Y of a one unit increase in Xj , holding all other predictors fixed.
In the advertising example, (3.19) becomes:

>sales = β0 + β1 × TV + β2 × radio + β3 × newspaper + e. (3.20)