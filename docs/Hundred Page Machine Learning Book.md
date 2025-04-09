# Hundred Page Machine Learning Book

## Machine Learning

Learning can be supervised, semi-supervised, unsupervised and reinforcement.

### Supervised Learning

The dataset is a collection of labeled examples. Each element of the set is called a feature vector. Each value in this vector is called a feature.

The label can either be an element belonging to a finite set of classes, or a real number.

The goal of a supervised learning algorithm is to use the dataset to produce a model that takes a feature vector as input and outputs information that allows deducing its label.

The two main supervised learning problems are **regression** and **classification**.

Most supervised learning algorithms such as **Support Vector Machine** are **model-based** and use the training data to create a model that has parameters learned from this data. Traning data can be discarded after a model has been built.

**Instance-based** learning algorithms such as **k-Nearest Neighbors** use the whole dataset as the model.

<!-- #### Classification

Classification is a problem of assigning a label to an unlabeled example. A classification learning algorithm takes a collection of labeled examples as inputs and produces a model that can take an unlabeled example as input and outputs a label.

A label is a member of a finite set of classes and the classification can be either binary or multiclass (binomial or multinomial). -->

### Unsupervised Learning

The dataset is a collection of unlabeled examples. The goal an unsupervised learning algorithm is to create a model that takes a feature vector as input and either transforms it into another vector or into a value.

In clustering, the model returns the id of the cluster for each feature vector.
In dimensionality reduction, the model returns a feature vector that has fewer features than the input vector.
In outlier detection, the output is a real number that indicates how a feature vector is different from a "typical" example in the dataset.

### Semi-Supervised Learning

The dataset contains both labeled and unlabeled examples, where the quantity of unlabeled examples is much higher than the number of labeled examples. The goal of a semi-supervised learning algorithm is the same as the goal of the supervised learning algorithm.

Using many unlabeled examples may help the learning algorithm to find a better model.

### Reinforcement Learning

Subfield of machine learning where the machine is constantly active and capable of perceiving the state of an environment as feature vectors. It can execute actions in every state that bring different rewards and can also move the machine to other states of the environment.

The goals of a reinforcement learning algorithm is to learn a policy, which is a function that takes the feature vector of a state as input and outputs an optimal action to execute in that state. The action is optimal if it maximizes the expected average reward.

### Summary

Any classification learning algorithm that builds a model implicity or explicitly creates a decision boundary. The form of the decision boundary determines the accuracy of the model, and it is this form and the way it is computed what differentiates one learning algorithm from another.

## Definitions

### Function

A function is a relation that associates each element x of a set X, the domain of the function, to a single element y of another set Y, the codomain of the function.

A function has a local minimum at $x = c$ if $f(x) \geq f(c)$ for every $x$ in some open interval around $x = c$. The minimum value among all local minima is the global minimum.

### $\max$ vs. $\argmax$

Operator $\max_{a\in\cal{A}}(f(a))$ returns the highest value for all elements in set $\cal{A}$.

However, operator $\argmax_{a\in\cal{A}}(f(a))$ returns the element of set $\cal{A}$ that maximizes $f(a)$.

### Derivative

A derivative $f'$ of a function $f$ is a function or value that describes how fast $f$ grows (or decreases).

If the derivative is a constant, the the function increases or decreases constantly at any point of its domain. If $f'$ is a function, then $f$ can grow at a different pace in different regions of its domain.

If the derivative $f'$ is positive at some point, then the function grows. Conversely, if the derivative is negative at some point, then it decreases. If the derivative is zero then the function's slope is horizontal at that point.

The chain rule is applied to differentiate nested functions. If $F(x) = f(g(x))$, then $F'(x) = f'(g(x))g'(x)$.

A **gradient** of a function is a vector of **partial derivatives**. A partial derivative is taken by finding the derivative and focusing on one of the function's inputs while treating the others as constants.

### Random Variables

Random variables can be discrete or continuous.

A **discrete random variable** takes on only a countable number of distinct values and its probability distribution is described by a **probability mass function** (pmf). The sum of probabilities equals 1.

A **continuous random variable** takes an infinite number of possible values in some interval. Its probability distribution is described by a **probability density function** (pdf), a function whose codomain is non-negative and its area is equal to 1.

The **expectation** of a discrete random variable is determined by the probabilities according to the pmf.

$\mathbb{E}[X] = \sum_{i=1}^k [x_i \cdot \Pr(X=x_i)]$

It is also called the **mean**, **average**, or **expected value** and denoted by $\mu$.

The expectation of a continuous random variable $X$ is a weighted average of all possible values of $X$, where each value is weighted by its probability density function. It is an integral that essentially sums up all the possible values of $X$ multiplied by their respective probabilities, resulting in the expectation.

$\mathbb{E}[X] = \int_\mathbb{R} x f_X(x) dx$

An integral is an equivalent of the summation over all values of the function when the function has continuous domain. It equals the area under the curve of the function (which for a pdf equals 1).

### Unbiased Estimator

An unbiased estimator is a statistic used to estimate a parameter in a statistical model. Specifically, if the expected value of the estimator equals the true value of the parameter being estimated, regardless of the sample size or any other factors, then the estimator is said to be unbiased.

Unbiased estimators provide, on average, accurate estimates of the parameters they're trying to estimate. Or, in other words, the average value of the estimator over all possible samples equals the true value of the parameter.

The sample mean is a measure of central tendency calculated as the average of a set of observations or data points. It is the sum of all the values in the sample and then divided by the total number of observations. This provides an estimate of the population mean when the sample is representative of the population from which it is drawn.

### Parameter Estimator

The Gaussian function can be used as a model of an unknown distribution of a random variable because it has all the properties of a probability function; this is the Gaussian or normal distribution.

Bayes' Rule can be applied iteratively to estimate the probability that a finite set of possible values takes on a certain random value. The initial guess of the probabilities for different values is called the prior.

The best value of the parameters given one example is obtained by using the principle of maximum a posteriori (MAP).

The principle of maximum a posteriori (MAP) estimation is a method used in Bayesian statistics to estimate the most probable value of an unknown parameter in a statistical model, given observed data.

In Bayesian statistics, a prior probability distribution describes prior knowledge about a parameter. Observed data may then be used to update our belief about the parameter, resulting in a posterior probability distribution. The MAP estimation seeks to find the value of the parameter that maximizes the posterior probability.

$\hat{\theta}_{MAP} = \argmax_\theta P(\theta|x)$

If the set of possible values for the parameters is not finite, then it needs to be optimized using a numerical optimization routine such as gradient descent. However, the natural logarithm of the right-hand side expression in MAP is usually optimized because the logarithm of a product becomes the sum of logarithms.

## Regression

Regression is a problem of predicting a real-valued label (target) given an unlabeled example. It is solved by a regression learning algorithm that takes a collection of labeled examples as inputs and produces a model that can take an unlabeled example as input and outputs a target.

### Linear Regression

Linear regression is a popular regression learning algorithm that learns a model which is a linear combination of features of the input example.

We want to build a model $f_{w,b} = wx+b$, where $w$ is a $D$-dimensional vector of parameters and $b$ is a real number.

The difference between the form of the linear model and the form of a Support Vector Machine is the missing sign operator.

Whereas the hyperplane in the SVM plays the role of the decision boundary, in linear regression the requirement is to choose it to be as close to all training examples as possible. The optimization procedure tries to minimize the following **objective** or **cost function**:

$\frac{1}{n} \sum_{i=1}^n (f_{w,b}(x_i) - y_i)^2$, where $n$ is the number of data points in the data set, and $y_i$ is the predicted value.

The $(f_{w,b}(x_i) - y_i)^2$ term is the **loss function**, more specifically **mean squared error loss**. In linear regression, the cost function is given by the average loss, also called **empirical risk**, an it is the average of all penalties obtained by applying the model to the training data.

Linear models rarely suffer from **overfitting**. The sum of squared errors is convenient because it has a continuous derivative, making the function smooth and thus easier to solve using an analytical optimization method.

### Logistic Regression

**Logistic regression** is actually not a regression but a classification learning algorithm. The misnomer is due to the fact that the mathematical formulation of logistic regression is similar to that of linear regression.

In **binary logistic regression** the negative label can be defined as 0 and the positive label as 1. One such continuous (activation) function whose codomain is (0,1) is the standard **logistic function**, or **sigmoid function**:

$f(x) = \frac{1}{1+\exp(-x)}$

The logistic regression model then looks like this:

$f_{w,b}(x) = \frac{1}{1+\exp(-(wx+b))}$.

The optimization criterion in logistic regression is called **maximum likelihood**. Ti maximize the likelihood of the training data according to a model:

$L_{w,b} = \prod_{i=1}^n f_{w,b}(x_i)^{y_i} (1-f_{w,b}(x_i)^{1-y_i})$.

Or:
$\begin{aligned}
~~  f_{w,b}(x) &~~~ \text{if} ~~ y_i = 1\\
~~ 1-f_{w,b}(x) &~~~ \text{if} ~~ y_i \neq 1
\end{aligned}$

The likelihood of observing $n$ labels for $n$ examples is the product of likelihoods of each observation.

However, it is more convenient to maximize the **log-likelihood** to avoid numerical overflow caused by the $\exp$ term used in the model:

$\begin{aligned}
LogL_{w,b} = \ln(L_{w,b}(x)) &= \sum_{i=1}^n [y_i \ln(f_{w,b}(x)) + (1-y_i)\ln(1-f_{w,b}(x))]
\end{aligned}$

This is a **strictly increasing function**, and thus maximizing this is the same as maximizing its argument.

Contrary to linear regression, there is no closed form solution to this optimization problem, so we have to use **numerical optimization** like **gradient descent**.

## Decision Tree

A **decision tree** is an **acyclic graph**, where a specific feature of the feature vector is examined in each branching node, down to a leaf node where a certain class is predicted. The model is described as:

$\frac{1}{n}\sum_{i=1}^n [y_i \ln(f_{ID3}(x_i) + (1-y_i) \ln(1-f_{ID3}(x_i)))]$,

where $f_{ID3}(x) = \Pr(y=1|x)$ is a **non-parametric model** of a decision tree. The formula is similar to logistic regression, but the difference is that the **ID3 (Iterative Dichotomiser 3)** learning algorithm makes an approximate optimization that works by recursively partitioning the dataset into subsets based on the values of input features, with the goal of maximizing the information gain at each step.

At each iteration, a feature that best divides the dataset into homogeneous subsets is selected using a measure like **entropy** or **Gini impurity**, and the dataset is split into subsets (branches) based on the possible values of that feature. With each split, ID3 constructs a decision tree where each node represents a decision, and each leaf represents a class label. The splitting process continues for smaller and smaller subsets until one of the stopping criterium is met:

1. Maximum depth is reached.
2. Cannot find a feature to split on.
3. All examples in the leaf node have the same classification.

The following **entropy** criterion, which is essentially a weighted sum of two entropies, is minimized during each split:

$H(S_-,S_+) = \frac{|S_-|}{S}H(S_-) + \frac{|S_+|}{S}H(S_+)$,

where $S$ is a set of examples, $|S|$ is its magnitude,

$
S_- = {(x,y)|(x,y) \in S, ~~ x^(j) \lt t} \text{, ~ and} \\
S_+ = {(x,y)|(x,y) \in S, ~~ x^(j) \geq t}
$,

and entropy $H(S)$ is defined as:

$H(S) = -f_{ID3} \ln(f_{ID3}) - (1 - f_{ID3}) \ln(1-f_{ID3})$.

This entropy-based split criterion reaches its minimum of 0 when all examples in $S$ have the sample label. On the other hand, the entropy is at its maximum of 1 when exactly one-half of examples in $S$ are labeled with 1, making such a leaf useless for classification.

ID3 has a tendency to create decision trees that suffer from overfitting because each split depends only on the local feature in the current iteration.

An ID3 variant called **C4.5** improves on ID3 by accepting both continuous and discrete features, handling incomplete examples, and solving overfitting by applying **pruning**. Pruning consists of going back through the tree once it has been created and removing branches that do not contribute significantly enough to the error reduction.

## Support Vector Machine

A **Support Vector Machine (SVM)** sees every feature vector as a point in a high-dimensional space, and determines a decision boundary (a hyperplane) that separates examples with positive labels from examples with negative labels.

The equation of the hyperplane is given by two parameters: a real-valued vector $w$ of the same dimension as an input feature vector $x$, and a real number $b$, giving: $wx - b = 0$.

The predicted label for some input feature vector x is given as:

$y = sign(wx - b)$.

The goal of the learning algorithm is to leverage the dataset and find the optimal values $w^\star$ and $b^\star$, so that a model $f(x)$ can be defined:

$f(x) = sign(w^\star x - b^\star)$.

The machine can find the optimal parameters by solving an optimization problem with the following constraints:

$
wx_i - b \geq +1  ~~~ \text{if} ~~ y_i = +1 \\
wx_i - b \leq -1  ~~~ \text{if} ~~ y_i = -1 \\
$

$\Rightarrow y_i(wx_i - b) >= 1$

The optimization problem becomes:

$\text{Minimize } \|w\| \text{ subject to } y_i(wx_i - b) \geq 1 \text{ for } i = 1,...,n$,

where $\|w\|$ is the magnitude of $w$, \
and $n$ is the number of examples.

Minimizing $\|w\|$ is equivalent to minimizing $\frac{1}{2}\|w\|^2$, leading to the following optimization problem for SVM:

$\min(\frac{1}{2}\|w\|^2), \text{ such that } y_i(wx_i-b)-1\geq 0, ~i=1,...,n$.

Traditionally this problem is solved with the method of Lagrange multipliers. It is convenient to solve an equivalent problem formulated thus:

$\max_{\alpha_1,...,\alpha_n} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{k=1}^n y_i\alpha_i(x_i k_i)y_k\alpha_k$,

subject to $\sum_{i=1}^n \alpha_i y_i = 0 \text{ and } \alpha_i \geq 0, ~ i=1,...,n$,

where $\alpha_i$ are called Lagrange multipliers. Formulated like this, the optimization problem becomes a convex quadratic optimization problem.

### Noise

In cases where the data is not linearly separable, a **hinge loss** function is introduced:

$\max(0, 1-y_i(wx_i-b))$.

This function is zero if the constraints given above are satisfied (i.e. if $wx_i$ lies on the correct side of the decision boundary). For the other points the function's value is proportional to the distance from the decision boundary.

The following cost function must then be minimized:

$$C\|w\|^2 + \frac{1}{n}\sum_{i=1}^n\max(0,1-y_i(wx_i-b)),$$

where $C$ is a hyperparameter that determines the tradeoff between increasing the size of the decision boundary and ensuring that each $x_i$ lies on the correct side of it. It regulates the tradeoff between classifying the training data well (minimizing emperical risk, when $C$ is decreased) and classifying future examples well (generalization, when $C$ is increased).

### Inherent Non-Linearity

When the data is not inherently linearly separable it may be possible to so in a space of higher dimensionality. Using a function to implicitly transform the original space into a higher one during the cost function optimization is called the **kernel trick**.

With the kernel trick it is possible to get rid of costly transformations of original feature vectors into higher-dimensional vectors before computing their dot-product. The trick is to replacing that with a simple operation on the original feature vectors that gives the same result.

**Kernel functions** (or **kernels**) can be used to efficiently work in higher-dimensional spaces without doing an explicity transformation mapping.

Multiple kernel functions exist, the most widely used of with is the **Radial Basis Function (RBF) kernel**:

$$k(x,x') = \exp(-\frac{\|x-x'\|^2}{2\sigma^2}),$$

where $\|x-x'\|^2$ is the squared **Euclidean distance** between two feature vectors, which is given by the following equation:

$$d(x_i,x_k) = \sqrt{\sum_{j=1}^D(x_i^{(j)}-x_k^{(j)})^2}.$$

The feature space of the RBF kernel has an infinite number of dimensions. Hyperparameter $\sigma$ can be adjusted to choose between a smooth or curvy decision boundary in the original space.

## k-Nearest Neighbors

**k-Nearest Neights (kNN)** is a non-parameteric learning algorithm that finds $k$ training example closest to input example $x$ and returns the majority label (classification) or average label (regression).

The closeness of two examples is given by a distance function such as Euclidean distance above or other popular distance metrics, such as: cosine similarity, Chebychev distance, Mahalanobis distance, or Hamming distance.

Negative **cosine similarity** is a measure of similarity of the directions of two vectors:

$$s(x_i,x_k) = cos(\angle(x_i,x_k))= \frac{\sum_{j=1}^Dx_i^{(j)}x_k^{(j)}}{\sqrt{\sum_{j=1}^{(j)}(x_i^{(j)})^2} \sqrt{\sum_{j=1}^{(j)}(x_k^{(j)})^2}}.$$

- If the angle between two vectors is $0\degree$, the two vectors point in the same direction and cosine similarity is 1.
- If two vectors are orthogonal, the cosine similarity is 0.
- For vectors pointing in opposite directions, the cosine similarity is -1.

## Learning Algorithms

The building blocks of any learning algorithm are:

1. A loss function.
2. An optimization criterion based on the loss function (such as a cost function).
3. An optimization routine leveraging training data to find a solution to the optimization criterion.

Some algorithms are designed to explicitly optimize a specific criterion (linear regression, logistic regression, support vector machine). Others optimize the criterion implicitly (decision tree learning, kNN).

### Gradient Descent

**Gradient descent** or **stochastic gradient descent** are the two most frequently used optimization algorithms used in cases where the optimization criterion is differentiable.

Gradient descent is an iterative optimization algorithm for finding the minimum of a function by starting at a random point and taking steps proportional to the negative of the gradient of the function at the current point.

In the case of a linear regression problem, we need to find values for $w$ and $b$ that minimize the error (mean squared error in this case):

$$l=\frac{1}{n}\sum_{i=1}^n(y_i-(wx_i+b))^2.$$

Gradient descent starts with calculating the partial derivative for each parameter:

$$\frac{\partial l}{\partial w} = \frac{1}{n}\sum_{i=1}^n -2x_i(y_i-(wx_i+b))$$
$$\frac{\partial l}{\partial b} = \frac{1}{n}\sum_{i=1}^n -2(y_i-(wx_i+b)).$$

Gradient descent proceeds in **epochs**; one epoch consists of using the training set in its entirety to update each parameter. The learning rate $\alpha$ controls the size of an update:

$$w \leftarrow w-\alpha\frac{\partial l}{\partial w},$$
$$b \leftarrow b-\alpha\frac{\partial l}{\partial b}.$$

The partial derivatives are subtracted from the parameter values because derivatives are indicators of growth of a function.

In the next epoch, the partial derivates are recalculated with the updated values of $w$ and $b$, and this process is continued until convergence (which usually takes many epochs).

Gradient descent is sensitive to the choice of learning rate $\alpha$ and is slow for large datasets. That's why several significant improvements have been proposed.

**Minibatch stochastic gradient descent** is a version that uses smaller batches (subsets) of the training data. **Adagrad** is a version of SGD that scales $\alpha$ for each parameter proportionally to the historical size of  the gradients. **Momentum** helps accelerate SGD by orienting the gradient descent in the relevant direction and reducing oscillations. In neural networks, SGD variants like **RMSProp** and **Adam** are frequently used.

Gradient descent and its variants are not machine learning algorithms; they are solvers of minimization problems in which the function to minimize has a gradient.

## Practical Considerations

### Feature Engineering

A **dataset** is a collection of **labeled examples**, where each element is called a **feature vector** in which each dimension contains a **feature** (value) that the describes the example somehow.

**Feature engineering** refers to the problem of transforming raw data into a dataset. It is a labor-intensive process that demands creativity and domain knowledge.

Any measurement can be used as a feature. The goal is to create **informative features** (or features with high **predictive power**) that allow a learning algorithm to build a model that does a good job of predicting labels of the data used for training.

A model has a **low bias** when it predicts the training data well and makes few mistakes when verifying with the examples.

#### **One-Hot Encoding**

When transforming categorical features into binary ones, increase the dimensionality of the feature vectors via **one-hot encoding**. Do not transform each category into a single digit (eg. 1, 2, 3) to avoid increasing dimensionality because this implies an order. However, if the order of a feature's values is not important, using ordered numbers will likely confuse the learning algorithm by finding incorrect regularity, and may lead to overfitting.

#### **Binning**

**Binning** or **bucketing** is the process of converting a continuous feature into multiple binary features (bins or buckets).

Binning might help the algorithm to learn using fewer examples because the exact value of the feature does not matter, only its range.

#### **Normalization**

Normalization is the process of converting a range of values that a feature can take into a standard range of values, such as $[-1,1]$ or $[0,1]$. The normalization formula is given as:

$$\bar{x}^{(j)} = \frac{x^{(j)}-\min^{(j)}}{\max^{(j)}-\min^{(j)}},$$

where $\min^{(j)}$ and $\max^{(j)}$ are the minimum and maximum value of the feature $j$ in the dataset, respectively.

Normalization can lead to **numerical stability** and an increased rate of **convergence**.

**Numerical stability** refers to the robustness and reliability of a numerical algorithm or computation in the face of small perturbations or errors in the input data, intermediate calculations, or rounding errors due to finite precision arithmetic.

**Convergence** refers to the process by which the algorithm reaches a stable or optimal solution and has iterated through its training process to a point where further iterations do not significantly change the model parameters or performance metrics.

#### **Standardization**

**Standardization** (or **z-score normalization**) is the procedure during which the feature values are rescaled so that they have the properties of a standard normalized distribution with mean $\mu=0$ and standard deviation $\sigma=1$. Standard scores of features are calculated thusly:

$$\hat{x}^{(j)}=\frac{x^{(j)}-\mu^{(j)}}{\sigma^{(j)}}.$$

Standardization is preferred over normalization in the following cases:

- For unsupervised learning algorithms.
- If a feature's values are distributed closely to a normalized distribution (bell curve).
- If a feature's values can be extremely high or low (**outliers**). Normalization would just squeeze the normal values into a very small range.

#### **Data Imputation**

If values for features are missing from the dataset there are a few approaches to take. The simplest ones are to remove those features from the dataset or using a learning algorithm that can deal with this kind of dataset. Another approach is to apply a **data imputation technique**.

One such technique consists of replacing the missing value of a feature by an **average feature value** in the dataset:

$$\hat{x}^{(j)} \leftarrow \frac{1}{m} \sum_{i=0}^n x_i^{(j)},$$

where $m<n$ is the number of examples in which the value of feature $j$ is present, while the summation excludes the examples in which the value of feature $j$ is absent.

Another technique is to replace the missing value with an **artificial value**. There are two options:

1. A value outside the normal range of values. This will let the algorithm learn what is best to do when a feature has a value significantly different from regular ones.
2. A value in the middle of the range. This will prevent the value from significantly affecting the prediction.

A more advanced technique is to use the missing value as the target variable for a **regression problem**. In this case, all the remaining features are used to form a feature vector $\hat{x}_i$, and set $\hat{y}_i \leftarrow x^{(j)}$, where $j$ is the feature with a missing value. Then $\hat{y}$ is predicted from $\hat{x}$ with a regression model.

If there is a significantly large dataset and just a few features with missing values, the dimensionality of the feature vector can be increased by adding a **binary indicator** feature for each feature with missing values: for each feature vector $x$, add the feature $j = D+1$ which is equal to 1 if the value of $j$ is present in $x$, and 0 otherwise. The missing value can then be replaced by 0 or any number.

### Learning Algorithm Selection

There are several factors to choosing one learning algorithm over another.

1. **Transparancy**. Most accurate learning algorithms such as neural networks or ensemble models are black boxes that are hard to grasp or visualize. Sometimes it is advantageous to choose a straightforward prediction algorithm like kNN, linear regression, or decision trees.
2. **Memory**: Incremental learning algorithms can be used in cases where a dataset is too large to be fully loaded into memory.
3. **Dataset size**: Some algorithms like neural networks or gradient boosting can handle a huge number of examples and millions of features. Others like SVM are more modest in their capacity.
4. **Feature type**: Some algorithms only apply to categorical or numerical features or require feature conversion first.
5. **Linearity**: If the data is linearly separable or can be modeled using a linear model then an SVM with linear kernel, logistic or linear regression can be good choices. Otherwise, deep neural networks or ensemble algorithms might work better.
6. **Convergence rate**: Simple algorithms like logistic and linear regression and decision trees converge much quicker than neural networks. Some algorithms like random forests are pleasingly parallelizable.
7. **Prediction speed**: When very high throughputs are required algorithms like SVM, linear and logistic regression, and some types of neural networks are extremely fast at prediction time. Others like kNN, ensemble algorithms, and deep or recurrent neural networks are slower. In modern libraries, kNN and ensemble algorithms are still pretty fast though.

### Underfitting / Overfitting

When a model makes many mistakes on the training data, it has a **high bias** or it **underfits**. The solution to the problem of underfitting is to try a model complex model or to engineer features with higher predictive power.

**Overfitting** or **high variance** is the problem where a model predicts the training data very well but **holdout data** (from the validation and testing datasets) quite poorly. To combat overfitting one of the following solutions may be applied:

- Use a simpler model (linear instead of polynomial regression, SVM with a linear kernel instead of RBF kernel, or a neural network with fewer layers).
- Reduce dimensionality of examples in the dataset.
- Add more training data.
- Regularization techniques.

### Regularization

**Regularization** refers to methods that force the learning algorithm to build a less complex model, which often leasds to a slightly higher bias but significantly lower variance. This phenomenon is also known as the **bias-variance tradeoff**.

**L1** and **L2** are the most commonly used types of regularization. To create a regularized model, the objective function is modified by adding a penalizing term whose value is higher when the model is more complex.

Let's consider the a linear regression objective function

$$\min_{w,b}\frac{1}{n}\sum_{i=1}^n (f_{w,b}(x_i)-y_i)^2$$

#### **L1 regularization**

L1 regularization is also known as **lasso regularization**. In practice L1 produces a **sparse model** where most of its parameters are equal or close to zero, and thus performs **feature selection** by deciding which features are essential for prediction and which are not.

The L1-regularized objective becomes:

$$\min_{w,b}\left[C|w|+\frac{1}{n}\sum_{i=1}^n (f_{w,b}(x_i)-y_i)^2\right],$$

with

$$|w|=\sum_{j=1}^D|w^{(j)}|,$$

where

$C$ is a hyperparameter that controls the weight of the regularizer. The higher $C$ becomes, the smaller $w^{(j)}$ becomes to minimize the objective, and the model will become very simple (which can lead to underfitting).

#### **L2 regularization**

L2 regularization is also called **ridge regularization** and usually gives better results for maximizing model performance on the holdout data. With L2, gradient descent can be used for optimizing the objective function.

The L2-regularized objective looks like this:

$$\min_{w,b}\left[C\|w\|^2+\frac{1}{n}\sum_{i=1}^n (f_{w,b}(x_i)-y_i)^2\right],$$

where

$$\|w\|^2=\sum_{j=1}^D(w^{(j)})^2.$$

L1 and L2 can be combined which is known as **elastic net regularization**. Other regularization methods (used in neural networks) are **dropout**, **batch-normalization**, **data augmentation**, and **early stopping**.

### Performance Metrics

Metrics are used assess a model's prediction accuracy, or manner of **generalization**.

For regression a model should perform better than the fit of a **mean model**. If so, then the mean squared error (MSE) is computed for the training and test data. If the MSE of the model on the test data is substantially higher than the MSE on the training data there is a sign of overfitting.

For classification the most widely used metrics are:

- Confusion Matrix
- Accuracy
- Cost-Sensitive Accuracy
- Precision / Recall
- Receiver Operating Characteristic

#### **Confusion Matrix**

A **confusion matrix** summarizes how successful a classification model is at predicting examples belonging to various classes. On one axis is the label that the model predicted and on the other the actual label. For each example, the number of **true positives** (TP), **true negatives** (TN), **false positives** (FP), and **false negatives** (FN) is being tracked.

A confusion matrix is used to calculate **precision** and **recall**. Precision is the ratio of correct positive predictions to the overall number of positive predictions:

$$precision = \frac{TP}{TP + FP}$$

Recall is the ratio of correct positive predictions to the overall number of positive examples:

$$recall = \frac{TP}{TP+FN}.$$

In practice it is necessary to choose between a high precision or high recall, which can be done by various means:

- Assigning a higher weighting to the examples of a specific class.
- Tuning hyperparameters to maximize precision or recall on the validation set.
- Varying the decision threshold for algorithms that return probabilities of classes.

**Accuracy** is a useful metric when errors in predicting all classes is equally important, such as for multiclass classification. It is given as follows:

$$accuracy = \frac{TP+TN}{TP+TN+FP+FN}$$

**Cost-sensitive accuracy** is a metric which may be valuable in the situation in which different classes have different importance. In this case, the FP and FN counts are multiplied by a cost value before calculating the accuracy using the equation above.

#### **Receiver Operating Characteristic (ROC)**

The **Receiver Operating Characteristic (ROC)** curve is a commonly used method to assess the performance of classification models that return a confidence score such as logistic regression, neural networks, and decision trees.

ROC curves use a combination of the **true positive rate (TPR)** or **sensitivity** and **false positive rate (FPR)** or the proportion of negative examples predicted incorrectly (1 - **specificity**). These are defined as:

$$\text{TPR} = \frac{TP}{TP + FN}$$
$$\text{FPR} = \frac{FP}{FP + TN}$$

The ROC curve visually demonstrates the **trade-off between sensitivity and specificity** across different threshold values. A diagonal line from the bottom-left corner to the top-right corner represents random guessing (no discrimination), while a curve above this line indicates better-than-random performance.

The **Area Under the ROC Curve (AUC)** quantifies the overall performance of the model across all possible threshold settings. It provides a single scalar value that represents the probability that the model will rank a randomly chosen positive instance higher than a randomly chosen negative instance. AUC values range from 0 to 1, where 0.5 indicates random performance (no discrimination) and 1 indicates perfect discrimination.

### Hyperparameter Tuning

One typical way to find the best combination of hyperparameters is to use **grid search**, which is the process of training several models at once with different hyperparameter values. A common trick is to use a logarithmic scale for the hyperparameter(s). Then the best performing model is kept according to the metric, and then values close to the best ones can be explored further.

This process is time-consuming, especially for large datasets or a large number of hyperparameters. More efficient techniques are **random search** and **Bayesian hyperparameter optimization**.

Random search is similar to grid search but uses a statistical distribution for each hyperparameter from which values are randomly sampled.

Bayesian techniques use past evaluation results to choose the next hyperparameter values to try, thus limiting the number of expensive optimizations of the objective function.

Other algorithmic tuning methods include **gradient-based techniques** and **evolutionary optimization techniques**.

### Cross-Validation

**Cross-validation** can be used to simulate a validation dataset in situations where it is unavailable due to lack of examples.

The training set is split into several subsets or **folds** In a typical five-fold cross-validation the training data is randomly split into five folds ${F_1,F_2,...,F_5}$. To train the first model $f_1$, all examples from folds $F_2, F_3, F_4$ and $F_5$ are used to train and the examples from $F_1$ are used as the validation set on which the metric of interest is computed. This process is continued for all of the 5 models and the five metric values are averaged to obtain the final performance metric.

Grid search can be combined with cross-validation to find the best values of hyperparameters for a model. Then the entire training set can be used to build the final model and the test set is used for assessment.

## Neural Network

The logistic regression model, or rather its generalizationn for multiclass classification, called the **softmax regression model** is a standard unit in a neural network.

A **neural network (NN)** $y=f_{NN}(\bold{x})$ is a nested function which for 3 network **layers** would look like this:

$$y=f_{NN}(\bold{x})=f_3(\bold{f}_2(\bold{f}_1(x))),$$

where $\bold{f}_1$ and $\bold{f}_2$ are vector functions of the following form:

$$\bold{f}_l(\bold{z}) = \bold{g}_l(\bold{W}_l \bold{z} + \bold{b}_l),$$

where $l$ is the layer index, and $\bold{g}_l$ is a nonlinear **activation function**. Weight and bias parameters $\bold{W}_l$ and $\bold{b}_l$ for each layer are learning using gradient descent by optimizing a particular **cost function**. Each row $\bold{w}_{l,u}$ ($u$ for unit) of matrix $\bold{W}_l$ is a vector of the same dimensionality as $\bold{z}$.

The $\bold{W}_l$ term in the expression $\bold{W}_l\bold{z}+\bold{b}_l$ is a matrix, where each row $u$ corresponds to a vector of parameters $\bold{w}_{l,u}$. The dimensionality of vector $\bold{w}_{l,u}$ equals the number of units in layer $l-1$.

Operation $\bold{W}_l\bold{z}$ returns a vector $\bold{a}_l = [\bold{w}_{l,1}\bold{z},\bold{w}_{l,2}\bold{z},...,\bold{w}_{l,n_l}\bold{z}]$, and the sum $\bold{a}_l+\bold{b}_l$ gives a $n_l$-dimensional vector $\bold{c}_l$. Then, function $\bold{g}_l(\bold{c}_l)$ produces the vector $\bold{y}_l = [y_l^{(1)},y_l^{(2)},...,y_l^{(n_l)}]$, where $n_l$ is the number of units in layer $l$.

### Multilayer Perceptron

The **multilayer perceptron (MLP)** or **vanilla neural network** is one particular configuration of of neural networks called **feed-forward neural networks (FFNN)**. An FFNN can be a regression or a classification model, depending on the activation used in the output layer.

A neural network consists of a connected combination of **nodes** logically organized into one or more **layers**. In an multilayer perceptor all outputs of one layer are connected to each input of the next layer. This architecture is called **fully-connected**.

In each node a linear transformation is applied to its input vector, and then an activation function $g$ to obtain the output value (a real number). For any of the **hidden layers** the formula applied is given as:

$$y_l^{(u)} \leftarrow g_l(\bold{w}_{l,u},\bold{y}_{l-1}+b_{l,u}).$$

Each node has its parameters $\bold{w}_{l,u}$ and $b_{l,u}$, and activation function $g_l$, where $u$ is the index of the node (unit), and $l$ is the index of the layer; and an activation function $g_l$, where l is the index of the layer. The vector $\bold{y}_{l-1}$ in each node is defined as

$$[y^{(1)}_{l-1},y^{(2)}_{l-1},y^{(3)}_{l-1},y^{(4)}_{l-1}].$$

For the first layer, the $\bold{y}_{l-1}$ factor in the equation above is instead replaced by vector $\bold{x} = x^{(1)},...,x^{(D)}$.

The last layer of a neural network usually contains only one node. If the activation function $g_{last}$ of this node is linear, then the neural network is a regression model. If it is a logistic function, then the neural network is a binary classification model.

#### Activation function

Any mathematical function can be chosen as activation function as long as it is differentiable. This property is essential to be able to use gradient descent to find the optimal parameters $\bold{w}_{l,u}$ and $b_{l,u}$ for all $l$ and $u$. Having nonlinear components in the function $f_{NN}$ is to allow the neural network to approximate nonlinear functions which it could not accomplish with only linear functions, no matter how many layers it would have.

Popular activation functions are the **logistic function (Sigmoid)** and **hyperbolic tangent function (TanH)** that both range from -1 to 1, and **rectified linear unit**:

$$\text{Sigmoid}(z) = \frac{1}{1+\exp(-x)},$$

$$\text{TanH}(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}},$$

$$\text{ReLu}(z) = \begin{cases}
    0 ~~ \text{ if } z < 0\\
    z ~~ \text{ otherwise}
\end{cases}$$

#### Deep Learning

**Deep learning** traditionally refers to neural networks that have more than two hidden layers. Networks that have many layers suffer from the problems of **exploding gradient** and **vanishing gradient**. However, due to many improvements to mitigate these issues the modern term **deep learning** means a neural network that uses the modern algorithmic and mathematical toolkit independently of how deep the neural network is.

Gradients can accumulate as they are propagated backward through the network during the training process, and certain activation functions (such as the sigmoid function) can saturate and produce very small gradients for large input values, exacerbating the problem of **exploding gradients**. It can be mitigated by applying techniques such as **gradient clipping** and **L1 and L2 regularization**, and using activation functions that do not suffer from saturation issues.

**Vanishing gradient** arises during the process of **backpropagation**; an algorithm for computing gradients on neural networks using partial derivatives of complex functions and the chain rule. During backpropagation, parameters receive an update proportional to the partial derivative of the cost function with respect to the current parameter. In some cases the gradient will be **vanishingly small**, preventing some parameters to change value. This can lead to an exponential gradient decrease with respect to the number of layers, effectively leading to the earlier layers to train very minimally.

Several improvements have been combined together to allow effective training of very deep neural networks. Techniques include ReLU (Rectified Linear Unit), LSTM (Long Short-Term Memory),**skip connections** used in **residual neural networks**, and modifications to the gradient descent algorithm.

### Convolutional Neural Network

When training on images the input is very high-dimensional because each pixel is a feature. A **convolutional neural network CNN** is a special kind of FFNN that significantly reduces the number of parameters in a deep neural network without losing much in the quality of the model.

The idea is to train the neural network to recognize regions of the same information as well as the edges so that it can predict an object represented in an input image. Since most of the important information in the image is local, the image can be split into square patches using a **moving window** approach. Multiple smaller regression models can then be trained at once, where each model receives a **patch** as input, and has as goal to learn a specific kind of pattern in that input.

Each small regression model applies a **convolution operation** to detect some pattern in the **input feature map** (or input image). The convolution operation involves sliding (convolving) a **filter/kernel matrix** $\bold{F}$ over the input feature map and computing the element-wise multiplication between the filter and the corresponding region of the input or **patch matrix** $\bold{P}$, followed by summing up the results to produce a single value. This process is repeated across the entire input feature map, with configurable **stride** (horizontal/vertical pixel shift of the filter) and **padding** (additional zeroed rows and columns around the input feature map).

The **output feature map** produced by the convolution operation represents features extracted from the input data, with each element in the output feature map corresponding to a local region in the input. By learning appropriate filters during training, CNNs can automatically extract hierarchical features from the input data, making them effective for tasks such as image classification, object detection, and segmentation.

One layer $l$ of a convolutional neural network consists of multiple convolution filters with their own bias parameters, and the output of $l$ will consist of $size_l$ matrices, one for each filter. Each filter convolves across the input image and convolution is computed at each iteration. The filter matrix and bias values are trainable parameters that are optimized using gradient descent with backpropagation. A nonlinearity in the form of the ReLU activation function is applied to the sum of the convolution and bias term for all hidden layers. The activation function of the output layer depends on the task.

A subsequent convolution layer $l+1$ treats the output of preceding layer $l$ as a collection of $size_l$ image matrices. Such a collection is called a **volume**, and the size of that collection is called the volume's **depth**. Each filter of layer $l+1$ convolves this whole volume, which is simply the sum of convolutions of the corresponding patches that the volume consists of.

**Pooling** is a downsampling operation that reduces the spatial dimensions of the input feature maps, while retaining important information. Pooling is typically applied after convolutional layers to progressively reduce the spatial size of the representation, leading to a smaller number of parameters and computational requirements in deeper layers of the network. There are several types of pooling operations, with the most common ones being max pooling and average pooling. A pooling operation can be configured with parameters such as the size of the pooling window, the stride, and padding.

### Recurrent Neural Network

**Recurrent Neural Networks (RNNs)** are a type of artificial neural network designed to model sequential data by maintaining a form of memory across time steps. A **sequence** is a matrix where each row is a feature vector and the row order is important.

Unlike traditional feedforward neural networks, which process each input independently, RNNs have connections that loop back on themselves, allowing them to incorporate information about previous time steps into their computations. This makes them well-suited for tasks involving sequential data, such as time series forecasting, natural language processing, and speech recognition.

An RNNs computes a hidden **state** $h_t$​ at each time step $t$ for each layer using an activation function $g_1$ (typically *hyperbolic tangent* or *ReLU*) applied to the linear combination of the input $x_t$ and the hidden state from the previous step $h_{t−1}$, along with a bias term $b_t$. The hidden state update can be expressed as:

$$h_t = g_1(\bold{W}_h x_t+\bold{U}_h h_{t-1}+\bold{b}_h),$$

where $\bold{W}_t$ and $\bold{U}_t$ are weight matrices.

The hidden state $h_t$ may be used to compute an output $y_t$ at time step $t$ using another linear transformation with weight matrix $\bold{V}_t$, bias vector $\bold{c}$ and activation function $\bold{g}_2$:

$$\bold{y}_t = \bold{g}_2(\bold{V}_t\bold{h}_t+\bold{c}),$$

where $\bold{g}_2$ is typically the **softmax activation function**. Given an input vector $z=[z_1,z_2,...,z_k]$, where $z_i$​ represents the score or logit for class $i$, the softmax function computes the probability $\sigma_i$​ for each class $i$ as follows:

$\sigma(z)=\frac{\exp(z_i)}{\sum_{j=1}^k\exp(z_j)}$.

In other words, the softmax function exponentiates each element of the input vector $z$, which ensures that all scores are positive, and then normalizes them by dividing by the sum of all exponentiated scores across all classes. This normalization ensures that the resulting probabilities sum up to 1, making them interpretable as probabilities.

To train RNN models, a special backpropagation algorithm is used called **Backpropagation Through Time (BPTT)** that takes into account the sequential nature of the data by *unfolding* the network over time.

The gradients of the loss function with respect to the network parameters (weights and biases) are computed using the *chain rule*, and the parameters are updated using an optimization algorithm such as *gradient descent*.

RNNs have the ability to model dependencies over long sequences, but they suffer from certain limitations, such as difficulty in capturing long-term dependencies (vanishing or exploding gradients), and they may struggle with processing inputs of varying lengths. To address some of these limitations, so-called *gated* variants of RNNs such as **Long Short-Term Memory (LSTM)** networks and **Gated Recurrent Units (GRUs)** have been developed, which are designed to better capture long-term dependencies and mitigate the vanishing gradient problem.

#### **Minimal Gated Unit**

A simple but effective one is the **Minimal Gated Unit (MGU)** that is composed of a memory cell and a forget gate. It computes an input update gate to control how much of the new input information is added to the current hidden state, and then computes a **candidate activation** that represents the new information. The final hidden state is a linear combination of the previous hidden state and the candidate activation, with the input update gate controlling the proportion of each component. MGUs are similar in spirit to GRUs but have fewer parameters and computational complexity, making them attractive for applications where efficiency is important.

The input update gate is computed using a sigmoid function applied to a linear transformation of the input $x_t$ and the previous hidden state $h_{t-1}$:

$$\Gamma_t = \sigma(\bold{W}_\Gamma x_t+\bold{U}_\Gamma h_{t-1}+\bold{b}_\Gamma),$$

where $\bold{W}_\Gamma$, $\bold{U}_\Gamma$ and $a_\Gamma$ are the trainable parameters of this linear transformation.

The candidate activation function $\tilde{h}_t$ is typically computed using a $tanh$ activation function applied to a linear transformation of the input $x_t$ and the previous hidden state $h_{t-1}$:

$$\tilde{h}_t = tanh(\bold{W}_cx_t+\bold{U}_c\bold{h}_{t-1}+\bold{b}_c).$$

Finally, the hidden state at time step $t$ is updated using a combination of the input gate and the candidate activation

$$\bold{h}_t = (1-\Gamma_t)h_{t-1}+\Gamma_t\tilde{h}_t,$$

and the output $y_t$ for a layer at time step $t$ becomes:

$$y_t = \sigma(\bold{V}\bold{h}_t + \bold{c}),$$

where $\sigma$ is the softmax activation function.

A gated unit takes an input and stores it for some time, which is equivalent to applying the identity function. When a network with gated units is trained with backpropagation through time, the gradient does not vanish because the derivative of the identity function is constant.

Other important extensions to vanilla recurrent neural networks are **Bi-RNNs** that processes input sequences in both forward and backward directions simultaneously, RNNs that incorporate **attention mechanisms**, and **sequence-to-sequence (seq2seq)** models used in **natural language processing (NLP)** tasks. A generalization of a recurrent neural network is a **Recursive Neural Network (RecNN)**.

## Kernel Regression

In linear regression, if the data does not have the form of a straight line polynomial regression could help. However, if the input is a high-dimensional feature vector (higher than 3 dimensions), finding the right polynomial would prove difficult.

**Kernel regression** is a non-parametric method where there are no parameters to learn but is based on the data itself, and has the following form:

$$f(x) = \frac{1}{n} \sum_{i=1}^n w_iy_i$$

where

$$w_i = \frac{n k(\frac{x_i-x}{b})}{\sum_{l=1}^n k(\frac{x_l-x}{b})}.$$

Function $k(\cdots)$ is called a **kernel** which plays the role of a *similarity function*; the values of coefficients $w_i$ are higher when $x$ is similar to $x_i$ and lower when they are dissimilar. The most frequently used kernel is the **Gaussian kernel**:

$$k(z) = \frac{1}{\sqrt{2\pi}} \exp(-0.5z^2).$$

Hyperparameter $b$ above is tuned using the validation set by running the model with a specific value of $b$ and calculatiung the mean squared error. The value of $b$ has a big influence on the smoothness of the shape of the polynomial regression line (higher values yield more smooth curves).

## Classification

### Multiclass Classification

In **multiclass classification** the label can be one of $C$ classes. Most learning algorithms can be either naturally converted to a multiclass case or they return a score that can be used in a **one versus rest** strategy.

Logistic regression can be naturally extended to multiclass learning problems by replacing the *sigmoid function* with the *softmax function*. The kNN algorithm is also straightforward to extend to the multiclass case; just find the $k$ closest examples for input $x$ and return the class that occurs most frequently.

The idea of **one versus rest** is to transform a multiclass problem into $C$ binary classification problems and build $C$ binary classifiers to yield $C$ predictions. Then pick the prediction of a non-zero class which is *the most certain*. In logistic regression, the model returns not a label but a score between 0 and 1 that can be interpreted as the probability that the label is positive (certainty of prediction). In SVM, the analog of certainty is the distance from the input $x$ to the decision boundary, and the larger the distance, the more certain the prediction.

### One-Class Classification

**One-class classification** (also known as **unary classification** or **class modeling**) attempts to identify objects of a specific class among all objects by learning from a training set that contains only the objects of that class and no other classifications. These learning algorithms are used for outlier, anomaly, and novelty detection. The most widely used one-class learning algorithms are **one-class Gaussian**, **one-class k-means**, **one-class kNN**, and **one-class SVM**.

In one-class Gaussian the data is modeled as if it came from a Gaussian distribution, specifically the **multivariate normal distribution (MND)**. The probability density function (pdf) for MND is given as:

$$f_{\bold{\mu},\bold{\Sigma}}(\bold{x}) = \frac{\exp(\\frac{1}{2}(\bold{x}-\bold{\mu}^\top \bold{\Sigma}^{-1}(\bold{x}-\bold{\mu})))}{\sqrt{(2\pi)^D|\bold{\Sigma}|}},$$

where $f_{\bold{\mu},\bold{\Sigma}}(\bold{x})$ returns the probability density corresponding to the input vector $\bold{x}$, $|\bold{\Sigma}| \equiv \det\bold{\Sigma}$ is the determinant of $\bold{\Sigma}$, and $\bold{\Sigma}^{-1}$ is the inverse of that matrix.

The **maximum likelihood** criterion is optimized to find the optimal values for vector $\bold{\mu}$ and matrix $\bold{\Sigma}$. In practice, the numbers in vector $\bold{\mu}$ determine the place where the curve of the Gaussian distribution is centered, while the numbers in $\bold{\Sigma}$ determine the shape of the curve.

One the model is parametrized by $\bold{\mu}$ and $\bold{\Sigma}$ learned for the data, the likelihood of every input $\bold{x}$ is predicted by using model $f_{\bold{\mu},\bold{\Sigma}}(\bold{x})$. Only if the likelihood is above a certain threshold, the example is predicted to belong to the target class, otherwise it is classified as an outlier.

When the data has a more complex shape, a more advanced algorithm can use a combination of several Gaussians. In this case there are more parameters to learn from the data, namely one $\bold{\mu}$ and one $\bold{\Sigma}$ for each Gaussian, and the parameters to form one probability distribution function for the **Gaussian mixture**.

One-class k-means and one-class kNN are based on a similar principle: build some model of the data and then define a threshold to decide whether the new feature vector looks similar to other examples according to the model.

One-class SVM attempts to either separate all training examples from the origin (in the feature space) and maximize the distance from the hyperplane to the origin, or to obtain a spherical boundary around the data by minimizing the volume of this hypersphere.

### Multi-Label Classification

**Multi-label classification** applies to situations where more than one label is needed to described an example from the dataset.

If the number of possible values for labels is high but they are all of similar nature (like tags), then each labeled example can be transformed into several labeled examples, one per label. Those new examples all have the same feature vector but only one label and thus becomes a multiclass classification problem that can be solved using the *one-versus-rest strategy*. In this case there is a new *threshold* hyperparameter; if the prediction score for some label is above this threshold, then that label is predicted for the input feature vector. The value of the threshold is chosen using the validation set.

Algorithms that can naturally be made multiclass like decision trees, logistic regression, and neural networks, can be applied to multi-label classification problems as well.

Neural networks can naturally train multi-label classification models by using the **binary cross-entropy** cost function. This is defined as:

$$-(y_{i,l}\ln(\hat{y}_{i,l}) + (1-y_{i,l}) \ln(1-\hat{y}_{i,l})),$$

where $\hat{y}_{i,l}$ is the probability that example $\bold{x}_i$ has label $l$. The minimization criterion is simply the average of all binary cross-entropy terms across all training examples and all labels of those examples.

When the number of possible values each label can take is small, fake classes can be created for each *combination* of the original classes. This is only possible with a small number of combinations, otherwise much more training data is needed to compensate for an increased set of classes.

Contrary to the previous methods that predict each label independently of one another, this approach has the advantage that the labels remain correlated. Correlation between labels can be essential in many problems.

## Ensemble Learning

**Ensemble learning** is a paradigm that focuses on training a large number of low-accuracy models and then combining the predictions given by them to obtain a high-accuracy **meta-model**.

Low-accuracy models are usually learned by **weak learners**; learning algorithms that cannot learn complex models and are thus typically fast to train and at prediction time such as *decision trees*. The idea is that if the trees are not identical and each tree is at least slightly better than random guesses, then then high accuracy can be obtained by combining a large number of such trees. A prediction for input $\bold{x}$ is obtained by combining the individual predictions of each weak model with some sort of weighted vote.

**Boosting** consists of using original training data and iteratively creating multiple models by using a weak learner, where each new model attempts to fix the errors that the previous models made. The final **ensemble model** is a certain combination of those multiple weak models built iteratively.

**Bagging** consists of creating many slightly different copies of the training data and then applying the weak learner to each copy to obtain multiple weak models and then combining them.

### Random Forest

**Random forest** is a popular and effective algorithm based on the idea of *bagging*. Given a training set, create $B$ random samples $\mathcal{S}_b$ of the training set and a decision tree model $f_b$ using each sample $\mathcal{S}_b$ as the training set. To sample $\mathcal{S}_b$ for some $b$ perform **sampling with replacement** and keep picking examples at random until $|\mathcal{S}_b|=n$. After training there are $B$ decision trees. The prediction for a new sample $\bold{x}$ is obtained as the average of $B$ predictions:

$$y \leftarrow \hat{f}(\bold{x}) \equiv \frac{1}{B}\sum_{b=1}^B f_b(\bold{x}),$$

in the case of regression, or by taking the majority vote in the case of classification.

Random forest uses a modified tree learning algorithm that insepects a random subset of the features at each split in the learning process. The reason for doing this is to avoid the correlation of the trees, as correlated predictors cannot help in improving the accuracy of prediction. Correlation will make bad models more likely to agree, which hampers the majority vote or average metric.

The most important hyperparameters to tune are the number of trees, $B$, and the size of the random subset of the features to consider at each split.

Random forest is effective because the *variance* of the final model is kept low by using multiple samples of the original dataset. Low variance means low *overfitting*. However, undesirable artifacts like noise, outliers, and over- or underrepresented examples may occur due to the random sampling process. The effect of these artifacts can be reduced by creating multiple random samples with replacement of the training set.

### Gradient Boosting

**Gradient boosting** is another effective ensemble learning algorithm. In this technique we start with an initial constant model $f_0$ and compute its  **residuals**. Residuals describe how well the target of each training example is predicted by the current model $f$. Another tree is then trained to fix the errors of the current model and it is added to the existing model with some weight $\alpha$. Each additional tree added to the model partially fixes the errors made by the previous trees until the maximum number $M$ (a hyperparameter) of trees are combined.

Gradient boosting is one of the most powerful machine learning algorithms because it creates very accurate models, and is capable of handling huge datasets with millions of features. It usually outperforms random forest in accuracy but can be slower in training due to its sequential nature.

This algorithm is called *gradient* boosting because a proxy of the gradient descent is used in the form of residuals that show how the model has to be adjusted to minimize the error (or residual).

Boosting reduces the bias (or underfitting) instead of the variance. As such, boosting can overfit though this can be mitigated by tuning the depth and the number of trees.

The three principle hyperparameters to train are the number of trees $M$, the learning rate $\alpha$, and the depth of trees $d$.

#### Gradient Boosting For Regression

For regression, choose a constant model $f = f_0$ first:

$$f = f_0(\bold{x}) \equiv \frac{1}{n} \sum_{i=1}^n y_i.$$

Then the labels for each example $i=1,...,n$ in the training set is modified as follows:

$\hat{y}_i \leftarrow y_i - f(\bold{x}_i)$,

where $\hat{y}_i$ is the residual, a new label for example $\bold{x}_i$. The modified training set with residuals instead of original labels is used to build a new decision tree model $f_1$. The boosting model is now defined as $f \equiv f_0 +\alpha f_1$, where $\alpha$ is the learning rate. This process continues until the predefined maximum $M$ of trees are combined.

#### Gradient Boosting For Classification

In the binary classification case, the prediction of the ensemble of decision trees is modeled using the sigmoid function:

$Pr(y=1|\bold{x},f) \equiv \frac{1}{1+\exp(e^{-f(\bold{x})})}$,

where $f(\bold{x}) \equiv \sum_{m=1}^M f_m(\bold{x})$ and $f_m$ is a regression tree. The maximum likelihood principle is applied by tring to find such and $f$ that maximizes

$$L_f = \sum_{i=1}^n \ln[Pr(y_i=1|\bold{x}_i,f)].$$

The algorithm starts with the initial constant model $f= f_0 = \frac{p}{1-p}$, where $p=\frac{1}{n} \sum_{i=1}^Ny_i$. Then, at each iteration $m$ a new tree $f_m$ is added to the model. To find the best $f_m$, first the partial derivative $g_i$ of the current model is calculated for each $i=1,...,n$:

$$g_i = \frac{\partial L_f}{\partial f},$$

where $f$ is the ensemble classifier model built at previous iteration $m-1$. To calculate $g_i$ we need to find the derivatives of $\ln[Pr(y_i=1|\bold{x}_i,f)]$ with respect to $f$ for all $i$.

Notice that

$$\ln[Pr(y_i=1|\bold{x}_i,f)] \equiv \ln\left[ \frac{1}{1+\exp(-f(\bold{x}_i))} \right].$$

The derivative of the right-hand term with respect to $f$ equals

$$\frac{1}{\exp(f(\bold{x}_i))+1}.$$

The training set is then transformed by replacing the original label $y_i$ with the corresponding partial derivative $g_i$ and building a new tree $f_m$ using the transformed training set. The optimal updat step $\rho_m$ is found as follows:

$$\rho_m \leftarrow \argmax_p(L_{f + \rho f_m}).$$

At the end of iteration $m$, the ensemble model $f$ is updated by adding new tree $f_m$:

$$f \leftarrow f + \alpha\rho_m f_m.$$

Iteration continues until $m=M$ and then ensemble model $f$ is returned.

## Sequence Labeling

**Sequence labeling** is the problem of automatically assigning a label to each element of a sequence.

For an example $i$ we define a set of input features and output labels:

$$\bold{X}_i = [x_i^{(1)}, x_i^{(2)},..., x_i^{(n_i)}], \\
\bold{Y}_i = [y_i^{(1)}, y_i^{(2)}, ..., y_i^{(n_i)}], \\
y_i \in {1,2,...,C}$$

where $n_i$ is the length of the sequence of example $i$, and $C$ is the number of possible classes.

As shown, a recurrent neural network can be used to label a sequence. At each time step $t$, it reads an input feature vector $\bold{x}_i^{(t)}$, and the last recurrent layer outputs a label $y_{last}^{(t)}$ which may be a vector or scalar.

### Conditional Random Fields

The model called **Conditional Random Fields (CRF)** is a very effective alternative that often performs well in practice for the feature vectors that have many informative features. This model can be seen as a generalization of logistic regresion to sequences.

CRFs model the conditional probability distribution of output variables (labels) given input variables (features) in a structured prediction framework.

CRFs define a set of feature functions $f={f_k(x,y)}$ that capture the compatibility between input features and output labels, encoding dependencies and patterns in the data. Each feature function computes a real-valued score based on the input features and output labels.

A set of parameters $w={w_k}$ parameterize the feature functions. Each parameter $w_k$ learned from the training data​ represents the weight or importance of the corresponding feature function.

The conditional probability of output labels given input features is modeled as:

$$Pr(\bold{y}|\bold{x};\bold{w})=\frac{1}{Z(\bold{x};\bold{w})} \exp⁡(\sum_k w_k f_k(\bold{x},\bold{y})),$$

where $Z(\bold{x};\bold{w})$ is the normalization factor (partition function) that ensures that the probabilities sum up to $1$ over all possible label sequences.

Given the input features $x$ and learned parameters $w$, the goal of prediction is to find the most probable output label sequence $\hat{\bold{y}}$ given the input: $\hat{\bold{y}} = \argmax_y Pr(\bold{y}|\bold{x};\bold{w})$. Inference in CRFs can be performed efficiently using dynamic programming algorithms such as the *Viterbi algorithm* or *belief propagation*.

In practice CRF models have been outperformed by bidirectional deep gated RNNs. CRFs are also significantly slower in training which makes them difficult to apply to large trainings sets.

### Sequence-to-Sequence Learning

**Sequence-to-sequence learning (seq2seq)** is a generalization of the sequence labeling problem. In seq2seq, $\bold{X}_i$ and $\bold{Y}_i$ can have different lengths. These models are applied to machine translation, conversational interfaces, text summarization, spelling correction, and other problems.

Many seq2seq learning problems are currently best solved by neural networks that have an **encoder** and a **decoder**.

The role of the encoder is to read the input an generate some sort of state that can be seen as a numerical representation of the *meaning* of the input the machine can work with. The meaning of some input is usually a vector or a matrix that contains real numbers called the **embedding** of the input.

The decoder is another neural network that takes an embedding (produced by the encoder) as input and generates a sequence of outputs. To produce a sequence of outputs, the decoder takes a *start of sequence* input feature vector $\bold{x}^{(0)}$ (typically all zeroes), produces the first output $\bold{y}^{(1)}$, updates its state by combining the embedding and $\bold{x}^{(0)}$, and then uses $\bold{y}^{(1)}$ as its next input $\bold{x}^{(1)}$.

Both encoder and decoder are traned simultaneously using the training data. The errors at the decoder are propagated to the encoder via backpropagation.

The accuracy of a seq2seq model can be improved by using an **attention** mechanism. It is implemented by an additional set of parameters that combined some information from the encoder and the current state of the decoder to generate the label. Attention allows for even better retention of long-term dependencies than provided by gated units and bidirectional recurrent neural networks.

## Active Learning

**Active learning** is a supervised learning paradigm that is applied when obtaining labeled examples is costly or difficult. The idea is to start learning with relatively few labeled examples and a large number of unlabeled ones, and the to label only those examples that contribute the most to the model quality.

There are several strategies of active learning, the most common of which is **data density and uncertainty based**.

### Data Density Based

In this strategy the current model $f$, trained using the existing labeled examples, is applied to each of the remaining unlabeled examples.

For each unlabeled example $\bold{x}$, the following importance score is computed:

$$density(\bold{x}) \cdot uncertainty_f(\bold{x}).$$

Density reflects how many examples surround \bold{x} in its close proximity, while $uncertainty_f(\bold{x})$ reflects how uncertain the prediction of the model $f$ is for $\bold{x}$.

A typical measure of uncertainty in multiclass classification is **entropy**:

$$H_f(\bold{x}) = -\sum_{c=1}^C Pr(y^{(c)};f(\bold{x})) \ln[Pr(y^{(c)};f(\bold{x}))],$$

where $Pr(y^{(c)};f(\bold{x}))$ is the probability score that model $f$ assigns to class $y^{c}$ when classifying $\bold{x}$.

The model is most uncertain if for each $y^{c}, f(y^{(c)}) = \frac{1}{C}$ and the entropy is at its maximum of 1. On the other hand, if for some $y^{c}, f(y^{(c)}) = 1$, then the model is certain about the class $y^{(c)}$ and the entropy is at its minimum of 0.

Density for example $\bold{x}$ is obtained by taking the average of the distance from $\bold{x}$ to each of its $k$ nearest neighbors (where $k$ is a hyperparameter).

Once we know the importance score of each unlabeled example, the one with the highest score is picked and the data engineer can ask an expert to annotate it. This newly annotated example is then added to the training set and the model rebuilt. This process continues until some stopping criterion is satisfied like the maximum requests of the expert or model performance.

### Other Active Learning Strategies

The **support vector-based** strategy consists of building an SVM model using the labeled data. The data engineer asks the expert to annotate the unlabeled example that lies closest to the hyperplane that separates the two classes. This closest example will be the least certain and would thus contribute the most to the reduction of possible places where the true hyperplane that we want to find could lie.

The **query by committee** strategy consists of training multiple models using different methods and then asking an expert to label an example on which those models disagree the most.

Some yet other strategies try to select examples to label so that the *variance* or the *bias* of the model are reduced the most.

## Semi-Supervised Learning

In **semi-supervised learning (SSL)** the goal is to leverage a large number of unlabeled examples to improve the model performance without needing additional labeled examples.

One frequently used SSL method is called **self-learning**, where a learning algorithm is used to build the initial model using the labeled examples. This model is then applied to all unlabeled examples and label them using the model. If the confidence score of prediction for some unlabeled example $\bold{x}$ is higher than some (emperically chosen) threshold then it is added to the training set, and the model is retrained. This process continues until a stopping condition is satisfied.

This method can bring some improvement to the model compared to just using the initially labeled dataset, but the increase in performance is usually very minimal. In practice the quality of the model could even decrease!

However, a more recent semi-supervised learning neural network architecture called a **ladder network** attained remarkable performance. A ladder network is basically an improved **autoencoder**.

### Autoencoder

An autoencoder is a feed-forward neural network with an encoder-decoder architecture. It is trained to reconstruct its input, where the training example is a pair $(\bold{x},\bold{x})$, and we want the output $\hat{\bold{x}}$ of the model $f(\bold{x})$ to be as similar to the input $\bold{x}$ as possible.

An autoencoder's network has a so-called **bottleneck layer** in the middle that contains the embedding of the $D$-dimensional input vector; the embedding layer usually has much fewer units than $D$. The goal of the decoder is to reconstruct the input feature vector from this embedding. The cost function is usually either the mean squared error or binary cross-entropy. The former is given by:

$$\frac{1}{n} \sum_{i=1}^n \| \bold{x}_i - f(\bold{x}_i) \|^2,$$

where $\|\bold{x}_i-f(\bold{x}_i)\|$ is the Euclidean distance between two vectors.

### Denoising Autoencoder

A **denoising autoencoder** corrupts the left-hand side $\bold{x}$ in the training example $(\bold{x},\bold{x})$ by adding some random perturbation to the features. If our examples are grayscale images with pixels represented as values between 0 and 1, usually a **Gaussian noise** is added to each feature. For each feature $j$ of the input feature vector $\bold{x}$ the noise value $n^{(j)}$ is sampled from the **Gaussian distribution**:

$$n^{(j)} \sim \mathcal{N}(\mu,\sigma^2),$$

where the notation $\sim$ means *sampled from*, and $\mathcal{N}(\mu,\sigma^2)$ denotes the Gaussian distribution with mean $\mu$ and standard deviation $\sigma$ whose probability distribution function is given by:

$$f_\theta(z) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(z-\mu)^2}{2\sigma^2}\right),$$

where $\pi$ is the constant and $\bold{\theta} \equiv [\mu,\sigma]$ is a hyperparameter. The new, corrupted value of the feature $x^{(j)}$ is given by $x^{(j)} + n^{(j)}$.

### Ladder Network

A **ladder network** is a *denoising autoencoder* with an upgrade. The encoder and the decoder have the same number of layers. The bottleneck layer is used directly to predict the label using the softmax activation function.

The network has serveral cost functions. For each layer $l$ of the encoder and the corresponding layer $l$ of the decoder, one cost $C_d^l$ penalizes the difference between the outputs of the two layers using squared Euclidean distance. When a labeled example is used during training, another negative log-likelihood cost function $C_c$ penalizes the error in prediction of the label. The combined cost function $C_c + \sum_{l=1}^L \lambda_l C_d^l$ (averaged over all examples in the batch) is optimized by the minibatch stochastic gradient descent with backpropagation. The hyperparameters $\lambda_l$ for each layer $l$ determine the tradeoff between the classification and encoding-decoding cost.

In the ladder network, not just the input is corrupted with the noise, but also the output of each encoder layer (during training). During prediction, new input $\bold{x}$ is not corrupted.

## One-Shot Learning

**One-shot learning** aims to classify objects from one or only a few examples. A typical application is to recognize that two images are the same or different.

A **siamese neural network (SNN)** is an effective way to solve this problem. It can be implemented as any kind of neural network, a convolutional neural network, a recurrent neural network, or an multilayer perceptron. The network only takes one image as input at a time.

An SNN is trained using the **triplet loss** function, which is defined as follows for an example $i$:

$$\max(\|f(A_i) - f(P_i) \|^2 - \| f(A_i) - f(N_i) \|^2 + \alpha, 0),$$

where $A$ is an image called the **anchor**, $P$ is an image called the **positive**, and $N$ is an image called the **negative**. Each training example $i$ is now a triplet $(A_i,P_i,N_i)$.

The cost function is defined as the average triplet loss:

$$\frac{1}{n} \sum_{i=1}^n \max(\|f(A_i) - f(P_i)\|^2 - \|f(A_i)-f(N_i)\|^2 + \alpha, 0),$$

where $\alpha$ is a positive-valued hyperparameter.

Intuitively, $\|f(A)-f(P)\|^2$ is low when our neural network outputs similar embedding vectors for $A$ and $P$, and $\|f(A_i)-f(N_i)\|^2$ is high when the embedding for images of two different subjects are different.

If the model works correctly, then

$$m=\|f(A_i) - f(P_i)\|^2 - \|f(A_i)-f(N_i)\|^2$$

will always be negative, because we subtract a high value from a small value. By setting $\alpha$ higher, we force the term $m$ to be even smaller to make sure that the model learned to recognize the two same subjects and two different subjects with a high margin. If $m$ is not small enough, then the cost will be positive because of the $\alpha$ parameter, and the model parameters will be adjusted in backpropagation.

A better way to create triplets for training is to use the current model after several epochs of learning and to find candidates for $N$ that are similar to $A$ and $P$ according to that model.

Building an SNN involves deciding on the architecture of the neural network first. CNN is a typical choice if the inputs are images. Given an example, calculating the average triplet loss involves applying the model consecutively to $A$, then to $P$, then to $N$, and then computing the triplet loss for that example. This process is repeated for all triplets in the batch and then compute the cost. Gradient descent with backpropagation will propagate the cost through the network to update its parameters.

In practice, more than one example for each subject is necessary for the model to be accurate enough. With the final model, to decide whether two images $A$ and $\hat{A}$ belong to the same subject (person), check if $\|f(A)-f(\hat{A})\|^2$ is less than $\tau$ (a hyperparameter).

## Zero-Shot Learning

In **zero-shot learning (ZSL)** we want to train a model so that it is able to predict labels that weren't in the training data.

The trick is to use embeddings that represent both the input *and* the output. **Word embeddings** can be learned from data, and they are usually compared using cosine similarity metrics. Word embeddings have such a property that each dimension of the embedding represents a specific feature of the meaning of the word. The values for each embedding are obtained using a specific training procedure applied to a *vast text corpus*.

In a classification problem, output label $y_i$ can be replaced for each example $i$ with its word embedding, and a multi-label model that predicts word embeddings trained. To get the label for a new example $\bold{x}$, model $f$ is applied to $\bold{x}$ and embedding $\hat{\bold{y}}$ is used to search among all English words whose embeddings are the most similar to $\hat{\bold{y}}$ using **cosine similarity**.

### Cosine Similarity

In cosine similarity with word embeddings, each word in a vocabulary is represented as a high-dimensional vector in a continuous vector space, typically using techniques like *Word2Vec*, *GloVe*, or *FastText*. The dimensions of these vectors capture semantic and syntactic properties of words based on their contexts in large text corpora.

Given two word vectors $\bold{v}_1$​ and $\bold{v}_2$​, the cosine similarity between them is calculated as the cosine of the angle between the two vectors:

$$sim(\bold{v}_1, \bold{v}_2) = \frac{\bold{v}_1\cdot\bold{v}_2}{\|\bold{v}_1\|\|\bold{v}_2\|},$$

where $\cdot$ represents the dot product of the two vectors, and $\|\cdots\|$ represents the Euclidean norm (magnitude) of a vector. Positive cosine similarity values indicate similarity, while negative values indicate dissimilarity.

## Advanced Practice

Some specific contexts and how to handle them are described here.

### Imbalanced Datasets

It often occurs that examples of some class will be underrepresented in the training data. A few *fraudulent* examples risk being misclassified in order to classify more numerous examples of the majority class correctly. This problem is observed for most learning algorithms applied to **imbalanced datasets**.

However, some algorithms are less sensitive to the problem of an imbalanced dataset. Decision trees, as well as random forest and gradient boosting, often perform well on imbalanced datasets.

Setting the cost of misclassification of examples of the minority class higher, the model will try harder to avoid misclassifying those examples, but this will incur the cost of misclassification of some examples of the majority class.

Some learning algorithms allow providing weights for every class and takes this information into account when looking for the best hyperplane. If a learning algorithm doesn't allow weighting classes, the technique of **oversampling** could be applied. It consists of increasing the importance of examples of some class by making multiple copies of the examples of that class. An opposite approach, **undersampling**, is to randomly remove from the training set some examples of the majority class.

### Synthetic examples

Yet another technique is to create **synthetic examples** by randomly sampling feature values of several examples of the minority class and combining them to obtain a new example of that class.

There are two popular algorithms that oversample the minority class by creating synthetic examples: the **synthetic minority oversampling technique (SMOTE)** and the **adaptive synthetic sampling method (ADASYN)**.

SMOTE and ADASYN work similarly in many ways. For a given example $\bold{x}_i$ of the minority class, create a set $\mathcal{S}_k$ of $k$ nearest neighbors of this example and then create a synthetic example $x_{new}$:

$$x_{new} \equiv x_i + \lambda(x_{zi}-x_i),$$

where $x_{zi}$ is an example of the minority class chosen randomly from $\mathcal{S}_k$. The interpolation hyperparameter $\lambda$ is a random number in the range $[0, 1]$.

Both SMOTE and ADASYN randomly pick all possible $x_i$ in the dataset. In ADASYN, the number of synthetic examples generated for each $x_i$ is proportional to the number of
examples in $\mathcal{S}_k$ which are not from the minority class. Therefore, more synthetic examples
are generated in the area where the examples of the minority class are rare.

## Combining Models

Ensemble algorithms like *random forest* typically combine models of the same nature. They boost performance by combining hundreds of weak models. Additional performance gain can sometimes be achieved by combining strong models made with different learning algorithms.

There are three typical ways to combined models: averaging, majority vote, and stacking.

**Averaging** works for regression as well as for classification models that return scores. Simply apply all models to the input $\bold{x}$ and then average the predictions.

**Majority vote** works for classification models. Apply all base models to the input $\bold{x}$ and then return the majority class among all predictions.

**Stacking** consists of building a meta-model that takes the output of base models as input. To combine classifiers $f_1$ and $f_2$ that both predict the same set of classes and create a training example $(\hat{\bold{x}}_i, \hat{y}_i)$ for the stacked model, set $\hat{y}_i = y_i$ and $\hat{\bold{x}_i} = [f_1(\bold{x}), f_2(\bold{x})]$.

If some of the base models return a score instead of a class, these values can be used as features as well.

To train the stacked model, it is recommended to use examples from the training set and tune the hyperparameters of the stacked model using cross-validation.

Combining multiple models can bring better performance because when several *uncorrelated* strong models agree then they are more likely to agree on the correct outcome. Ideally, base models should be obtained using different features or using algorithms of a different nature.

## Training Neural Networks

One challenging aspect in neural network training is how to convert a dataset into the input that the netork can work with.

For text a good way is to represent tokens is by using **word embeddings**. For a multilayer perceptor, to convert texts to vectors the *bag of words* approach might work well, especially for larger texts.

The advantage of a modern neural network architecture over an older established one becomes less significant as the data is preprocessed, cleaned, normalized, and as a larger training set is created. Modern neural network architectures are a result of the collaboration of several computer scientists and such models could be almost infeasible to implement on your own, and require too much computing power to train. *Time spent trying to replicate results from a recent scientific breakthrough may not be worth it and might be better spent on building the solution around a less modern but stable model **and** getting more training data*.

Once the architecture has been chosen, it is recommended to start with one or two layers, train a model to see if it fits the training data well (has a low bias). If not, gradually increase the size of each layer and the number of layers until the model perfectly fits the training data. Once this is the case, if the model does not perform well on the validation data (has a high variance), then add regularization to the model. If after adding regularization the model does not fit the training data anymore, then slightly increase the size of the network. Continue this process iteratively until the model fits both training and validation data well enough according to a chosen metric.

## Advanced Regularization

In neural networks, specific regularizers can be used besides L1 and L2 regularization: **dropout**, **early stopping**, and **batch-normalization**.

The concept of **dropout** is very simple. Each time a training example is ran through the network, temporarily exclude at random some units from the computation. The higher the percentage of units excluded the higher the regularization effect. The dropout parameter is in the range $[0, 1]$ and it has to be found experimentally by tuning it on the validation data.

**Early stopping** is the way to train a neural network by saving the preliminary model after every epoch and assessing the performance of the preliminary model on the validation set. Recall that as the number of epochs
increases, the cost decreases, which means that the model fits the training data well. However, at some point, after some epoch $e$, the model can start overfitting: the cost
keeps decreasing, but the performance of the model on the validation data deteriorates. Training can be stopped once a decreased performance on the validation set is observed. Models saved after each epoch are called **checkpoints**.

**Batch normalization** (or batch standardization) is a technique that consists of standardizing the outputs of each layer before the units of the subsequent layer receive them as input. In practice, batch normalization results in faster and more stable training, as well as some regularization effect. So it's always a good idea to try to use batch normalization.

Another regularization technique that can be applied not just to neural networks, but to virtually any learning algorithm, is called **data augmentation**. This technique is often used to regularize models that work with images. Once you have your original labeled training set, you can create a synthetic example from an original example by applying various transformations to the original image: zooming it slightly, rotating, flipping, darkening, and
so on. This often results in increased performance of the model in practice.

## Handling Multiple Inputs

Multiple inputs are also known as **multimodal data**, that is, multiple feature vectors for each example. It's hard to adapt **shallow learning** algorithms to work with multimodal data, but one shallow model could be trained on one type of feature and another model on another type, and the using some model combination technique.

If the problem cannot be divided problem into two independent subproblems, an attempt can be made at vectorizing each input (by applying the corresponding feature engineering method) and then simply concatenating two feature vectors together to form one wider feature vector.

With neural networks there is more flexibility. Two subnetworks can be built, one for each type of input. For example, a CNN subnetwork would read the image while an RNN
subnetwork would read the text. Both subnetworks have as their last layer an embedding: CNN has an embedding of the image, while RNN has an embedding of the text. These two embeddings can be concatenated and then a classification layer, such as softmax or sigmoid, is added on top of it.

## Handling Multiple Outputs

In some problems we want to predict multiple outputs for one input. Some problems with multiple outputs can
be effectively converted into a multi-label classification problem. However, in some cases the outputs are multimodal, and their combinations cannot be effectively enumerated.

Handling a situation where a model should detect an object on an image and returning its coordinates can be done as follows. Create one subnetwork that works as an encoder. It will read an input image using one or several convolution layers. The encoder's last layer would be the embedding of the image.

Then add two other subnetworks on top of the embedding layer: one that takes the embedding vector as input and predicts the coordinates of an object. This first subnetwork can have a *ReLU* as the last layer, which is a good choice for *predicting positive real numbers*, such as coordinates; this subnetwork could use the mean squared error cost $C_1$.

The second subnetwork will take the same embedding vector as input and predict the probabilities for each tag. This second subnetwork can have a *Softmax* as the last layer, which is appropriate for the *probabilistic output*, and use the averaged *negative log-likelihood* cost $C_2$ (also called **cross-entropy** cost).

Unfortunately it is impossible to optimize the two cost functions at the same time. By trying to optimize one we risk hurting the second one and the other way around. One solution is to add another hyperparameter $\gamma$ in the range $(0,1)$ and define the combined cost function as $\gamma C_1 + (1−\gamma)C_2$. Then tune the value for $\gamma$ on the validation data just like any other hyperparameter.

## Transfer Learning

Transfer learning is probably where neural networks have a unique advantage over the shallow models. In transfer learning, you pick an existing model trained on some dataset, and you adapt this model to predict examples from another dataset, different from the one the model was built on.

Transfer learning in neural networks works like this:

1. Build a deep model on the original big dataset.
2. Compile a much smaller labeled dataset for the second model.
3. Remove the last one or several layers from the first model. Usually, these are layers responsible for the classification or regression that usually follow the embedding layer.
4. Replace the removed layers with new layers adapted for the new problem.
5. Freeze the parameters of the layers remaining from the first model.
6. Use the smaller labeled dataset and gradient descent to train the parameters of only the new layers.

Often it is possible to find deep models for visual problem online. Download one that has high chances to be of use for the problem, remove several last layers (the quantity of layers to remove is a hyperparameter), add new prediction layers and train the model.

Transfer learning can help in situations when your problem requires a labeled dataset that is very costly to obtain, but when it is possible to get another dataset for which labels are more readily available. Much fewer annotated examples would be required than if the original problem was solved from scratch.

## Unsupervised Learning

Unsupervised learning deals with problems in which data doesn't have labels. The absence of labels representing
the desired behavior for a model means the absence of a solid reference point to judge the quality of that model.

### Density Estimation

Density estimation is a problem of modeling the probability density function (pdf) of the unknown probability distribution from which the dataset has been drawn. It is useful for novelty or intrusion detection.

For a one-class classification problem the pdf can be estimated with a **parametric** model, or more precisely a multivariate normal distribution (MND). Models can also be **non-parametric**, like the one used in kernel regression. It turns out that the same approach can work for density estimation.

Let $\{x_i\}_{i=1}^N$ be a one-dimensional dataset whose samples $x_i$ are drawn from an unknown pdf $f$ with $x_i\in\R$ for all $i=1,...,N$. The kernel model of $f$, denoted as $\hat{f}_b$ is given by

$$\hat{f}_b=\frac{1}{Nb}\sum_{i=1}^N k\left( \frac{x-x_i}{b} \right),$$

where $b$ is a hyperparameter controlling the tradeoff between the bias and variance of the model, and $k$ is a kernel such as the Gaussian kernel given by:

$$k(z) =\frac{1}{\sqrt{2\pi}}\exp(\frac{-z^2}{2}).$$

The value of $b$ should be such that the difference between the real shape of $f$ and the shape of model $\hat{f}_b$ is minimized. A reasonable choice of measure of this difference is called the **mean integrated squared error (MISE)**:

$$\text{MISE}(b) = \mathbb{E}\left[ \int_\R{(\hat{f}_b(x) - f(x))^2 ~ dx} \right]$$

Intuitively, we square the difference between the real pdf $f$ and the model of it $\hat{f}_b$. The integral $\int_\R$ replaces the summation $\sum_{i=1}^N$ used in the mean squared error, while the **expectation operator** $\mathbb{E}$ replaces the average $\frac{1}{N}$.

When the loss function has a continuous domain such as $(\hat{f}_b(x) - f(x))^2$, the summation has to be replaced with an integral. The expectation operation $\mathbb{E}$ means that we want $b$ to be optimial for all possible realizations of the training set $\{x_i\}_{i=1}^N$. That is important because $\hat{f}_b$ is defined on a *finite* sample of some probability distribution, while the real pdf $f$ is defined on an infinite domain (the set $\R$).

The right-hand side of the equation above can be rewritten as:

$$\mathbb{E}\left[ \int_\R \hat{f}_b^2(x) dx \right] - 2\mathbb{E} \left[ \int_\R \hat{f}_b(x) f(x) dx \right] + \mathbb{E} \left[ \int_\R f(x)^2dx \right].$$

The third term in this summation is independent of $b$ and thus be factored out. An unbiased estimator of the first term is given by $\int_\R\hat{f}_b^2(x)dx$ while the unbiased estimator of the second term can be approximated by **cross-validation**

$$-\frac{2}{N}\sum_{i=1}^N\hat{f}_b^{(i)}(x_i),$$

where $\hat{f}_b^{(i)}$ is a kernel model of $f$ computed on our training set with the example $x_i$ excluded.

> In statistics, the term $\sum_{i=1}^N\hat{f}_b^{(i)}(x_i)$ is known as the **leave one out estimate**. It is a form of cross-validation in which each fold consists of one example. Because $f$ is a pdf, the term $\int_\R \hat{f}_b(x) f(x) dx$ is the expected value of the function $\hat{f}_b$, and *this leave one out estimate* is an unbiased estimator of $\mathbb{E} \left[ \int_\R \hat{f}_b(x) f(x) dx \right]$.

To find the optimal value $b^*$ for b, the cost is minimized as:

$$\int_\R \hat{f}_b^2(x)dx - \frac{2}{N}\sum_{i=1}^N \hat{f}_b^{(i)}(x_i),$$

and $b^*$ can be found using **grid search**, where it is picked at the minimum of the grid search curve.

### Clustering

Clustering is a problem of learning to assign labels to examples by leveraging an unlabeled dataset. Deciding on whether the learned model is optimal is much more complicated than in supervised learning.

There are various clustering algorithms whose performance depends on the unknown properties of the probability distribution that the dataset was drawn from.

Some of the most useful and widely used clustering algorithms are k-means, HDBSCAN, and the Gaussian mixture model. The first two compute so-called **hard clustering** in which each example can belong to only one cluster.

#### K-Means

In **k-means** clustering $k$ number of clusters are chosen and $k$ feature vectors called **centroids** are randomly placed into the feature space.

Then the distance from each sample $\bold{x}$ to each centroid $\bold{c}$ is computed using some metric, like the Euclidean distance and the closest centroid is assigned to each sample. For each centroid, the average feature vector of the samples labeled with it is calculated. These average feature vectors become the new locations of the centroids.

Then recompute the distance from each sample to each centroid, modify the assignment and repeat the procedure until the assignments don't change after the centroid locations were recomputed. The model is the list of assignments of centroids ids to the samples.

The initial position of centroids influence the final positions, so two runs of k-means can result in two different models. Some variants of k-means compute the initial positions of centroids based on some properties of the dataset.

#### (H)DBSCAN

While k-means and similar algorithms are centroid-based, **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is a density-based clustering algorithm. Instead of guessing how many clusters are needed, DBSCAN defines two hyperparameters: $\eta$ and $n$. It starts by picking an example $\bold{x}$ from the dataset at random and assigning it to cluster 1. Then count how many examples have the distance from $\bold{x}$ less than or equal to $\eta$. If this quantity is greater than or equal to $n$, then put all these $\eta$-neighbors to the same cluster 1. Next, examine each member of cluster 1 and find their respective $\eta$-neighbors. If some member of cluster 1 has $n$ or more $\eta$-neighbors, expand cluster 1 by adding those $\eta$-neighbors to the cluster. Continue expanding cluster 1 until there are no more examples to put in and then pick another example from the dataset that does not belong to any cluster and put it to cluster 2. This process is continued until all examples either belong to some cluster or are marked as outliers. An outlier is an example whose $\eta$-neighborhood contains less than $n$ examples.

The advantage of DBSCAN is that it can build clusters that have an arbitrary shape, while k-means and other centroid-based algorithms create clusters that have a shape of a hypersphere. An obvious drawback of DBSCAN is that it has two hyperparameters and choosing good values for them (especially $\eta$) could be challenging. Furthermore, having $\eta$ fixed, the clustering algorithm cannot effectively deal with clusters of varying density.

**HDBSCAN** is a follow-up clustering algorithm that keeps the advantages of DBSCAN by removing the need to decide on the value of $\eta$. The algorithm is capable of building clusters of varying density. HDBSCAN only has one important hyperparameter: $n$, the minimum number of examples to put in a cluster. This hyperparameter is relatively simple to choose by intuition. HDBSCAN has very fast implementations and can deal with millions of examples effectively. Modern implementations of k-means are much faster than HDBSCAN, but the qualities of the latter may outweigh its drawbacks for many practical tasks.

> It recommended to always try applying HDBSCAN on the data first, and k-means for more specific cases.

#### Prediction Strength

There are various techniques for selecting $k$. None of them is proven optimal. Most of those techniques require the analyst to make an educated guess by looking at some metrics or by examining cluster assignments visually, but one numerical way of determining the reasonable number of clusters is based on the concept of **prediction strength**.

The idea is to split the data into a training set $\mathcal{S}_{tr}$ of length $N_{tr}$ and a test set $\mathcal{S}_{te}$ of length $N_{te}$, similarly to how it is done in supervised learning. Then fix $k$, the number of clusters, and run a clustering algorithm $C$ on sets $\mathcal{S}_{tr}$ and $\mathcal{S}_{te}$ and obtain the clustering results $C(\mathcal{S}_{tr},k)$ and $C(\mathcal{S}_{te},k)$.

Let $A$ be the clustering $C(\mathcal{S}_{tr},k)$ built using the training set. The clusters in $A$ can be seen
as regions. If an example falls within one of those regions, then that example belongs to some specific cluster.

Define the $N_{te} \times N_{te}$ **co-membership matrix** $\bold{D}[A, \mathcal{S}_{te}]$ as follows:

$$D[A,\mathcal{S}_{te}]^{(i,i')} = 1$$

if and only if example $\bold{x}_i$ and $\bold{x}_{i'}$ from the test set belong to the same cluster according to the clustering $A$. Otherwise:

$$D[A,\mathcal{S}_{te}]^{(i,i')} = 0.$$

Now a clustering $A$ with $k$ clusters has been built using the *training set*, and additionally a co-membership matrix that indicates whether two examples from the *test set* belong to the same cluster in $A$.

Intuitively, if the quantity $k$ is the reasonable number of clusters, then two examples that belong to the same cluster in clustering $C(\mathcal{S}_{te},k)$ will most likely also belong to the same cluster in clustering $C(\mathcal{S}_{tr},k)$. On the other hand, if $k$ is not reasonable (too high or too low), then training data-based and test data-based clustering will likely be inconsistent.

More formally, the **prediction strength** for the number of clusters $k$ is given by:

$$ps(k) \equiv \min_{j=1,...,k} \frac{1}{|A_j|(|A_j|-1)} \sum_{i,i'\in A_j} \bold{D}[A,\mathcal{S}_{te}]^{(i,i')}$$

where $A\equiv C(\mathcal{S}_{tr},k)$, $A_j$ is the $j\th$ cluster from the clustering $C(\mathcal{S}_{te},k)$ and $|A_j|$ is the number of examples in cluster $A_j$.

Given a clustering $C(\mathcal{S}_{tr},k)$, compute for each test cluster the proportion of observation pairs in that cluster that are also assigned to the same cluster by the training set centroids. The **prediction strength** is the minimum of this quantity over the $k$ test clusters.

> A reasonable number of clusters is the largest $k$ such that $ps(k) > 0.8$.

For non-deterministic clustering algorithms such as k-means, which can generate different clusterings depending on the initial position of centroids, it is recommended to do multiple runs of the clustering algorithm for the same $k$ and compute the average prediction strength $\bar{ps}(k)$ over multiple runs.

Another effective method to estimate the number of clusters is the **gap statistic** method. Other, less automatic methods include the **elbow method** and the **average silhoutte method**.

Both DBSCAN and k-means compute so-called **hard clustering**, in which each example belongs to only one cluster.

#### Gaussian Mixture Model Estimation

**Gaussian Mixture Model (GMM)** allows each example to be a member of several clusters with different *membership score*. Computing a GMM is very similar to doing model-based density estimation. In GMM, instead of having just one multivariate normal distribution (MND), we have a weighted sum of several MNDs:

$$f_X=\sum_{j=1}^k \phi_j f_{\bold{\mu}_j​,\bold{\Sigma}_j},$$

where $f_{\bold{\mu}_j​,\bold{\Sigma}_j}$ is the $j\th$ MND, and $\phi_j$ is its weight in the sum. The values of parameters $\bold{\mu}_j$, $\bold{\Sigma}_j$, and $\phi_j$, for all $j=1,...,k$ are obtained using the **expectation maximization (EM)** algorithm to optimize the **maximum likelihood** criterion.

There are $k$ Gaussian distributions defined for $j=1,...,k$:

$$f(x|\mu_j,\sigma_j^2) = \frac{1}{\sqrt{2\pi\sigma_j^2}} \exp{(-\frac{(x-\mu_j)^2}{2\sigma_j^2})}$$

where $f(x|\mu_j,\sigma_j^2)$ is a probability distribution function defining the likelihood of $X=x$.

The expectation maximization algorithm is used to estimate parameters $\mu_j$, $\sigma_j^2$, and $\phi_j$ for all $j=1,...,k$. The parameters $\phi_j$ are useful for the density estimation and less useful for clustering.

Expectation maximization works as follows. An initial guess is made for all $\mu_j$ and $\sigma_j^2$, and $\phi_1 = ... = \phi_j = ... = \phi_k = \frac{1}{k}$ for each $\phi_j$, $j\in1,...,k$.

At each iteration of EM, the following 4 steps are executed:

1. For all $i=1,...,N$ and $j=1,...,k$ calculate the likelihood of each $x_i$:

$$f(x_i|\mu_j,\sigma_j^2) = \frac{1}{\sqrt{2\pi\sigma_j^2}} \exp{(-\frac{(x_i-\mu_j)^2}{2\sigma_j^2})}.$$

2. Using **Bayes' Rule**, for each example $x_i$, calculate the likelihood $b_i^{(j)}$ that the example belongs to cluster $j\in\{1,...,k\}$, or in other words, the likelihood that the example was drawn from the Gaussian j:

$$b_i^{(j)} \leftarrow \frac{f(x_i|\mu_j,\sigma_j^2)\phi_j}{\sum_{j=1}^k f(x_i|\mu_j,\sigma_j^2)\phi_j}.$$

Parameter $\phi_j$ reflects how likely it is that Gaussian distribution $j$ with parameters $\mu_j$ and $\sigma_j^2$ may have produced this dataset. That is why all $\phi$ was set to an equal amount of $\frac{1}{k}$; since it is unknown how likely each of the Gaussians is, this is reflected by setting their likelihood to an equal amount.

3. Compute the new values of $\mu_j$ and $\sigma_j^2$, $j\in\{1,...,k\}$ as:

$$\mu_j \leftarrow \frac{\sum_{i=1}^N b_i^{(j)}x_i}{\sum_{i=1}^N b_i^{(j)}} ~~ \text{and} ~~ \sigma_j^2 \leftarrow \frac{b_i^{(j)}(x_i-\mu_j)^2}{\sum_{i=1}^N b_i^{(j)}}.$$

4. Update $\phi_j, j\in\{1,...,k\}$ as:

$$\phi_j \leftarrow \frac{1}{N} \sum_{i=1}^N b_i^{(j)}.$$

Steps 1-4 are executed iteratively until the values of $\mu_j$ and $\sigma_j^2$ do not change much anymore; for example the delta is below some threshold $\epsilon$.

The EM algorithm is very similar to the k-means algorithm: start with random clusters, then iteratively update each cluster's parameters by averaging the data that is assigned to that cluster. The only difference in the case of GMM is that the assignment of an example $x_i$ to the cluster $j$ is **soft**: $x_i$ belongs to cluster $j$ with probability $b_i^{(j)}$. This is why the new values for $\mu_j$ and $\sigma_j^2$ are calculated as a **weighted average** with weights $b_i^{(j)}$.

Once parameters $\mu_j$ and $\sigma_j^2$ have been learned for each cluster $j$, the membership score of example $x$ in cluster $j$ is given by $f(x|\mu_j,\sigma_j^2)$.

The equations above described one-dimensional data. The extension to $D$-dimensional data ($D>1$) is straightforward: instead of variance $\sigma^2$ the multinomial normal distribution (MND) is parameterized by the covariance matrix $\Sigma$ instead.

Whereas k-means have clusters that can only be circular, the clusters in GMM have ellipoid forms with arbitrary elongation and rotation. The values in the covariance matrix control these properties.

> To choose the right $k$ in GMM, there is no clear-cut method. It is recommended to try different $k$ values and build a different model $f_{tr}^k$ for each $k$ on the training data. Then pick the value of $k$ that maximizes the likelihood of examples in the test set:
> $$\argmax_k \prod_{i=1}^{|N_{te}|} f_{tr}^k(\bold{x}_i)$$
> where $|N_{te}|$ is the size of the test set.

For some datasets other clustering algorithms like **spectral clustering** and **hierarchical clustering** may be more appropriate. However, in most practical cases, k-means, HDBSCAN, and GMM are satisfactory.

### Dimensionality Reduction

**Dimensionality reduction** techniques are used less in
practice now than in the past. The most frequent use case for dimensionality reduction is data visualization: humans can only interpret a maximum of three dimensions on a plot.

Reducing data to lower dimensionality and figuring out which quality of the original example each new feature in the reduced feature space reflects allows use of simpler algorithms. Dimensionality reduction removes redundant or highly correlated features and also reduces the noise in the data.

Three widely used techniques of dimensionality reduction are **principal component analysis (PCA)**, **uniform manifold approximation and projection (UMAP)**, and **autoencoders**.

In an autoencoder, the low-dimensional output of the bottleneck layer can be used as the vector of reduced dimensionality that represents the high-dimensional input feature vector. This low-dimensional vector represents the essential information contained in the input vector because the autoencoder is capable of reconstructing the input feature vector based on the bottleneck layer output alone.

#### Principal Component Analysis

**Principal component analysis (PCA)** is one of the oldest dimensionality reduction methods.

Principal components are vectors that define a new coordinate system in which the first axis goes in the direction of the *highest variance* in the data. The second axis is orthogonal to the first one and goes in the direction of the *second highest variance* in the data. If the data is three-dimensional then the third axis would be orthogonal to both the first and the second axes and would go in the direction of the third highest variance, etc.

In order to reduce the dimensionality of data to $D_{new} < D$, pick $D_{new}$ largest principal components and project data points on them.

When data is very high-dimensional, it often happens in
practice that the first two or three principal components account for most of the variation in the data, so by displaying the data on a 2D or 3D plot it is indeed possible to see very high-dimensional data and its properties.

#### Uniform Manifold Approximation and Projection

The idea of most modern dimensionality reduction algorithms is similar: first a similarity metric is designed for two examples. For visualization purposes, besides the Euclidean distance, this similarity metric often reflects some local properties of the two examples, such as the density of other examples around them.

In **Uniform Manifold Approximation and Projection (UMAP)**, the similarity metric $w$ is defined as:

$$w(\bold{x}_i, \bold{x}_j) \equiv w_i(\bold{x}_i,\bold{x}_j) + w_j(\bold{x}_j,\bold{x}_i) - w_i(\bold{x}_i,\bold{x}_j)w_j(\bold{x}_j,\bold{x}_i).$$

Function $w_i(\bold{x}_i,\bold{x}_j)$ is defined as:

$$w_i(\bold{x}_i,\bold{x}_j) \equiv \exp\left(-\frac{d(\bold{x}_i,\bold{x}_j) - \rho_i}{\sigma_i}\right),$$

where $d(\bold{x}_i,\bold{x}_j)$ is the Euclidean distance between two examples, $\rho_i$ is the distance from $\bold{x}_i$ to its closest neighbor, and $\sigma_i$ is the distance from $\bold{x}_i$ to its $k^{\th}$ closest neighbor (where $k$ is a hyperparameter).

The UMAP similarity metric varies in the range from 0 to 1 and is symmetric, which means that $w(\bold{x}_i,\bold{x}_j) \equiv w(\bold{x}_j,\bold{x}_i).$

Let $w$ denote the similarity of two examples in the original high-dimensional space and let $w^\prime$ be the similarity given by the similarity metric equation in the new low-dimensionality space.

> A **fuzzy set** is a generalization of a set. For each element $x$ in a fuzzy set $\mathcal{S}$, there's a membership function $\mu_\mathcal{S}(x)\in[0,1]$ that defines the *membership strength* of $x$ to the set $\mathcal{S}$. $x$ *weakly belongs to* a fuzzy set $\mathcal{S}$ if $\mu_\mathcal{S}(x)$ is close to zero. On the other hand, if $\mu_\mathcal{S}(x)$ is close to 1, then $x$ has a strong membership in $\mathcal{S}$. If $\mu(x)=1$ for all $x\in\mathcal{S}$, then a fuzzy set $\mathcal{S}$ becomes equivalent to a normal, nonfuzzy set.

Because the values of $w$ and $w^\prime$ lie in the range between 0 and 1, $w(\bold{x}_i,\bold{x}_j)$ can be seen as membership of the pair of examples $(\bold{x}_i,\bold{x}_j)$ in a certain fuzzy set. The same can be said about $w^\prime$. The notion of similarity of two fuzzy sets is called **fuzzy set cross-entropy** and is defined as:

$$C_{w,w^\prime} = \sum_{i=1}^N \sum_{j=1}^N \left[w(\bold{x}_i,\bold{x}_j)\ln\left( \frac{w(\bold{x}_i,\bold{x}_j)}{w^\prime(\bold{x}_i^\prime,\bold{x}_j^\prime)} \right) + (1-w(\bold{x}_i,\bold{x}_j))\ln\left(\frac{1-w(\bold{x}_i,\bold{x}_j)}{1-w^\prime(\bold{x}_i^\prime,\bold{x}_j^\prime)}\right)\right],$$

where $\bold{x}^\prime$ us the low-dimensional version of the original higher-dimensional example $\bold{x}$.

The unknown parameters are $\bold{X}^\prime_i$ for all $i = 1,...,N$, the low-dimensional examples that need to be found. They can be computed with gradient descent by minimizing $C_{w,w^\prime}$.

> In practice, *UMAP* is slightly slower than *Principle Component Analysis* but faster than *autoencoder*.

### Outlier Detection

**Outlier detection** is the problem of detecting the examples in the dataset that are very different from what a typical example in the dataset looks like. Techniques like autoencoder and one-class classifier learning are suited to solving this problem.

If an autoencoder is used it can be trained on the dataset and use this model during prediction to reconstruct the example from the *bottleneck layer*. The model will unlikely be capable of reconstructing an outlier. In other words, if it fails it means that the example is an outlier.

In one-class classification, the model either predicts that the input example belongs to the class or not, and thus is an outlier.

## Metric Learning

A metric is a function of two variables that satisfies the following three conditions:

$$
\begin{align*}
\text{1.} &~~~ d(\bold{x},\bold{x}^\prime) \geq 0 & \text{non-negativity}\\
\text{2.} &~~~ d(\bold{x},\bold{x}^\prime) \leq d(\bold{x},\bold{z}) + d(\bold{z},\bold{x}^\prime) & \text{triangle inequality}\\
\text{3.} &~~~ d(\bold{x},\bold{x}^\prime) = d(\bold{x}^\prime,\bold{x}) & \text{symmetry}
\end{align*}
$$

The most frequently used metrics of similarity between two feature vectors are **Euclidean distance** and **cosine similarity**.

> The fact that one metric can work better than another depending on the dataset indicates that none of them are perfect in general. Choosing a good metric can be done by learning it from data.

The equation for the Euclidean distance between two feature vectors $\bold{x}$ and $\bold{x}^\prime$

$$d(\bold{x},\bold{x}^\prime) = \|\bold{x} - \bold{x}^\prime\| \equiv \sqrt{(\bold{x} - \bold{x}^\prime)^2} = \sqrt{(\bold{x} - \bold{x}^\prime)(\bold{x} - \bold{x}^\prime)}$$

can be modified to make it parametrizable and then learn the parameters for thisv metric from data as follows:

$$d_{\bold{A}}(\bold{x},\bold{x}^\prime) = \|\bold{x} - \bold{x}^\prime\|_{\bold{A}} \equiv \sqrt{(\bold{x} - \bold{x}^\prime)^\intercal\bold{A}(\bold{x} - \bold{x}^\prime)},$$

where $\bold{A}$ is a $D \times D$ matrix, where $D$ is the dimension.

If $\bold{A}$ is the identity matrix with $D=3$, where

$$\bold{A} \equiv \left[
\begin{matrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{matrix} \right],$$

then $d_{\bold{A}}$ is the Euclidean distance.

In order to satisfy the *non-negativity* and *triangle inequality* conditions for a metric, matrix $\bold{A}$ has to be *positive semidefinite*; a generalization of the notion of a non-negative real number to matrices. Any positive semidefinite matrix $\bold{M}$ satisfies:

$$\bold{z}^\intercal\bold{M}\bold{z} \geq 0,$$

for any vector $\bold{z}$ having the same dimensionality as the number of rows and columns in $\bold{M}$. To satisfy the *symmetry* condition, simply take $(d(\bold{x},\bold{x}^\prime) + d(\bold{x}^\prime,\bold{x}))/2$.

To build the training data for a metric learning problem, manually create two sets $\mathcal{S}$ and $\mathcal{D}$. Assume an unanotated set $\mathcal{X} = \{\bold{x}_i\}_{i=1}^N$. Then, a pair of examples $(\bold{x}_i,\bold{x}_k)$ belongs to $\mathcal{S}$ if $\bold{x}_i$ and $\bold{x}_k$ are similar and to $\mathcal{D}$ otherwise.

Training the matrix of parameters $\bold{A}$ from the data involves finding a positive semidefinite matrix $\bold{A}$ that solves the following optimization problem:

$$\min_A \sum_{(\bold{x}_i,\bold{x}_k)\in\mathcal{S}} \|\bold{x}-\bold{x}^\prime\|_\bold{A}^2$$

such that

$$\|\bold{x}-\bold{x}^\prime\|_\bold{A} \geq c,$$

where $c$ is a positive scalar constant.

The solution to this optimization problem is found by gradient descent with a modification that ensures that the found matrix $\bold{A}$ is positive semidefinite.

> **One-shot learning** with **siamese networks** and **triplet loss** can be seen as metric learning problem: the pairs of pictures of the same person belong to set $\mathcal{S}$, while pairs of random pictures belong to set $\mathcal{D}$.

> There are many other ways to learn a metric as well, including non-linear and kernel-based. However, the one described above as well as one-shot learning should suffice for most applications.

## Rank Learning

**Rank learning** is a supervised learning problem. One frequent problem solved using this technique is the optimization of results returned by a search engine for a query.

The goal of the learning algorithm is to find a ranking function $f$ that outputs values that can be used to rank *documents*. An ideal ranking function would output values that induce the same ranking of documents as given by the labels for each training example.

A labeled example $\mathcal{X}_i, i = 1,...,N$, is a collection of feature vectors with labels:

$$\mathcal{X}_i = \{(\bold{x}_{i,j},y_{i,j})\}_{j=1}^{r_i},$$

where $N$ is the size of training set $\mathcal{X}$, and $r_i$ is the size of a ranked collection of  documents. The features in feature vector $\bold{x}_{i,j}$ represent the document $j=1,...,r_i$.

> For example, $x_{i,j}^{(1)}$ could represent how recent a document is, $x_{i,j}^{(2)}$ could represent the word hit, $x_{i,j}^{(3)}$ the document size, etc. The label $y_{i,j}$ could be the rank $(1,2,...,r_i)$ or a score.

There are three approaches to solve the ranking problem: **pointwise**, **pairwise**, and **listwise**.

### Pointwise Ranking

In the pointwise approach each training example is transformed into multiple examples; one per document. The learning problem becomes a standard supervised learning problem, either regression or logistic regression. In each example $(\bold{x},y)$ of the pointwise learning problem, $\bold{x}$ is the feature vector of some document, and $y$ is the original score (if $y_{i,j}$ is a score) or a synthetic score obtained from the ranking.

Any supervised learning algorithm can be used. The solution is usually approximate because each document is considered in isolation, while the original ranking given by the labels $y_{i,j}$ of the original training set could optimize the positions of the whole set of documents.

### Pairwise Ranking

In the pairwise ranking approach a pair of documents are considered in isolation. Given a pair of documents $(\bold{x}_i,\bold{x}_k)$ a model $f$ is built which outputs a value close to 1 if the rank of $\bold{x}_i$ is higher than the rank of $\bold{x}_k$; and a value close to 0 otherwise. At test time, the final ranking for an unlabeled example $\mathcal{X}$ is obtained by aggregating the predictions for all pairs of documents in $\mathcal{X}$.

> The pairwise method works better than pointwise, but is still suffers from isolation problems.

### Listwise Ranking

In a listwise approach the model is optimized directly on some metric that reflects the quality of ranking. There are various metrics for asssessing search engine result ranking, including *precision* and *recall*. A popular metric that combines both is called **mean average precision (MAP)**.

To define mean average precision, a collection of of search results for a query is examined by judges (or *rankers*) and each one is assigned relevancy labels. Labels could be binary (relevant/irrelevant) or on some scale (1-10) where higher values correspond to higher relevancy. Then, the ranking model is tested on this collection.

The **precision** of the model for some query is given by:

$$\text{precision} = \frac{m + n}{n}$$

$$\text{precision} = \frac{|\{\text{relevant documents}\} \cap \{\text{retrieved documents}\}|}{|\{\text{retrieved documents}\}|},$$

where $|\cdot|$ means "number of". The **average precision (AveP)** metric is defined for a ranked collection of ducments returned by a search engine for some query $q$ as:

$$\text{AveP}(q) = \frac{\sum_{k=1}^n (P(k)\cdot\text{rel}(k))}{|\{\text{relevant documents}\}|},$$

where $n$ is the number of retrieved documents, $P(k)$ denotes the precision computed for the top $k$ search results returned for the query by the model, $\text{rel}(k)$ is an indicator function that returns 1 if the item at rank $k$ is a relevant document (according to judges) and 0 otherwise.

Finally, the mean average precision for a collection of search queries of size $Q$ is given by:

$$\text{MAP} = \frac{\sum_{q=1}^Q\text{AveP}(q)}{Q}.$$

#### LambdaMART

One state of the art ranking algorithm called **LambdaMART** implements a listwise approach, and uses gradient boosting to train ranking function $h(\bold{x})$. Then, binary model $f(\bold{x}_i,\bold{x}_k)$ that predicts whether the document $\bold{x}_i$ should have a higher rank than document $\bold{x}_k$ (for the same search query) is given by a sigmoid with a hyperparameter $\alpha$:

$$f(\bold{x}_i,\bold{x}_k) \equiv \frac{1}{1+\exp(h(\bold{x}_i) - h(\bold{x}_k))\alpha}.$$

As with many models that predict probability, the cost function is cross-entropy computed using model $f$. Function $h$ is built by combining multiple regression trees to try to minimize the cost.

> Recall that in gradient boosting a tree is added to the model to reduce the error that the current model makes on the training data. For the classification problem, the derivative of the cost function replaced the real labels of training examples.

LamdaMART works similarly but replaces the real gradient with a combination of the gradient and another factor that depends on the metric, such as *mean average precision*. This factor modifies the original gradient by increasing or decreasing it so that the metric value is improved. This means that in LambdaMART the metric is optimized directly.

> Typical supervised learning algorithms optimize the cost instead of the metric because metrics are usually not differentiable.

To build a ranked list of results based on the predictions of the model $f$ any ranker can be used, and a straightforward approach is to use an existing sorting algorithm to sort the documents.

## Recommendation Learning

**Learning to recommend** is an approach to building *recommender systems*. This is a system that will recommend content to a user based on their consumption history.

Two traditional approaches to do this are **content-based filtering** and **collaborative filtering**.

### Content-Based Filtering

This approach consists of learning what users like based on the description of the consumed content. We can create one training *per user* and add articles to this dataset as a feature vector $\bold{x}$ and whether the user has recently consumed this article as label $y$. Then a model of each user is built and can regularly examine each new piece of content to determine whether a specific user would consume it or not.

> Content-based approaches have a problematic limitation where a user can get trapped in a so-called filter bubble, and only suggests new content that is very similar to what was already consumed.

### Collaborative Filtering

In collaborative filtering the recommendations to one user are computed based on what other users consume or rate.

Information on user preferences is organized in a matrix, where each row corresponds to a user and each column corresponds to a piece of content that that user rated or consumed. This matrix is usually huge and extremely sparse, which makes it difficult to make meaningful recommendations. Another drawback of this approach is that the content of the recommended items is ignored.

### Hybrid Approach

Most real-world recommender systems use a hybrid approach by combining recommendations obtained by the content-based and collaborative filtering models. Two effective recommender system learning algorithms are **factorization machines (FM)** and **denoising autoencoders**.

### Factorization Machine

**Factorization machine** is a relatively new kind of algorithm, explicitly designed for sparse datasets.

Trying to fit a regression or classifcation model to an extremely sparse dataset would result in poor generalization. Factorization machines approach this problem differently.

A factorization machine is defined as follows:

$$f(\bold{x}) \equiv b + \sum_{i=1}^D w_ix_i + \sum_{i=1}^D \sum_{j=i+1}^D(\bold{v}_i\cdot\bold{v}_j)x_ix_j.$$

where $b$ and $w_i, i=1,...,D$, are scalar parameters similar to those used in linear regression. Vectors $\bold{v}_i$ are $k$-dimensional vectors of **factors**, where $k \ll D$ is a hyperparameter. The expression $\bold{v}_i\cdot\bold{v}_j$ is a dot-product of the $i^{\th}$ and $j^{\th}$ vectors of factors.

> Instead of looking for one wide vector of parameters, which poorly reflects interactions between features because of sparsity, it is completed by additional parameters that apply to pairwise interactions $x_ix_j$ between features. However, instead of having a parameter $w_{i,j}$ for each interaction, which would add $D(D-1)$ number of new parameters to the model, $w_{i,j}$ is factorized into $\bold{v}_i\bold{v}_j$ by adding only $Dk \ll D(D-1)$ parameters to the model.

The loss function could be squared error loss (for regression) or hinge loss. For classification with $y \in \{-1,+1\}$, with hinge or logistic loss the prediction is made as:

$$y = \text{sign}(f(x)).$$

Logistic loss is defined as:

$$loss(f(\bold{x}),y) = \frac{1}{\ln2}\ln(1+\epsilon^{-yf(\bold{x})}).$$

Gradient descent can be used to optimize the average loss. The **one versus rest** strategy can be used to convert this multiclass problem into five binary classification problems.

### Denoising Autoencoders

A **denoising autoencoder** is a neural network that reconstructs its input from the *bottleneck layer*. The fact that the input is corrupted by noise while the output should not be makes denoising autoencoders an ideal tool to build a recommender model.

> The idea is to suggest new content as if they were removed from the complete set of preferred movies by some corruption process. The goal of the denoising autoencoder is to reconstruct those "removed" items.

At training time, randomly replace some of the (non-zero) rated content features in the input feature with zeros. Train the autoencoder to reconstruct the uncorrupted input.

At prediction time, build a feature vector for the user which will include uncorrupted rated content features as well as the handcrafted features. Use the trained denoising autoencoder to reconstruct the uncorrupted input. Recommend content to the user that has the highest scores at the model's output.

## Word Embeddings

**Word embeddings** are feature vectors that represent words. One algorithm that works well in practice is **word2vec**, and in particular a variant called **skip-gram**.

In word embedded learning the goal is to build a model which can be used to convert a one-hot encoding of a word into a word embedding.

The context of a word lets you predict the word they surround. It's also how a machine can learn that various words have a similar meaning; because they share similar contexts in multiple texts.Moreover, this also works the other way around: a word can predict the context that surround it.

A skip-gram is a piece of sentence with a certain window size of words surrounding the predicted word. A skip-gram with a window size of 7 (3+1+3) is defined like this:

$$[\bold{x}_{-3},\bold{x}_{-2},\bold{x}_{-1},\bold{x},\bold{x}_{+1},\bold{x}_{+2},\bold{x}_{+3}].$$

The skip-gram model is a fully-connected network like the multilayer perceptron. The neural network has to learn to predict the context of words of the skip-gram given the central word.

This kind of learning is called **self-supervised**: the labeled examples get extracted from the unlabeled data such as text.

The activation function used in the output layer is *softmax*. The cost function is the negative log-likelihood. The embedding for a word is obtained as the output of the embedded layer when the one-hot encoding of this word is given as the input to the model.

Because of the large number of parameters in the word2vec models, two techniques are used to increase the computational efficiency: *hierarchical softmax* and *negative sampling*. Hierarchical softmax is an efficient way of computing softmax that consists of representing the outputs of softmax as leaves of a binary tree. The idea of negative sampling is only to update a random sample of all outputs per iteration of gradient decent.

## Other Machine Learning Algorithms

- Topic Modeling
  - Latent Dirichlet Allocation (LDA)
- Gaussian Processes (GP)
- Generalized Linear Model (GLM)
- Probabilistic Graphical Model (PGM)
  - Conditional Random Fields (CRF)
  - Bayesian networks
  - Belief networks
  - Probabilistic independence networks
- Markov Chain Monte Carlo (MCMC)
- Generative Adversarial Networks (GAN)
- Genetic Algorithms (GA)
- Reinforcement Learning(RL)
  - Q-learning
