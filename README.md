# Questions around strict saddle and nonlinear conjugate gradient

Starting point: [Paper by Chan--Renous-Legoubin and Royer on nonlinear CG (NCG) for general nonconvex functions with applications to nonconvex regression](https://arxiv.org/abs/2201.08568)


## Question 1: Can we analyze NCG on strict saddle functions?

- Definition of strict saddle: [A paper by Rong Ge et al on matrix problems](https://proceedings.mlr.press/v70/ge17a.html)
- See also [Wright and Ma, Chapter 7](https://book-wright-ma.github.io/Book-WM-20210422.pdf)

*Step 1.1: Analyze a method combining NCG with negative curvature in the spirit of (Wright and Ma, section 9.3) for generic nonconvex functions assuming access to Lipschitz constants.*

*Step 1.2: Adapt the analysis to the strict saddle setting.*

*Step 1.3: Replace the use of Lipschitz constants by line searches.* (see Wright and Ma or [Royer and Wright](https://arxiv.org/abs/1706.03131) for illustration).*

## Question 2: Can we show that NCG avoids strict saddle points almost surely?

- Original result for gradient descent: [First-order methods almost always avoid strict saddle points](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=GR_DsT0AAAAJ&citation_for_view=GR_DsT0AAAAJ:0urtJCGzaFQC)
- Recent revisit by EPFL group (adding line searches): [Gradient descent avoids strict saddles with a simple line-search method too](https://arxiv.org/abs/2507.13804)
- More on the theorem by the same EPFL group: [On center-stable theorem](https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=o6tPrZAAAAAJ&sortby=pubdate&citation_for_view=o6tPrZAAAAAJ:NhqRSupF_l8C)
- Advanced results show complexity bounds for perturbed variants of gradient descent [How to escape saddle points efficiently](https://proceedings.mlr.press/v70/jin17a.html). See also [A practical randomized trust-region method to escape saddle points in high dimension](https://arxiv.org/abs/2603.15494) for a recent work by the Boumal group on such results in the context of trust-region methods and [Understanding the Implicit Regularization of Gradient Descent in Over-parameterized Models](https://arxiv.org/abs/2505.17304?) for an alternate perturbation technique.

*Step 2.1. Adapt the GD analysis to show that NCG also escapes saddle points (with fixed stepsize).*

*Step 2.2. Study the line-search paper to show that NCG with line search also escapes saddle points.*

*Step 2.3. Propose a perturbed variant?*

## Question 3: Can we show that the nonconvex regression problem is strict saddle?

- Same references than in question 1, but different focus (how to show that a function is strict saddle)

*Step 3.1: Consider the nonconvex regression problem with A=[1 0;-1 0;0 1;0 -1] and b=[1;-1;1;1] as a toy example.*
