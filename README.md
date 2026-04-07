# Questions around strict saddle and nonlinear conjugate gradient

Starting point: [Paper by Chan--Renous-Legoubin and Royer on nonlinear CG (NCG) for general nonconvex functions with applications to nonconvex regression](https://arxiv.org/abs/2201.08568)


## Question 1: Can we analyze NCG on strict saddle functions?

- Definition of strict saddle: [A paper by Rong Ge et al on matrix problems](https://proceedings.mlr.press/v70/ge17a.html)
- See also [Wright and Ma, Chapter 7](https://book-wright-ma.github.io/Book-WM-20210422.pdf)

*First step: Analyze a method combining NCG with negative curvature in the spirit of (Wright and Ma, section 9.3)*

## Question 2: Can we show that NCG avoids strict saddle points almost surely?

- Original result for gradient descent: [First-order methods almost always avoid strict saddle points](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=GR_DsT0AAAAJ&citation_for_view=GR_DsT0AAAAJ:0urtJCGzaFQC)
- Recent revisit by EPFL group (adding line searches): [Gradient descent avoids strict saddles with a simple line-search method too](https://arxiv.org/abs/2507.13804)
- More on the theorem by the same EPFL group: [On center-stable theorem](https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=o6tPrZAAAAAJ&sortby=pubdate&citation_for_view=o6tPrZAAAAAJ:NhqRSupF_l8C)

*First step: Understand the line-search variant to see what assumptions are needed.*

## Question 3: Can we show that the nonconvex regression problem is strict saddle?

- Same references than in question 1, but different focus (how to show that a function is strict saddle)

*First step: Consider the nonconvex regression problem with A=[1 0;-1 0;0 1;0 -1] and b=[1;-1;1;1] as a toy example.*
