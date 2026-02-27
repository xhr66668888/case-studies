**[Submission Instruction]{style="color: blue"}:** Submit your work as a
PDF (**handwritten scanned or typed, then converted to PDF**). Include
the required **screenshots** for each question.

1.  **PCA: Explained Variance + Low-D Representation**

    Assume PCA has already been run on a **mean-centered** dataset with
    **5 features**. The eigenvalues of the covariance matrix (sorted
    from largest to smallest) are:
    $$\lambda_1 = 5, \; \lambda_2 = 3, \; \lambda_3 = 2, \; \lambda_4 = 1, \; \lambda_5 = 1.$$

    The corresponding orthonormal eigenvectors are:
    $$v_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \;
        v_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}, \;
        v_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 0 \end{bmatrix}, \;
        v_4 = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}, \;
        v_5 = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 1 \end{bmatrix}.$$

    **[Part A]{style="color: blue"}:** Find the smallest $d$ such that
    the explained variance ratio is at least 80%:
    $$\frac{\lambda_1 + \cdots + \lambda_d}{\lambda_1 + \lambda_2 + \lambda_3 + \lambda_4 + \lambda_5} \ge 0.80.$$

    **[Part B]{style="color: blue"}:** For the **mean-centered** sample
    $$x = \begin{bmatrix} 2 \\ -1 \\ 3 \\ 0 \\ 1 \end{bmatrix},$$
    compute its $d$-dimensional PCA representation $$z = V_d^\top x,$$
    where $V_d = [v_1 \; v_2 \; \cdots \; v_d]$.

2.  **Kernel PCA: Does a Feature Map Define a Valid Kernel?**

    In Kernel PCA, a **kernel** is a function
    $k : \mathbb{R}^m \times \mathbb{R}^m \to \mathbb{R}$ that measures
    similarity between two inputs $x$ and $z$. A kernel is especially
    useful if it lets us compute a **high-dimensional dot product**
    using only the original low-dimensional variables (the kernel
    trick).

    Consider 2D inputs $x = [x_1, x_2]$ and the feature map:
    $$\phi(x) = \begin{bmatrix} 1 \\ \sqrt{2}x_1 \\ \sqrt{2}x_2 \\ x_1^2 \\ \sqrt{2}x_1x_2 \\ x_2^2 \end{bmatrix}.$$

    Define the kernel by: $$k(x, z) = \phi(x)^\top \phi(z),$$ where
    $x = [x_1, x_2]$ and $z = [z_1, z_2]$. **[Part
    A]{style="color: blue"}:** Compute the **high-dimensional dot
    product** $\phi(x)^\top \phi(z)$ explicitly (expand and simplify).

    **[Part B]{style="color: blue"}:** Define the following
    **low-dimensional kernel expression**:
    $$\tilde{k}(x, z) = (x^\top z + 1)^2.$$ Expand $\tilde{k}(x, z)$
    into terms involving $x_1, x_2, z_1, z_2$.

    **[Part C]{style="color: blue"}:** Compare your final expressions
    from Parts A and B. Based on this comparison, does the kernel
    definition $k(x, z) = \phi(x)^\top \phi(z)$ correctly demonstrate
    the **kernel trick** (i.e., a high-dimensional dot product computed
    using a low-dimensional dot product)?

3.  **PCA + Scaling (Notebook: Part I) --- Key Interpretation**

    This question is based on the notebook:
    `Case_Study_Dimensionality_Reduction_UCI_Wine_Dataset.ipynb`

    **[Part A]{style="color: blue"}:** Does feature scaling matter for
    PCA? Briefly explain **why**.

    **[Part B]{style="color: blue"}:** The notebook applies PCA(2) after
    three scalers: **Min-Max**, **Standard**, and **Robust**. Which
    scaler is the **most suitable** for this implementation (justify
    using the **class separability** you observe in the PCA(2) plots)?
    Keep your answer concise.

    **[Note]{style="color: blue"}:** In scikit-learn, `PCA` performs
    **mean-centering internally** on the input it receives (i.e., it
    subtracts the per-feature mean of the data passed to `PCA.fit`).

4.  **Kernel PCA (RBF) --- Effect of $\gamma$**

    This question is based on the notebook:
    `Case_Study_Dimensionality_Reduction_UCI_Wine_Dataset.ipynb`

    **[Task]{style="color: blue"}:** In the Kernel PCA (RBF) section,
    run: $$\gamma \in \{0.1, 0.5, 1\}.$$ Identify which $\gamma$
    produces the **best 2D class separability** (by visual inspection),
    include a **screenshot** of that plot, and briefly explain how
    changing $\gamma$ affects the embedding.

    **[Reference]{style="color: blue"}:**
    <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html>

5.  **UMAP --- Parameter Sensitivity**

    This question is based on the notebook:
    `Case_Study_Dimensionality_Reduction_UCI_Wine_Dataset.ipynb`

    **[Task]{style="color: blue"}:** In the UMAP section, try the grid:
    $$n\_neighbors \in \{5, 10, 15\}, \quad min\_dist \in \{0.1, 0.5, 1\}.$$
    Find the configuration that gives the **best class separability**
    (by visual inspection), include a **screenshot** of that plot, and
    briefly discuss what changes when you vary $n\_neighbors$ and
    $min\_dist$.

    **[Reference]{style="color: blue"}:**
    <https://umap-learn.readthedocs.io/en/latest/parameters.html#min-dist>
