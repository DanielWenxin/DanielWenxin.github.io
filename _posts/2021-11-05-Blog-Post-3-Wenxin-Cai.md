# Wenxin's Blog Post 4

For Blog Post 4, we are aim to write a tutorial about *spectral clustering* algorithm for clustering data points. Each of the below parts will pose to you one or more specific tasks. You should plan to both:

- Achieve these tasks using clean, efficient, and well-documented Python and 
- Write, in your own words, about how to understand what's going on.  

> Remember, your aim is not just to write and understand the algorithm, but to explain to someone else how they could do the same. 

***Note***: your blog post doesn't have to contain a lot of math. It's ok for you to give explanations like "this function is an approximation of this other function according to the math in the written assignment." 

### Notation

In all the math below: 

- Boldface capital letters like $\mathbf{A}$ refer to matrices (2d arrays of numbers). 
- Boldface lowercase letters like $\mathbf{v}$ refer to vectors (1d arrays of numbers). 
- $\mathbf{A}\mathbf{B}$ refers to a matrix-matrix product (`A@B`). $\mathbf{A}\mathbf{v}$ refers to a matrix-vector product (`A@v`). 

### Comments and Docstrings

You should plan to comment all of your code. Docstrings are not required except in Part G. 

## Introduction

Here, we are gonna learn *spectral clustering*. To start, let's look at some examples where we do not need spectral clustering. 


```python
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
```


```python
n = 200
np.random.seed(1111)
X, y = datasets.make_blobs(n_samples=n, shuffle=True, random_state=None, centers = 2, cluster_std = 2.0)
plt.scatter(X[:,0], X[:,1])

plt.savefig("P1.png") 
```


    
![P1.png](/images/P1.png)
    


*Clustering* refers to the task of separating this data set into the two natural "blobs." K-means is a very common way to achieve this task, which has good performance on circular-ish blobs like these: 


```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 2)
km.fit(X)

plt.scatter(X[:,0], X[:,1], c = km.predict(X))

plt.savefig("P2.png") 
```


    
![P2.png](/images/P2.png)
    


### Harder Clustering

That was all well and good, but what if our data is "shaped weird"? 


```python
np.random.seed(1234)
n = 200
X, y = datasets.make_moons(n_samples=n, shuffle=True, noise=0.05, random_state=None)
plt.scatter(X[:,0], X[:,1])

plt.savefig("P3.png") 
```


    
![P3.png](/images/P3.png)
    


We can still make out two meaningful clusters in the data, but now they aren't blobs but crescents. As before, the Euclidean coordinates of the data points are contained in the matrix `X`, while the labels of each point are contained in `y`. Now k-means won't work so well, because k-means is, by design, looking for circular clusters. 


```python
km = KMeans(n_clusters = 2)
km.fit(X)
plt.scatter(X[:,0], X[:,1], c = km.predict(X))

plt.savefig("P4.png") 
```


    
![P4.png](/images/P4.png)
    


Whoops! That's not right! 

As we'll see, spectral clustering is able to correctly cluster the two crescents. In the following problems, you will derive and implement spectral clustering. 

## Part A

In this part, we are going to create the similarity matrix 𝐀. 
Specifically, we tend to analyze the distances between each pair of coordinates given in X. 

Thus, we'll use a parameter `epsilon`. So if the distance between the coordinate `X[i]` and `X[j]` is smaller than the critical value `epsilon`, the corresponding `A[i,j]` entry should equal to `1`, and `0` otherwise.

Notice that the diagonal entries of the similarity matrix 𝐀 should always be equal to zero. (Good to use `np.fill_diagonal()`)

### Hint: 
You can find the function to compute the distance matrix.

Please use `epsilon = 0.4` for this part.


```python
import sklearn
Dis = sklearn.metrics.pairwise_distances(X, metric='euclidean')
# We define Dis as the distance matrix of X by applying the function sklearn.metrics.pairwise_distances

A = 1*(Dis < 0.4)
# We construct A as the matrix with boolean entries, and convert the entries less than 0.4 into 1, and others into 0.
np.fill_diagonal(A,0)
# The diagonal entries should all equal to zero
A
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 1, 0],
           ...,
           [0, 0, 0, ..., 0, 1, 1],
           [0, 0, 1, ..., 1, 0, 1],
           [0, 0, 0, ..., 1, 1, 0]])



## Part B

Now we have already derived the similarity matrix 𝐀 where each entry showing whether the distance between coordinates is close or not. Then, we are gonna cluster the coordinates in X into two groups based on the similarity matrix 𝐀.

### Definition:

1. *degree* of $i$ : $d_i = \sum_{j = 1}^n a_{ij}$ 
(the $i$th row-sum of $\mathbf{A}$)

2. label matrix y : The clustering information are contained in matrix `y`. Specifically, if `y[i] = 1`, then we can conclude that the point i is clustered in C1; if `y[i] = 0`, it is clustered in C0.

3. The *binary norm cut objective* of a matrix $\mathbf{A}$ is the function 

$$N_{\mathbf{A}}(C_0, C_1)\equiv \mathbf{cut}(C_0, C_1)\left(\frac{1}{\mathbf{vol}(C_0)} + \frac{1}{\mathbf{vol}(C_1)}\right)\;.$$
In this formula, we have

- $\mathbf{cut}(C_0, C_1) \equiv \sum_{i \in C_0, j \in C_1} a_{ij}$ is the *cut* of the clusters $C_0$ and $C_1$. 
- $\mathbf{vol}(C_0) \equiv \sum_{i \in C_0}d_i$, where $d_i = \sum_{j = 1}^n a_{ij}$ is the *degree* of row $i$ as we defined above, showing the total number of all other points closed. Therefore, The *volume* of cluster $C_0$ is a measure of the size of the cluster. 

#### What's Cut Term ? 

In this section, we are gonna write a function called cut(A,y) to return the cut term. 

Specifically, the cut term $\mathbf{cut}(C_0, C_1)$ represents the level of connections between points from cluster C0 and C1. Therefore, to calculate the cut term, we can simply search out all these pairs of points in different clusters and sum up the corresponding entries in `A[i,j]`.


```python
def cut(A,y):
    count = 0          # Define a variable count starting from 0
    for i in range(200):          # Looping over each label in y
        for j in range(200):      # Looping over each label in y
            if y[i] != y[j]:      # If two labels are different
                count += A[i,j]   
                # We update count by adding up the corresponding entry in the similarity matrix 𝐀  
    count = count/2
    # Since we may count the corresponding entries twice, we tend to divide it by two
    return count
```

Now we are going to compute the cut term for the cluster memerships matrix we are given. After that, we are gonna create a vector containing random numbers either 0 or 1, and then compute the corresponding cut term value. 

You will see that the "true" cut term value should be smaller. 


```python
print(cut(A,y))
# print the cut objective for the true clusters y

RandomNum = np.random.randint(2, size=200)
# We generate a random vector of random labels of length n

print(cut(A,RandomNum))
# print the cut objective for the random labels
```

    13.0
    1150.0


#### What's the Volume Term ?

The volumn term of the clusters basically represents how bug the cluster is. In other words, if the cluster C1 is small, then the value of $\mathbf{vol}(C_0)$ would be small, indicating that $\frac{1}{\mathbf{vol}(C_0)}$ will be large.

So here we can write the function `vols(A,y)` which returns a tuple where the first entry shows the volume of cluster 0 and the second entry shows the volume of cluster 1. 


```python
def vols(A,y):
    AllDegree = np.sum(A,axis=0)      # Compute all row-sum degree for each row in A, named it AllDegree
    ClusterC0 = AllDegree[y == 0]     # We select these labels in y contained in the cluster 0 and select out all 
                                      # these corresponding row-sum degree
    ClusterC1 = AllDegree[y == 1]     # We select these labels in y contained in the cluster 1 and select out all 
                                      # these corresponding row-sum degree
    return (np.sum(ClusterC0), np.sum(ClusterC1))
    # Sum up all these row-sum degree in cluster 1 and cluster 0, and return
```

#### Compute the binary normalized cut objective

Then we can write a function `normcut(A,y)` to compute the value of binary normalized cut objective from the formula given above.


```python
def normcut(A,y):
    CutNum = cut(A,y)                    # Derive the Cut Term through applying the cut function
    v0,v1 = vols(A,y)                    # Derive the 𝐯𝐨𝐥(𝐶0) and 𝐯𝐨𝐥(𝐶1) through applying the vols function
    Bnorm = CutNum*((1/v0) + (1/v1))     
    # Plug into the formula we are given above, and get the binary norm cut objective
    return Bnorm
```

Again, we are going to compute the binary normalized cut objective by using the cluster memerships matrix y we are given. After that, we are gonna create a vector containing random numbers either 0 or 1, and then compute the corresponding binary normalized cut objective. 


```python
print(normcut(A,y))        # Derive the normcut objective using the true labels y
print(normcut(A,RandomNum))   # Derive the normcut objective using both the fake labels RandomNum
```

    0.011518412331615225
    1.0240023597759158


We observed that the true value is smaller

## Part C

In this part, we are trying to minimize our `normcut(A,y)` term by finding a new label vector `y`. Here is the math trick.
Here's the trick: define a new vector $\mathbf{z} \in \mathbb{R}^n$ such that: 

$$
z_i = 
\begin{cases}
    \frac{1}{\mathbf{vol}(C_0)} &\quad \text{if } y_i = 0 \\ 
    -\frac{1}{\mathbf{vol}(C_1)} &\quad \text{if } y_i = 1 \\ 
\end{cases}
$$

Note that the signs of  the elements of $\mathbf{z}$ contain all the information from $\mathbf{y}$: if $i$ is in cluster $C_0$, then $y_i = 0$ and $z_i > 0$. 


#### So first, let's write the function `transform(A,y)` to compute the new label vector by defnition above. 


```python
def transform(A,y):
    z = np.zeros(200)       # create the default matrix z with all zero entries.
    v0,v1 = vols(A,y)       # Derive the 𝐯𝐨𝐥(𝐶0) and 𝐯𝐨𝐥(𝐶1) through applying the vols function
    z[y == 0] = 1/v0        # Set these cluster 0 labels with value 1/v0 in matrix z.
    z[y == 1] = -1/v1       # Set these cluster 1 labels with value -1/v1 in matrix z.
    return z                # Return the resulting matrix z.
```

Then, we are given that 
$$\mathbf{N}_{\mathbf{A}}(C_0, C_1) = \frac{\mathbf{z}^T (\mathbf{D} - \mathbf{A})\mathbf{z}}{\mathbf{z}^T\mathbf{D}\mathbf{z}}\;$$

where $\mathbf{D}$ is the diagonal matrix with nonzero entries $d_{ii} = d_i$, and  where $d_i = \sum_{j = 1}^n a_i$ is the degree (row-sum) from before.  



#### So next, we can apply the new label z to compute the right side of the equation, and then we can compare to the left side of the equation and see whether they are equal.


```python
z = transform(A,y)          # Derive the new label matrix through applying the function transform
LeftSide = normcut(A,y)     # Set the leftside as the binary norm cut objective derived from normcut function
D = np.zeros_like(A,dtype=float) # matrix of zeros of same shape as A
AllDegree = np.sum(A,axis=0)     # All all the row-sum degree
D[:200,:200] = np.diag(AllDegree)      # Insert these degrees into the diagonal entries of D
RightSide = ((z.T) @ (D-A) @ z)/((z.T) @ D @ z)
# Derive the matrix followed the formula provided
np.isclose(LeftSide,RightSide)     # Compare the leftside and the rightside
```




    True



#### Then we can check whether $\mathbf{z}^T\mathbf{D}\mathbb{1} = 0$, where $\mathbb{1}$ is the vector of `n` ones (i.e. `np.ones(n)`). 


```python
ONE = np.ones(n)
Result = (z.T) @ D @ ONE  # create the Result matrix based on the instruction
np.isclose(Result,0)      # check the identity Z.T @ D @ 1 is close to 0
```




    True



## Part D

From last part we can notice that our goal is to make the normcut term as small as possible. In other words, minimizing the function 

$$ R_\mathbf{A}(\mathbf{z})\equiv \frac{\mathbf{z}^T (\mathbf{D} - \mathbf{A})\mathbf{z}}{\mathbf{z}^T\mathbf{D}\mathbf{z}} $$

subject to the condition $\mathbf{z}^T\mathbf{D}\mathbb{1} = 0$. 

Thus, we can achieve this by substituting $\mathbf{z}$ each time and to get the more appropriate one.

We are given the function `orth_obj` and `orth`


```python
def orth(u, v):
    return (u @ v) / (v @ v) * v

e = np.ones(n) 

d = D @ e

def orth_obj(z):
    z_o = z - orth(z, d)
    return (z_o @ (D - A) @ z_o)/(z_o @ D @ z_o)
```

So we can minimize this function `orth_obj` by changing z each time. 

Hint: we can use `minimize` function from `scipy.optimize`


```python
import scipy
z_ = scipy.optimize.minimize(orth_obj,z,method='nelder-mead').x
# Use the minimize function from scipy.optimize to minimize the function orth_obj with respect to z
```

## Part E

Now the `z_` that we derived above contains all the label and clustering information for points. So we can plot the data, and use different colors for `z_[i] < 0` and `z_min[i] >= 0`


```python
plt.scatter(X[:,0], X[:,1], c = z_ < 0)
# Plot the original data, using one color for points 
# such that z_min[i] < 0 and another color for points such that z_min[i] >= 0.

plt.savefig("P5.png") 
```


    
![P5.png](/images/P5.png)
    


## Part F

Here is an more efficient method of finding the appropriate `z_` by using eigenvalues and eigenvectors of matrices.

Specifically, we can use the Rayleigh-Ritz Theorem, which shows that 
$$ (\mathbf{D} - \mathbf{A}) \mathbf{z} = \lambda \mathbf{D}\mathbf{z}\;, \quad \mathbf{z}^T\mathbf{D}\mathbb{1} = 0$$

which is equivalent to the standard eigenvalue problem 

$$ \mathbf{D}^{-1}(\mathbf{D} - \mathbf{A}) \mathbf{z} = \lambda \mathbf{z}\;, \quad \mathbf{z}^T\mathbb{1} = 0\;.$$

Since $\mathbb{1}$ is the eigenvector with the smallest eigenvalue, so the `z_` we want should be the eigenvector correspond to the second-smallest eigenvalue.

So here we are gonna create the Laplacian matrix of the similarity matrix 𝐀, $\mathbf{L} = \mathbf{D}^{-1}(\mathbf{D} - \mathbf{A})$. Solving this equation and find the eigenvector correspond to the second-smallest eigenvalue.


```python
D_inv = np.linalg.inv(D)     # Derive the inverse matrix of D
L = D_inv @ (D - A)          # Define L as the Laplacian matrix of the similarity matrix 𝐀
Lam,U = np.linalg.eig(L)     # Get all the possible eigenvalues and eigenvectors of L 
ix = Lam.argsort()           # Sort these eigenvalues
U = U[:,ix]
z_eig = U[:,1]               # Get the eigenvector corresponding to its second-smallest eigenvalue
```

#### Plot by using z_eig


```python
plt.scatter(X[:,0], X[:,1], c = z_eig < 0)
# Plot the original data, using one color for points 
# such that z_eig[i] < 0 and another color for points such that z_eig[i] >= 0.

plt.savefig("P6.png") 
```


    
![P6.png](/images/P6.png)
    


## Part G

In this part, we are gonna write a function `spectral_clustering(X, epsilon)` by combined all these parts we've done before, where X is a matrix that contains all the coordinates of points and epsilon is the critical value distance between two coordinates.

Notice: You should definitely aim to keep your solution under 10, very compact lines.


```python
def spectral_clustering(X, epsilon):
    """
    For function spectral_clustering, we have two imputs:
      - X, a matrix that contains all the coordinates of points
      - epsilon, the critical value distance between two coordinates

    The function will return a matrix containing the labels of these points, where these with negative labels
    are clustered into one group, and these with positive labels are clustered into one group.
    """
    
    n = np.shape(X)[0]                        # Set n as the number of coordinates in X
    
    A = 1*(sklearn.metrics.pairwise_distances(X, metric='euclidean') < epsilon)
    
    # Derive the similarity matrix A by applying sklearn.metrics.pairwise_distances
    
    np.fill_diagonal(A,0)
    
    # Diagonal should all be 0
    
    D = np.zeros_like(A,dtype=float)         # Set D as the default all-zero matrix
    
    D[:n,:n] = np.diag(np.sum(A,axis=0))     # Fill the diagonal of D by all the row-sum degrees
    
    Lam,U = np.linalg.eig((np.linalg.inv(D)) @ (D - A))   
    # Derive the corresponding eigenvalue and eigenvector by the given formula
    
    z_eig = U[:,Lam.argsort()][:,1]
    # Get the eigenvector corresponding to its second-smallest eigenvalue
    
    return z_eig 
```

## Part H

Do some experiments in this part. You can use `make_moons` we are given, adjusting the parameter value of `noise` and `n_samples` to see what happens.  


```python
np.random.seed(1234)
n = 1000
X, y = datasets.make_moons(n_samples=n, shuffle=True, noise=0.09, random_state=None)
# Plug in the parameters into the function where n_samples = 1000 and noise = 0.09, etc

plt.scatter(X[:,0], X[:,1], c = spectral_clustering(X, 0.4) < 0)
# Plot the scatter where color is the return value of the spectral_clustering function

plt.savefig("P7.png") 
```


    
![P7.png](/images/P7.png)
    


## Part I

Try to use another data set, the bull's eye!


```python
n = 1000
X, y = datasets.make_circles(n_samples=n, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
# Plug in the parameters into the function where n_samples = 1000 and noise = 0.05, etc

plt.scatter(X[:,0], X[:,1], c = spectral_clustering(X, 0.52) < 0)
# Plot the scatter where color is the return value of the spectral_clustering function

plt.savefig("P8.png") 
```


    
![P8.png](/images/P8.png)
    


There are two concentric circles. As before k-means will not do well here at all.


```python
km = KMeans(n_clusters = 2)
km.fit(X)
plt.scatter(X[:,0], X[:,1], c = km.predict(X))

plt.savefig("P9.png") 
```


    
![P9.png](/images/P9.png)
    


Can the function separate these circles successfully? Now you can make some experiments by adjusting the value of the parameter `epsilon` in your function spectral_clustering, and see for which value of `epsilon` is appropriate. 

### Attempt #1


```python
n = 1000
X, y = datasets.make_circles(n_samples=n, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
# Plug in the parameters into the function where n_samples = 1000 and noise = 0.05, etc

plt.scatter(X[:,0], X[:,1], c = spectral_clustering(X, 0.3) < 0)
# Plot the scatter where color is the return value of the spectral_clustering function

plt.savefig("P10.png") 
```


    
![P10.png](/images/P10.png)
    


### Attempt #2


```python
n = 1000
X, y = datasets.make_circles(n_samples=n, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
# Plug in the parameters into the function where n_samples = 1000 and noise = 0.05, etc

plt.scatter(X[:,0], X[:,1], c = spectral_clustering(X, 0.4) < 0)
# Plot the scatter where color is the return value of the spectral_clustering function

plt.savefig("P11.png") 
```


    
![P11.png](/images/P11.png)
    


### Attempt #3


```python
n = 1000
X, y = datasets.make_circles(n_samples=n, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
# Plug in the parameters into the function where n_samples = 1000 and noise = 0.05, etc

plt.scatter(X[:,0], X[:,1], c = spectral_clustering(X, 0.5) < 0)
# Plot the scatter where color is the return value of the spectral_clustering function

plt.savefig("P12.png") 
```


    
![P12.png](/images/P12.png)
    


### Attempt #4


```python
n = 1000
X, y = datasets.make_circles(n_samples=n, shuffle=True, noise=0.05, random_state=None, factor = 0.4)
# Plug in the parameters into the function where n_samples = 1000 and noise = 0.05, etc

plt.scatter(X[:,0], X[:,1], c = spectral_clustering(X, 0.7) < 0)
# Plot the scatter where color is the return value of the spectral_clustering function

plt.savefig("P13.png") 
```


    
![P13.png](/images/P13.png)
    


#### From the experiments shown above, roughly speaking, we can see that it can separate two rings successfully for the values of epsilon ranges between 0.4 and 0.5. 

## Part J

Great work! Turn this notebook into a blog post with plenty of helpful explanation for your reader. Remember that your blog post should be entirely in your own words, without any copying and pasting from this notebook. Remember also that extreme mathematical detail is not required. 
