## New Algorithms for Learning Incoherent and Overcomplete Dictionaries

| Sanjeev Arora   | Rong Ge †    | Ankur Moitra   |
|-----------------|--------------|----------------|
|                 | May 27, 2014 |                |

## Abstract

In sparse recovery we are given a matrix A ∈ R n × m ('the dictionary') and a vector of the form AX where X is sparse , and the goal is to recover X . This is a central notion in signal processing, statistics and machine learning. But in applications such as sparse coding , edge detection, compression and super resolution, the dictionary A is unknown and has to be learned from random examples of the form Y = AX where X is drawn from an appropriate distribution - this is the dictionary learning problem. In most settings, A is overcomplete : it has more columns than rows. This paper presents a polynomial-time algorithm for learning overcomplete dictionaries; the only previously known algorithm with provable guarantees is the recent work of [48] who gave an algorithm for the undercomplete case, which is rarely the case in applications. Our algorithm applies to incoherent dictionaries which have been a central object of study since they were introduced in seminal work of [18]. In particular, a dictionary is µ -incoherent if each pair of columns has inner product at most µ/ √ n .

The algorithm makes natural stochastic assumptions about the unknown sparse vector X , which can contain k ≤ c min( √ n/µ log n, m 1 / 2 -η ) non-zero entries (for any η &gt; 0). This is close to the best k allowable by the best sparse recovery algorithms even if one knows the dictionary A exactly . Moreover, both the running time and sample complexity depend on log 1 //epsilon1 , where /epsilon1 is the target accuracy, and so our algorithms converge very quickly to the true dictionary. Our algorithm can also tolerate substantial amounts of noise provided it is incoherent with respect to the dictionary (e.g., Gaussian). In the noisy setting, our running time and sample complexity depend polynomially on 1 //epsilon1 , and this is necessary.

## 1 Introduction

Finding sparse representations for data -signals, images, natural language- is a major focus of computational harmonic analysis [20, 41]. This requires having the right dictionary A ∈ R n × m for the dataset, which allows each data point to be written as a sparse linear combination of the columns of A . For images, popular choices for the dictionary include sinusoids, wavelets, ridgelets, curvelets, etc. [41] and each one is useful for different types of features: wavelets for impulsive events, ridgelets for discontinuities in edges, curvelets for smooth curves, etc. It is common to

∗ arora@cs.princeton.edu, Princeton University, Computer Science Department and Center for Computational Intractability

† rongge@microsoft.com, Microsoft Research, New England.Part of this work was done while the author was a graduate student at Princeton University and was supported in part by NSF grants CCF-0832797, CCF-1117309, CCF-1302518, DMS-1317308, and Simons Investigator

‡ moitra@mit.edu, Massachusetts Institute of Technology, Department of Mathematics and CSAIL Part of this work was done while the author was a postdoc at the Institute for Advanced Study and was supported in part by NSF grant No. DMS-0835373 and by an NSF Computing and Innovation Fellowship.

combine such hand-designed bases into a single dictionary, which is 'redundant' or 'overcomplete' because m /greatermuch n . This can allow sparse representation even if an image contains many different 'types' of features jumbled together. In machine learning dictionaries are also used for feature selection [45] and for building classifiers on top of sparse coding primitives [35].

In many settings hand-designed dictionaries do not do as well as dictionaries that are fit to the dataset using automated methods. In image processing such discovered dictionaries are used to perform denoising [21], edge-detection [40], super-resolution [52] and compression. The problem of discovering the best dictionary to a dataset is called dictionary learning and also referred to as sparse coding in machine learning. Dictionary learning is also a basic building block in the design of deep learning systems [46]. See [3, 20] for further applications. In fact, the dictionary learning problem was identified by [44] as part of a study on internal image representations in the visual cortex. Their work suggested that basis vectors in learned dictionaries often correspond to well-known image filters such as Gabor filters.

Our goal is to design an algorithm for this problem with provable guarantees in the same spirit as recent work on nonnegative matrix factorization [7], topic models [8, 6] and mixtures models [43, 12]. (We will later discuss why current algorithms in [39], [22], [4], [36], [38] do not come with such guarantees.) Designing such algorithms for dictionary learning has proved challenging. Even if the dictionary is completely known, it can be NP-hard to represent a vector u as a sparse linear combination of the columns of A [16]. However for many natural types of dictionaries, the problem of finding a sparse representation is computationally easy. The pioneering work of [18], [17] and [29] (building on the uncertainty principle of [19]) presented a number of important examples (in fact, the ones we used above) of dictionaries that are incoherent and showed that /lscript 1 -minimization can find a sparse representation in a known, incoherent dictionary if one exists.

Definition 1.1 ( µ -incoherent) . An n × m matrix A whose columns are unit vectors is µ -incoherent if ∀ i = j we have 〈 A i , A j 〉 ≤ µ/ √ n. We will refer to A as incoherent if µ is O (log n ).

/negationslash

A randomly chosen dictionary is incoherent with high probability (even if m = n 100 ). [18] gave many other important examples of incoherent dictionaries, such as one constructed from spikes and sines , as well as those built up from wavelets and sines, or even wavelets and ridgelets. There is a rich body of literature devoted to incoherent dictionaries (see additional references in [25]). [18] proved that given u = Av where v has k nonzero entries, where k ≤ √ n/ 2 µ , basis pursuit (solvable by a linear program) recovers v exactly and it is unique. [25] (and subsequently [50]) gave algorithms for recovering v even in the presence of additive noise. [49] gave a more general exact recovery condition (ERC) under which the sparse recovery problem for incoherent dictionaries can be algorithmically solved. All of these require n &gt; k 2 µ 2 . In a foundational work, [13] showed that basis pursuit solves the sparse recovery problem even for n = O ( k log( m/k )) if A satisfies the weaker restricted isometry property [14]. Also if A is a full-rank square matrix, then we can compute v from A -1 u , trivially. But our focus here will be on incoherent and overcomplete dictionaries; extending these results to RIP matrices is left as a major open problem.

The main result in this paper is an algorithm that provably learns an unknown, incoherent dictionary from random samples Y = AX where X is a vector with at most k ≤ c min( √ n/µ log n, m 1 / 2 -η ) non-zero entries (for any η &gt; 0, small enough constant c &gt; 0 depending on η ). Hence we can allow almost as many non-zeros in the hidden vector X as the best sparse recovery algorithms which assume that the dictionary A is known . The precise requirements that we place on the distributional model are described in Section 1.2. We can relax some of these conditions at the cost of increased running time or requiring X to be more sparse. Finally, our algorithm can tolerate a substantial amount of additive noise, an important consideration in most applications including sparse coding, provided it is independent and uncorrelated with the dictionary.

## 1.1 Related Works

Algorithms used in practice Dictionary learning is solved in practice by variants of alternating minimization . [39] gave the first approach; subsequent popular approaches include the method of optimal directions (MOD) of [22], and K-SVD of [4]. The general idea is to maintain a guess for A and X and at every step either update X (using basis pursuit) or update A by, say, solving a least squares problem. Provable guarantees for such algorithms have proved difficult because the initial guesses may be very far from the true dictionary, causing basis pursuit to behave erratically. Also, the algorithms could converge to a dictionary that is not incoherent, and thus unusable for sparse recovery. (In practice, these heuristics do often work.)

Algorithms with guarantees An elegant paper of [48] shows how to provably recover A exactly if it has full column rank, and X has at most √ n nonzeros. However, requiring A to be full column rank precludes most interesting applications where the dictionary is redundant and hence cannot have full column rank (see [18, 20, 41]). Moreover, the algorithm in [48] is not noise tolerant.

After the initial announcement of this work, [2, 1] independently gave provable algorithms for learning overcomplete and incoherent dictionaries. Their first paper [2] requires the entries in X to be independent random ± 1 variables. Their second [1] gives an algorithm -a version of alternating minimization- that converges to the correct dictionary given a good initial dictionary (such a good initialization can only be found using [2] in special cases, or more generally using this paper). Unlike our algorithms, theirs assume the sparsity of X is at most n 1 / 4 or n 1 / 6 (assumption A4 in both papers), which are far from the n 1 / 2 limit of incoherent dictionaries. The main change from the initial version of our paper is that we have improved the dependence of our algorithms from poly(1 //epsilon1 ) to log 1 //epsilon1 (see Section 5).

After this work, [11] give an quasi-polynomial time algorithm for dictionary learning using sumof-squares SDP hierarchy. The algorithm can output an approximate dictionary even when sparsity is almost linear in the dimensions with weaker assumptions.

Independent Component Analysis When the entries of X are independent, algorithms for independent component analysis or ICA [15] can recover A . [23] gave a provable algorithm that recovers A up to arbitrary accuracy, provided entries in X are non-Gaussian (when X is Gaussian, A is only determined up to rotations anyway). Subsequent works considered the overcomplete case and gave provable algorithms even when A is n × m with m&gt;n [37, 28].

However, these algorithms are incomparable to ours since the algorithms are relying on different assumptions (independence vs. sparsity). With sparsity assumption, we can make much weaker assumptions on how X is generated. In particular, all these algorithms require the support Ω of the vector X to be at least 3-wise independent ( Pr [ u, v, w ∈ Ω] = Pr [ u ∈ Ω] Pr [ v ∈ Ω] Pr [ w ∈ Ω]) in the undercomplete case and 4-wise independence in the overcomplete case . Our algorithm only requires the support S to have bounded moments ( Pr [ u, v, w ∈ Ω] ≤ Λ Pr [ u ∈ Ω] Pr [ v ∈ Ω] Pr [ w ∈ Ω] where Λ is a large constant or even a polynomial depending on m,n,k , see Definition 1.5). Also, because our algorithm relies on the sparsity constraint, we are able to get almost exact recover in the noiseless case (see Theorem 1.4 and Section 5). This kind of guarantee is impossible for ICA without sparsity assumption.

## 1.2 Our Results

A range of results are possible which trade off more assumptions with better performance. We give two illustrative ones: the first makes the most assumptions but has the best performance; the

second has the weakest assumptions and somewhat worse performance. The theorem statements will be cleaner if we use asymptotic notation: the parameters k, n, m will go to infinity and the constants denoted as ' O (1)' are arbitrary so long as they do not grow with these parameters.

First we define the class of distributions that the k -sparse vectors must be drawn from. We will be interested in distributions on k -sparse vectors in R m where each coordinate is nonzero with probability Θ( k/m ) (the constant in Θ( · ) can differ among coordinates).

Definition 1.2 (Distribution class Γ and its moments) . The distribution is in class Γ if (i) each nonzero X i has expectation 0 and lies in [ -C, -1] ∪ [1 , C ] where C = O (1). (ii) Conditioned on any subset of coordinates in X being nonzero, the values X i are independent of each other.

/negationslash

The distribution has bounded /lscript -wise moments if the probability that X is nonzero in any subset S of /lscript coordinates is at most c /lscript times ∏ i ∈ S Pr [ X i = 0] where c = O (1).

Remark: (i) The bounded moments condition trivially holds for any constant /lscript if the set of nonzero locations is a random subset of size k . The values of these nonzero locations are allowed to be distributed very differently from one another. (ii) The requirement that nonzero X i 's be bounded away from zero in magnitude is similar in spirit to the Spike-and-Slab Sparse Coding (S3C) model of [27], which also encourages nonzero latent variables to be bounded away from zero to avoid degeneracy issues that arise when some coefficients are much larger than others. (iii) In the rest of the paper we will be focusing on the case when C = 1, all the proofs generalize directly to the case C &gt; 1 by losing constant factors in the guarantees.

Because of symmetry in the problem, we can only hope to learn dictionary A up to permutation and sign-flips. We say two dictionaries are column-wise /epsilon1 -close, if after appropriate permutation and flipping the corresponding columns are within distance /epsilon1 .

Definition 1.3. Two dictionaries A,B ∈ R n × m are column-wise /epsilon1 -close, if there exists a permutation π and θ ∈ {± 1 } m such that ‖ ( A i ) -θ i ( B ) π ( i ) ‖ ≤ /epsilon1 .

Later when we are talking about two dictionaries that are /epsilon1 -close, we always assume the columns are ordered correctly so that ‖ A i -B i ‖ ≤ /epsilon1 .

Theorem 1.4. There is a polynomial time algorithm to learn a µ -incoherent dictionary A from random examples. With high probability the algorithm returns a dictionary ˆ A that is column-wise /epsilon1 close to A given random samples of the form Y = AX , where X ∈ R n is chosen according to some distribution in Γ and A is in R n × m :

- If k ≤ c min( m 2 / 5 , √ n µ log n ) and the distribution has bounded 3 -wise moments, c &gt; 0 is a universal constant, then the algorithm requires p 1 samples and runs in time ˜ O ( p 2 1 n ) .
- Even if each sample is of the form Y ( i ) = AX ( i ) + η i , where η i 's are independent spherical Gaussian noise with standard deviation σ = o ( √ n ) , the algorithms above still succeed provided the number of samples is at least p 3 and p 4 respectively.
- If k ≤ c min( m ( /lscript -1) / (2 /lscript -1) , √ n µ log n ) and the distribution has bounded /lscript -wise moments, c &gt; 0 is a constant only depending on /lscript , then the algorithm requires p 2 samples and runs in time ˜ O ( p 2 2 n )

In particular p 1 = Ω(( m 2 /k 2 ) log m + mk 2 log m + m log m log 1 //epsilon1 ) and p 2 = Ω(( m/k ) /lscript -1 log m + mk 2 log m log 1 //epsilon1 ) and p 3 and p 4 are larger by a σ 2 //epsilon1 2 factor.

√ n

Remark: The sparsity that our algorithm can tolerate - the minimum of µ log n and m 1 / 2 -η -approaches the sparsity that the best known algorithms require even if A is known . Although the running time and sample complexity of the algorithm are relatively large polynomials, there are many ways to optimize the algorithm. See the discussion in Section 7.

Now we describe the other result which requires fewer assumptions on how the samples are generated, but require more stringent bounds on the sparsity:

/negationslash

Definition 1.5 (Distribution class D ) . A distribution is in class D if (i) the events X i = 0 have weakly bounded second and third moments, in the sense that Pr [ X i = 0 and X j = 0] ≤ n /epsilon1 Pr [ X i = 0] Pr [ X j = 0], Pr [ X i , X j , X t = 0] ≤ o ( n 1 / 4 ) Pr [ X i = 0] Pr [ X j = 0] Pr [ X t = 0]. (ii) Each nonzero X i is in [ -C, -1] ∪ [1 , C ] where C = O (1).

/negationslash

/negationslash

/negationslash

/negationslash

/negationslash

/negationslash

/negationslash

/negationslash

The following theorem is proved similarly to Theorem 1.4, and is sketched in Appendix B.

Theorem 1.6. There is a polynomial time algorithm to learn a µ -incoherent dictionary A from random examples of the form Y = AX , where X is chosen according to some distribution in D . If k ≤ c min( m 1 / 4 , n 1 / 4 -/epsilon1/ 2 √ µ ) and we are given p ≥ Ω(max( m 2 /k 2 log m, mn 3 / 2 log m log n k 2 µ )) samples , then the algorithm succeeds with high probability, and the output dictionary is column-wise /epsilon1 = O ( k √ µ/n 1 / 4 -/epsilon1/ 2 ) close to the true dictionary. The algorithm runs in time ˜ O ( p 2 n + m 2 p ) . The algorithm is also noise-tolerant as in Theorem 1.4.

## 1.3 Proof Outline

/negationslash

The key observation in the algorithm is that we can test whether two samples share the same dictionary element (see Section 2). Given this information, we can build a graph whose vertices are the samples, and edges correspond to samples that share the same dictionary element. A large cluster in this graph corresponds to the set of all samples with X i = 0. In Section 3 we give an algorithm for finding all the large clusters. Then we show how to recover the dictionary given the clusters in Section 4. This allows us to get a rough estimate of the dictionary matrix. Section 5 gives an algorithm for refining the solution in the noiseless case. The three main parts of the techniques are:

/negationslash

Overlapping Clustering: Heuristics such as MOD [22] or K-SVD [4] have a cyclic dependence: If we knew A , we could solve for X and if we knew all of the X 's we could solve for A . Our main idea is to break this cycle by (without knowing A ) finding all of the samples where X i = 0. We can think of this as a cluster C i . Although our strategy is to cluster a random graph, what is crucial is that we are looking for an overlapping clustering since each sample X belongs to k clusters! Many of the algorithms which have been designed for finding overlapping clusterings (e.g. [9], [10]) have a poor dependence on the maximum number of clusters that a node can belong to. Instead, we give a simple combinatorial algorithm based on triplet (or higher-order) tests that recovers the underlying, overlapping clustering. In order to prove correctness of our combinatorial algorithm, we rely on tools from discrete geometry, namely the piercing number [42, 5].

/negationslash

Recovering the Dictionary: Next, we observe that there are a number of natural algorithms for recovering the dictionary once we know the clusters C i . We can think of a random sample from C i as applying a filter to the samples we are given, and filtering out only those samples where X i = 0. The claim is that this distribution will have a much larger variance along the direction A i than along other directions, and this allows us to recovery the dictionary either using a certain averaging algorithm, or by computing the largest singular vector of the samples in C i . In fact, this latter approach is similar to K-SVD [4] and hence our analysis yields insights into why these heuristics work.

Fast Convergence: The above approach yields provable algorithms for dictionary learning whose running time and sample complexity depend polynomially on 1 //epsilon1 . However once we have a suitably good approximation to the true dictionary, can we converge at a much faster rate? We analyze a simple alternating minimization algorithm Iterative Average and we derive a formula for its updates where we can analyze it by thinking of it instead as a noisy version of the matrix power method (see Lemma 5.6). This analysis is inspired by recent work on analyzing alternating minimization for the matrix completion problem [34, 32], and we obtain algorithms whose running time and sample complexity depends on log 1 //epsilon1 . Hence we get algorithms that converge rapidly to the true dictionary while simultaneously being able to handle almost the same sparsity as in the sparse recovery problem where A is known!

/negationslash

NOTATION: Throughout this paper, we will use Y ( i ) to denote the i th sample and X ( i ) as the vector that generated it - i.e. Y ( i ) = AX ( i ) . Let Ω ( i ) denote the support of X ( i ) . For a vector X let X i be the i th coordinate. For a matrix A ∈ R n × m (especially the dictionary matrix), we use A i to denote the i -th column (the i -th dictionary element). Also, for a set S ⊂ { 1 , 2 , ..., m } , we use A S to denote the submatrix of A with columns in S . We will use ‖ A ‖ F to denote the Frobenius norm and ‖ A ‖ to denote the spectral norm. Moreover we will use Γ to denote the distribution on k -sparse vectors X that is used to generate our samples, and Γ i will denote the restriction of this distribution to vectors X where X i = 0. When we are working with a graph G we will use Γ G ( u ) to denote the set of neighbors of u in G . Throughout the paper 'with high probability' means the probability is at least 1 -n -∆ for large enough ∆.

## 2 The Connection Graph

In this part we show how to test whether two samples share the same dictionary element, i.e., whether the supports Ω ( i ) and Ω ( j ) intersect. The idea is we can check the inner-product of Y ( i ) and Y ( j ) , which can be decomposed into the sum of inner-products of dictionary elements

<!-- formula-not-decoded -->

If the supports are disjoint, then each of the terms above is small since 〈 A p , A q 〉 ≤ µ/ √ n by the incoherence assumption. To prove the sum is indeed small, we will appeal to the classic HansonWright inequality:

Theorem 2.1 (Hanson-Wright) . [31] Let X be a vector of independent, sub-Gaussian random variables with mean zero and variance one. Let M be a symmetric matrix. Then

<!-- formula-not-decoded -->

This will allow us to determine if Ω (1) and Ω (2) intersect but with false negatives:

Lemma 2.2. Suppose kµ &lt; √ n C ′ log n for large enough constant C ′ (depending on C in Definition 1.2). Then if Ω ( i ) and Ω ( j ) are disjoint, with high probability |〈 Y ( i ) , Y ( j ) 〉| &lt; 1 / 2 .

Proof: Let N be the k × k submatrix resulting from restricting A T A to the locations where X ( i ) and X ( j ) are non-zero. Set M to be a 2 k × 2 k matrix where the k × k submatrices in the top-left and bottom-right are zero, and the k × k submatrices in the bottom-left and top-right are (1 / 2) N and (1 / 2) N T respectively. Here we think of the vector X as being a length 2 k vector whose first

k entries are the non-zero entries in X ( i ) and whose last k entries are the non-zero entries in X ( j ) . And by construction, we have that

<!-- formula-not-decoded -->

We can now appeal to the Hanson-Wright inequality (above). Note that since Ω ( i ) and Ω ( j ) do not intersect, the entries in M are each at most µ/ √ n and so the Frobenius norm of M is at most µk √ 2 n . This is also an upper-bound on the spectral norm of M . We can set t = 1 / 2, and for kµ &lt; √ n/C ′ log n both terms in the minimum are Ω(log n ) and this implies the lemma. /squaresolid

We will also make use of a weaker bound (but whose conditions allow us to make fewer distributional assumptions):

Lemma 2.3. If k 2 µ &lt; √ n/ 2 then |〈 Y ( i ) , Y ( j ) 〉| &gt; 1 / 2 implies that Ω ( i ) and Ω ( j ) intersect

Proof: Suppose Ω ( i ) and Ω ( j ) are disjoint. Then the following upper bound holds:

<!-- formula-not-decoded -->

/negationslash and this implies the lemma. /squaresolid

This only works up to k = O ( n 1 / 4 / √ µ ). In comparison, the stronger bound of Lemma 2.2 makes use of the randomness of the signs of X and works up to k = O ( √ n/µ log n ).

In our algorithm, we build the following graph:

Definition 2.4. Given p samples Y (1) , Y (2) , ..., Y ( p ) , build a connection graph on p nodes where i and j are connected by an edge if and only if |〈 Y ( i ) , Y ( j ) 〉| &gt; 1 / 2.

This graph will 'miss' some edges, since if a pair X ( i ) and X ( j ) have intersecting support we do not necessarily meet the above condition. But by Lemma 2.2 (with high probability) this graph will not have any false positives:

Corollary 2.5. With high probability, each edge ( i, j ) present in the connection graph corresponds to a pair where Ω ( i ) and Ω ( j ) have non-empty intersection.

Consider a sample Y (1) for which there is an edge to both Y (2) and Y (3) . This means that there is some coordinate i in both Ω (1) and Ω (2) and some coordinate i ′ in both Ω (1) and Ω (3) . However the challenge is that we do not immediately know if Ω (1) , Ω (2) and Ω (3) have a common intersection or not.

## 3 Overlapping Clustering

/negationslash

Our goal in this section is to determine which samples Y have X i = 0 just from the connection graph. To do this, we will identify a combinatorial condition that allows us to decide whether or not a set of three samples Y (1) , Y (2) and Y (3) that have supports Ω (1) , Ω (2) and Ω (3) respectively have a common intersection or not. From this condition, it is straightforward to give an algorithm that correctly groups together all of the samples Y that have X i = 0. In order to reduce the number of letters used we will focus on the first three samples Y (1) , Y (2) and Y (3) although all the claims and lemmas hold for all triples.

/negationslash

/negationslash

/negationslash

Suppose we are given two samples Y (1) and Y (2) with supports Ω (1) and Ω (2) where Ω (1) ∩ Ω (2) = { i } . We will prove that this pair can be used to recover all the samples Y for which X i = 0. This will follow because we will show that the expected number of common neighbors between Y (1) , Y (2) and Y will be large if X i = 0 and otherwise will be small. So throughout this subsection let us consider a sample Y = AX and let Ω be its support. We will need the following elementary claim.

<!-- formula-not-decoded -->

/negationslash

Proof: Using ideas similar to Lemma 2.2, we can show if | Ω ∩ Ω (1) | = 1 (that is, the new sample has a unique intersection with Ω (1) ), then |〈 Y, Y (1) 〉| &gt; 1 / 2.

Now let i ∈ Ω (1) ∩ Ω (2) ∩ Ω (3) , let E be the event that Ω ∩ Ω (1) = Ω ∩ Ω (2) = Ω ∩ Ω (3) = { i } . Clearly, when event E happens, for all j = 1 , 2 , 3 , |〈 Y, Y ( j ) 〉| &gt; 1 / 2. The probability of E is at least

<!-- formula-not-decoded -->

Here we used bounded second moment property for the conditional probability and union bound.

/squaresolid

This claim establishes a lower bound on the expected number of common neighbors of a triple, if they have a common intersection. Next we establish an upper bound, if they don't have a common intersection. Suppose Ω (1) ∩ Ω (2) ∩ Ω (3) = ∅ . In principle we should be concerned that Ω could still intersect each of Ω (1) , Ω (2) and Ω (3) in different locations. Let a = | Ω (1) ∩ Ω (2) | , b = | Ω (1) ∩ Ω (3) | and c = | Ω (2) ∩ Ω (3) | .

Lemma 3.2. Suppose that Ω (1) ∩ Ω (2) ∩ Ω (3) = ∅ . Then the probability that Ω intersects each of Ω (1) , Ω (2) and Ω (3) is at most

<!-- formula-not-decoded -->

Proof: We can break up the event whose probability we would like to bound into two (not necessarily disjoint) events: (1) the probability that Ω intersects each of Ω (1) , Ω (2) and Ω (3) disjointly (i.e. it contains a point i ∈ Ω (1) but i / ∈ Ω (2) , Ω (3) , and similarly for the other sets ). (2) the probability that Ω contains a point in the common intersection of two of the sets, and one point from the remaining set. Clearly if Ω intersects the each of Ω (1) , Ω (2) and Ω (3) then at least one of these two events must occur.

The probability of the first event is at most the probability that Ω contains at least one element from each of three disjoint sets of size at most k . The probability that Ω contains an element of just one such set is at most the expected intersection which is k 2 m , and since the expected intersection of Ω with each of these sets are non-positively correlated (because they are disjoint) we have that the probability of the first event can be bounded by k 6 m 3 .

Similarly, for the second event: consider the probability that Ω contains an element in Ω (1) ∩ Ω (2) . Since Ω (1) ∩ Ω (2) ∩ Ω (3) = ∅ , then Ω must also contain an element in Ω (3) too. The expected intersection of Ω and Ω (1) ∩ Ω (2) is ka m and the expected intersection of Ω and Ω (3) is k 2 m , and again the expectations are non-positively correlated since the two sets Ω (1) ∩ Ω (2) and Ω (3) are disjoint by assumption. Repeating this argument for the other pairs completes the proof of the lemma. /squaresolid

Note that if Γ has bounded higher order moment, the probability that two sets of size k intersect in at least Q elements is at most ( k 2 m ) Q . Hence we can assume that with high probability there is no pair of samples whose supports intersect in more than a constant number of locations. When Γ only has bounded 3-wise moment see Appendix A.

## Algorithm 1 OverlappingCluster , Input: p samples Y (1) , Y (2) , ..., Y ( p )

1. Compute a graph G on p nodes where there is an edge between i and j iff |〈 Y ( i ) , Y ( j ) 〉| &gt; 1 / 2
2. Set T = pk 10 m
3. Repeat Ω( m log 2 m ) times:
4. Choose a random edge ( u, v ) in G
5. Set S u,v = { w : | Γ G ( u ) ∩ Γ G ( v ) ∩ Γ G ( w ) | ≥ T } ∪ { u, v }
6. Delete any set S u,v where u, v are contained in a strictly smaller set S a,b (also delete any duplicates)
7. Output the remaining sets S u,v

/negationslash

Let us quantitatively compare our lower and upper bound: If k ≤ cm 2 / 5 then the expected number of common neighbors for a triple with Ω (1) ∩ Ω (2) ∩ Ω (3) = ∅ is much larger than the expected number of common neighbors of a triple whose common intersection is empty. Under this condition, if we take p = O ( m 2 /k 2 log n ) samples each triple with a common intersection will have at least T common neighbors, and each triple whose common intersection is empty will have less than T/ 2 common neighbors.

Hence we can search for a triple with a common intersection as follows: We can find a pair of samples Y (1) and Y (2) whose supports intersect. We can take a neighbor Y (3) of Y (1) in the connection graph (at random), and by counting the number of common neighbors of Y (1) , Y (2) and Y (3) we can decide whether or not their supports have a common intersection.

Definition 3.3. We will call a pair of samples Y (1) and Y (2) an identifying pair for coordinate i if the intersection of Ω (1) and Ω (2) is exactly { i } .

Theorem 3.4. The output of OverlappingCluster is an overlapping clustering where each set corresponds to some i and contains all Y ( j ) for which i ∈ Ω ( j ) . The algorithm runs in time ˜ O ( p 2 n ) and succeeds with high probability if k ≤ c min( m 2 / 5 , √ n µ log n ) and if p = Ω( m 2 log m k 2 )

Proof: We can use Lemma 2.2 to conclude that each edge in G corresponds to a pair whose support intersects. We can appeal to Lemma 3.2 and Claim 3.1 to conclude that for p = Ω( m 2 /k 2 log m ), with high probability each triple with a common intersection has at least T common neighbors, and each triple without a common intersection has at most T/ 2 common neighbors.

In fact, for a random edge ( Y (1) , Y (2) ), the probability that the common intersection of Ω (1) and Ω (2) is exactly { i } is Ω(1 /m ) because we know that they do intersect, and that intersection has a constant probability of being size one and it is uniformly distributed over m possible locations. Appealing to a coupon collector argument we conclude that if the inner loop is run at least Ω( m log 2 m ) times then the algorithm finds an identifying pair ( u, v ) for each column A i with high probability.

Note that we may have pairs that are not an identifying triple for some coordinate i . However, any other pair ( u, v ) found by the algorithm must have a common intersection. Consider for example a pair ( u, v ) where u and v have a common intersection { i, j } . Then we know that there is some other pair ( a, b ) which is an identifying pair for i and hence S a,b ⊂ S u,v . (In fact this containment is strict, since S u,v will also contain a set corresponding to an identifying pair for j too). Hence the second-to-last step in the algorithm will necessarily delete all such non-identifying pairs S u,v .

What is the running time of this algorithm? We need O ( p 2 n ) time to build the connection graph, and the loop takes ˜ O ( pmn ) time. Finally, the deletion step requires time ˜ O ( m 2 ) since there

will be ˜ O ( m ) pairs found in the previous step and for each pair of pairs, we can delete S u,v if and only if there is a strictly smaller S a,b that contains u and v . This concludes the proof of correctness of the algorithm, and its running time analysis. /squaresolid

## 4 Recovering the Dictionary

## 4.1 Finding the Relative Signs

/negationslash

/negationslash

Here we show how to recover the column A i once we have learned which samples Y have X i = 0. We will refer to this set of samples as the 'cluster' C i . The key observation is that if Ω (1) and Ω (2) uniquely intersect in index i then the sign of 〈 Y (1) , Y (2) 〉 is equal to the sign of X (1) i X (2) i . If there are enough such pairs Y (1) and Y (2) , we can determine not only which samples Y have X i = 0 but also which pairs of samples Y and Y ′ have X i , X ′ i = 0 and sign( X i ) = sign( X ′ i ). This is the main step of the algorithm OverlappingAverage .

/negationslash

Theorem 4.1. If the input to OverlappingAverage C 1 , ..., C m are the true clusters { j : i ∈ Ω ( j ) } up to permutation, then the algorithm outputs a dictionary ˆ A that is column-wise /epsilon1 -close to A with high probability if k ≤ min( √ m, √ n µ ) and if p = Ω ( max( m 2 log m/k 2 , m log m//epsilon1 2 ) ) Furthermore the algorithm runs in time O ( p 2 ) .

Intuitively, the algorithm works because the sets C ± i correctly identify samples with the same sign. This is summarized in the following lemma.

<!-- formula-not-decoded -->

Proof: It suffices to prove the lemma at the start of Step 8, since this step only takes the complement of C ± i with respect to C i . Appealing to Lemma 2.2 we conclude that Ω ( u ) and Ω ( v ) uniquely intersect in coordinate i then the sign of 〈 Y ( u ) , Y ( v ) 〉 is equal to the sign of X ( u ) i X ( v ) i . Hence when Algorithm 2 adds an element to C ± i it must have the same sign as the i th component of X ( u i ) . What remains is to prove that each node v ∈ C i is correctly labeled. We will do this by showing that for any such vertex, there is a length two path of labeled pairs that connects u i to v , and this is true because the number of labeled pairs is large. We need the following simple claim:

Claim 4.3. If p &gt; m 2 log m/k 2 then with high probability any two clusters share at most 2 pk 2 /m 2 nodes in common.

This follows since the probability that a node is contained in any fixed pair of clusters is at most k 2 /m 2 . Then for any node u ∈ C i , we would like to lower bound the number of labeled pairs it has in C i . Since u is in at most k -1 other clusters C i 1 , ..., C i k -1 , the number of pairs u, v where v ∈ C i that are not labeled for C i is at most

<!-- formula-not-decoded -->

Therefore for a fixed node u for at least a 2 / 3 fraction of the other nodes w ∈ C i the pair u, w is labeled. Hence we conclude that for each pair of nodes u i , v ∈ C i the number of w for which both u i , w and w,v are labeled is at least |C i | / 3 &gt; 0 and so for every v , there is a labeled path of length two connecting u i to v . /squaresolid

Using this lemma, we are ready to prove Algorithm 2 correctly learns all columns of A .

Algorithm 2 OverlappingAverage , Input: p samples Y (1) , Y (2) , ...Y ( p ) and overlapping clusters C 1 , C 2 , ..., C m

## 1. For each C i

2. For each pair ( u, v ) ∈ C i that does not appear in any other C j ( X ( u ) and X ( v ) have a unique intersection)

3.

4. Choose an arbitrary u i ∈ C i , and set C ± i = { u i }
2. Label the pair +1 if 〈 Y ( u ) , Y ( v ) 〉 &gt; 0 and otherwise label it -1.
5. For each v ∈ C i
7. Else if there is w ∈ C i where the pairs u i , w and v, w have the same label, add v to C ± i .
6. If the pair u i , v is labeled +1 add v to C ± i
8. If |C ± i | ≤ |C i | / 2 set C ± i ←C i \C ± i .
9. Let ˆ A i = ∑ v ∈C ± i Y ( v ) / ‖ ∑ v ∈C ± i Y ( v ) ‖
10. Output ˆ A , where each column is ˆ A i for some i

Proof: We can invoke Lemma 4.2 and conclude that C ± i is either { u : X ( u ) i &gt; 0 } or { u : X ( u ) i &lt; 0 } , whichever set is larger. Let us suppose that it is the former. Then each Y ( u ) in C ± i is an independent sample from the distribution conditioned on X i &gt; 0, which we call Γ + i . We have that E Γ + i [ AX ] = cA i where c is a constant in [1 , C ] because E Γ + i [ X j ] = 0 for all j = i .

/negationslash

Let us compute the variance:

<!-- formula-not-decoded -->

/negationslash

/negationslash

Note that there are no cross-terms because the signs of each X j are independent. Furthermore we can bound the norm of each vector Y ( u ) via incoherence. We can conclude that if |C ± i | &gt; C 2 k log m//epsilon1 2 , then with high probability ‖ ˆ A i -A i ‖ ≤ /epsilon1 using vector Bernstein's inequality ([30], Theorem 12). This latter condition holds because we set C ± i to itself or its complement based on which one is larger. /squaresolid

## 4.2 An Approach via SVD

Here we give an alternative algorithm for recovering the dictionary based instead on SVD. Intuitively if we take all the samples whose support contains index j , then every such sample Y ( i ) has a component along direction A j . Therefore direction A j should have the largest variance and can be found by SVD. The advantage is that methods like K-SVD which are quite popular in practice also rely on finding directions of maximum variance, so the analysis we provide here yields insights into why these approaches work. However, the crucial difference is that we rely on finding the correct overlapping clustering in the first step of our dictionary learning algorithms, whereas K-SVD and approaches like approximate it via their current guess for the dictionary.

/negationslash

Let us fix some notation: Let Γ i be the distribution conditioned on X i = 0. Then once we have found the overlapping clustering, each cluster is a set of random samples from Γ i . Also let α = |〈 u, A i 〉| .

/negationslash

<!-- formula-not-decoded -->

Note that R 2 i is the projected variance of Γ i on the direction u = A i . Our goal is to show that for any u = A i (i.e. α = 1), the variance is strictly smaller.

/negationslash

/negationslash

Lemma 4.5. The projected variance of Γ i on u is at most

<!-- formula-not-decoded -->

Proof: Let u || and u ⊥ be the components of u in the direction of A i and perpendicular to A i . Then we want bound E Γ i [ 〈 u, Y 〉 2 ] where Y is sampled from Γ i . Since the signs of each X j are independent, we can write

<!-- formula-not-decoded -->

Since α = ‖ u || ‖ we have:

/negationslash

<!-- formula-not-decoded -->

Also E Γ i [ X 2 j ] = ( k -1) / ( m -1). Let v be the unit vector in the direction u ⊥ . We can write

<!-- formula-not-decoded -->

/negationslash where A -i denotes the dictionary A with the i th column removed. The maximum over v of v T A -i A T -i v is just the largest singular value of A -i A T -i which is the same as the largest singular value of A T -i A -i which by the Greshgorin Disk Theorem (see e.g. [33]) is at most 1 + µ √ n m .

And hence we can bound

<!-- formula-not-decoded -->

Also since |〈 u || , A j 〉| = α |〈 A i , A j 〉| ≤ αµ/ √ n we obtain:

/negationslash

<!-- formula-not-decoded -->

/negationslash and this concludes the proof of the lemma. /squaresolid

Definition 4.6. Let ζ = max { µk √ n , √ k m } , so the expression in Lemma 4.5 can be be an upper bounded by α 2 R 2 i +2 α √ 1 -α 2 · ζ +(1 -α 2 ) ζ 2 .

We will show that an approach based on SVD recovers the true dictionary up to additive accuracy ± ζ . Note that here ζ is a parameter that converges to zero as the size of the problem increases, but is not a function of the number of samples. So unlike the algorithm in the previous subsection, we cannot make the error in our algorithm arbitrarily small by increasing the number of samples, but this algorithm has the advantage that it succeeds even when E [ X i ] = 0.

/negationslash

Corollary 4.7. The maximum singular value of Γ i is at least R i and the direction u satisfies ‖ u -A i ‖ ≤ O ( ζ ) . Furthermore the second largest singular value is bounded by O ( R 2 i ζ 2 ) .

## Algorithm 3 OverlappingSVD , Input: p samples Y (1) , Y (2) , ...Y ( p )

1. Run OverlappingCluster (or OverlappingCluster2 ) on the p samples
2. Let C 1 , C 2 , ... C m be the m returned overlapping clusters
3. Compute ˆ Σ i = 1 |C i | ∑ Y ∈C i Y Y T
4. Compute the first singular value ˆ A i of ˆ Σ i
5. Output ˆ A , where each column is ˆ A i for some i

Proof: The bound in Lemma 4.5 is only an upper bound, however the direction α = 1 has variance R 2 i &gt; 1 and hence the direction of maximum variance must correspond to α ∈ [1 -O ( ζ 2 ) , 1]. Then we can appeal to the variational characterization of singular values (see [33]) that

<!-- formula-not-decoded -->

Then condition that α ∈ [ -O ( ζ ) , O ( ζ )] for the second singular value implies the second part of the corollary. /squaresolid

Since we have a lower bound on the separation between the first and second singular values of Σ i , we can apply Wedin's Theorem and show that we can recover A i approximately even in the presence of noise.

Theorem 4.8 (Wedin) . [51] Let δ = σ 1 ( M ) -σ 2 ( M ) and let M ′ = M + E and furthermore let v 1 and v ′ 1 be the first singular vectors of M and M ′ respectively. Then

<!-- formula-not-decoded -->

Hence even if we do not have access to Σ i but rather an approximation to it ˆ Σ i (e.g. an empirical covariance matrix computed from our samples), we can use the above perturbation bound to show that we can still recover a direction that is close to A i - and in fact converges to A i as we take more and more samples.

Theorem 4.9. If the input to OverlappingSVD is the correct clustering, then the algorithm outputs a dictionary ˆ A so that for each i , ‖ A i -ˆ A i ‖ ≤ ζ with high probability if k ≤ c min( √ m, √ n µ log n ) and if

<!-- formula-not-decoded -->

Proof: Appealing to Theorem 3.4, we have that with high probability the call to OverlappingCluster returns the correct overlapping clustering. Then given n log n samples from the distribu-

ζ 2 tion Γ i the classic result of Rudelson implies that the computed empirical covariance matrix ˆ Σ i is close in spectral norm to the true co-variance matrix [47]. This, combined with the separation of the first and second singular values established in Corollary 4.7 and Wedin's Theorem 4.8 imply that we recover each column of A up to an additive accuracy of /epsilon1 and this implies the theorem. Note that since we only need to compute the first singular vector, this can be done via power iteration [26] and hence the bottleneck in the running time is the call to OverlappingCluster . /squaresolid

Algorithm 4 IterativeAverage , Input: Initial estimation B , ‖ B i -A i ‖ ≤ /epsilon1 , q samples (independent of B ) Y (1) , Y (2) , ...Y ( q )

1. For each sample i , let Ω ( i ) = { j : |〈 Y ( i ) , B j 〉| &gt; 1 / 2 }
2. For each dictionary element j
3. Let C + j be the set of samples that have inner product more than 1 / 2 with B ( j ) ( C + j = { i : 〈 Y ( i ) , B j 〉 &gt; 1 / 2 } )
5. Let ˆ X ( i ) = B + Ω ( i ) Y ( i )
4. For each sample i in C + j
6. Let Q i,j = Y ( i ) -∑ t ∈ Ω ( i ) \{ j } B t ˆ X ( i ) t
7. Let B ′ j = ∑ i ∈C + j Q i,j / ‖ ∑ i ∈C + j Q i,j ‖ .
8. Output B ′ .

## 4.3 Noise Tolerance

Here we elaborate on why the algorithm can tolerate noise provided that the noise is uncorrelated with the dictionary (e.g. Gaussian noise). The observation is that in constructing the connection graph, we only make use of the inner products between pairs of samples Y (1) and Y (2) , the value of which is roughly preserved under various noise models. In turn, the overlapping clustering is a purely combinatorial algorithm that only makes use of the connection graph. Finally, we recover the dictionary A using singular value decomposition, which is well-known to be stable under noise (e.g. Wedin's Theorem 4.8).

## 5 Refining the Solution

Earlier sections gave noise-tolerant algorithms for the dictionary learning problem with sample complexity O (poly( n, m, k ) //epsilon1 2 ). This dependency on /epsilon1 is necessary for any noise-tolerant algorithm since even if the dictionary has only one vector, we need O (1 //epsilon1 2 ) samples to estimate the vector in presence of noise. However when Y is exactly equal to AX we can hope to recover the dictionary with better running time and much fewer samples. In particular, [24] recently established that /lscript 1 -minimization is locally correct for incoherent dictionaries, therefore it seems plausible that given a very good estimate for A there is some algorithm that computes a refined estimate of A whose running time and sample complexity have a better dependence on /epsilon1 .

In this section we analyze the local-convergence of an algorithm that is similar to K-SVD [4]; see Algorithm 4 IterativeAverage . Recall B S denotes the submatrix of B whose columns are indices in S ; also, P + = ( P T P ) -1 P T is the left-pseudoinverse of the matrix P . Hence P + P = I , PP + is the projection matrix to the span of columns of P .

The key lemma of this section shows the error decreases by a constant factor in each round of IterativeAverage (provided that it was suitably small to begin with). Let /epsilon1 0 ≤ 1 / 100 k .

Theorem 5.1. Suppose the dictionary A is µ -incoherent with µ/ √ n &lt; 1 /k log k , initial solution is /epsilon1 &lt; /epsilon1 0 close to the true solution (i.e. for all i ‖ B i -A i ‖ ≤ /epsilon1 ). With high probability the output of IterativeAverage is a dictionary B ′ that satisfies ‖ B ′ i -A i ‖ ≤ (1 -δ ) /epsilon1 , where δ is a universal positive constant. Moreover, the algorithm runs in time O ( qnk 2 ) and succeeds with high probability when number of samples q = Ω( m log 2 m ) .

We will analyze the update made to the first column B 1 , and the same argument will work for all columns (and hence we can apply a union bound to complete the proof). To simplify the proof, we will let ξ denote arbitrarily small constants (whose precise value will change from line to line). First, we establish some basic claims that will be the basis for our analysis of IterativeAverage .

Claim 5.2. Suppose A is a µ incoherent matrix with µ/ √ n &lt; 1 /k log k . If for all i , ‖ B i -A i ‖ ≤ /epsilon1 0 then IterativeAverage recovers the correct support for each sample (i.e. Ω ( i ) = supp ( X ( i ) ) ) and the correct sign (i.e. C + j = { j : X ( i ) j &gt; 0 } ) 1

Proof: We can compute 〈 Y ( i ) , B 1 〉 = ∑ j ∈ Ω ( i ) X ( i ) j 〈 A j , B 1 〉 and the total contribution of all of the terms besides X ( i ) 1 〈 A 1 , B 1 〉 for j = 1 is at most 1 / 3. This implies the claim. /squaresolid

/negationslash

To simplify the notation, let us permute the samples so that C + 1 = { 1 , 2 , ..., l } . The probability that X ( i ) 1 &gt; 0 is Θ( k/m ) and so for q = Θ( m log 2 m ) samples with high probability the number of samples l where X ( i ) 1 &gt; 0 is Ω( qk/m ) = Ω( k log 2 m ).

Claim 5.3. The set of columns { B i } i is µ ′ = µ + O ( k/ √ n ) -incoherent where µ ′ / √ n ≤ 1 / 10 k .

Definition 5.4. Let M i be the matrix (0 , B Ω ( i ) \{ 1 } ) B + Ω ( i ) .

Then we can write Q i, 1 = ( I -M i ) Y ( i ) . Let us establish some basic properties of M i that we will need in our analysis:

Claim 5.5. M i has the following properties: (1) M i B 1 = 0 (2) For all j ∈ Ω ( i ) \{ 1 } , M i B j = B j and (3) ‖ M i ‖ ≤ 1 + ξ

Proof: The first and second property follow immediately from the definition of M i , and the third property follows from the Gershgorin disk theorem. /squaresolid

For the time being, we will consider the vector ˆ B 1 = ∑ l i =1 Q i, 1 / ∑ l i =1 X ( i ) 1 . We cannot compute this vector directly (note that ˆ B 1 and B ′ 1 are in general different) but first we will show that ˆ B 1 and A 1 are suitably close. To accomplish this, we will first find a convenient expression for the error:

## Lemma 5.6.

<!-- formula-not-decoded -->

Proof: The proof is mostly carefully reorganizing terms and using properties of M i 's to simplify the expression.

Let us first compute ˆ B 1 -B 1 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

1 Notice that this is not a 'with high probability' statement, the support is always correctly recovered. That is why we use Ω ( i ) both in the algorithm and for the true support

The last equality uses the first and second properties of M i from the above claim. Consequently we have

<!-- formula-not-decoded -->

And this is our desired expression.

/squaresolid

We will analyze the two terms in the above equation separately. The second term is the most straightforward to bound, since it is the sum of independent vector-valued random variables (after we condition on the support Ω ( i ) of each sample in C + 1 .

Claim 5.7. If l &gt; Ω( k log 2 m ) , then with high probability the second term of Equation (1) is bounded by /epsilon1/ 100 .

Proof: The denominator is at least l and the numeratoris the sum of at most lk independent random vectors with mean zero, and whose length is at most 3 C/epsilon1 . We can invoke the vector Bernstein's inequality [30], and conclude that the sum is bounded by O ( C √ lk log m/epsilon1 ) with high probability. After normalization the second term is bounded by /epsilon1/ 100. /squaresolid

All that remains is to bound the first term. Note that the coefficient of ‖ M i ( A 1 -B 1 ) ‖ is independent of the support, and so the first term will converge to its expectation - namely E [ ‖ M i ( A 1 -B 1 ) ‖ ]. So it suffices to bound this expectation.

<!-- formula-not-decoded -->

Proof: We will break up A 1 -B 1 onto its component ( x ) in the direction B 1 and its orthogonal component ( y ) in B ⊥ 1 . First we bound the norm of x :

<!-- formula-not-decoded -->

Next we consider the component y . Consider the supports Ω (1) and Ω (2) of two random samples from C + 1 . These sets certainly intersect at least once, since both contain { 1 } . Yet with probability at least 2 / 3 this is their only intersection (e.g. see Claim 4.3). If so, let S = (Ω (1) ∪ Ω (2) ) \{ 1 } . Recall that ‖ B T S ‖ ≤ 1 + ξ . However B T S y is the concatenation of B T Ω (1) y and B T Ω (2) y and so we conclude that ‖ B T Ω (1) y ‖ + ‖ B T Ω (2) y ‖ ≤ (1 + ξ ) √ 2. Since the spectral norm of (0 , B Ω ( i ) \{ 1 } ) is bounded, we conclude that ‖ M 1 y ‖ + ‖ M 2 y ‖ ≤ (1 + ξ ) √ 2. This implies that

<!-- formula-not-decoded -->

And this is indeed at most (1 -δ ) /epsilon1 which concludes the proof of the lemma. /squaresolid

Combining the two claims, we know that with high probability ˆ B 1 has distance at most (1 -δ ) /epsilon1 to A 1 . However, B ′ 1 is not equal to ˆ B 1 (and we cannot compute ˆ B 1 because we do not know the normalization factor). The key observation here is ˆ B 1 is a multiple of B ′ 1 , the vector B ′ 1 and A 1 all have unit norm, so if ˆ B 1 is close to A 1 the vector B ′ 1 must also be close to A 1 .

Claim 5.9. If x and y are unit vectors, and x ′ is a multiple of x then ‖ x ′ -y ‖ ≤ /epsilon1 &lt; 1 implies that ‖ x -y ‖ ≤ /epsilon1 √ 1 + /epsilon1 2

## Algorithm 5 OverlappingCluster2 , Input: p samples Y (1) , Y (2) , ..., Y ( p ) , integer /lscript

1. Compute a graph G on p nodes where there is an edge between i and j iff |〈 Y ( i ) , Y ( j ) 〉| &gt; 1 / 2
2. Set T = pk Cm 2 /lscript
3. Repeat Ω( k /lscript -2 m log 2 m ) times:
4. Choose a random node u in G , and /lscript -1 neighbors u 1 , u 2 , ...u /lscript -1
6. Set S u 1 ,u 2 ,...u /lscript -1 = { w s.t. | Γ G ( u ) ∩ Γ G ( u 1 ) ∩ ... ∩ Γ G ( w ) | ≥ T } ∪ { u 1 , u 2 , ...u /lscript -1 }
5. If | Γ G ( u ) ∩ Γ G ( u 1 ) ∩ ... ∩ Γ G ( u /lscript -1 ) | ≥ T
7. Delete any set S u 1 ,u 2 ,...u /lscript -1 if u 1 , u 2 , ...u /lscript -1 are contained in a strictly smaller set S v 1 ,v 2 ,...v /lscript -1
8. Output the remaining sets S u 1 ,u 2 ,...u /lscript -1

Proof: We have that ‖ x -y ‖ 2 = sin 2 θ +(1 -cos θ ) 2 where θ is the angle between x and y . Note that sin θ ≤ ‖ x ′ -y ‖ ≤ /epsilon1 so hence ‖ x -y ‖ ≤ √ /epsilon1 2 +(1 -√ 1 -/epsilon1 2 ) 2 . Note that for 0 ≤ a ≤ 1 we have 1 -a ≤ √ 1 -a and this implies the claim. /squaresolid

This concludes the proof of Theorem 5.1. To bound the running time, observe that for each sample, the main computations involve computing the pseudo-inverse of a n × k matrix, which takes O ( nk 2 ) time.

## 6 A Higher Order Algorithm

Here we extend the algorithm OverlappingCluster presented in Section 3 to succeed even when k ≤ c min( m 1 / 2 -η , √ n/µ log n ). The premise of OverlappingCluster is that we can distinguish whether or not a triple of samples Y (1) , Y (2) , Y (3) has a common intersection based on their number of common neighbors in the connection graph. However for k = ω ( m 2 / 5 ) this is no longer true! But we will instead consider higher-order groups of sets. In particular, for any η &gt; 0 there is an /lscript so that we can distinguish whether an /lscript -tuple of samples Y (1) , Y (2) , ..., Y ( /lscript ) has a common intersection or not based on their number of common neighbors, and this test succeeds even for k = Ω( m 1 / 2 -η ).

The main technical challenge is in showing that if the sets Ω (1) , Ω (2) , ..., Ω ( /lscript ) do not have a common intersection, that we can upper bound the probability that a random set Ω intersects each of them. To accomplish this, we will need to bound the number of ways of piercing /lscript sets Ω (1) , Ω (2) , ..., Ω ( /lscript ) that have bounded pairwise intersections by at most s points (see definitions below and Lemma 6.4), and this is the key to analyzing our higher order algorithm OverlappingCluster2 . We will defer the proofs of the key lemmas and the description of the algorithm in this section to Appendix 6.

Nevertheless what we need is an analogue of Claim 3.1 and Lemma 3.2. The first is easy, but what about an analogue of Lemma 3.2? To analyze the probability that a set Ω intersects each of the sets Ω (1) , Ω (2) , ..., Ω ( /lscript ) we will rely on the following standard definition:

Definition 6.1. Given a collection of sets Ω (1) , Ω (2) , ..., Ω ( /lscript ) , the piercing number is the minimum number of points p 1 , p 2 , ..., p r so that each set contains at least one point p i .

The notion of piercing number is well-studied in combinatorics (see e.g. [42]). However, one is usually interested in upper-bounding the piercing number. For example, a classic result of Alon and Kleitman concerns the ( p, q )-problem [5]: Suppose we are given a collection of sets that has

the property that each choice of p of them has a subset of q which intersect. Then how large can the piercing number be? Alon and Kleitman proved that the piercing number is at most a fixed constant c ( p, q ) independent of the number of sets [5].

However, here our interest in piercing number is not in bounding the minimum number of points needed but rather in analyzing how many ways there are of piercing a collection of sets with at most s points, since this will directly yield bounds on the probability that Ω intersects each of Ω (1) , Ω (2) , ..., Ω ( /lscript ) . We will need as a condition that each pair of sets has bounded intersection, and this holds in our model with high-probability.

Claim 6.2. With high probability, the intersection of any pair Ω (1) , Ω (2) has size at most Q

Definition 6.3. We will call a set of /lscript sets a ( k, Q ) family if each set has size at most k and the intersection of each pair of sets has size at most Q .

Lemma 6.4. The number of ways of piercing ( k, Q ) family (of /lscript sets) with s points is at most ( /lscriptk ) s . And crucially if /lscript ≥ s +1 , then the number of ways of piercing it with s points is at most Qs ( s +1)( /lscriptk ) s -1 .

Proof: The first part of the lemma is the obvious upper bound. Now let us assume /lscript ≥ s + 1: Then given a set of s points that pierce the sets, we can partition the /lscript sets into s sets based on which of the s points is hits the set. (In general, a set may be hit by more than one point, but we can break ties arbitrarily). Let us fix any s + 1 of the /lscript sets, and let U be the the union of the pairwise intersections of each of these sets. Then U has size at most Qs ( s + 1). Furthermore by the Pigeon Hole Principle, there must be a pair of these sets that is hit by the same point. Hence one of the s points must belong to the set U , and we can remove this point and appeal to the first part of the lemma (removing any sets that are hit by this point). This concludes the proof of the second part of the lemma, too. /squaresolid

/negationslash

Theorem 6.5. The algorithm OverlappingCluster2 ( /lscript ) finds an overlapping clustering where each set corresponds to some i and contains all Y ( j ) for which X ( j ) i = 0 . The algorithm runs in time ˜ O ( k /lscript -2 mp + p 2 n ) and succeeds with high probability if k ≤ c min( m ( /lscript -1) / (2 /lscript -1) , √ n µ log n ) and if p = Ω( m 2 /k 2 log m + k /lscript -2 m log 2 m )

In order to prove this theorem we first give an analogue of Claim 3.1:

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

The proof of this claim is identical to the proof of Claim 3.1. Next we give the crucial corollary of Lemma 6.4.

Corollary 6.7. The probability that Ω hits each set in a ( k, Q ) family (of /lscript sets) is at most

<!-- formula-not-decoded -->

where C s is a constant depending polynomially on s .

Proof: We can break up the probability of the event that Ω hits each set in a ( k, Q ) family into another family of events. Let us consider the probability that X pierces the family with s ≤ /lscript -1 points or s ≥ /lscript points. In the former case, we can invoke the second part of Lemma 6.4 and the probability that X hits any particular set of s points is at most ( k/m ) s . In the latter case, we can invoke the first part of Lemma 6.4. /squaresolid

Note that if k ≤ m 1 / 2 then k/m is always greater than or equal to k s -1 ( k/m ) s . And so asymptotically the largest term in the above sum is ( k 2 /m ) /lscript which we want to be asymptotically smaller than k/m which is the probability in Claim 6.6. So if k ≤ cm ( /lscript -1) / (2 /lscript -1) then above bound is o ( k/m ) which is asymptotically smaller than the probability that a given set of /lscript nodes that have a common intersection are each connected to a random (new) node in the connection graph. So again, we can distinguish between whether or not an /lscript -tuple has a common intersection or not and this immediately yields a new overlapping clustering algorithm that works for k almost as large as √ m , although the running time depends on how close k is to this bound.

## 7 Discussion

This paper shows it is possible to provably get around the chicken-and-egg problem inherent in dictionary learning: not knowing A seems to prevents recovering X 's and vice versa. By using combinatorial techniques to recover the support of each X without knowing the dictionary, our algorithm suggests a new way to design algorithms.

Currently the running time is ˜ O ( p 2 n ) time, which may be too slow for large-scale problems. But our algorithm suggests more heuristic versions of recovering the support that are more efficient. One alternative is to construct the connection graph G and then find the overlapping clustering by running a truncated power method [53] on e i + e j (a vector that is one on indices i , j and zero elsewhere and ( i, j ) is an edge). In experiments, this recovers a good enough approximation to the true clustering that can then be used to smartly initialize KSVD so that it does not have to start from scratch. In practice, this yields a hybrid method that converges much more quickly and succeeds more often. Thus we feel that in practice the best algorithm may use algorithmic ideas presented here.

/negationslash

We note that for dictionary learning, making stochastic assumptions seems unavoidable. Interestingly, our experiments help to corroborate some of the assumptions. For instance, the condition E [ X i | X i = 0] = 0 used in our best analysis also seems necessary for KSVD; empirically we have seen its performance degrade when this is violated.

## Acknowledgements

We would like to thank Aditya Bhaskara, Tengyu Ma and Sushant Sachdeva for numerous helpful discussions throughout various stages of this work.

## References

- [1] A. Agarwal, A. Anandkumar, P. Jain, P. Netrapalli, and R. Tandon. Learning sparsely used overcomplete dictionaries via alternating minimization. In arxiv:1310.7991 , 2013.
- [2] A. Agarwal, A. Anandkumar, and P. Netrapalli. Exact recovery of sparsely used overcomplete dictionaries. In arxiv:1309.1952 , 2013.
- [3] M. Aharon. Overcomplete dictionaries for sparse representation of signals. In PhD Thesis , 2006.
- [4] M. Aharon, M. Elad, and A. Bruckstein. K-svd: An algorithm for designing overcomplete dictionaries for sparse representation. In IEEE Trans. on Signal Processing , pages 4311-4322, 2006.
- [5] N. Alon and D. Kleitman. Piercing convex sets and the hadwigder debrunner ( p, q )-problem. In Advances in Mathematics , pages 103-112, 1992.
- [6] A. Anandkumar, D. Foster, D. Hsu, S. Kakade, and Y. Liu. A spectral algorithm for latent dirichlet allocation. In NIPS , pages 926-934, 2012.
- [7] S. Arora, R. Ge, R. Kannan, and A. Moitra. Computing a nonnegative matrix factorization provably. In STOC , pages 145-162, 2012.
- [8] S. Arora, R. Ge, and A. Moitra. Learning topic models - going beyond svd. In FOCS , pages 1-10, 2012.
- [9] S. Arora, R. Ge, S. Sachdeva, and G. Schoenebeck. Finding overlapping communities in social networks: Towards a rigorous approach. In EC , 2012.
- [10] M. Balcan, C. Borgs, M. Braverman, J. Chayes, and S-H Teng. Finding endogenously formed communities. In SODA , 2013.
- [11] Boaz Barak, John Kelner, and David Steurer. Dictionary learning using sum-of-square hierarchy. unpublished manuscript , 2014.
- [12] M. Belkin and K. Sinha. Polynomial learning of distribution families. In FOCS , pages 103-112, 2010.
- [13] E. Candes, J. Romberg, and T. Tao. Stable signal recovery from incomplete and inaccurate measurements. In Communications of Pure and Applied Math , pages 1207-1223, 2006.
- [14] E. Candes and T. Tao. Decoding by linear programming. In IEEE Trans. on Information Theory , pages 4203-4215, 2005.
- [15] P. Comon. Independent component analysis: A new concept? In Signal Processing , pages 287-314, 1994.
- [16] G. Davis, S. Mallat, and M. Avellaneda. Greedy adaptive approximations. In J. of Constructive Approximation , pages 57-98, 1997.
- [17] D. Donoho and M. Elad. Optimally sparse representation in general (non-orthogonal) dictionaries via /lscript 1 -minimization. In PNAS , pages 2197-2202, 2003.

- [18] D. Donoho and X. Huo. Uncertainty principles and ideal atomic decomposition. In IEEE Trans. on Information Theory , pages 2845-2862, 1999.
- [19] D. Donoho and P. Stark. Uncertainty principles and signal recovery. In SIAM J. on Appl. Math , pages 906-931, 1999.
- [20] M. Elad. Sparse and redundant representations. In Springer , 2010.
- [21] M. Elad and M. Aharon. Image denoising via sparse and redundant representations over learned dictionaries. In IEEE Trans. on Signal Processing , pages 3736-3745, 2006.
- [22] K. Engan, S. Aase, and J. Hakon-Husoy. Method of optimal directions for frame design. In ICASSP , pages 2443-2446, 1999.
- [23] A. Frieze, M. Jerrum, and R. Kannan. Learning linear transformations. In FOCS , pages 359-368, 1996.
- [24] Q. Geng, H. Wang, and J. Wright. On the local correctness of /lscript 1 -minimization for dictionary learning. In arxiv:1101.5672 , 2013.
- [25] A. Gilbert, S. Muthukrishnan, and M. Strauss. Approximation of functions over redundant dictionaries using coherence. In SODA , 2003.
- [26] G. Golub and C. van Loan. Matrix computations. In The Johns Hopkins University Press , 1996.
- [27] I. J. Goodfellow, A. Courville, and Y.Bengio. Large-scale feature learning with spike-and-slab sparse coding. In ICML , pages 718-726, 2012.
- [28] N. Goyal, S. Vempala, and Y. Xiao. Fourier pca. In STOC , 2014.
- [29] R. Gribonval and M. Nielsen. Sparse representations in unions of bases. In IEEE Transactions on Information Theory , pages 3320-3325, 2003.
- [30] D. Gross. Recovering low-rank matrices from few coefficients in any basis. In arxiv:0910.1879 , 2009.
- [31] D. Hanson and F. Wright. A bound on tail probabilities for quadratic forms in independent random variables. In Annals of Math. Stat. , pages 1079-1083, 1971.
- [32] M. Hardt. On the provable convergence of alternating minimization for matrix completion. In arxiv:1312.0925 , 2013.
- [33] R. Horn and C. Johnson. Matrix analysis. In Cambridge University Press , 1990.
- [34] P. Jain, P. Netrapalli, and S. Sanghavi. Low rank matrix completion using alternating minimization. In STOC , pages 665-674, 2013.
- [35] K. Kavukcuoglu, M. Ranzato, and Y. LeCun. Fast inference in sparse coding algorithms with applications to object recognition. In NYU Tech Report , 2008.
- [36] K. Kreutz-Delgado, J. Murray, K. Engan B. Rao, T. Lee, and T. Sejnowski. Dictionary learning algorithms for sparse representation. In Neural Computation , 2003.

- [37] L. De Lathauwer, J Castaing, and J. Cardoso. Fourth-order cumulant-based blind identification of underdetermined mixtures. In IEEE Trans. on Signal Processing , pages 2965-2973, 2007.
- [38] H. Lee, A. Battle, R. Raina, and A. Ng. Efficient sparse coding algorithms. In NIPS , 2006.
- [39] M. Lewicki and T. Sejnowski. Learning overcomplete representations. In Neural Computation , pages 337-365, 2000.
- [40] J. Mairal, M. Leordeanu, F. Bach, M. Herbert, and J. Ponce. Discriminative sparse image models for class-specific edge detection and image interpretation. In ECCV , 2008.
- [41] S. Mallat. A wavelet tour of signal processing. In Academic-Press , 1998.
- [42] J. Matousek. Lectures on discrete geometry. In Springer , 2002.
- [43] A. Moitra and G. Valiant. Setting the polynomial learnability of mixtures of gaussians. In FOCS , pages 93-102, 2010.
- [44] B. Olshausen and B. Field. Sparse coding with an overcomplete basis set: A strategy employed by v1? In Vision Research , pages 3331-3325, 1997.
- [45] M. Pontil, A. Argyriou, and T. Evgeniou. Multi-task feature learning. In NIPS , 2007.
- [46] M. Ranzato, Y. Boureau, and Y. LeCun. Sparse feature learning for deep belief networks. In NIPS , 2007.
- [47] M. Rudelson. Random vectors in the isotropic position. In J. of Functional Analysis , pages 60-72, 1999.
- [48] D. Spielman, H. Wang, and J. Wright. Exact recovery of sparsely-used dictionaries. In Journal of Machine Learning Research , 2012.
- [49] J. Tropp. Greed is good: Algorithmic results for sparse approximation. In IEEE Transactions on Information Theory , pages 2231-2242, 2004.
- [50] J. Tropp, A. Gilbert, S. Muthukrishnan, and M. Strauss. Improved sparse approximation over quasi-incoherent dictionaries. In IEEE International Conf. on Image Processing , 2003.
- [51] P. Wedin. Perturbation bounds in connection with singular value decompositions. In BIT , pages 99-111, 1972.
- [52] J. Yang, J. Wright, T. Huong, and Y. Ma. Image super-resolution as sparse representation of raw image patches. In CVPR , 2008.
- [53] X. Yuan and T. Zhang. Truncated power method for sparse eigenvalue problems. In Journal of Machine Learning Research , pages 899-925, 2013.

## A Clustering Using Only Bounded 3-wise Moment

When the support of X has only bounded 3-wise moment, it is possible to have two supports Ω with large intersection. In that case checking the number of common neighbors cannot correctly identify whether the three samples have a common intersection. In particular, there might be false positives (three samples with no common intersection but has many common neighbors) but no false negatives (still all samples with common intersection will have many common neighbors). The algorithm can still work in this case, because it is unlikely for the two supports to have a very large intersection:

Lemma A.1. Suppose Γ has bounded 3-wise moments, k = cm 2 / 5 for some small enough constant c &gt; 0 . For any set Ω of size k , the probability that a random support Ω ′ from Γ has intersection larger than m 1 / 5 / 100 with Ω is at most O ( m -6 / 5 ) .

Proof: Let T be the number of triples in the intersection of Ω and Ω ′ . For any triple in Ω, the probability that it is also in Ω ′ is at most O ( k 3 /m 3 ) by bounded 3-wise moment. Therefore E [ T ] ≤ ( k 3 ) O ( k 3 /m 3 ) = O ( k 6 /m 3 ).

On the other hand, whenever Ω and Ω ′ has more than m 1 / 5 / 100 intersections, T is larger than ( m 1 / 5 / 100 3 ) . By Markov's inequality we know Pr [ | Ω ∩ Ω ′ | ≥ m 1 / 5 / 100] ≤ O ( m -6 / 5 ). /squaresolid

Since the probability of having false positives is small (but not negligible), we can do a simple trimming operation when we are computing the set S u,v in Algorithm 1. We shall change the definition of S u,v as follows:

1. Set S ′ u,v = { w : | Γ G ( u ) ∩ Γ G ( v ) ∩ Γ G ( w ) | ≥ T } ∪ { u, v } .
2. Set S u,v = { w : w ∈ S ′ u,v and | Γ G ( w ) ∩ S ′ u,v | ≥ T } .

Now S ′ u,v is the same as the old definition and may have false positives. However, intuitively the false positives are not in the cluster so they cannot have many connections to the cluster, and will be filtered out in the second step. In particular, we have the following lemma:

Lemma A.2. If ( u, v ) is an indentifying pair (as defined in Definition 3.3) for i , then with high probability S u,v is the set C i = { j : i ∈ Ω ( j ) } .

Proof: First we argue the set S ′ u,v is the union of C i with a small set. By Claim 3.1 and Chernoff bound, for all w ∈ C i u, v, w has more than T common neighbors, so w ∈ S ′ u,v . On the other hand, if w /negationslash∈ C i but w ∈ S ′ u,v , then by Lemma 3.2 we know Ω ( w ) must have a large intersection with either Ω ( u ) or Ω ( v ) , which has probability only O ( m -6 / 5 ) by Lemma A.1. Therefore again by concentration bounds with high probability | S ′ u,v \C i | ≤ p/m /lessmuch T .

Now consider the second step. For the samples in C i , the probability that they are connected to another random sample in C i is 1 -O ( k 2 /m ), so by concentration bounds with high probability they have at least T neighbors in C i , and they will not be filtered and are still in S u,v . On the other hand, for any vertex w /negationslash∈ C i , the expected number of edges from w to C i is only O ( k 2 /m ) |C i | /lessmuch T , and by concentration property, they are concentrated around the expectation with high probability. So for any w ∈ S ′ u,v \C i , it can only have O ( pk 3 /m 2 ) edges to C i , and O ( p/m ) edges to S ′ u,v \C i . The total number of edges to S ′ u,v is much less than T , so all of those vertices are going to be removed, and S u,v = C . /squaresolid

This lemma ensures after we pick enough random pairs, with high probability all the correct clusters C i 's are among the S u,v 's. There can be 'bad' sets, but same as before all those sets contains some of the C i , so will be removed at the end of the algorithm:

Claim A.3. For any pair ( u, v ) with i ∈ Ω ( u ) ∩ Ω ( v ) , let C i = { j : i ∈ Ω ( j ) } , then with high probability C i ⊆ S u,v .

Proof: This is essentially in the proof of the previous lemma. As before by Claim 3.1 we know C i ⊆ S ′ u,v . Now for any sample in C i , the expected number of edges to C i is (1 -o (1)) |C i | , by concentration bounds we know the number of neighbors is larger than T with high probability. Then we apply union bound for all samples in , and conclude that S . /squaresolid

C i C i ⊆ u,v

## B Extensions: Proof Sketch of Theorem 1.6

Let us first examine how the conditions in the hypothesis of Theorem 1.4 were used in its proof and then discuss why they can be relaxed.

Our algorithm is based on three steps: constructing the connection graph, finding the overlapping clustering, and recovering the dictionary. However if we invoke Lemma 2.3 (as opposed to Lemma 2.2) then the properties we need of the connection graph follow from each X being at most k sparse for k ≤ n 1 / 4 / √ µ without any distributional assumptions.

Furthermore, the crucial steps in finding the overlapping clustering are bounds on the probability that a sample X intersects a triple with a common intersection, and the probability that it does so when there is no common intersection (Claim 3.1 and Lemma 3.2). Indeed, these bounds hold whenever the probability of two sets intersecting in two or more locations is smaller (by, say, a factor of k ) than the probability of the sets intersecting once. This can be true even if elements in the sets have significant positive correlation (but for the ease of exposition, we have emphasized the simplest models at the expense of generality). Lastly, Algorithm 2 we can instead consider the difference between the averages for S i and C i \ S i and this succeeds even if E [ X i ] is non-zero. This last step does use the condition that the variables X i are independent, but if we instead use Algorithm 3 we can circumvent this assumption and still recover a dictionary that is close to the true one.

Finally, the 'bounded away from zero' assumption in Definition 1.2 can be relaxed: the resulting algorithm recovers a dictionary that is close enough to the true one and still allows sparse recovery. This is because when the distribution has the anti-concentration property, a slight variant of Algorithm 1 can still find most (instead of all) columns with X i = 0.

Using the ideas from this part, we give a proof sketch for Theorem 1.6

/negationslash

Proof: [sketch for Theorem 1.6] The proof follows the same steps as the proof of Theorem 4.9. There are a few steps that needs to be modified:

1. Invoke Lemma 2.3 instead of Lemma 2.2.
2. For Lemma 3.2, use the weaker bound on the 4-th moment. This is still OK because k is smaller now.
3. In Definition 4.4, redefine R 2 i to be E x ∈ D i [ 〈 A i , Ax 〉 2 ].
4. In Lemma 4.5, use the bound R 2 i α 2 + α √ 1 -α 2 2 k √ µ/n 1 / 4 +(1 -α 2 ) k 2 µ/ √ n in order to take the correlations between X i 's into account.

/squaresolid

Remark: Based on different assumptions on the distribution, there are algorithms with different trade-offs. Theorem 1.6 is only used to illustrate the potential of our approach and does not try to achieve optimal trade-off in every case.

A major difference from class Γ is that the X i 's do not have expectation 0 and are not forbidden from taking values close to 0 (provided they do have reasonable probability of taking values away from 0). Another major difference is that the distribution of X i can depend upon the values of other nonzero coordinates. The weaker moment condition allows a fair bit of correlation among the set of nonzero coordinates.

It is also possible to relax the condition that each nonzero X i is in [ -C, -1] ∪ [1 , C ]. Instead we require X i has magnitude at most O (1), and has a weak anti-concentration property: for every δ &gt; 0 it has probability at least c δ &gt; 0 of exceeding δ in magnitude. This requires changing Algorithm 1 in the following ways:

For each set S , let T be the subset of vertices that have at least 1 -2 δ neighbors in S : T = { i ∈ S, | Γ G ( i ) ∩ S | ≥ (1 -2 δ ) | S | . Keep sets S that 1 -2 δ fraction of the vertices are in T ( | T | ≥ (1 -2 δ ) | S | ).Here the choice of δ depend on parameters µ, n, k , and effects the final accuracy of the algorithm. This ensures for any remaining S , there must be a single coordinate that every X ( i ) for i ∈ S is nonzero on.

In the last step, only output sets that are significantly different from the previously outputted sets (significantly different means the symmetric difference is at least pk/ 5 m )

## C Discussion: Overlapping Communities

/negationslash

There is a connection between the approach used here, and the recent work on algorithms for finding overlapping communities (see in particular [9], [10]). We can think of the set of samples Y for which X i = 0 as a 'community'. Then each sample is in more than one community, and indeed for our setting of parameters each sample is contained in k communities. We can think of the main approach of this paper as:

If we can find all of the overlapping communities, then we can learn an unknown dictionary.

So how can we find these overlapping communities? The recent papers [9], [10] pose deterministic conditions on what constitutes a community (e.g. each node outside of the community has fewer edges into the community than do other members of the community). These papers provide algorithms for finding all of the communities, provided these conditions are met. However for our setting of parameters, both of these algorithms would run in quasi-polynomial time. For example, the parameter ' d ' in the paper [9] is an upper-bound on how many communities a node can belong to, and the running time of the algorithms in [9] are quasi-polynomial in this parameter. But in our setting, each sample Y belongs to k communities - one for each non-zero value in X - and the most interesting setting here is when k is polynomially large. Similarly, the parameter ' θ ' in [10] can be thought of as: If node u is in community c , what is the ratio of the edges incident to u that leave the community c compared to the number that stay inside c ? Again, for our purposes this parameter ' θ ' is roughly k and the algorithms in [10] depend quasi-polynomially on this parameter.

Hence these algorithms would not suffice for our purposes because when applied to learning an unknown dictionary, their running time would depend quasi-polynomially on the sparsity k . In contrast, our algorithms run in polynomial time in all of the parameters, albeit for a more restricted notion of what constitutes a community (but one that seems quite natural from the perspective of dictionary learning). Our algorithm OverlappingCluster finds all of the overlapping 'communities' provided that whenever a triple of nodes shares a common community they have many

more common neighbors than if they do not all share a single community. The correctness of the algorithm is quite easy to prove, once this condition is met; but here the main work was in showing that our generative model meets these neighborhood conditions.