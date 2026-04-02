## Compressed Sensing

David L. Donoho , Member, IEEE

AbstractSuppose is an unknown vector in (a digital image or signal); we plan to measure general linear functionals of and then reconstruct. If is known to be compressible by transform coding with a known transform, and we reconstruct via the nonlinear procedure defined here, the number of measurements can be dramatically smaller than the size . Thus, certain natural classes of images with pixels need only /61 /40 /108/111/103 /40 /41/41 nonadaptive nonpixel samples for faithful recovery, as opposed to the usual pixel samples. 1

the important information about the signals/images-in effect, not acquiring that part of the data that would eventually just be 'thrown away' by lossy compression. Moreover, the protocols are nonadaptive and parallelizable; they do not require knowledge of the signal/image to be acquired in advance-other than knowledge that the data will be compressible-and do not attempt any 'understanding' of the underlying object to guide an active or adaptive sensing strategy. The measurements made in the compressed sensing protocol are holographic -thus, not simple pixel samples-and must be processed nonlinearly .

In specific applications, this principle might enable dramatically reduced measurement time, dramatically reduced sampling rates, or reduced use of analog-to-digital converter resources.

## A. Transform Compression Background

Our treatment is abstract and general, but depends on one specific assumption which is known to hold in many settings of signal and image processing: the principle of transform sparsity . Wesuppose that the object of interest is a vector , which can be a signal or image with samples or pixels, and that there is an orthonormal basis for which can be, for example, an orthonormal wavelet basis, a Fourier basis, or a local Fourier basis, depending on the application. (As explained later, the extension to tight frames such as curvelet or Gabor frames comes for free.) The object has transform coefficients , and these are assumed sparse in the sense that, for some and for some Yi

/49 /52 /53 /50 More specifically, suppose has a sparse representation in some orthonormal basis (e.g., wavelet, Fourier) or tight frame (e.g., curvelet, Gabor)-so the coefficients belong to an ball for /48 /49 . The most important coefficients in that expansion allow reconstruction with /50 error /40 /49 /50 /49 /41 . It is possible to design /61 /40 /108/111/103/40 /41/41 nonadaptive measurements allowing reconstruction with accuracy comparable to that attainable with direct knowledge of the most important coefficients. Moreover, a good approximation to those important coefficients is extracted from the measurements by solving a linear programBasis Pursuit in signal processing. The nonadaptive measurements have the character of 'random' linear combinations of basis/frame elements. Our results use the notions of optimal recovery, of -widths, and information-based complexity. We estimate the Gel'fand -widths of balls in high-dimensional Euclidean space in the case /48 /49 , and give a criterion identifying near-optimal subspaces for Gel'fand -widths. We show that 'most' subspaces are near-optimal, and show that convex optimization (Basis Pursuit) is a near-optimal way to extract information derived from these near-optimal subspaces. 1 p &lt; N N p &lt;

Index TermsAdaptive sampling, almost-spherical sections of Banach spaces, Basis Pursuit, eigenvalues of random matrices, Gel'fand -widths, information-based complexity, integrated sensing and processing, minimum /49 -norm decomposition, optimal recovery, Quotient-of-a-Subspace theorem, sparse solution of linear equations.

## I. INTRODUCTION

A S our modern technology-driven civilization acquires and exploits ever-increasing amounts of data, 'everyone' now knows that most of the data we acquire 'can be thrown away' with almost no perceptual loss-witness the broad success of lossy compression formats for sounds, images, and specialized technical data. The phenomenon of ubiquitous compressibility raises very natural questions: why go to so much effort to acquire all the data when most of what we get will be thrown away? Can we not just directly measure the part that will not end up being thrown away?

In this paper, we design compressed data acquisition protocols which perform as if it were possible to directly acquire just

Manuscript received September 18, 2004; revised December 15, 2005.

The author is with the Department of Statistics, Stanford University, Stanford, CA 94305 USA (e-mail: donoho@stanford.edu).

Communicated by A. Hłst-Madsen, Associate Editor for Detection and Estimation.

Digital Object Identifier 10.1109/TIT.2006.871582

<!-- formula-not-decoded -->

Such constraints are actually obeyed on natural classes of signals and images; this is the primary reason for the success of standard compression tools based on transform coding [1]. To fix ideas, we mention two simple examples of constraint.

- Bounded Variation model for images . Here image brightness is viewed as an underlying function on the unit square , which obeys (essentially)

<!-- formula-not-decoded -->

The digital data of interest consist of pixel samples of produced by averaging over pixels. We take a wavelet point of view; the data are seen as a superposition of contributions from various scales. Let denote the component of the data at scale , and let denote the orthonormal basis of wavelets at scale , containing elements. The corresponding coefficients obey . 4j 3

- Bump Algebra model for spectra . Here a spectrum (e.g., mass spectrum or magnetic resonance spectrum) is modeled as digital samples of an underlying function on the real line which is a superposition of so-called spectral lines of varying positions, amplitudes, and linewidths. Formally (f(i/n))

<!-- formula-not-decoded -->

Here the parameters are line locations, are amplitudes/polarities, and are linewidths, and represents a lineshape, for example the Gaussian, although other profiles could be considered. We assume the constraint where , which in applications represents an energy or total mass constraint. Again we take a wavelet viewpoint, this time specifically using smooth wavelets. The data can be represented as a superposition of contributions from various scales. Let denote the component of the spectrum at scale , and let denote the orthonormal basis of wavelets at scale , containing elements. The corresponding coefficients again obey , [2]. @i &lt; R

While in these two examples, the constraint appeared, other constraints with can appear naturally as well; see below. For some readers, the use of norms with may seem initially strange; it is now well understood that the norms with such small are natural mathematical measures of sparsity [3], [4]. As decreases below , more and more sparsity is being required. Also, from this viewpoint, an constraint based on requires no sparsity at all. &lt; 1

Note that in each of these examples, we also allowed for separating the object of interest into subbands, each one of which obeys an constraint. In practice, in the following we stick with the view that the object of interest is a coefficient vector obeying the constraint (I.1), which may mean, from an application viewpoint, that our methods correspond to treating various subbands separately, as in these examples.

The key implication of the constraint is sparsity of the transform coefficients. Indeed, we have trivially that, if denotes the vector with everything except the largest coefficients set to

<!-- formula-not-decoded -->

for , with a constant depending only on . Thus, for example, to approximate with error , we need to keep only the biggest terms in . 0,1, 2,

## B. Optimal Recovery/Information-Based Complexity Background

Our question now becomes: if is an unknown signal whose transform coefficient vector obeys (I.1), can we make a reduced number of measurements which will allow faithful reconstruction of . Such questions have been discussed (for other types of assumptions about ) under the names of Optimal Recovery [5] and Information-Based Complexity [6]; we now adopt their viewpoint, and partially adopt their notation, without making a special effort to be really orthodox. We 6

use 'OR/IBC' as a generic label for work taking place in those fields, admittedly being less than encyclopedic about various scholarly contributions.

We have a class of possible objects of interest, and are interested in designing an information operator that samples pieces of information about , and an algorithm that offers an approximate reconstruction of . Here the information operator takes the form An R" + Rm

<!-- formula-not-decoded -->

where the are sampling kernels, not necessarily sampling pixels or other simple features of the signal; however, they are nonadaptive, i.e., fixed independently of . The algorithm is an unspecified, possibly nonlinear reconstruction operator. Si

We are interested in the error of reconstruction and in the behavior of optimal information and optimal algorithms. Hence, we consider the minimax error as a standard of comparison

<!-- formula-not-decoded -->

So here, all possible methods of nonadaptive linear sampling are allowed, and all possible methods of reconstruction are allowed.

In our application, the class of objects of interest is the set of objects where obeys (I.1) for a given and . Denote then p

<!-- formula-not-decoded -->

Our goal is to evaluate and to have practical schemes which come close to attaining it.

## C. Four Surprises

Here is the main quantitative phenomenon of interest for this paper.

Theorem 1: Let be a sequence of problem sizes with , , and , , . Then for there is so that mn

<!-- formula-not-decoded -->

Wefind this surprising in four ways. First, compare (I.3) with (I.2). We see that the forms are similar, under the calibration . In words: the quality of approximation to which could be gotten by using the biggest transform coefficients can be gotten by using the pieces of nonadaptive information provided by . The surprise is that we would not know in advance which transform coefficients are likely to be the important ones in this approximation, yet the optimal information operator is nonadaptive, depending at most on the class and not on the specific object. In some sense this nonadaptive information is just as powerful as knowing the best transform coefficients. = ~ Nlog(m)

This seems even more surprising when we note that for objects , the transform representation is the optimal one: no other representation can do as well at characterizing by a few coefficients [3], [7]. Surely then, one imagines, the sampling kernels underlying the optimal information

operator must be simply measuring individual transform coefficients? Actually, no: the information operator is measuring very complex 'holographic' functionals which in some sense mix together all the coefficients in a big soup. Compare (VI.1) below. ( Holography is a process where a three-dimensional (3-D) image generates by interferometry a two-dimensional (2-D) transform. Each value in the 2-D transform domain is influenced by each part of the whole 3-D object. The 3-D object can later be reconstructed by interferometry from all or even a part of the 2-D transform domain. Leaving aside the specifics of 2-D/3-D and the process of interferometry, we perceive an analogy here, in which an object is transformed to a compressed domain, and each compressed domain component is a combination of all parts of the original object.)

Another surprise is that, if we enlarged our class of information operators to allow adaptive ones, e.g., operators in which certain measurements are made in response to earlier measurements, we could scarcely do better. Define the minimax error under adaptive information allowing adaptive operators where, for , each kernel is allowed to depend on the information gathered at previous stages . Formally setting &lt; j

<!-- formula-not-decoded -->

we have the following.

Theorem 2: For , there is so that for &lt; = 2 R &gt; 0

<!-- formula-not-decoded -->

So adaptive information is of minimal help-despite the quite natural expectation that an adaptive method ought to be able iteratively somehow 'localize' and then 'close in' on the 'big coefficients.'

An additional surprise is that, in the already-interesting case , Theorems 1 and 2 are easily derivable from known results in OR/IBC and approximation theory! However, the derivations are indirect; so although they have what seem to the author as fairly important implications, very little seems known at present about good nonadaptive information operators or about concrete algorithms matched to them.

Our goal in this paper is to give direct arguments which cover the case of highly compressible objects, to give direct information about near-optimal information operators and about concrete, computationally tractable algorithms for using this near-optimal information.

## D. Geometry and Widths

From our viewpoint, the phenomenon described in Theorem 1 concerns the geometry of high-dimensional convex and nonconvex 'balls.' To see the connection, note that the class is the image, under orthogonal transformation, of an ball. If this is convex and symmetric about the origin, as well as being orthosymmetric with respect to the axes = 1

provided by the wavelet basis; if , this is again symmetric about the origin and orthosymmetric, while not being convex, but still star-shaped.

To develop this geometric viewpoint further, we consider two notions of -width; see [5].

Definition 1.1: The Gel'fand -width of with respect to the norm is defined as

<!-- formula-not-decoded -->

where the infimum is over -dimensional linear subspaces of , and denotes the orthocomplement of with respect to the standard Euclidean inner product.

In words, we look for a subspace such that 'trapping' in that subspace causes to be small. Our interest in Gel'fand -widths derives from an equivalence between optimal recovery for nonadaptive information and such -widths, well known in the case [5], and in the present setting extending as follows. = 1

Theorem 3: For and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus the Gel'fand -widths either exactly or nearly equal the value of optimal information. Ultimately, the bracketing with constant will be for us just as good as equality, owing to the unspecified constant factors in (I.3). We will typically only be interested in near-optimal performance, i.e., in obtaining to within constant factors.

It is relatively rare to see the Gel'fand -widths studied directly [8]; more commonly, one sees results about the Kolmogorov -widths.

Definition 1.2: Let be a bounded set. The Kolmogorov -width of with respect the norm is defined as X Rm

<!-- formula-not-decoded -->

where the infimum is over -dimensional linear subspaces of .

In words, measures the quality of approximation of possible by -dimensional subspaces .

In the case , there is an important duality relationship between Kolmogorov widths and Gel'fand widths which allows us to infer properties of from published results on . To state it, let be defined in the obvious way, based on approximation in rather than norm. Also, for given and , let and be the standard dual indices , . Also, let denote the standard unit ball of . Then [8] 2 1/p'

<!-- formula-not-decoded -->

In particular

The asymptotic properties of the left-hand side have been determined by Garnaev and Gluskin [9]. This follows major work by Kashin [10], who developed a slightly weaker version of this result in the course of determining the Kolmogorov -widths of Sobolev spaces. See the original papers, or Pinkus's book [8] for more details.

Theorem 4: ( Kashin, Garnaev, and Gluskin (KGG)) For all and &gt;

<!-- formula-not-decoded -->

Theorem 1 now follows in the case by applying KGG with the duality formula (I.6) and the equivalence formula (I.4). The case of Theorem 1 does not allow use of duality and the whole range is approached differently in this paper.

## E. Mysteries …

Because of the indirect manner by which the KGG result implies Theorem 1, we really do not learn much about the phenomenon of interest in this way. The arguments of Kashin, Garnaev, and Gluskin show that there exist near-optimal -dimensional subspaces for the Kolmogorov widths; they arise as the null spaces of certain matrices with entries which are known to exist by counting the number of matrices lacking certain properties, the total number of matrices with entries, and comparing. The interpretability of this approach is limited.

The implicitness of the information operator is matched by the abstractness of the reconstruction algorithm. Based on OR/IBC theory we know that the so-called central algorithm is optimal. This 'algorithm' asks us to consider, for given information , the collection of all objects which could have given rise to the data

Defining now the center of a set the central algorithm is

<!-- formula-not-decoded -->

and it obeys, when the information is optimal,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

see Section III below.

This abstract viewpoint unfortunately does not translate into a practical approach (at least in the case of the , ). The set is a section of the ball , and finding the center of this section does not correspond to a standard tractable computational problem. Moreover, this assumes we know and , which would typically not be the case.

## F. Results

Our paper develops two main types of results.

- Near-Optimal Information. We directly consider the problem of near-optimal subspaces for Gel'fand -widths of , and introduce three structural conditions (CS1-CS3) on an -bymatrix which imply that its nullspace is near-optimal. We show that the vast majority of -subspaces of are near-optimal, and random sampling yields near-optimal information operators with overwhelmingly high probability.
- Near-Optimal Algorithm. Westudy a simple nonlinear reconstruction algorithm: simply minimize the norm of the coefficients subject to satisfying the measurements. This has been studied in the signal processing literature under the name Basis Pursuit; it can be computed by linear programming. We show that this method gives near-optimal results for all . &lt;

In short, we provide a large supply of near-optimal information operators and a near-optimal reconstruction procedure based on linear programming, which, perhaps unexpectedly, works even for the nonconvex case . &lt; p

For a taste of the type of result we obtain, consider a specific information/algorithm combination.

- CS Information. Let be an matrix generated by randomly sampling the columns, with different columns independent and identically distributed (i.i.d.) random uniform on . With overwhelming probability for large , has properties CS1-CS3 discussed in detail in Section II-A below; assume we have achieved such a favorable draw. Let be the basis matrix with basis vector as the th column. The CS Information operator is the matrix . ICS
- -minimization. To reconstruct from CS Information, we solve the convex optimization problem

<!-- formula-not-decoded -->

In words, we look for the object having coefficients with smallest norm that is consistent with the information .

To evaluate the quality of an information operator , set

<!-- formula-not-decoded -->

To evaluate the quality of a combined algorithm/information pair , set

<!-- formula-not-decoded -->

Theorem 5: Let , be a sequence of problem sizes obeying , , ; and let be a corresponding sequence of operators deriving from CS matrices with underlying parameters and (see Section II below). Let . There exists so that is near-optimal: mn mn ~ A &lt; p &lt; = ICS 0 ICS

<!-- formula-not-decoded -->

for , . Moreover, the algorithm delivering the solution to is near-optimal: A1,n

<!-- formula-not-decoded -->

for , .

Thus, for large , we have a simple description of near-optimal information and a tractable near-optimal reconstruction algorithm.

## G. Potential Applications

To see the potential implications, recall first the Bump Algebra model for spectra. In this context, our result says that, for a spectrometer based on the information operator in Theorem 5, it is really only necessary to take measurements to get an accurate reconstruction of such spectra, rather than the nominal measurements. However, they must then be processed nonlinearly.

Recall the Bounded Variation model for images. In that context, a result paralleling Theorem 5 says that for a specialized imaging device based on a near-optimal information operator it is really only necessary to take measurements to get an accurate reconstruction of images with pixels, rather than the nominal measurements. =

The calculations underlying these results will be given below, along with a result showing that for cartoon-like images (which may model certain kinds of simple natural imagery, like brain scans), the number of measurements for an -pixel image is only .

## H. Contents

Section II introduces a set of conditions CS1-CS3 for near-optimality of an information operator. Section III considers abstract near-optimal algorithms, and proves Theorems 1-3. Section IV shows that solving the convex optimization problem gives a near-optimal algorithm whenever . Section V points out immediate extensions to weakconditions and to tight frames. Section VI sketches potential implications in image, signal, and array processing. Section VII, building on work in [11], shows that conditions CS1-CS3 are satisfied for 'most' information operators.

Finally, in Section VIII, we note the ongoing work by two groups (Gilbert et al. [12] and CandŁs et al. [13], [14]), which although not written in the -widths/OR/IBC tradition, imply (as we explain), closely related results.

## II. INFORMATION

Consider information operators constructed as follows. With the orthogonal matrix whose columns are the basis elements , and with certain -bymatrices obeying conditions specified below, we construct corresponding information operators . Everything will be completely transparent to the choice of orthogonal matrix and hence we will assume that is the identity throughout this section. =

In view of the relation between Gel'fand -widths and minimax errors, we may work with -widths. Let denote as usual the nullspace . We define the width of a set relative to an operator

<!-- formula-not-decoded -->

In words, this is the radius of the section of cut out by the nullspace . In general, the Gel'fand -width is the smallest value of obtainable by choice of

<!-- formula-not-decoded -->

We will show for all large and the existence of by matrices where with dependent at most on and the ratio .

## A. Conditions CS1-CS3

In the following, with let denote a submatrix of obtained by selecting just the indicated columns of . We let denote the range of in . Finally, we consider a family of quotient norms on ; with denoting the norm on vectors indexed by m}|J

<!-- formula-not-decoded -->

These describe the minimal -norm representation of achievable using only specified subsets of columns of .

We define three conditions to impose on an matrix , indexed by strictly positive parameters , , and .

CS1: The minimal singular value of exceeds uniformly in . 0

CS2: On each subspace we have the inequality

<!-- formula-not-decoded -->

uniformly in

.

CS3: On each subspace uniformly in . pn

CS1 demands a certain quantitative degree of linear independence among all small groups of columns. CS2 says that linear combinations of small groups of columns give vectors that look much like random noise, at least as far as the comparison of and norms is concerned. It will be implied by a geometric fact: every slices through the ball in such a way that the resulting convex section is actually close to spherical. CS3 says that for every vector in some , the associated quotient norm is never dramatically smaller than the simple norm on .

It turns out that matrices satisfying these conditions are ubiquitous for large and when we choose the and properly. Of course, for any finite and , all norms are equivalent and almost any arbitrary matrix can trivially satisfy these conditions simply by taking very small and , very large. However, the definition of 'very small' and 'very large' would have to

depend on for this trivial argument to work. We claim something deeper is true: it is possible to choose and independent of and of . &lt;

Consider the set

<!-- formula-not-decoded -->

of all matrices having unit-normalized columns. On this set, measure frequency of occurrence with the natural uniform measure (the product measure, uniform on each factor ).

Theorem 6: Let be a sequence of problem sizes with , , and , , and . There exist and depending only on and so that, for each the proportion of all matrices satisfying CS1-CS3 with parameters and eventually exceeds . &gt; X

We will discuss and prove this in Section VII. The proof will show that the proportion of matrices not satisfying the condition decays exponentially fast in .

For later use, we will leave the constants and implicit and speak simply of CS matrices, meaning matrices that satisfy the given conditions with values of parameters of the type described by this theorem, namely, with and not depending on and permitting the above ubiquity.

## B. Near-Optimality of CS Matrices

We now show that the CS conditions imply near-optimality of widths induced by CS matrices.

Theorem 7: Let be a sequence of problem sizes with and . Consider a sequence of by matrices obeying the conditions CS1-CS3 with and positive and independent of . Then for each , there is so that for ~ A C = no

Proof: Consider the optimization problem

<!-- formula-not-decoded -->

Our goal is to bound the value of

Choose so that . Let denote the indices of the largest values in . Without loss of generality, suppose coordinates are ordered so that comes first among the entries, and partition . Clearly Lpn/

<!-- formula-not-decoded -->

while, because each entry in is at least as big as any entry in , (I.2) gives

<!-- formula-not-decoded -->

A similar argument for approximation gives, in case

<!-- formula-not-decoded -->

Now . Hence, with , we have . As and , we can invoke CS3, getting

On the other hand, again using and , invoke CS2, getting € = k o

<!-- formula-not-decoded -->

Combining these with the above

<!-- formula-not-decoded -->

with . Recalling , and invoking CS1 we have = S1,ppl-1/p ,

In short, with c1/n1

<!-- formula-not-decoded -->

The theorem follows with

<!-- formula-not-decoded -->

## III. ALGORITHMS

Given an information operator , we must design a reconstruction algorithm which delivers reconstructions compatible in quality with the estimates for the Gel'fand -widths. As discussed in the Introduction, the optimal method in the OR/IBC framework is the so-called central algorithm, which unfortunately, is typically not efficiently computable in our setting. We now describe an alternate abstract approach, allowing us to prove Theorem 1.

## A. Feasible-Point Methods

Another general abstract algorithm from the OR/IBC literature is the so-called feasible-point method , which aims simply to find any reconstruction compatible with the observed information and constraints.

As in the case of the central algorithm, we consider, for given information , the collection of all objects which could have given rise to the information €

<!-- formula-not-decoded -->

In the feasible-point method, we simply select any member of , by whatever means. One can show, adapting standard OR/IBC arguments in [15], [6], [8] the following.

Lemma 3.1: Let where and is an optimal information operator, and let be any element of . Then for = &lt; p &lt; 1

<!-- formula-not-decoded -->

In short, any feasible point is within a factor two of optimal.

Proof: Wefirst justify our claims for optimality of the central algorithm, and then show that a feasible point is near to the central algorithm. Let again denote the result of the central algorithm. Now

<!-- formula-not-decoded -->

/114 /97/100/105/117/115 Now the feasible point obeys ; hence, Xpsml

Now clearly, in the special case when is only known to lie in and is measured, the minimax error is exactly /114 /97/100/105/117/115 . Since this error is achieved by the central algorithm for each such , the minimax error over all is achieved by the central algorithm. This minimax error is R) (R) =

But the triangle inequality gives

<!-- formula-not-decoded -->

hence, as Xp,R(yn)

<!-- formula-not-decoded -->

More generally, if the information operator is only nearoptimal, then the same argument gives

<!-- formula-not-decoded -->

A popular choice of feasible point is to take an element of least norm , i.e., a solution of the problem

<!-- formula-not-decoded -->

where here is the vector of transform coefficients, . A nice feature of this approach is that it is not necessary to know the radius of the ball ; the element of least norm will always lie inside it. For later use, call the solution . By the preceding lemma, this procedure is near-minimax: 0 €

with where for given and =

## B. Proof of Theorem 3

Before proceeding, it is convenient to prove Theorem 3. Note that the case is well known in OR/IBC so we only need to give an argument for (though it happens that our argument works for as well). The key point will be to apply the -triangle inequality 1

valid for ; this inequality is well known in interpolation theory [17] through Peetre and Sparr's work, and is easy to verify directly. p

Suppose without loss of generality that there is an optimal subspace , which is fixed and given in this proof. As we just saw

Now

/114 /97/100/105/117/115 so clearly . Now suppose without loss of generality that and attain the radius bound, i.e., they satisfy and, for they satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then define . Set and . By the -triangle inequality =

and so

<!-- formula-not-decoded -->

Hence . However, so belongs to . Hence, and

<!-- formula-not-decoded -->

## C. Proof of Theorem 1

We are now in a position to prove Theorem 1 of the Introduction.

First, in the case , we have already explained in the Introduction that the theorem of Garnaev and Gluskin implies =

the result by duality. In the case , we need only to show a lower bound and an upper bound of the same order.

For the lower bound, we consider the entropy numbers, defined as follows. Let be a set and let be the smallest number such that an -net for can be built using a net of cardinality at most . From Carl's theorem [18]-see the exposition in Pisier's book [19]-there is a constant so that the Gel'fand -widths dominate the entropy numbers.

<!-- formula-not-decoded -->

Secondly, the entropy numbers obey [20], [21]

<!-- formula-not-decoded -->

At the same time, the combination of Theorems 7 and 6 shows that

<!-- formula-not-decoded -->

Applying now the Feasible Point method, we have

<!-- formula-not-decoded -->

with immediate extensions to for all . We conclude that

<!-- formula-not-decoded -->

as was to be proven.

## D. Proof of Theorem 2

Now is an opportune time to prove Theorem 2. We note that in the case of , this is known [22]-[25]. The argument is the same for , and we simply repeat it. Suppose that , and consider the adaptively constructed subspace according to whatever algorithm is in force. When the algorithm terminates, we have an -dimensional information vector and a subspace consisting of objects which would all give that information vector. For all objects in , the adaptive information therefore turns out the same. Now the minimax error associated with that information is exactly /114 /97/100/105/117/115 ; but this cannot be smaller than 1 Vo

The result follows by comparing with .

<!-- formula-not-decoded -->

## IV. BASIS PURSUIT

The least-norm method of the previous section has two drawbacks. First, it requires that one know ; we prefer an algorithm which works for . Second, if , the least-norm problem invokes a nonconvex optimization procedure, and would be considered intractable. In this section, we correct both drawbacks. &lt; p &lt;

A. The Case

In the case , is a convex optimization problem. Written in an equivalent form, with being the optimization variable, gives

<!-- formula-not-decoded -->

This can be formulated as a linear programming problem: let be the by matrix . The linear program

<!-- formula-not-decoded -->

has a solution , say, a vector in which can be partitioned as ; then solves . The reconstruction is . This linear program is typically considered computationally tractable. In fact, this problem has been studied in the signal analysis literature under the name Basis Pursuit [26]; in that work, very large-scale underdetermined problems-e.g., with and -weresolved successfully using interior-point optimization methods. 2

As far as performance goes, we already know that this procedure is near-optimal in case ; from (III.2) we have the following.

Corollary 4.1: Suppose that is an information operator achieving, for some then the Basis Pursuit algorithm achieves, for all (yn) = Al,n

<!-- formula-not-decoded -->

In particular, we have a universal algorithm for dealing with any class -i.e., any , any , any . First, apply a near-optimal information operator; second, reconstruct by Basis Pursuit. The result obeys

<!-- formula-not-decoded -->

for a constant depending at most on . The inequality can be put to use as follows. Fix . Suppose the unknown object is known to be highly compressible, say obeying the a priori bound , . Let . For any such object, rather than making measurements, we only need to make measurements, and our reconstruction obeys &lt; Q ~ Ke log(m) Ke =

While the case is already significant and interesting, the case is of interest because it corresponds to data which are more highly compressible, offering more impressive performance in Theorem 1, because the exponent is even stronger than in the case. Later in this section, we extend the same interpretation of to performance over throughout . 1/p = 1 (R)

## B. Relation Between and Minimization

The general OR/IBC theory would suggest that to handle cases where , we would need to solve the nonconvex optimization problem , which seems intractable. However, in the current situation at least, a small miracle happens: solving is again near-optimal. To understand this, we first take a small detour, examining the relation between and the extreme case of the spaces. Let us define (Pi

<!-- formula-not-decoded -->

where is just the number of nonzeros in . Again, since the work of Peetre and Sparr [16], the importance of and the relation with for is well understood; see [17] for more detail. &lt; p

Ordinarily, solving such a problem involving the norm requires combinatorial optimization; one enumerates all sparse subsets of searching for one which allows a solution . However, when has a sparse solution, will find it. m

Theorem 8: Suppose that satisfies CS1-CS3 with given positive constants , . There is a constant depending only on and and not on or so that, if some solution to has at most nonzeros, then and both have the same unique solution. y P1)

In words, although the system of equations is massively underdetermined, minimization and sparse solution coincide-when the result is sufficiently sparse.

There is by now an extensive literature exhibiting results on equivalence of and minimization [27]-[34]. In the early literature on this subject, equivalence was found under conditions involving sparsity constraints allowing nonzeros. While it may seem surprising that any results of this kind are possible, the sparsity constraint is, ultimately, disappointingly small. A major breakthrough was the contribution of CandŁs, Romberg, and Tao [13] which studied the matrices built by taking rows at random from an by Fourier matrix and gave an order bound, showing that dramatically weaker sparsity conditions were needed than the results known previously. In [11], it was shown that for 'nearly all' by matrices with , equivalence held for nonzeros, . The above result says effectively that for 'nearly all' by matrices with , equivalence held up to nonzeros, where . = p = Olpn / log(n)) 9 pn m &lt;

Our argument, in parallel with [11], shows that the nullspace has a very special structure for obeying the conditions in question. When is sparse, the only element in a given affine subspace which can have small norm is itself.

To prove Theorem 8, we first need a lemma about the nonsparsity of elements in the nullspace of . Let and, for a given vector , let denote the mutilated vector with entries . Define the concentration m} €

This measures the fraction of norm which can be concentrated on a certain subset for a vector in the nullspace of . This concentration cannot be large if is small.

Lemma 4.1: Suppose that satisfies CS1-CS3, with constants and . There is a constant depending on the so that if satisfies

<!-- formula-not-decoded -->

then

Proof: This is a variation on the argument for Theorem 7. Let . Assume without loss of generality that is the most concentrated subset of cardinality , and that the columns of are numbered so that ; partition . We again consider , and have . We again invoke CS2-CS3, getting = {1, ~u =

<!-- formula-not-decoded -->

We invoke CS1, getting

Now, of course, . Combining all these

<!-- formula-not-decoded -->

Proof of Theorem 8: Suppose that and is supported on a subset . = C {1,

We first show that if , is the only minimizer of . Suppose that is a solution to , obeying

Then where . We have

Invoking the definition of twice

<!-- formula-not-decoded -->

i.e., .

Now recall the constant of Lemma 4.1. Define so that and . Lemma 4.1 shows that implies .

## C. Near-Optimality of Basis Pursuit for

Wenowreturn to the claimed near-optimality of Basis Pursuit throughout the range .

Theorem 9: Suppose that satisifies CS1-CS3 with constants and . There is so that a solution to a problem instance of with obeys

<!-- formula-not-decoded -->

The proof requires an stability lemma, showing the stability of minimization under small perturbations as measured in norm. For and stability lemmas, see [33]-[35]; however, note that those lemmas do not suffice for our needs in this proof.

Lemma 4.2: Let be a vector in and be the corresponding mutilated vector with entries . Suppose that where . Consider the instance of defined by ; the solution of this instance of obeys &lt; Vo = P1

<!-- formula-not-decoded -->

Proof of Lemma 4.2: Put for short , and set . By definition of ß = 00

<!-- formula-not-decoded -->

while

As solves (P1)

and of course

Hence,

Finally

Combining the above, setting , and , we get

<!-- formula-not-decoded -->

and (IV.2) follows.

Proof of Theorem 9: We use the same general framework as in Theorem 7. Let where . Let be the solution to , and set .

Let as in Lemma 4.1 and set . Let index the largest amplitude entries in . From and (II.4) we have = Ileollp

<!-- formula-not-decoded -->

and Lemma 4.1 provides

<!-- formula-not-decoded -->

Applying Lemma 4.2

<!-- formula-not-decoded -->

The vector lies in and has . Hence, 0 = = =

<!-- formula-not-decoded -->

We conclude by homogeneity that

<!-- formula-not-decoded -->

Combining this with (IV.3),

<!-- formula-not-decoded -->

## V. IMMEDIATE EXTENSIONS

Before continuing, we mention two immediate extensions to the results so far; they are of interest below and elsewhere.

## A. Tight Frames

Our main results so far have been stated in the context of making an orthonormal basis. In fact, the results hold for tight frames . These are collections of vectors which, when joined together as columns in an matrix have . It follows that, if , then we have the Parseval relation m X m' &lt; m' ) =

and the reconstruction formula . In fact, Theorems 7 and 9 only need the Parseval relation in the proof. Hence, the same results hold without change when the relation between and involves a tight frame. In particular, if is an matrix satisfying CS1-CS3, then defines a near-optimal information operator on , and solution of the optimization problem =

<!-- formula-not-decoded -->

defines a near-optimal reconstruction algorithm .

A referee remarked that there is no need to restrict attention to tight frames here; if we have a general frame the same results go through, with constants involving the frame bounds. This is true and potentially very useful, although we will not use it in what follows.

## B. Weak Balls

Our main results so far have been stated for spaces, but the proofs hold for weak balls as well . The weak ball of radius consists of vectors whose decreasing rearrangements obey

<!-- formula-not-decoded -->

Conversely, for a given , the smallest for which these inequalities all hold is defined to be the norm: . The 'weak' moniker derives from . Weak constraints have the following key property: if denotes a mutilated version of the vector with all except the largest items set to zero, then the inequality Hlellwep =

<!-- formula-not-decoded -->

is valid for and , with . In fact, Theorems 7 and 9 only needed (V.1) in the proof, together with (implicitly) . Hence, we can state results for spaces defined using only weaknorms, and the proofs apply without change. 1 = =

## VI. STYLIZED APPLICATIONS

We sketch three potential applications of the above abstract theory. In each case, we exhibit that a certain class of functions

has expansion coefficients in a basis or frame that obey a particular or weak embedding, and then apply the above abstract theory.

## A. Bump Algebra

Consider the class of functions which are restrictions to the unit interval of functions belonging to the Bump Algebra [2], with bump norm . This was mentioned in the Introduction, which observed that the wavelet coefficients at level obey where depends only on the wavelet used. Here and later we use standard wavelet analysis notations as in [36], [37], [2]. B 2-j

We consider two ways of approximating functions in . In the classic linear scheme, we fix a 'finest scale' and measure the resumØ coefficients where , with a smooth function integrating to . Think of these as point samples at scale after applying an antialiasing filter. We reconstruct by giving an approximation error 2-j1

<!-- formula-not-decoded -->

with depending only on the chosen wavelet. There are coefficients associated with the unit interval, and so the approximation error obeys =

<!-- formula-not-decoded -->

In the compressed sensing scheme, we need also wavelets where is an oscillating function with mean zero. We pick a coarsest scale . We measure the resumØ coefficients -there are of these-and then let denote an enumeration of the detail wavelet coefficients . The dimension of is . The norm satisfies 0 = : 0 &lt; k &lt; =2j1 (@j,k m

<!-- formula-not-decoded -->

This establishes the constraint on norm needed for our theory. We take and apply a near-optimal information operator for this and (described in more detail later). Weapply the near-optimal algorithm of minimization, getting the error estimate

<!-- formula-not-decoded -->

with independent of . The overall reconstruction

<!-- formula-not-decoded -->

has error

<!-- formula-not-decoded -->

again with independent of . This is of the same order of magnitude as the error of linear sampling. €

The compressed sensing scheme takes a total of samples of resumØ coefficients and samples associated with detail coefficients, for a total pieces of information. It achieves error comparable to classical sampling based on samples. Thus, it needs dramatically fewer samples for comparable accuracy: roughly speaking, only the cube root of the number of samples of linear sampling. 2j1

To achieve this dramatic reduction in sampling, we need an information operator based on some satisfying CS1-CS3. The underlying measurement kernels will be of the form

<!-- formula-not-decoded -->

where the collection is simply an enumeration of the wavelets , with and .

## B. Images of Bounded Variation

We consider now the model with images of Bounded Variation from the Introduction. Let denote the class of functions with domain , having total variation at most [38], and bounded in absolute value by as well. In the Introduction, it was mentioned that the wavelet coefficients at level obey where depends only on the wavelet used. It is also true that . € f(s) € B .

Weagain consider two ways of approximating functions in . The classic linear scheme uses a 2-D version of the scheme we have already discussed. We again fix a 'finest scale' and measure the resumØ coefficients where now is a pair of integers , . indexing position. We use the Haar scaling function

Wereconstruct by giving an approximation error

<!-- formula-not-decoded -->

There are coefficients associated with the unit square, and so the approximation error obeys =

<!-- formula-not-decoded -->

In the compressed sensing scheme, we need also Haar wavelets where is an oscillating function with mean zero which is either horizontally oriented , vertically oriented , or diagonally oriented . We pick a 'coarsest scale' , and measure the resumØ coefficients -there are of these. Then let be the concatenation of the detail wavelet coefficients . The dimension of is . The norm obeys = = 0 =

<!-- formula-not-decoded -->

This establishes the constraint on norm needed for applying our theory. We take and apply a near-optimal information operator for this and . We apply the near4j0 og

optimal algorithm of minimization to the resulting information, getting the error estimate

<!-- formula-not-decoded -->

with independent of . The overall reconstruction

<!-- formula-not-decoded -->

has error

<!-- formula-not-decoded -->

again with independent of . This is of the same order of magnitude as the error of linear sampling. €

The compressed sensing scheme takes a total of samples of resumØ coefficients and samples associated with detail coefficients, for a total pieces of measured information. It achieves error comparable to classical sampling with samples. Thus, just as we have seen in the Bump Algebra case, we need dramatically fewer samples for comparable accuracy: roughly speaking, only the square root of the number of samples of linear sampling. C . c4jo 4j1/2

## C. Piecewise Images With Edges

We now consider an example where , and we can apply the extensions to tight frames and to weakmentioned earlier. Again in the image processing setting, we use the -model discussed in CandŁs and Donoho [39], [40]. Consider the collection of piecewise smooth , with values, first and second partial derivatives bounded by , away from an exceptional set which is a union of curves having first and second derivatives in an appropriate parametrization; the curves have total length . More colorfully, such images are cartoons -patches of uniform smooth behavior separated by piecewise-smooth curvilinear boundaries. They might be reasonable models for certain kinds of technical imagery-e.g., in radiology. C2,2(B, L)

The curvelets tight frame [40] is a collection of smooth frame elements offering a Parseval relation and reconstruction formula

The frame elements have a multiscale organization, and frame coefficients grouped by scale obey the weak constraint

<!-- formula-not-decoded -->

compare [40]. For such objects, classical linear sampling at scale by smooth 2-D scaling functions gives

<!-- formula-not-decoded -->

This is no better than the performance of linear sampling for the Bounded Variation case, despite the piecewise character of ; the possible discontinuities in are responsible for the inability of linear sampling to improve its performance over compared to Bounded Variation. C2

In the compressed sensing scheme, we pick a coarsest scale . We measure the resumØ coefficients in a smooth wavelet expansion-there are of these-and then let denote a concatentation of the finer scale curvelet coefficients . The dimension of is , with due to overcompleteness of curvelets. The weak 'norm' obeys Jo = 4j0 c(4j1

<!-- formula-not-decoded -->

with depending on and . This establishes the constraint on weak norm needed for our theory. We take and apply a near-optimal information operator for this and . We apply the near-optimal algorithm of minimization to the resulting information, getting the error estimate

<!-- formula-not-decoded -->

with absolute. The overall reconstruction has error

<!-- formula-not-decoded -->

again with independent of . This is of the same order of magnitude as the error of linear sampling.

The compressed sensing scheme takes a total of samples of resumØ coefficients and samples associated with detail coefficients, for a total pieces of information. It achieves error comparable to classical sampling based on samples. Thus, even more so than in the Bump Algebra case, we need dramatically fewer samples for comparable accuracy: roughly speaking, only the fourth root of the number of samples of linear sampling.

## VII. NEARLY ALL MATRICES ARE CS MATRICES

We may reformulate Theorem 6 as follows.

Theorem 10: Let , be a sequence of problem sizes with , , for and . Let be a matrix with columns drawn independently and uniformly at random from . Then for some and , conditions CS1-CS3 hold for with overwhelming probability for all large . &lt; mn &gt; 0

Indeed, note that the probability measure on induced by sampling columns i.i.d. uniform on is exactly the natural uniform measure on . Hence, Theorem 6 follows immediately from Theorem 10.

In effect matrices satisfying the CS conditions are so ubiquitous that it is reasonable to generate them by sampling at random from a uniform probability distribution.

The proof of Theorem 10 is conducted over Sections VIIA-C; it proceeds by studying events , , where CS1 Holds , etc. It will be shown that for parameters and 0

then defining and , we have =

Since, when occurs, our random draw has produced a matrix obeying CS1-CS3 with parameters and , this proves Theorem 10. The proof actually shows that for some , , ; the convergence is exponentially fast.

## A. Control of Minimal Eigenvalue

The following lemma allows us to choose positive constants and so that condition CS1 holds with overwhelming probability. 01

Lemma7.1: Consider sequences of with . Define the event mn) (n,

<!-- formula-not-decoded -->

Then, for each , for sufficiently small

<!-- formula-not-decoded -->

The proof involves three ideas. The first idea is that the event of interest for Lemma 7.1 is representable in terms of events indexed by individual subsets

<!-- formula-not-decoded -->

Our plan is to bound the probability of occurrence of every . The second idea is that for a specific subset , we get large deviations bounds on the minimum eigenvalue; this can be stated as follows.

Lemma 7.2: For , let denote the event that the minimum eigenvalue exceeds . Then there is so that for sufficiently small and all &gt;

<!-- formula-not-decoded -->

uniformly in .

This was derived in [11] and in [35], using the concentration of measure property of singular values of random matrices, e.g., see Szarek's work [41], [42].

The third and final idea is that bounds for individual subsets can control simultaneous behavior over all . This is expressed as follows.

Lemma 7.3: Suppose we have events all obeying, for some fixed and Qn,J &gt;

for each with . Pick with and with . Then for all sufficiently large &gt; &gt; &lt; ß

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Our main goal of this subsection, Lemma 7.1, now follows by combining these three ideas.

It remains only to prove Lemma 7.3. Let

<!-- formula-not-decoded -->

We note that, by Boole's inequality

<!-- formula-not-decoded -->

the last inequality following because each member is of cardinality , since , as soon as . Also, of course m 2 &lt; n 2 3 &gt;

<!-- formula-not-decoded -->

so we get . Taking as given, we get the desired conclusion. log(#J) &lt;

## B. Spherical Sections Property

We now show that condition CS2 can be made overwhelmingly likely for large by choice of and sufficiently small but still positive. Our approach derives from [11], which applied an important result from Banach space theory, the almost spherical sections phenomenon. This says that slicing the unit ball in a Banach space by intersection with an appropriate finite-dimensional linear subspace will result in a slice that is effectively spherical [43], [44]. We develop a quantitative refinement of this principle for the norm in , showing that, with overwhelming probability, every operator for affords a spherical section of the ball. The basic argument we use originates from work of Milman, Kashin, and others [44], [10], [45]; we refine an argument in Pisier [19] and, as in [11], draw inferences that may be novel. We conclude that not only do almost-spherical sections exist, but they are so ubiquitous that every with will generate them. pn / log(m)

Definition 7.1: Let . We say that offers an -isometry between and if

<!-- formula-not-decoded -->

The following lemma shows that condition CS2 is a generic property of matrices.

Lemma 7.4: Consider the event that every with offers an -isometry between and . For each , there is so that p(e)

<!-- formula-not-decoded -->

To prove this, we first need a lemma about individual subsets proven in [11].

Lemma 7.5: Fix . Choose so that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Choose so that

<!-- formula-not-decoded -->

and let denote the difference between the two sides. For a subset in , let denote the event that furnishes an -isometry to . Then as 4J

<!-- formula-not-decoded -->

Now note that the event of interest for Lemma 7.4 is

<!-- formula-not-decoded -->

to finish, apply the individual Lemma 7.5 together with the combining principle in Lemma 7.3.

## C. Quotient Norm Inequalities

We now show that, for , for sufficiently small , nearly all large by matrices have property CS3. Our argument borrows heavily from [11] which the reader may find helpful. We here make no attempt to provide intuition or to compare with closely related notions in the local theory of Banach spaces (e.g., Milman's Quotient of a Subspace Theorem [19]).

Let be any collection of indices in ; is a linear subspace of , and on this subspace a subset of possible sign patterns can be realized, i.e., sequences of 's generated by

<!-- formula-not-decoded -->

CS3 will follow if we can show that for every , some approximation to satisfies for . &lt; 1 y €

Lemma7.6: Uniform Sign-Pattern Embedding. Fix . Then for , set

<!-- formula-not-decoded -->

For sufficiently small , there is an event with , as . On this event, for each subset with , for each sign pattern in , there is a vector with 03 (03, 8)

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

In words, a small multiple of any sign pattern almost lives in the dual ball .

Before proving this result, we indicate how it gives the property CS3; namely, that , and imply

<!-- formula-not-decoded -->

Consider the convex optimization problem

<!-- formula-not-decoded -->

This can be written as a linear program, by the same sort of construction as given for (IV.1). By the duality theorem for linear programming, the value of the primal program is at least the value of the dual

<!-- formula-not-decoded -->

Lemma 7.6 gives us a supply of dual-feasible vectors and hence a lower bound on the dual program. Take ; we can find which is dual-feasible and obeys =

<!-- formula-not-decoded -->

picking appropriately and taking into account the spherical sections theorem, for sufficiently large , we have ; (VII.7) follows with . 3 / =

1) Proof of Uniform Sign-Pattern Embedding: The proof of Lemma 7.6 follows closely a similar result in [11] that considered the case . Our idea here is to adapt the argument for the case to the case, with changes reflecting the different choice of , , and the sparsity bound . We leave out large parts of the argument, as they are identical to the corresponding parts in [11]. The bulk of our effort goes to produce the following lemma, which demonstrates approximate embedding of a single sign pattern in the dual ball. n m &lt; m &lt; m ~ Am?

Lemma 7.7: Individual Sign-Pattern Embedding. Let , let , with , , , as in the statement of Lemma 7.6. Let . Given a collection , there is an iterative algorithm, described below, producing a vector as output which obeys 1 &lt; i &lt; m

<!-- formula-not-decoded -->

Let be i.i.d. uniform on ; there is an event described below, having probability controlled by Slo

<!-- formula-not-decoded -->

for which can be explicitly given in terms of and . On this event

<!-- formula-not-decoded -->

Lemma 7.7 will be proven in a section of its own. We now show that it implies Lemma 7.6.

We recall a standard implication of so-called VapnikCervonenkis theory [46]

<!-- formula-not-decoded -->

Notice that if , then

<!-- formula-not-decoded -->

while also

<!-- formula-not-decoded -->

Hence, the total number of sign patterns generated by operators obeys

<!-- formula-not-decoded -->

Now furnished by Lemma 7.7 is positive, so pick with . Define &gt;

<!-- formula-not-decoded -->

where denotes the instance of the event (called in the statement of Lemma 7.7) generated by a specific , combination. On the event , every sign pattern associated with any obeying is almost dual feasible. Now

<!-- formula-not-decoded -->

which tends to zero exponentially rapidly.

## D. Proof of Individual Sign-Pattern Embedding

1) An Embedding Algorithm: The companion paper [11] introduced an algorithm to create a dual feasible point starting from a nearby almost-feasible point . It worked as follows.

Let be the collection of indices with

<!-- formula-not-decoded -->

and then set where denotes the least-squares projector . In effect, identify the indices where exceeds half the forbidden level , and 'kill' those indices. 'Io

Continue this process, producing , , etc., with stage-dependent thresholds successively closer to . Set

<!-- formula-not-decoded -->

and, putting ,

<!-- formula-not-decoded -->

If is empty, then the process terminates, and set . Termination must occur at stage . At termination

<!-- formula-not-decoded -->

Hence, is definitely dual feasible. The only question is how close to it is.

2) Analysis Framework: Also in [11] bounds were developed for two key descriptors of the algorithm trajectory

<!-- formula-not-decoded -->

and

Weadapt the arguments deployed there. We define bounds and for and , of the form Ve;n

<!-- formula-not-decoded -->

here and , where will be defined later. We define subevents

<!-- formula-not-decoded -->

Now define this event implies, because

<!-- formula-not-decoded -->

We will show that, for chosen in conjunction with

<!-- formula-not-decoded -->

This implies

Put

<!-- formula-not-decoded -->

and note that this depends quite weakly on . Recall that the event is defined in terms of and . On the event , . Lemma 7.1 implicitly defined a quantity lowerbounding the minimum eigenvalue of &lt;

<!-- formula-not-decoded -->

and the lemma follows.

3) Large Deviations: Define the events so that

<!-- formula-not-decoded -->

every where . Pick so that . Pick so that À1(p1/2 A,

<!-- formula-not-decoded -->

With this choice of , when the event occurs,

<!-- formula-not-decoded -->

Also, on , (say) for . In [11], an analysis framework was developed in which a family of random variables i.i.d. was introduced, and it was shown that /8j = 2-j-1 2-j-1

and

<!-- formula-not-decoded -->

That paper also stated two simple large deviations bounds.

Lemma 7.8: Let be i.i.d. , ,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Applying this, we note that the event

<!-- formula-not-decoded -->

stated in terms of variables, is equivalent to an event

<!-- formula-not-decoded -->

stated in terms of standard random variables, where and

<!-- formula-not-decoded -->

We therefore have for the inequality

<!-- formula-not-decoded -->

Now and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since , the term of most concern in is at ; the other terms are always better. Also in fact does not depend on . Focusing now on , we may write

<!-- formula-not-decoded -->

Recalling that and putting

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

A similar analysis holds for the

<!-- formula-not-decoded -->

## VIII. CONCLUSION

A. Summary

We have described an abstract framework for compressed sensing of objects which can be represented as vectors . We assume the object of interest is a priori compressible so that for a known basis or frame and . Starting from an by matrix with satisfying conditions CS1-CS3, and with the matrix of an orthonormal basis or tight frame underlying , we define the information operator . Starting from the -tuple of measured information , we reconstruct an approximation to by solving m

<!-- formula-not-decoded -->

The proposed reconstruction rule uses convex optimization and is computationally tractable. Also, the needed matrices satisfying CS1-CS3 can be constructed by random sampling from a uniform distribution on the columns of .

We give error bounds showing that despite the apparent undersampling , good accuracy reconstruction is possible for compressible objects, and we explain the near-optimality of these bounds using ideas from Optimal Recovery and Information-Based Complexity. We even show that the results are stable under small measurement errors in the data ( small). Potential applications are sketched related to imaging and spectroscopy. In(so)ll

## B. Alternative Formulation

Weremark that the CS1-CS3 conditions are not the only way to obtain our results. Our proof of Theorem 9 really shows the following.

Theorem 11: Suppose that an matrix obeys the following conditions, with constants , and . &lt;

- A1: The maximal concentration (defined in Section IV-B) obeys v(4, J)

<!-- formula-not-decoded -->

- A2: The width (defined in Section II) obeys

<!-- formula-not-decoded -->

Let . For some and all , the solution of obeys the estimate

<!-- formula-not-decoded -->

In short, a different approach might exhibit operators with good widths over balls only, and low concentration on 'thin' sets. Another way to see that the conditions CS1-CS3 can no doubt be approached differently is to compare the results in [11], [35]; the second paper proves results which partially overlap those in the first, by using a different technique.

## C. The Partial Fourier Ensemble

We briefly discuss two recent articles which do not fit in the -widths tradition followed here, and so were not easy to cite earlier with due prominence.

First, and closest to our viewpoint, is the breakthrough paper of CandŁs, Romberg, and Tao [13]. This was discussed in Section IV-B; the result of [13] showed that minimization can be used to exactly recover sparse sequences from the Fourier transform at randomly chosen frequencies, whenever the sequence has fewer than nonzeros, for some . Second is the article of Gilbert et al. [12], which showed that a different nonlinear reconstruction algorithm can be used to recover approximations to a vector in which is nearly as good as the best -term approximation in norm, using about random but nonuniform samples in the frequency domain; here is (it seems) an upper bound on the norm of .

These papers both point to the partial Fourier ensemble, i.e., the collection of matrices made by sampling rows out of the Fourier matrix, as concrete examples of working within the CS framework; that is, generating near-optimal subspaces for Gel'fand -widths, and allowing minimization to reconstruct from such information for all . &lt;

Now [13] (in effect) proves that if , then in the partial Fourier ensemble with uniform measure, the maximal concentration condition A1 (VIII.1) holds with overwhelming probability for large (for appropriate constants , ). On the other hand, the results in [12] seem to show that condition A2 (VIII.2) holds in the partial Fourier ensemble with overwhelming probability for large , when it is sampled with a certain nonuniform probability measure. Although the two papers [13], [12] refer to different random ensembles of partial Fourier matrices, both reinforce the idea that interesting relatively concrete families of operators can be developed for compressed sensing applications. In fact, CandŁs has informed us of some recent results he obtained with Tao [47] indicating that, modulo polylog factors, A2 holds for the uniformly sampled partial Fourier ensemble. This seems a very significant advance. An 9

Note Added in Proof

In the months since the paper was written, several groups have conducted numerical experiments on synthetic and real data for the method described here and related methods. They have explored applicability to important sensor problems, and studied applications issues such as stability in the presence of noise. The reader may wish to consult the forthcoming Special Issue on Sparse Representation of the EURASIP journal Applied Signal Processing , or look for papers presented at a special session in ICASSP 2005, or the workshop on sparse representation held in May 2005 at the University of Maryland Center for Scientific Computing and Applied Mathematics, or the workshop in November 2005 at Spars05, UniversitØ de Rennes.

Areferee has pointed out that Compressed Sensing is in some respects similar to problems arising in data stream processing, where one wants to learn basic properties [e.g., moments, histogram] of a datastream without storing the stream. In short, one wants to make relatively few measurements while inferring relatively much detail. The notions of 'Iceberg queries' in large databases and 'heavy hitters' in data streams may provide points of entry into that literature.

## ACKNOWLEDGMENT

In spring 2004, Emmanuel CandŁs told the present author about his ideas for using the partial Fourier ensemble in 'undersampled imaging'; some of these were published in [13]; see also the presentation [14]. More recently, CandŁs informed the present author of the results in [47] referred to above. It is a pleasure to acknowledge the inspiring nature of these conversations. The author would also like to thank Anna Gilbert for telling him about her work [12] finding the B-best Fourier coefficients by nonadaptive sampling, and to thank Emmanuel CandŁs for conversations clarifying Gilbert's work. Thanks to the referees for numerous suggestions which helped to clarify the exposition and argumentation. Anna Gilbert offered helpful pointers to the data stream processing literature.

## REFERENCES

- [1] D. L. Donoho, M. Vetterli, R. A. DeVore, and I. C. Daubechies, 'Data compression and harmonic analysis,' IEEE Trans. Inf. Theory , vol. 44, no. 6, pp. 2435-2476, Oct. 1998.
- [2] Y. Meyer, Wavelets and Operators . Cambridge, U.K.: Cambridge Univ. Press, 1993.
- [3] D. L. Donoho, 'Unconditional bases are optimal bases for data compression and for statistical estimation,' Appl. Comput. Harmonic Anal. , vol. 1, pp. 100-115, 1993.
- [4] , 'Sparse components of images and optimal atomic decomposition,' Constructive Approx. , vol. 17, pp. 353-382, 2001.
- [5] A. Pinkus, ' /110 -widths and optimal recovery in approximation theory,' in Proc. Symp. Applied Mathematics , vol. 36, C. de Boor, Ed., Providence, RI, 1986, pp. 51-66. [6] J. F. Traub and H. Woziakowski, A General Theory of Optimal Algorithms . New York: Academic, 1980.
- [7] D. L. Donoho, 'Unconditional bases and bit-level compression,' Appl. Comput. Harmonic Anal. , vol. 3, pp. 388-92, 1996.
- [8] A. Pinkus, /110 -Widths in Approximation Theory . New York: SpringerVerlag, 1985. [9] A. Y. Garnaev and E. D. Gluskin, 'On widths of the Euclidean ball' (in English), Sov. Math.-Dokl. , vol. 30, pp. 200-203, 1984.
- [10] B. S. Kashin, 'Diameters of certain finite-dimensional sets in classes of smooth functions,' Izv. Akad. Nauk SSSR, Ser. Mat. , vol. 41, no. 2, pp. 334-351, 1977.

- [11] D. L. Donoho, 'For most large underdetermined systems of linear equations, the minimal /96 -norm solution is also the sparsest solution,' Commun. Pure Appl. Math. , to be published.
- [12] A. C. Gilbert, S. Guha, P. Indyk, S. Muthukrishnan, and M. Strauss, 'Near-optimal sparse fourier representations via sampling,' in Proc 34th ACM Symp. Theory of Computing , MontrØal, QC, Canada, May 2002, pp. 152-161.
- [13] E. J. CandŁs, J. Romberg, and T. Tao, 'Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information.,' IEEE Trans. Inf. Theory , to be published.
- [14] E. J. CandŁs, 'Robust Uncertainty Principles and Signal Recovery,' presented at the 2nd Int. Conf. Computational Harmonic Anaysis, Nashville, TN, May 2004.
- [15] C. A. Micchelli and T. J. Rivlin, 'A survey of optimal recovery,' in Optimal Estimation in Approximation Theory , C. A. Micchelli and T. J. Rivlin, Eds: Plenum, 1977, pp. 1-54.
- [16] J. Peetre and G. Sparr, 'Interpolation of normed abelian groups,' Ann. Math. Pure Appl. , ser. 4, vol. 92, pp. 217-262, 1972.
- [17] J. Bergh and J. Löfström, Interpolation Spaces. An Introduction . Berlin, Germany: Springer-Verlag, 1976.
- [18] B. Carl, 'Entropy numbers /115 -numbers, and eigenvalue problems,' J. Funct. Anal. , vol. 41, pp. 290-306, 1981.
- [19] G. Pisier, The Volume of Convex Bodies and Banach Space Geometry . Cambridge, U.K.: Cambridge Univ. Press, 1989.
- [20] C. Schütt, 'Entropy numbers of diagonal operators between symmetric Banach spaces,' J. Approx. Theory , vol. 40, pp. 121-128, 1984.
- [21] T. Kuhn, 'A lower estimate for entropy numbers,' J. Approx. Theory , vol. 110, pp. 120-124, 2001.
- [22] S. Gal and C. Micchelli, 'Optimal sequential and nonsequential procedures for evaluating a functional,' Appl. Anal. , vol. 10, pp. 105-120, 1980.
- [23] A. A. Melkman and C. A. Micchelli, 'Optimal estimation of linear operators from inaccurate data,' SIAM J. Numer. Anal. , vol. 16, pp. 87-105, 1979.
- [24] M. A. Kon and E. Novak, 'The adaption problem for approximating linear operators,' Bull. Amer. Math. Soc. , vol. 23, pp. 159-165, 1990.
- [25] E. Novak, 'On the power of adaption,' J. Complexity , vol. 12, pp. 199-237, 1996.
- [26] S. Chen, D. L. Donoho, and M. A. Saunders, 'Atomic decomposition by basis pursuit,' SIAM J. Sci Comp. , vol. 20, no. 1, pp. 33-61, 1999.
- [27] D. L. Donoho and X. Huo, 'Uncertainty principles and ideal atomic decomposition,' IEEE Trans. Inf. Theory , vol. 47, no. 7, pp. 2845-62, Nov. 2001.
- [28] M. Elad and A. M. Bruckstein, 'A generalized uncertainty principle and sparse representations in pairs of bases,' IEEE Trans. Inf. Theory , vol. 49, no. 9, pp. 2558-2567, Sep. 2002.
- [29] D. L. Donoho and M. Elad, 'Optimally sparse representation from overcomplete dictionaries via /96 norm minimization,' Proc. Natl. Acad. Sci. USA , vol. 100, no. 5, pp. 2197-2002, Mar. 2002.
- [30] R. Gribonval and M. Nielsen, 'Sparse representations in unions of bases,' IEEE Trans Inf Theory , vol. 49, no. 12, pp. 3320-3325, Dec. 2003.
- [31] J. J. Fuchs, 'On sparse representation in arbitrary redundant bases,' IEEE Trans. Inf. Theory , vol. 50, no. 6, pp. 1341-1344, Jun. 2002.
- [32] J. A. Tropp, 'Greed is good: Algorithmic results for sparse approximation,' IEEE Trans Inf. Theory , vol. 50, no. 10, pp. 2231-2242, Oct. 2004.
- [33] , 'Just relax: Convex programming methods for identifying sparse signals in noise,' IEEE Trans Inf. Theory , vol. 52, no. 3, pp. 1030-1051, Mar. 2006.
- [34] D. L. Donoho, M. Elad, and V. Temlyakov, 'Stable recovery of sparse overcomplete representations in the presence of noise,' IEEE Trans. Inf. Theory , vol. 52, no. 1, pp. 6-18, Jan. 2006.
- [35] D. L. Donoho, 'For most underdetermined systems of linear equations, the minimal /96 -norm near-solution approximates the sparsest near-solution,' Commun. Pure Appl. Math. , to be published.
- [36] I. C. Daubechies, Ten Lectures on Wavelets . Philadelphia, PA: SIAM, 1992.
- [37] S. Mallat, A Wavelet Tour of Signal Processing . San Diego, CA: Academic, 1998.
- [38] A. Cohen, R. DeVore, P. Petrushev, and H. Xu, 'Nonlinear approximation and the space /66/86 /40 /82 /41 ,' Amer. J. Math. , vol. 121, pp. 587-628, 1999.
- [39] E. J. CandŁs and D. L. Donoho, 'Curvelets-A surprisingly effective nonadaptive representation for objects with edges,' in Curves and Surfaces , C. Rabut, A. Cohen, and L. L. Schumaker, Eds. Nashville, TN: Vanderbilt Univ. Press, 2000.
- [40] , 'New tight frames of curvelets and optimal representations of objects with piecewise /67 singularities,' Comm. Pure Appl. Math. , vol. LVII, pp. 219-266, 2004.
- [41] S. J. Szarek, 'Spaces with large distances to /96 and random matrices,' Amer. J. Math. , vol. 112, pp. 819-842, 1990.
- [42] , 'Condition numbers of random matrices,' J. Complexity , vol. 7, pp. 131-149, 1991.
- [43] A. Dvoretsky, 'Some results on convex bodies and banach spaces,' in Proc. Symp. Linear Spaces , Jerusalem, Israel, 1961, pp. 123-160.
- [44] T. Figiel, J. Lindenstrauss, and V . D. Milman, 'The dimension of almostspherical sections of convex bodies,' Acta Math. , vol. 139, pp. 53-94, 1977.
- [45] V. D. Milman and G. Schechtman, Asymptotic Theory of Finite-Dimensional Normed Spaces (Leecture Notes in Mathematics) . Berlin, Germany: Springer-Verlag, 1986, vol. 1200.
- [46] D. Pollard, Empirical Processes: Theory and Applications . Hayward, CA: Inst. Math. Statist., vol. 2, NSF-CBMS Regional Conference Series in Probability and Statistics.
- [47] E. J. CandŁs and T. Tao, 'Near-optimal signal recovery from random projections: Universal encoding strategies,' Applied and Computational Mathematics, Calif. Inst. Technol., Tech. Rep., 2004.