## Robust Uncertainty Principles: Exact Signal Reconstruction from Highly Incomplete Frequency Information

Emmanuel Candes † , Justin Romberg † , and Terence Tao /sharp

† Applied and Computational Mathematics, Caltech, Pasadena, CA 91125 /sharp Department of Mathematics, University of California, Los Angeles, CA 90095

June 10, 2004

## Abstract

This paper considers the model problem of reconstructing an object from incomplete frequency samples. Consider a discrete-time signal f ∈ C N and a randomly chosen set of frequencies Ω of mean size τN . Is it possible to reconstruct f from the partial knowledge of its Fourier coefficients on the set Ω?

A typical result of this paper is as follows: for each M &gt; 0, suppose that f obeys

<!-- formula-not-decoded -->

/negationslash then with probability at least 1 -O ( N -M ), f can be reconstructed exactly as the solution to the /lscript 1 minimization problem

<!-- formula-not-decoded -->

In short, exact recovery may be obtained by solving a convex optimization problem. We give numerical values for α which depends on the desired probability of success; except for the logarithmic factor, the condition on the size of the support is sharp.

The methodology extends to a variety of other setups and higher dimensions. For example, we show how one can reconstruct a piecewise constant (one or two-dimensional) object from incomplete frequency samples-provided that the number of jumps (discontinuities) obeys the condition above-by minimizing other convex functionals such as the total-variation of f .

Keywords. Random matrices, free probability, sparsity, trigonometric expansions, uncertainty principle, convex optimization, duality in optimization, total-variation minimization, image reconstruction, linear programming.

Acknowledgments. E. C. is partially supported by a National Science Foundation grant DMS 01-40698 (FRG) and by an Alfred P. Sloan Fellowship. J. R. is supported by National Science Foundation grants DMS 01-40698 and ITR ACI-0204932. T. T. is a Clay Prize Fellow and is supported in part by grants from the Packard Foundation. E. C. and T.T. thank the Institute for Pure and Applied Mathematics at UCLA for their warm hospitality. E. C. would like to thank Amos Ron and David Donoho for stimulating conversations, and Po-Shen Loh for early numerical experiments on a related project.

## 1 Introduction

In many applications of practical interest, we often wish to reconstruct an object (a discrete signal, a discrete image, etc.) from incomplete Fourier samples. In a discrete setting, we may pose the problem as follows; let ˆ f be the Fourier transform of a discrete object f ( t ), t ∈ Z d N := { 0 , 1 , . . . , N -1 } d ,

<!-- formula-not-decoded -->

The problem is then to recover f from partial frequency information, namely, from ˆ f ( ω ), where ω = ( ω 1 , . . . , ω d ) belongs to some set Ω of cardinality less than N d -the size of the discrete object.

In this paper, we show that we can recover f exactly from observations ˆ f | Ω on small set of frequencies provided that f is sparse . The recovery consists of solving a straightforward optimization problem that finds f /sharp of minimal complexity with ˆ f /sharp ( ω ) = ˆ f ( ω ), ∀ ω ∈ Ω.

## 1.1 A puzzling numerical experiment

This idea is best motivated by an experiment with surprisingly positive results. Consider a simplified version of the classical 'tomography' problem in medical imaging: we wish to reconstruct a 2D image f ( t 1 , t 2 ) from samples ˆ f | Ω of its discrete Fourier transform on a star-shaped domain Ω [4]. Our choice of domain is not contrived; many real imaging devices can collect high-resolution samples along radial lines at relatively few angles. Figure 1(b) illustrates a typical case where one gathers 512 samples along each of 22 radial lines.

Frequently discussed approaches in the literature of medical imaging for reconstructing an object from 'polar' frequency samples are the so-called filtered backprojection algorithms. In a nutshell, one assumes that the Fourier coefficients at all of the unobserved frequencies are zero (thus reconstructing the image of 'minimal energy' under the observation constraints). This strategy does not perform very well, and could hardly be used for medical diagnostic [15]. The reconstructed image, shown in Figure 1(c), has severe nonlocal artifacts caused by the angular undersampling. A good reconstruction algorithm, it seems, would have to guess the values of the missing Fourier coefficients. In other words, one would need to interpolate ˆ f ( ω 1 , ω 2 ). This is highly problematic, however; predictions of Fourier coefficients from their neighbors are very delicate, due to the global and highly oscillatory nature of the Fourier transform. Going back to our example, we can see the problem immediately. To recover frequency information near ( ω 1 , ω 2 ), where ω 1 is near ± π , we would need to interpolate ˆ f at the Nyquist rate 2 π/N . However, we only have samples at rate about π/ 22; the sampling rate is almost 50 times smaller than the Nyquist rate!

We propose instead a strategy based on convex optimization. Let ‖ g ‖ BV be the totalvariation norm of a two-dimensional object g which for discrete data g ( t 1 , t 2 ), 0 ≤ t 1 , t 2 ≤ N -1, takes the form

<!-- formula-not-decoded -->

Figure 1: Example of a simple recovery problem. (a) The Logan-Shepp phantom test image. (b) Sampling 'domain' in the frequency plane; Fourier coefficients are sampled along 22 approximately radial lines. (c) Minimum energy reconstruction obtained by setting unobserved Fourier coefficients to zero. (d) Reconstruction obtained by minimizing the total-variation, as in (1.1). The reconstruction is an exact replica of the image in (a).

<!-- image -->

where D 1 is the finite difference D 1 g = g ( t 1 , t 2 ) -g ( t 1 -1 , t 2 ) and D 2 g = g ( t 1 , t 2 ) -g ( t 1 , t 2 -1). To recover f from partial Fourier samples, we find a solution f /sharp to the optimization problem

<!-- formula-not-decoded -->

In a nutshell, given partial observation ˆ f | Ω , we seek a solution f /sharp with minimum complexityhere Total Variation (TV)-and whose 'visible' coefficients match those of the unknown object f . Our hope here is to partially erase some of the artifacts classical reconstruction methods exhibit (which tend to have large TV norm) while maintaining fidelity to the observed data via the constraints on the Fourier coefficients of the reconstruction.

When we use (1.1) for the recovery problem illustrated in Figure 1 (with the popular LoganShepp phantom as a test image), the results are surprising. The reconstruction is exact ; that is, f /sharp = f ! Now this numerical result is not special to this phantom. In fact, we performed a series of experiments of this type and obtained perfect reconstruction on many similar test phantoms.

## 1.2 Main Results

This paper is about a quantitative understanding of this very special phenomenon. For which classes of signals/images can we expect perfect reconstruction? What are the tradeoffs between complexity and number of samples? In order to answer these questions, we first develop a fundamental mathematical understanding of a special one-dimensional model problem; we then exhibit reconstruction strategies which are shown to exactly reconstruct the unknown signal and can be deployed in many related and sophisticated reconstruction setups.

For a signal f ∈ C N , we define the classical discrete transform Fourier transform F f = ˆ f : C N → C N by

If we are given the value of the Fourier coefficients ˆ f ( k ) for all frequencies k ∈ Z N , then one can obviously reconstruct f exactly via the Fourier inversion formula

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now suppose that we are only given the Fourier coefficients ˆ f | Ω sampled in some partial subset Ω /subsetnoteql Z N of all frequencies (here and below we abuse notations and identify the frequencies ω k = 2 πk/N with the corresponding integers whenever convenient). Of course, this is not enough information by itself to reconstruct f exactly, since f has N degrees of freedom and we are only specifying | Ω | &lt; N of those degrees (here and below | Ω | denotes the cardinality of Ω).

Suppose, however, that we also specify that f is supported on a small (but a priori unknown) subset T of Z N ; that is, we assume that f can be written as a sparse superposition of spikes

<!-- formula-not-decoded -->

If | T | is small enough, we can recover f exactly:

Theorem 1.1 Suppose that the signal length N is a prime integer. Let Ω be a subset of { 0 , . . . , N -1 } , and let f be a vector supported on T such that

<!-- formula-not-decoded -->

Then f can be reconstructed uniquely from Ω and ˆ f | Ω . Conversely, if Ω is not the set of all N frequencies, then there exist distinct vectors f, g such that | supp( f ) | , | supp( g ) | ≤ 1 2 | Ω | +1 and such that ˆ f | Ω = ˆ g | Ω .

Proof We will need the following lemma [18], from which we see that with knowledge of T , we can reconstruct f uniquely (using linear algebra) from ˆ f | Ω :

Lemma 1.2 ([18], Corollary 1.4) Let N be a prime integer and T, Ω be subsets of Z N . Put /lscript 2 ( T ) (resp. /lscript 2 (Ω) ) to be the space of signals that are zero outside of T (resp. Ω ). The restricted Fourier transform F T → Ω : /lscript 2 ( T ) → /lscript 2 (Ω) is defined as

<!-- formula-not-decoded -->

If | T | = | Ω | , then F T → Ω is a bijection; as a consequence, we thus see that F T → Ω is injective for | T | ≤ | Ω | and surjective for | T | ≥ | Ω | . Clearly, the same claims hold if the Fourier transform F is replaced by the inverse Fourier transform F -1 .

To prove Theorem 1.1, we start with the former claim. Suppose for contradiction that there were two objects f, g such that ˆ f | Ω = ˆ g | Ω and | supp( f ) | , | supp( g ) | ≤ 1 2 | Ω | . Then the Fourier transform of f -g vanishes on Ω, and | supp( f -g ) | ≤ | Ω | . By Lemma 1.2 we see that F supp ( f -g ) → Ω is injective, and thus f -g = 0. The uniqueness claim follows.

Now we prove the latter claim. Since | Ω | &lt; N , we can find disjoint subsets T, S of Ω such that | T | , | S | ≤ 1 2 | Ω | +1 and | T | + | S | = | Ω | +1. Let k 0 be some frequency which does not lie in Ω. Applying Lemma 1.2, we have that F T ∪ S → Ω ∪{ k 0 } is a bijection, and thus we can find a vector h supported on T ∪ S whose Fourier transform vanishes on Ω but is non-zero on k 0 ; in particular, h is not identically zero. The claim now follows by taking f := h | T and g := -h | S .

Note that if N is not prime, the lemma (and hence the theorem) fails, essentially because of the presence of non-trivial subgroups of Z N with addition modulo N ; see [6], [18] for further discussion. However, it is plausible to think that Lemma 1.2 continues to hold for non-prime N if T and Ω are assumed to be generic - in particular, they are not subgroups of Z N , or cosets of subgroups. If T and Ω are selected uniformly at random, then it is expected that the theorem holds with probability very close to one; one can indeed presumably quantify this statement by adapting the arguments given above but we will not do so here. However, we refer the reader to section 1.6 for a rapid presentation of informal arguments pointing out in this direction.

A refinement of the argument in Theorem 1.1 shows that for fixed sets T , S , Ω in Z N , the space of vectors f, g supported on T , S such that ˆ f | Ω = ˆ g | Ω has dimension | T ∪ S | - | Ω |

when | T ∪ S | ≥ | Ω | , and has dimension | T ∩ S | otherwise. In particular, if we let Σ( N t ) denote those vectors whose support has size at most N t , then set of the vectors in Σ( N t ) which cannot be reconstructed uniquely in this class from the Fourier coefficients sampled at Ω, is contained in a finite union of linear spaces of dimension at most 2 N t -| Ω | . Since Σ( N t ) itself is a finite union of linear spaces of dimension N t , we thus see that recovery of f from ˆ f | Ω is in principle possible generically whenever | supp( f ) | = N t &lt; | Ω | ; once N t ≥ | Ω | , however, it is clear from simple degrees-of-freedom arguments that unique recovery is no longer possible. While our methods do not quite attain this theoretical upper bound for correct recovery, our numerical experiements suggest that they do come within a constant factor of this bound (see Figure 2).

Theorem 1.1 asserts that f can be reconstructed from ˆ f | Ω if | T | ≤ | Ω | / 2 (and that this bound is the best possible). In principle, we can recover f exactly by solving the combinatorial optimization problem

<!-- formula-not-decoded -->

/negationslash where ‖ g ‖ /lscript 0 is the number of nonzero terms # { t, g ( t ) = 0 } . Solving (1.4) directly is infeasible even for modest-sized signals. The algorithm would let T run over all subsets of { 0 , . . . , N -1 } of cardinality | T | ≤ 1 2 | Ω | and for each T , checking whether f was in the range of F T → Ω or not, and then inverting the relevant minor of the Fourier matrix to recover f once T was determined. It is well-known that this procedure would clearly be very computationally expensive, however, since there are exponentially many subsets to check; for instance, for | Ω | ∼ N/ 2, this number scales like 4 N · 3 -3 N/ 4 ! As an aside comment, note that it is not clear how to make this algorithm robust, especially since the results in [18] do not provide any effective lower bound on the determinant of the minors of the Fourier matrix, see section 6 for a discussion of this point.

A more computationally efficient strategy for recovering f from Ω and ˆ f | Ω is to solve the convex problem

<!-- formula-not-decoded -->

The key result in this paper is that the solutions to ( P 0 ) and ( P 1 ) are equivalent for an overwhelming percentage of the choices for T and Ω with | T | ≤ α · | Ω | / log N ( α &gt; 0 is a constant): in these cases, solving the convex problem ( P 1 ) recovers f exactly .

To establish this upper bound, we will assume that the observed Fourier coefficients are randomly sampled . To make this precise, we introduce a probability parameter 0 &lt; τ &lt; 1, and consider the sequence ( I k ) 1 ≤ k ≤ N of independent Bernoulli random variables

<!-- formula-not-decoded -->

We then define the random set of frequencies Ω as

<!-- formula-not-decoded -->

Clearly, | Ω | follows the binomial distribution and

<!-- formula-not-decoded -->

In fact, classical large deviations arguments (or the central limit theorem) tell us that with high probability, the size of | Ω | is very close to τN . Our main theorem can now be stated as follows.

Theorem 1.3 Let f ∈ C N be a discrete signal and Ω be the random set defined in (1.7) . For a given accuracy parameter M , if f is supported on T and

<!-- formula-not-decoded -->

then with probability at least 1 -O ( N -M ) , the minimizer to the problem (1.5) is unique and is equal to f .

In light of (1.8) we see that (1.9) is essentially | T | ∼ | Ω | , modulo a constant and a logarithmic factor. Indeed, an easy modification to the second part of Theorem 1.1 shows that the condition (1.9) cannot be weakened to (for instance) | supp( f ) | ≤ ( 1 2 + ε ) τN , for any /epsilon1 &gt; 0. The paper gives an explicit value of α ( M ), namely, α ( M ) /equivasymptotic 1 / [29 . 6( M +1)] although we have not pursued the question of exactly what the optimal value might be.

In Section 5, we present numerical results which suggest that in practice, we can expect to recover f more than 50% of the time if | T | ≤ | Ω | / 4. For | T | ≤ | Ω | / 8, the recovery rate is above 90%. Empircally, the constants 1 / 4 and 1 / 8 do not seem to vary for N in the range of a few hundred to a few thousand.

## 1.3 For Almost Every Ω

As the theorem suggests, there exist sets Ω and functions f for which the /lscript 1 -minimization procedure does not recover f correctly, even if | supp( f ) | is much smaller than | Ω | . We sketch two counter-examples:

- Dirac's comb. Suppose that N is a perfect square and consider the picket-fence signal which consists of spikes of unit height and with uniform spacing equal to √ N . This signal is often used as an extremal point for uncertainty principles [6, 7] as one of its remarkable properties is its invariance through the Fourier transform. Hence suppose that Ω is the set of all frequencies but the multiples of √ N , namely, | Ω | = N -√ N . Then ˆ f | Ω = 0 and obviously the reconstruction is identically zero.
- Note that the problem here does not really have anything to do with /lscript 1 -minimization per se; f cannot be reconstructed from its Fourier samples on Ω thereby showing that Theorem 1.1 does not work 'as is' for arbitrary sample sizes.
- Box signals . The example above suggests that in some sense | T | must not be greater than about √ | Ω | . In fact, there exist more extreme examples. Assume the sample size N is large and consider for example the indicator function f of the interval T := { t : -N -0 . 01 &lt; t &lt; N 0 . 01 } and let Ω be the set Ω := { k : N/ 3 &lt; k &lt; 2 N/ 3 } . Let h be a function whose Fourier transform ˆ h is a non-negative bump function adapted to the interval { k : -N/ 6 &lt; k &lt; N/ 6 } which equals 1 when -N/ 12 &lt; k &lt; N/ 12. Then | h ( t ) | 2 has Fourier transform vanishing in Ω, and is rapidly decreasing away from t = 0; in particular we have | h ( t ) | 2 = O ( N -100 ) for t /negationslash∈ T . On the other hand,

one easily computes that | h (0) | 2 &gt; c for some absolute constant c &gt; 0. Because of this, the signal f -ε | h | 2 will have smaller /lscript 1 -norm than f for ε &gt; 0 sufficiently small (and N sufficiently large), while still having the same Fourier coefficients as f on Ω. Thus in this case f is not the minimizer to the problem ( P 1 ), despite the fact that the support of f is much smaller than that of Ω.

The above counterexamples relied heavily on the special choice of Ω (and to a lesser extent of supp( f )); in particular, it needed the fact that the complement of Ω contained a large interval (or more generally, a long arithmetic progression). But for most sets Ω, large arithmetic progressions in the complement do not exist, and the problem largely disappears. In short, Theorem 1.3 essentially says is that for most sets | T | ∼ | Ω | , the inequality holds.

## 1.4 Extensions

As mentioned earlier, results on our model problem extend easily to higher dimensions as well as to other setups. To be concrete consider the problem of recovering a one-dimensional piecewise constant signal via

<!-- formula-not-decoded -->

where we adopt the convention that g ( -1) = g ( N -1). In a nutshell, model (1.5) is obtained from (1.10) after differentiation. Indeed, let δ be the vector of first difference δ ( t ) = g ( t ) -g ( t -1), and note that ∑ δ ( t ) = 0. Obviously,

<!-- formula-not-decoded -->

/negationslash and, therefore, with υ ( ω ) = (1 -e -iω ) -1 , the problem is identical to

<!-- formula-not-decoded -->

which is precisely what we have been studying.

/negationslash

We now explore versions of Theorem 1.3 in higher dimensions. To be concrete, consider the two-dimensional situation (statements in arbitrary dimensions are exactly of the same flavor):

Corollary 1.4 Put T = { t, f ( t ) = f ( t -1) } . Under the assumptions of Theorem 1.3, the minimizer to the problem (1.10) is unique and is equal f with probability at least 1 -O ( N -M ) -provided, of course, that f be adjusted so that ∑ f ( t ) = ˆ f (0) .

Theorem 1.5 Put N = n 2 . We let f ( t 1 , t 2 ) , 1 ≤ t 1 , t 2 ≤ n be a discrete signal and Ω be the random set defined as in (1.7) . Assume that for a given accuracy parameter M , f is supported on T obeying (1.9) . Then with probability at least 1 -O ( N -M ) , the minimizer to the problem (1.5) is unique and is equal to f .

We will not prove this result as the strategy is exactly parallel to that of Theorem 1.3. Just as in the one-dimensional case, a similar statement for piecewise constant functions

exists provided, of course, that the support of f be replaced by { ( t 1 , t 2 ) : | D 1 f ( t 1 , t 2 ) | 2 + | D 2 f ( t 1 , t 2 ) | 2 = 0 } . We omit the details.

/negationslash

We hope that we managed to suggest that there actually are a variety of results similar to Theorem 1.3, and we only selected a few instances. As a matter of fact, those provide a precise quantitative understanding of the 'surprising result' discussed at the beginning of this paper.

## 1.5 Relationship to Uncertainty Principles

From a certain point of view, our results are connected to the so-called uncertainty principles [6, 7] which say that it is difficult to localize a signal f ∈ C N both in time and frequency at the same time. Indeed, classical arguments show that f is the unique minimizer of ( P 1 ) if and only if

/negationslash

<!-- formula-not-decoded -->

Put T = supp( f ) and apply the triangle inequality

<!-- formula-not-decoded -->

Hence, a sufficient condition to establish that f is our unique solution would be to show that

/negationslash

<!-- formula-not-decoded -->

or equivalently ∑ T | h ( t ) | &lt; 1 2 ‖ h ‖ /lscript 1 . The connection with the uncertainty principle is now explicit; f is the unique minimizer if it is impossible to 'concentrate' half of the /lscript 1 norm of a signal that is missing frequency components in Ω on a 'small' set T . For example, [6] guarantees exact reconstruction if

<!-- formula-not-decoded -->

Take | Ω | &lt; N/ 2, then that condition says that | T | must be zero which, of course, is far from being the content of Theorem 1.3. In truth, this paper does not follow this classical approach. Instead, we will use duality theory to study the solution of ( P 1 ).

## 1.6 Robust Uncertainty Principles

Underlying our analysis is a new notion of uncertainty principle which holds for almost any pair (supp( f ) , supp( ˆ f )). With T = supp( f ) and Ω = supp( ˆ f ), the classical discrete uncertainty principle [6] says that

<!-- formula-not-decoded -->

with equality obtained for signals such as the Dirac's comb. As we mentioned above, such extremal signals correspond to very special pairs ( T, Ω). However, for most choices of T

and Ω, the analysis presented in this paper shows that it is impossible to find f such that T = supp( f ) and Ω = supp( ˆ f ) unless

<!-- formula-not-decoded -->

which is considerably stronger than (1.11). Here, the statement 'most pairs' says again that the probability of selecting a random pair ( T, Ω) violating (1.12) is at most O ( N -M ). (We are of course aware of numerical studies in [6] pointing out the lack of sharpness of the uncertainty principle when T is random.)

In some sense, (1.12) is the typical uncertainty relation one can generally expect (as opposed to (1.11)), hence, justifying the title of this paper. Because of space limitation, we are unable to belaborate on this fact and its implications any further, but will do so in a companion paper.

## 1.7 Connections with existing work

The idea of relaxing a combinatorial problem into a convex problem is not new and goes back a long way. For example, [5, 16] used the idea of minimizing /lscript 1 norms to recover spike trains. The motivation is that this makes available a host of computationally feasible procedures. For example, a convex problem of the type (1.5) can be practically solved using techniques of linear programming such as interior point methods [3].

Now, there exists some evidence that in special situations the unique solution to an /lscript 1 minimization problem coincides with that of the unique minimizer of the /lscript 0 problem. For example, a series of beautiful papers [7, 8, 9, 12, 14] is concerned with a special setup where one is given a dictionary D of vectors (waveforms) of C N , D = ( d k ) 1 ≤ k ≤ M and one seeks sparse representations of a signal f ∈ C N as a superposition of elements of D

<!-- formula-not-decoded -->

Suppose that the number of elements M from D is greater than the sample size N , then there are many ways in which one can represent f as a superposition of elements from D and one would want to find the 'sparsest' one. Consider the solution which minimizes the /lscript 0 norm of α subject to the constraint (1.13) and that which minimizes the /lscript 1 norm. A typical result of this body of work is as follows: suppose that s can be synthesized out of very few elements from D , then the solution to both problems are unique and are equal. We also refer to [19, 20] for very recent results along these lines.

This literature certainly influenced our thinking in the sense it made us suspect that results such as Theorem 1.3 were actually possible. However, we would like to emphasize that the claims presented in this paper are of a substantially different nature. We give essentially two reasons:

- First, our model problem is different since we need to 'guess' a signal from incomplete data, as opposed to finding the sparsest expansion of a fully specified signal.
- And second, our approach is decidedly probabilistic-as opposed to deterministicand thus calls for very different techniques. For example, underlying our analysis are delicate estimates about the size of random matrices, which may be of independent interest.

Besides the wonderful properties of /lscript 1 , there is a second line of research connected to our findings. We can think of recovering a sparse superposition of spikes from an incomplete set of observations in the Fourier domain as a spectral estimation problem proviso swapping time and frequency: ˆ f is a superposition of a few complex sinusoids whose frequency and amplitude we need to determine from a few samples. From this point of view, our work is related to [10, 11] and [21] where the authors study sampling patterns allowing the exact reconstruction of a signal. These references show that the locations and amplitudes of a sequence of | T | spikes can be recovered exactly from 2 | T | +1 consecutive Fourier coefficients (in [21] for example, the recovery requires solving a system of equations and factoring a polynomial). Our results, namely, Theorems 1.1 and 1.3 are quite distinct and far more general since they address the radically different situation in which we do not have the freedom to choose the samples at our convenience.

Finally, it is interesting to note that our results and the references above are also related to recent work [22] in finding near-best B -term Fourier approximations (which is in some sense the dual to our recovery problem). The algorithm in [22, 23], which operates by estimating the frequencies present in the signal from a small number of randomly placed samples, produces with high probability an approximation in sublinear time with error within a constant of the best B -term approximation. First, in [23] the samples are again selected to be equispaced whereas we are not at liberty to choose the frequency samples at all since they are specified a priori . And second, we wish to produce as a result an entire signal or image of size N , so a sublinear algorithm is an impossibility.

## 2 Strategy

It is clear that at least one minimizer to ( P 1 ) exists. On the other hand, it is not apparent why this minimizer should be unique, and why it should equal f . In this section, we outline our strategy for answering these questions. Using duality theory, we will be able to derive necessary and sufficient conditions for ( P 1 ) to recover f . We note that a similar duality approach was independently developed in [13] for finding sparse approximations from general dictionaries.

## 2.1 Duality

To get a feel for the line of argumentation, consider first the case where f is real-valued. Then (1.5) can be written as the linear program

<!-- formula-not-decoded -->

where g + ( t ) = max( g ( t ) , 0), g -( t ) = -min( g ( t ) , 0), and the matrix F Ω contains only the rows of the Fourier transform matrix corresponding to entries in Ω. The corresponding Lagrangian is

<!-- formula-not-decoded -->

(2.2)

with µ + , µ -≥ 0. At a minimum (˜ g + , ˜ g -), there will be a saddle point in L , and we will have

<!-- formula-not-decoded -->

Then for f to be the minimum of (2.1), we need

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with µ + , µ -≥ 0. In fact, for f to be the unique minimizer of (2.1), it is necessary and sufficient for there to exist a λ such that for P ( t ) = ( F ∗ Ω λ )( t ), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, to show that f /sharp is unique and is equal to f , it suffices to find a trigonometric polynomial P whose Fourier transform is supported in Ω-in other words, which only uses frequencies in Ω-and which matches sgn( f ) on supp( f ), and has magnitude strictly less than 1 elsewhere. The following lemma generalizes for the case where f is complex-valued.

Lemma 2.1 Let Ω ⊂ Z N . For a vector f ∈ C N , define the 'sign' vector sgn ( f ) by sgn ( f )( t ) := f ( t ) / | f ( t ) | when t ∈ supp( f ) and sgn ( f ) = 0 otherwise. Suppose there exists a vector P whose Fourier transform ˆ P is supported in Ω such that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

- Then if F supp ( f ) → Ω is injective, the minimizer f /sharp to the problem ( P 1 ) (1.5) is unique and is equal to f .
- Conversely, if f is the unique minimizer of ( P 1 ) , then there exists a vector P with the above properties.

Proof We may assume that Ω is non-empty and that f is non-zero since the claims are trivial otherwise.

Suppose first that such a function P exists. Let g be any vector not equal to f with ˆ g | Ω = ˆ f | Ω . Write h := g -f , then ˆ h vanishes on Ω. Observe that for any t ∈ supp( f ) we have

<!-- formula-not-decoded -->

while for t /negationslash∈ supp( f ) we have | g ( t ) | = | h ( t ) | ≥ Re( h ( t ) P ( t )) since | P ( t ) | &lt; 1. Thus

<!-- formula-not-decoded -->

However, the Parseval's formula gives

<!-- formula-not-decoded -->

since ˆ P is supported on Ω and ˆ h vanishes on Ω. Thus ‖ g ‖ /lscript 1 ≥ ‖ f ‖ /lscript 1 . Now we check when equality can hold, i.e. when ‖ g ‖ /lscript 1 = ‖ f ‖ /lscript 1 . An inspection of the above argument shows that this forces | h ( t ) | = Re( h ( t ) P ( t )) for all t /negationslash∈ supp( f ). Since | P ( t ) | &lt; 1, this forces h to vanish outside of supp( f ). Since ˆ h vanishes on Ω, we thus see that h must vanish identically (this follows from the assumption about the injectivity of F supp ( f ) → Ω ) and so g = f . This shows that f is the unique minimizer f /sharp to the problem (1.5).

Conversely, suppose that f = f /sharp is the unique minimizer to (1.5). Without loss of generality we may normalize ‖ f ‖ /lscript 1 = 1. Then the closed unit ball B := { g : ‖ g ‖ /lscript 1 ≤ 1 } and the affine space V := { g : ˆ g | Ω = ˆ f | Ω } intersect at exactly one point, namely f . By the Hahn-Banach theorem we can thus find a function P such that the hyperplane Γ 1 := { g : ∑ Re( g ( t ) P ( t )) = 1 } contains V , and such that the half-space Γ ≤ 1 := { g : ∑ Re( g ( t ) P ( t )) ≤ 1 } contains B . By perturbing the hyperplane if necessary (and using the uniqueness of the intersection of B with V ) we may assume that Γ 1 ∩ B is contained in the minimal facet of B which contains f , namely { g ∈ B : supp( g ) ⊆ supp( f ) } .

Since B lies in Γ ≤ 1 , we see that sup t | P ( t ) | ≤ 1; since f ∈ Γ 1 ∩ B , we have P ( t ) = sgn( f )( t ) when t ∈ supp( f ). Since Γ 1 ∩ B is contained in the minimal facet of B containing f , we see that | P ( t ) | &lt; 1 when t /negationslash∈ supp( f ). Since Γ 1 contains V , we see from Parseval that ˆ P is supported in Ω. The claim follows.

Since the space of functions with Fourier transform supported in Ω has | Ω | degrees of freedom, and the condition that P match sgn( f ) on supp( f ) requires | supp( f ) | degrees of freedom, one now expects heuristically (if one ignores the open conditions that P has magnitude strictly less than 1 outside of supp( f )) that f /sharp should be unique and be equal to f whenever | supp( f ) | /lessmuch | Ω | ; in particular this gives an explicit procedure for recovering f from Ω and ˆ f | Ω .

## 2.2 Architecture of the Argument

Equipped with our duality theorem, we are now in a position to present the main ideas of the argument. Fix f . We may assume that τN &gt; M log N since the claim is vacuous otherwise (as we will see, α ( M ) = O (1 /M ) and thus (1.9) will force f ≡ 0, at which point it is clear that the solution to ( P 1 ) is equal to f = 0).

We let T ⊂ Z N denote the support of f , T := supp( f ). Let Ω be the random set defined by (1.7). Since τN &gt; M log N , a typical application of the large deviation theorem shows that the cardinality of Ω is if course close to that of its expected value, e.g.

<!-- formula-not-decoded -->

Slightly more precise estimates are possible, see [1]. It then follows that

<!-- formula-not-decoded -->

In the sequel it will be convenient to denote by B M the event {| Ω | &lt; (1 -/epsilon1 M ) | τN |} .

In light of Lemma 2.1, it suffices -with probability 1 -O ( N -M )- to (1) show that the matrix F supp ( f ) → Ω has full rank, and (2) construct a trigonometric polynomial P ( t ), 0 ≤ t ≤ N -1, whose Fourier transform is supported on Ω, matches sgn( f ) on T , and has magnitude strictly less than 1 outside of T . To do this we shall need some auxiliary linear transformations (i.e. matrices) as we will see next.

In this section, we will work with vectors restricted to the set T and it will be convenient to let /lscript 2 ( T ) denote the subspace of such restrictions (and similarly /lscript 2 ( Z N ) := C N ). With these notations, we let H : /lscript 2 ( T ) → /lscript 2 ( Z N ) denote the linear transform defined by

/negationslash

<!-- formula-not-decoded -->

Let ι : /lscript 2 ( T ) → /lscript 2 ( Z N ) be the obvious embedding of /lscript 2 ( T ) into /lscript 2 ( Z N ) (extending by zero outside of T ), and let ι ∗ : /lscript 2 ( Z N ) → /lscript 2 ( T ) be the dual restriction map, thus ι ∗ f := f | T . Observe that ι ∗ ι : /lscript 2 ( T ) → /lscript 2 ( T ) is simply the identity operator on /lscript 2 ( T ), and that the operator ι ∗ H : /lscript 2 ( T ) → /lscript 2 ( T ) is self-adjoint.

The key point is that the terms in (2.10) are rather oscillatory, since we have stripped out the non-oscillatory diagonal t = t ′ ; indeed, the main idea of the argument will be to use the randomization of Ω to treat H as a 'white noise' operator whose eventual effect will be negligible, especially if H is raised to a high power.

To see the relevance of the operator H to our problem, observe that for all f ∈ /lscript 2 ( T )

<!-- formula-not-decoded -->

with ˆ f ( ω ) the Fourier coefficient of f evaluated at the frequency ω . In particular, ( ι -1 | Ω | H ) f has Fourier transform supported in Ω. Next, suppose for the moment that the self-adjoint

operator ι ∗ ι -1 | Ω | ι ∗ H from /lscript 2 ( T ) to itself is invertible, and then set P ( t ), 0 ≤ t ≤ N -1, to be the trigonometric polynomial

<!-- formula-not-decoded -->

Then by the preceding discussion:

- Frequency support. P has Fourier transform supported in Ω;
- Spatial interpolation. P obeys

<!-- formula-not-decoded -->

and so P agrees with sgn( f ) on T .

Consider now the invertibility issue. By definition

<!-- formula-not-decoded -->

Hence, the invertibility of ι ∗ ι -1 | Ω | ι ∗ H implies that F T → Ω be injective. In summary, to prove the theorem it will suffice to show that:

- Invertibility. The operator ι ∗ ι -1 | Ω | ι ∗ H is invertible (with probability 1 -O ( N -M )).
- Magnitude on T c . The function P defined in (2.11) obeys the bound sup t ∈ T c | P ( t ) | &lt; 1 (with probability 1 -O ( N -M )).

We first consider the former claim.

## 3 Construction of the Dual Polynomial

## 3.1 Invertibility

We would like to establish invertibility of the matrix ι ∗ ι -1 | Ω | ι ∗ H with high probability. One obvious way to proceed would be to show that the operator norm or equivalently the largest eigenvalue of ι ∗ H is less than | Ω | . This is easily done if | supp( f ) | is extremely small (e.g. much less than √ | Ω | ), simply by estimating the operator norm directly by the Frobenius norm ‖ · ‖ F , which is easy to compute explicitly. Recall that for any squared matrix M , the Frobenius norm ‖ M ‖ F of M is defined by the formula

<!-- formula-not-decoded -->

and obeys ‖ M ‖ ≤ ‖ M ‖ F . However, this simple approach does not work well when | supp( f ) | is large, say equal to α · (log N ) -1 · | Ω | . In this case, we have to resort to estimating the Frobenius norm of a large power of ι ∗ H , taking advantage of cancellations arising from the randomness of the matrix coefficients of ι ∗ H .

We state the key estimate of this section.

Theorem 3.1 Put H 0 = ι ∗ H for short, where H is the operator defined by (2.10) . Set c τ := e log((1 -τ ) /τ ) and let

<!-- formula-not-decoded -->

Then

<!-- formula-not-decoded -->

In most interesting situations a n is less than b n which allows slightly to reformulate (3.1). Note that the classical Stirling approximation to n ! gives

<!-- formula-not-decoded -->

and, therefore, letting φ be the 'golden ratio' φ := (1 + √ 5) / 2, the 2 n th moment obeys

<!-- formula-not-decoded -->

provided that a n obeys

<!-- formula-not-decoded -->

Theorem 3.1 gives a precise estimate about the operator norm of H 0 . To see why this is true, assume that (3.3) holds; since H 0 is self-adjoint

<!-- formula-not-decoded -->

and, therefore,

<!-- formula-not-decoded -->

Now selecting n = /ceilingleft log | T |/ceilingright so that

<!-- formula-not-decoded -->

gives

<!-- formula-not-decoded -->

Formalizing matters, we proved

Corollary 3.2 Suppose | T | ≤ (log | τN | ) -1 | τN | . Then for any /epsilon1 &gt; 0 , we have

<!-- formula-not-decoded -->

Proof The Markov inequality above bounds the probability by (1 + /epsilon1 ) -2 n which goes to zero as n = /ceilingleft log | T |/ceilingright goes to infinity.

We now return to the study of the invertibility of ι ∗ ι -1 | Ω | H 0 . Letting α be a positive number 0 &lt; α &lt; 1, it follows from the Markov inequality that

<!-- formula-not-decoded -->

We then apply inequality (3.1) (recall ‖ H n 0 ‖ 2 F = Tr( H 2 n 0 )) and obtain

<!-- formula-not-decoded -->

We remark that the last inequality holds for any sample size | T | (proviso the condition (3.3)) and we now specialize (3.4) to selected values of | T | .

Suppose that | T | obeys

<!-- formula-not-decoded -->

Then

<!-- formula-not-decoded -->

We then have the following result.

Theorem 3.3 Assume that τ ≤ . 44 , say, and suppose that T obeys (3.5) . Then (3.3) holds for any n ≥ 4 , and therefore

<!-- formula-not-decoded -->

The only thing to establish is that T obeys (3.3). This is merely technical and the proof is in the Appendix.

With the notations of the previous section and especially (2.9), observe now that

<!-- formula-not-decoded -->

where we recall that B M := {| Ω | &lt; (1 -/epsilon1 M ) | τN |} has probability less than N -M . Suppose T obeys (3.5) with α M := α (1 -/epsilon1 M ) instead of α ,

<!-- formula-not-decoded -->

Corollary 3.4 Take n = ( M +1)log N . We see from the Neumann series that the operator ι ∗ ι -1 | Ω | ι ∗ H is invertible with probability at least 1 -(1+2 /γ 2 ) N -M since ι ∗ ι is the identity on vectors supported on T .

We have thus established the invertibility of ι ∗ ι -1 | Ω | ι ∗ H with high probability, and thus P is well defined with high probability. It remains to show that sup t/ ∈ T | P ( t ) | &lt; 1 with high probability.

## 3.2 Magnitude of the polynomial on the complement of T

We first develop an expression for P ( t ) by making use of the algebraic identity

<!-- formula-not-decoded -->

Indeed, we can write so that the inverse is given by the truncated Neumann series

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The point is that the remainder term R is quite small in the Frobenius norm: suppose that ‖ ι ∗ H ‖ F ≤ α · | Ω | , then

In particular, the matrix coefficients of R are all individually less than α n / (1 -α n ). Introduce the /lscript ∞ -norm of a matrix as ‖ M ‖ ∞ = sup ‖ x ‖ ∞ ≤ 1 ‖ Mx ‖ ∞ which is also given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, it follows from the Cauchy-Schwarz inequality that

<!-- formula-not-decoded -->

where # M (col) is of course the number of columns of M . This observation gives the crude estimate

<!-- formula-not-decoded -->

As we shall soon see, the bound (3.8) allows us to effectively neglect the R term in this formula; the only remaining difficulty will be to establish good bounds on the truncated Neumann series 1 | Ω | H ∑ n -1 m =0 1 | Ω | m ( ι ∗ H ) m .

## 3.3 Estimating the truncated Neumann series

From (2.11) we observe that on the complement of T

<!-- formula-not-decoded -->

since the ι component in (2.11) vanishes outside of T . Applying (3.7), we may rewrite P as

<!-- formula-not-decoded -->

where and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let a 0 , a 1 &gt; 0 be two numbers with a 0 + a 1 = 1. Then

<!-- formula-not-decoded -->

and the idea is to bound each term individually. Put Q 0 = S n -1 sgn( f ) so that P 1 = 1 | Ω | HRι ∗ (sgn( f ) + Q 0 ). With these notations, observe that

<!-- formula-not-decoded -->

Hence, bounds on the magnitude of P 1 will follow from bounds on ‖ HR ‖ ∞ together with bounds on the magnitude of ι ∗ Q 0 . It will be of course sufficient to derive bounds on ‖ Q 0 ‖ ∞ (since ‖ ι ∗ Q 0 ‖ ∞ ≤ ‖ Q 0 ‖ ∞ ) which will follow from those on P 0 since Q 0 is nearly equal to P 0 (they differ by only one very small term term).

Fix t ∈ T c and write P 0 ( t ) as

<!-- formula-not-decoded -->

The idea is to use moment estimates to control the size of each term X m ( t ).

Lemma 3.5 Set n = km . Then E | X m ( t 0 ) | 2 k obeys the same estimate as that in Theorem 3.1 (up to a multiplicative factor | T | -1 ), namely,

<!-- formula-not-decoded -->

In particular, following (3.2)

<!-- formula-not-decoded -->

where γ is as before.

The proof of these moment estimates mimics that of Theorem 3.1 and may be found in the Appendix.

Lemma 3.6 Fix a 0 = . 91 . Suppose that | T | obeys (3.5) and let B M be the set where | Ω | &lt; (1 -/epsilon1 M ) · | τN | with /epsilon1 M as in (2.9) . For each t ∈ Z N , there is a set A t with the property

<!-- formula-not-decoded -->

and

As a consequence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and similarly for Q 0 .

Proof We suppose that n is of the form n = 2 J -1 (this property is not crucial and only simply simplifies our exposition). For each m and k such that km ≥ n , it follows from (3.5) and (3.10) together with some simple calculations that

<!-- formula-not-decoded -->

Again | Ω | ≈ | τN | and we will develop a bound on the set B c M where | Ω | ≥ (1 -/epsilon1 M ) | τN | . On this set

<!-- formula-not-decoded -->

Fix β j &gt; 0, 0 ≤ j &lt; J , such that ∑ J -1 j =0 2 j β j ≤ a 0 . Obviously,

<!-- formula-not-decoded -->

where K j = 2 J -j . Observe that for each m with 2 j ≤ m&lt; 2 j +1 , K j m obeys n ≤ K j m&lt; 2 n and, therefore, (3.11) gives

<!-- formula-not-decoded -->

For example, taking β -K j j to be constant for all j , i.e. equal to β -n 0 , gives

<!-- formula-not-decoded -->

with ∑ J -1 j =0 2 j β j ≤ a 0 . Numerical calculations show that for β 0 = . 42, ∑ j 2 j β j ≤ . 91 which gives

<!-- formula-not-decoded -->

The claim for Q 0 is, of course, identical and the lemma follows.

Lemma 3.7 Fix a 1 = . 09 . Suppose that the pair ( α, N ) obeys | τN | 3 / 2 α n 1 -α n ≤ a 1 / 2 . Then

<!-- formula-not-decoded -->

on the event A ∩ {‖ ι ∗ H ‖ F ≤ α | Ω |} , for some A obeying P ( A ) ≥ 1 -O ( N -M ) .

Proof As we observed before, (1) ‖ P 1 ‖ ∞ ≤ ‖ H ‖ ∞ ‖ R ‖ ∞ (1 + ‖ Q 0 ‖ ∞ ), and (2) Q 0 obeys the bound stated in Lemma 3.6. Consider then the event {‖ Q 0 ‖ ∞ ≤ 1 } . On this event, ‖ P 1 ‖ ≤ a 1 if 1 | Ω | ‖ H ‖‖ R ‖ ∞ ≤ a 1 / 2. The matrix H obeys 1 | Ω | ‖ H ‖ ∞ ≤ | T | since H has | T | columns and each matrix element is bounded by | Ω | (note that far better bounds are possible). It then follows from (3.8) that

<!-- formula-not-decoded -->

with probability at least 1 -O ( N -M ). We then simply need to choose α and n such that the right hand-side is less than a 1 / 2.

## 3.4 Proof of Theorem 1.3

It is now clear that we have assembled all the intermediate results to prove our theorem. Indeed, we proved the invertibility of i ∗ i -1 | Ω | ι ∗ H with probability O ( N -M ) and | P ( t ) | &lt; 1 for all t ∈ T c (again with high probability), provided that α and n be selected appropriately as we now explain.

Fix M &gt; 0. We choose α = . 42 and n to be the nearest integer to ( M +1)log N .

1. From the discussion following Theorem 3.3, it follows that i ∗ i -| Ω | -1 ι ∗ H is invertible with probability O ( N -M ).
2. With this special choice, /epsilon1 n = 2[( M +1)log N ] 2 · N -( M +1) and, therefore, Lemma 3.6 implies that both P 0 and Q 0 are bounded by .91 outside of T c with probability at least 1 -[1 + 2(( M +1)log N ) 2 ] · N -M .
3. And finally, to prove that | P 1 ( t ) | &lt; . 09 outside T c , Lemma 3.6 assures that it is sufficient to have N 3 / 2 α n / (1 -α n ) ≤ . 045. Because log( . 42) ≈ -. 87 and log( . 045) ≈ -3 . 10, this condition is approximately equivalent to

<!-- formula-not-decoded -->

Take M ≥ 2, for example; then the above inequality is satisfied as soon as N ≥ 17.

To conclude, we proved that if T obeys

<!-- formula-not-decoded -->

then the reconstruction with probability exceeding 1 -O ([( M +1)log N ) 2 ] · N -M ). In other words, we may take α ( M ) in Theorem 1.3 to be of the form

<!-- formula-not-decoded -->

## 4 Moments of Random Matrices

## 4.1 A First Formula for the Expected Value of the Trace of ( H 0 ) 2 n

Recall that H 0 ( t, t ′ ), t, t ′ ∈ T , is the | T | × | T | matrix whose entries are defined by

/negationslash

<!-- formula-not-decoded -->

A diagonal element of the 2 n th power of H 0 may be expressed as

/negationslash

<!-- formula-not-decoded -->

where we adopt the convention that t 2 n +1 = t 1 whenever convenient and, therefore,

/negationslash

Using (1.7) and linearity of expectation, we can write this as

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

The idea is to use the independence of the I { ω j ∈ Ω } 's to simplify this expression substantially; however, one has to be careful with the fact that some of the ω j 's may be the same, at which point one loses independence of those indicator variables. These difficulties require a certain amount of notation. We let Z N = { 0 , 1 , . . . , N -1 } be the set of all frequencies as before, and let A be the finite set A := { 1 , . . . , 2 n } . For all ω := ( ω 1 , . . . , ω 2 n ), we define the equivalence relation ∼ ω on A by saying that j ∼ ω j ′ if and only if ω j = ω j ′ . We let P ( A ) be the set of all equivalence relations on A . Note that there is a partial ordering on the equivalence relations as one can say that ∼ 1 ≤∼ 2 if ∼ 1 is coarser than ∼ 2 , i.e. a ∼ 2 b implies a ∼ 1 b for all a, b ∈ A . Thus, the coarsest element in P ( A ) is the trivial equivalence relation in which all elements of A are equivalent (just one equivalence class), while the finest element is the equality relation =, i.e. each element of A belongs to a distinct class ( | A | equivalence classes).

For each equivalence relation ∼ in P , we can then define the sets Ω( ∼ ) ⊂ Z 2 n N by

<!-- formula-not-decoded -->

and the sets Ω ≤ ( ∼ ) ⊂ Z 2 n N by

<!-- formula-not-decoded -->

Thus the sets { Ω( ∼ ) : ∼∈ P} form a partition of Z 2 n N . The sets Ω ≤ ( ∼ ) can also be defined as

<!-- formula-not-decoded -->

For comparison, the sets Ω( ∼ ) can be defined as

<!-- formula-not-decoded -->

/negationslash

/negationslash

We give an example: suppose n = 2 and fix ∼ such that 1 ∼ 4 and 2 ∼ 3 (exactly 2 equivalence classes); then Ω( ∼ ) := { ω ∈ Z 4 N : ω 1 = ω 4 , ω 2 = ω 3 , and ω 1 = ω 2 } while Ω ≤ ( ∼ ) := { ω ∈ Z 4 N : ω 1 = ω 4 , ω 2 = ω 3 } .

Now, let us return to the computation of the expected value. Because the random variables I k (1.6) are independent and have all the same distribution, the quantity E [ ∏ 2 n j =1 I ω j ] depends only on the equivalence relation ∼ ω and not on the value of ω itself. Indeed, we have

<!-- formula-not-decoded -->

where A/ ∼ denotes the equivalence classes of ∼ . Thus we can rewrite the preceding expression as

/negationslash

<!-- formula-not-decoded -->

where ∼ ranges over all equivalence relations.

We would like to pause here and consider (4.2). Take n = 1, for example. There are only two equivalent classes on { 1 , 2 } and, therefore, the right hand-side is equal to

/negationslash

/negationslash

/negationslash

Our goal is to rewrite the expression inside the brackets so that the exclusion ω 1 = ω 2 does not appear any longer, i.e. we would like to rewrite the sum over ω ∈ Z 2 N : ω 1 = ω 2 in terms of sums over ω ∈ Z 2 N : ω 1 = ω 2 , and over ω ∈ Z 2 N . In this special case, this is quite easy as

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

/negationslash

The motivation is quite clear. Removing the exclusion allows to rewrite sums as product, e.g.

<!-- formula-not-decoded -->

and each factor is equal to either N or 0 depending on whether t 1 = t 2 or not.

The next section generalizes these ideas and develop an identity, which allows us to rewrite sums over Ω( ∼ ) in terms of sums over Ω ≤ ( ∼ ).

## 4.2 Inclusion-Exclusion formulae

Lemma 4.1 (Inclusion-Exclusion principle for equivalence classes) Let A and G be non-empty finite sets. For any equivalence class ∼∈ P ( A ) on ω ∈ G | A | , we have

<!-- formula-not-decoded -->

Thus, for instance, if A = { 1 , 2 , 3 } and ∼ is the equality relation, i.e. j ∼ k if and only if j = k , this identity is saying that

<!-- formula-not-decoded -->

where we have omitted the summands f ( ω 1 , ω 2 , ω 3 ) for brevity.

Proof By passing from A to the quotient space A/ ∼ if necessary we may assume that ∼ is the equality relation =. Now relabeling A as { 1 , . . . , n } , ∼ 1 as ∼ , and A ′ as A , it suffices to show that

<!-- formula-not-decoded -->

We prove this by induction on n . When n = 1 both sides are equal to ∑ ω ∈ G f ( ω ). Now suppose inductively that n &gt; 1 and the claim has already been proven for n -1. We observe that the left-hand side of (4.4) can be rewritten as

<!-- formula-not-decoded -->

where ω ′ := ( ω 1 , . . . , ω n -1 ). Applying the inductive hypothesis, this can be written as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now we work on the right-hand side of (4.4). If ∼ is an equivalence class on { 1 , . . . , n } , let ∼ ′ be the restriction of ∼ to { 1 , . . . , n -1 } . Observe that ∼ can be formed from ∼ ′ either by adjoining the singleton set { n } as a new equivalence class (in which case we write ∼ = {∼ ′ , { n }} , or by choosing a j ∈ { 1 , . . . , n -1 } and declaring n to be equivalent to j (in which case we write ∼ = {∼ ′ , { n }} / ( j = n )). Note that the latter construction can recover the same equivalence class ∼ in multiple ways if the equivalence class [ j ] ∼ ′ of j in ∼ ′ has

size larger than 1, however we can resolve this by weighting each j by 1 | [ j ] ∼′ | . Thus we have the identity

<!-- formula-not-decoded -->

for any complex-valued function F on P ( { 1 , . . . , n } ). Applying this to the right-hand side of (4.4), we see that we may rewrite this expression as the sum of

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and where we adopt the convention ω ′ = ( ω 1 , . . . , ω n -1 ). But observe that

<!-- formula-not-decoded -->

and thus the right-hand side of (4.4) matches (4.5) as desired.

## 4.3 Stirling Numbers

As emphasized earlier, our goal is to use our inclusion-exclusion formula to rewrite the sum (4.2) as a sum over Ω ≤ ( ∼ ). In order to do this, it is best to introduce another element of combinatorics, which will prove to be very useful.

For any n, k ≥ 0, we define the Stirling number of the second kind S ( n, k ) to be the number of equivalence relations on a set of n elements which have exactly k equivalence classes, thus

<!-- formula-not-decoded -->

Thus for instance S (0 , 0) = S (1 , 1) = S (2 , 1) = S (2 , 2) = 1, S (3 , 2) = 3, and so forth. We observe the basic recurrence

<!-- formula-not-decoded -->

This simply reflects the fact that if a is an element of A and ∼ is an equivalence relation on A with k equivalence classes, then either a is not equivalent to any other element of A (in which case ∼ has k -1 equivalence classes on A \{ a } ), or a is equivalent to one of the k equivalence classes of S \{ a } .

We now need an identity for the Stirling numbers 1 .

1 We found this identity by modifying a standard generating function identity for the Stirling numbers which involved the polylogarithm. It can also be obtained from the formula S ( n, k ) = 1 k ! ∑ k -1 i =0 ( -1) i ( k i ) ( k -i ) n , which can be verified inductively from (4.6).

Lemma 4.2 For any n ≥ 1 and 0 ≤ τ &lt; 1 / 2 , we have the identity

<!-- formula-not-decoded -->

Note that the condition 0 ≤ τ &lt; 1 / 2 ensures that the right-hand side is convergent.

Proof We prove this by induction on n . When n = 1 the left-hand side is equal to τ , and the right-hand side is equal to

<!-- formula-not-decoded -->

as desired. Now suppose inductively that n ≥ 1 and the claim has already been proven for n . Applying the operator ( τ 2 -τ ) d dτ to both sides (which can be justified by the hypothesis 0 ≤ τ &lt; 1 / 2) we obtain (after some computation)

<!-- formula-not-decoded -->

and the claim follows from (4.6).

We shall refer to the quantity in (4.7) as F n ( τ ), thus

<!-- formula-not-decoded -->

Thus we have

<!-- formula-not-decoded -->

and so forth. When τ is small we have the approximation F n ( τ ) ≈ ( -1) n +1 τ , which is worth keeping in mind. Some more rigorous bounds in this spirit are as follows.

Lemma 4.3 Let n ≥ 1 and 0 ≤ τ &lt; 1 / 2 . If τ 1 -τ ≤ e 1 -n , then we have | F n ( τ ) | ≤ τ 1 -τ . If instead τ 1 -τ &gt; e 1 -n , then

<!-- formula-not-decoded -->

Proof Elementary calculus shows that for x &gt; 0, the function g ( x ) = τ x x n -1 (1 -τ ) x is increasing for x &lt; x ∗ and decreasing for x &gt; x ∗ , where x ∗ := ( n -1) / log 1 -τ τ . If τ 1 -τ ≤ e 1 -n , then x ∗ ≤ 1, and so the alternating series F n ( τ ) = ∑ ∞ k =1 ( -1) n + k g ( k ) has magnitude at most g (1) = τ 1 -τ . Otherwise the series has magnitude at most

<!-- formula-not-decoded -->

and the claim follows.

Roughly speaking, this means that F n ( τ ) behaves like τ for n = O (log[1 /τ ]) and behaves like ( n/ log[1 /τ ]) n for n /greatermuch log[1 /τ ].

## 4.4 A Second Formula for the Expected Value of the Trace of H 2 n 0

Let us return to (4.2). The inner sum of (4.2) can be rewritten as

<!-- formula-not-decoded -->

with f ( ω ) := e i ∑ 1 ≤ j ≤ 2 n ω j ( t j -t j +1 ) . We prove the following useful identity:

## Lemma 4.4

<!-- formula-not-decoded -->

Proof Applying (4.3) and rearranging, we may rewrite this as

<!-- formula-not-decoded -->

where

Splitting A into equivalence classes A ′ of A/ ∼ 1 , we observe that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

splitting ∼ ′ based on the number of equivalence classes | A ′ / ∼ ′ | , we can write this as

<!-- formula-not-decoded -->

by (4.8). Gathering all this together, we have proven the identity (4.9).

We specialize (4.9) to the function f ( ω ) := exp( i ∑ 1 ≤ j ≤ 2 n ω j ( t j -t j +1 )) and obtain

/negationslash

<!-- formula-not-decoded -->

We now compute

<!-- formula-not-decoded -->

For every equivalence class A ′ ∈ A/ ∼ , let t A ′ denote the expression t A ′ := ∑ a ∈ A ′ ( t a -t a +1 ), and let ω A ′ denote the expression ω A ′ := ω a for any a ∈ A ′ (these are all equal since ω ∈ Ω ≤ ( ∼ )). Then

<!-- formula-not-decoded -->

We now see the importance of (4.10) as the inner sum equals | Z N | = N when t A ′ = 0 and vanishes otherwise. Hence, we proved the following:

Lemma 4.5 For every equivalence class A ′ ∈ A/ ∼ , let t A ′ := ∑ a ∈ A ′ ( t a -t a +1 ) . Then

/negationslash

<!-- formula-not-decoded -->

/negationslash

This formula will serve as a basis for all of our estimates. In particular, because of the constraint t j = t j +1 , we see that the summand vanishes if A/ ∼ contains any singleton equivalence classes. This means, in passing, that the only equivalence classes which contribute to the sum obey | A/ ∼ | ≤ n .

## 4.5 A First Bound on E [Tr( H 2 n 0 )]

Let ∼ be an equivalence which does not contain any singleton. Then the following inequality holds

<!-- formula-not-decoded -->

To see why this is true, observe that as linear combinations of t 1 , . . . , t 2 n , the expressions t j -t j +1 are all linearly independent of each other except for the constraint ∑ 2 n j =1 t j -t j +1 = 0. Thus we have | A/ ∼ | -1 independent constraints in the above sum, and so the number of t 's obeying the constraints is bounded by | T | 2 n -| A/ ∼| +1 .

/negationslash

All the equivalence classes in the sum (4.11) are without singletons as otherwise t A ′ = 0. Thus, for n, k ≥ 0, we let P ( n, k ) be the number of equivalence classes on a set of n elements which have exactly k equivalence classes and no singletons

<!-- formula-not-decoded -->

There is a simple recursion on these numbers, namely,

<!-- formula-not-decoded -->

which is valid for all n, k ≥ 0. This simply reflects the fact that if α is an element of A and ∼ is an equivalence relation on A with k equivalence classes, then either (1) α belongs to a class which has only one other element β of A (in which case ∼ has k -1 equivalence classes and no singleton on A \{ α, β } ), or α is equivalent to one of the k equivalence classes of A \{ α } , each of which having at least two elements.

With these notations, we established

<!-- formula-not-decoded -->

The following lemma provides an upper bound on those P ( n, k )'s.

Lemma 4.6 The numbers P ( n, k ) obey

<!-- formula-not-decoded -->

Proof The proof operates by induction. The bound (4.14) is obvious for n = 1. Suppose the claim is established for all pairs ( m,k ) with m ≤ n . We will show that this implies the property for m = n +1. Indeed,

<!-- formula-not-decoded -->

.

The claim follows since for λ ≥ 1+ √ 5 2 , we have λ n -1 + λ n -2 ≤ λ n

This lemma gives us an idea of how large the P (2 n, k )'s appearing in the sum (4.13) really are. To derive an upper bound on the whole sum, we also need to understand the behavior of ∏ A ′ ∈ A/ ∼ F | A ′ | ( τ ). This is the subject of our next section.

## 4.6 Convex analysis

We start with a useful and classical lemma.

Lemma 4.7 Let f be a convex function on [0 , 1] , say. Consider the problem

<!-- formula-not-decoded -->

Then the maximum value f ∗ is obtained by allocating one x j to 1 and all the others to 0, i.e. f ∗ = ( k -1) f (0) + f (1) .

Proof For each x j , 0 ≤ x j ≤ 1, the convexity of f implies

<!-- formula-not-decoded -->

Summing this inequality over all indices gives

<!-- formula-not-decoded -->

which is what we sought to establish.

Corollary 4.8 Suppose that f = log F is a convex function on [0 , 1] , say, and consider

<!-- formula-not-decoded -->

Then the maximum value F ∗ is obtained by allocating one x j to 1 and all the others to 0, i.e. F ∗ = ( F (0)) k -1 F (1) .

<!-- formula-not-decoded -->

Note that both the lemma and the corollary hold for 'discrete' functions; that is, suppose that f ( j ) obeys

<!-- formula-not-decoded -->

Then the maximum value of ∑ k j =1 f ( n j ) where the n j 's are now integer values obeying n j ≥ 0 and ∑ k j =1 n j = n is of course achieved by taking all the n j 's equal to zero but one equal to n .

With these preliminaries in place, recall now the bound obtained in Lemma 4.3,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Note that we voluntarily exchanged the subscripts, namely, τ and n to reflect the idea that we shall view G as a function of n while τ will serve as a parameter. It is clear that log G is convex and, therefore,

<!-- formula-not-decoded -->

obeys

<!-- formula-not-decoded -->

Set G = G τ/ (1 -τ ) for short. Then for any equivalence class such that | A/ ∼ | = k , the above argument yields

<!-- formula-not-decoded -->

which, on the one hand, gives

<!-- formula-not-decoded -->

On the other hand, P (2 n, k ) ≤ φ 2 n (2 n -1) . . . (2 n -2 k +1) (see Lemma 4.6) and, therefore,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

We prove that the summand f is in some sense convex.

Lemma 4.9 For each k ≤ n -1 , f obeys

<!-- formula-not-decoded -->

As a consequence of this lemma, the maximum of f ( k ), 1 ≤ k ≤ n is of course attained at either the left-end point ( k = 1) or the right-end point ( k = n ); in short,

<!-- formula-not-decoded -->

Proof We need to establish that for each 1 ≤ k ≤ n -1,

<!-- formula-not-decoded -->

Observe that with α = N G (2) / | T | and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Clearly

<!-- formula-not-decoded -->

and, therefore, it is sufficient to establish that ρ k +1 ρ k -1 ≥ 1. Put m = n -k , then

<!-- formula-not-decoded -->

It is now a simple exercise to check that for each m ≥ 2, the logarithm of the right-hand is nonnegative, i.e.

<!-- formula-not-decoded -->

We omit the proof of this fact.

## 4.7 Proof of Theorem 3.1

The previous section established

<!-- formula-not-decoded -->

where letting c τ := e log((1 -τ ) /τ )

<!-- formula-not-decoded -->

and

This is exactly the content of Theorem 3.1.

<!-- formula-not-decoded -->

## 5 Numerical Experiments

In this section, we present numerical experiments in order to derive empirical bounds on | T | relative to | Ω | for a signal f supported on T to be the unique minimizer of ( P 1 ). The results can be viewed as a set of practical guidelines for situations where one can expect perfect recovery from partial Fourier information using convex optimization.

Our experiments are of the following form:

1. Choose constants N (the length of the signal), N t (the number of spikes in the signal), and N ω (the number of observed frequencies).
2. Randomly generate the subdomain T by sampling { 0 , . . . , N -1 } N t times without replacement (we have | T | = N t ).
3. Randomly generate f by setting f ( t ) = 0 , t ∈ T c and drawing both the real and imaginary parts of f ( t ) , t ∈ T from independent Gaussian distributions with mean zero and variance one 2 .
4. Randomly generate the subdomain Ω of observed frequencies by again sampling { 0 , . . . , N -1 } N ω times without replacement ( | Ω | = N ω ).
5. Solve ( P 1 ), and compare the solution to f .

The /lscript 1 -norm is not strictly convex, so solving ( P 1 ) using a Newton-type method that relies on local quadratic approximations of ‖ · ‖ /lscript 1 is problematic. Instead, we use a very simple gradient descent with projection algorithm. The number of iterations needed for convergence is high (on the order of 10 5 ), but since we can rapidly project onto the constraint set (using two fast Fourier transforms), each iteration takes a short amount of time. As an indication, the algorithm typically converges in less than 10 seconds on a standard desktop computer for signals of length N = 1024.

Figure 2 illustrates the recovery rate for varying values of | T | and | Ω | for N = 512. From the plot, we can see that for | Ω | ≥ 32, if | T | ≤ | Ω | / 5, we recover f perfectly about 80% of the time. For | T | ≤ | Ω | / 8, the recovery rate is practically 100%. We remark that these numerical results are consistent with earlier findings [2].

One source of slack in the theoretical analysis is the way in which we choose the polynomial P ( t ) (as in (2.11)). Theorem 2.1 states that f is a minimizer of ( P 1 ) if and only if there exists any trigonometric polynomial that has P ( t ) = sgn( f )( t ) , t ∈ T and | P ( t ) | &lt; 1 , t ∈ T c . In (2.11) we choose P ( t ) that minimizes the /lscript 2 norm on T c under the linear constraints P ( t ) = sgn( f )( t ) , t ∈ T . However, the condition | P ( t ) | &lt; 1 suggests that a minimal /lscript ∞ choice would be more appropriate (but is seemingly intractable analytically).

Figure 3 illustrates how often the sufficient condition of P ( t ) chosen as (2.11) meets the constraint | P ( t ) | &lt; 1 , t ∈ T c for the same values of τ and | T | . The empirical bound on T is stronger by about a factor of two; for | T | ≤ | Ω | / 10, the success rate is very close to 100%.

2 The results here, as in the rest of the paper, seem to rely only on the sets T and Ω. The actual values that f takes on T can be arbitrary; choosing them to be random emphasizes this. Figures 2 remain the same if we take f ( t ) = 1 , t ∈ T , say.

<!-- image -->

Figure 2: Recovery experiment for N = 512. (a) The image intensity represents the percentage of the time solving ( P 1 ) recovered the signal f exactly as a function of | Ω | (vertical axis) and | T | / | Ω | (horizontal axis); in white regions, the signal is recovered approximately 100% of the time, in black regions, the signal is never recovered. For each | T | , | Ω | pair, 100 experiments were run. (b) Cross-section of the image in (a) at | Ω | = 64. We can see that we have perfect recovery with very high probability for | T | ≤ 16.

<!-- image -->

0.1

T

|

|

/

|

Ω

|

(b)

Figure 3: Sufficient condition test for N = 512. (a) The image intensity represents the percentage of the time P ( t ) chosen as in (2.11) meets the condition | P ( t ) | &lt; 1 , t ∈ T c . (b) A cross-section of the image in (a) at | Ω | = 64. Note that the axes are scaled differently than in Figure 2.

100

90

80

70

60

50

40

30

20

10

0

true

% suff.

0.05

0.15

0.2

0.25

<!-- image -->

<!-- image -->

(b)

Figure 4: Two more phantom examples for the recovery problem discussed in Section 1.1. On the left is the original phantom ((d) was created by drawing ten ellipses at random), in the center is the minimum energy reconstruction, and on the right is the minimum total-variation reconstruction. The minimum total-variation reconstructions are exact.

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

As a final example of the effectiveness of this recovery framework, we show two more results of the type presented in Section 1.1; piecewise constant phantoms reconstructed from Fourier samples on a star. The phantoms, along with the minimum energy and minimum total-variation reconstructions (which are exact), are shown in Figure 4. Note that the total-variation reconstruction is able to recover very subtle image features; for example, both the short and skinny ellipse in the upper right hand corner of Figure 4(d) and the very faint ellipse in the bottom center are preserved. (We invite the reader to check [4] for related types of experiments.)

## 6 Discussion

We would like to close this paper by offering a few comments about the results obtained in this paper and by discussing the possibility of generalizations and extensions.

## 6.1 Stability

In the introduction section, we argued that even if one knew the support T of f , the reconstruction might be unstable. Indeed with knowledge of T , a reasonable strategy might be to recover f by the method of least-squares, namely,

<!-- formula-not-decoded -->

In practice, the matrix inversion might be problematic. Now observe that with the notations of this paper

<!-- formula-not-decoded -->

Hence, for stability we would need 1 | Ω | H 0 ≤ 1 -δ for some δ &gt; 0. This is of course exactly the problem we studied, compare Theorem 3.3. In fact, selecting α M as suggested in the proof of our main theorem (see section 3.4) gives 1 | Ω | H 0 ≤ . 42 with probability at least 1 -O ( N -M ). This shows that selecting | T | as to obey (1.9), | T | ≈ | Ω | / log N actually provides stability.

## 6.2 Robustness

An important question concerns the robustness of the reconstruction procedure vis a vis measurement errors. For example, we might want to consider the model problem which says that instead of observing the Fourier coefficients of f , one is given those of f + h where h is some small perturbation. Then one might still want to reconstruct f via

<!-- formula-not-decoded -->

In this setup, of course, one cannot expect exact recovery. Instead, one would like to know whether or not our reconstruction strategy is well-behaved or more precisely, how far is the minimizer f /sharp from the true object f . In short, what is the typical size of the error? Our preliminary calculations suggest that the reconstruction is robust in the sense that the error ‖ f -f /sharp ‖ 1 is small for small perturbations h obeying ‖ h ‖ 1 ≤ δ , say. We hope to be able to report on these early findings in a follow-up paper.

## 6.3 Extensions

Finally, work in progress shows that similar exact reconstruction phenomena hold for other synthesis/measurement pairs. Suppose one is given a pair of of bases ( B 1 , B 2 ) and randomly selected coefficients of an object f in one basis, say B 2 . (From this broader viewpoint, the special cases discussed in this paper assume that B 1 is the canonical basis of R N or R N × R N (spikes in 1D, 2D), or is the basis of Heavysides as in the Total-variation reconstructions, and B 2 is the standard 1D, 2D Fourier basis.) Then, it seems that f can be recovered exactly provided that it may be synthesized as a sparse superposition of elements in B 1 . The relationship between the number of nonzero terms in B 1 and the number of observed coefficients depends upon the incoherence between the two bases [7]. The more incoherent, the fewer coefficients needed. Again, we hope to report on such extensions in a separate publication.

Figure 5: Behavior of the left and right-hand side of (7.1) for two values of n

<!-- image -->

## 7 Appendix

## 7.1 Proof of Theorem 3.3

We need to prove that for τ ≤ . 44 and n ≥ 4,

<!-- formula-not-decoded -->

Now (2 n -1) 2 n = (2 n ) 2 n e -1 /epsilon1 n where /epsilon1 n ≤ e 1 / 2 n , say. We may then rewrite the previous inequality as

<!-- formula-not-decoded -->

where s τ = τ 1 -τ c τ . Because | T | ≤ α 2 γ 2 | τN | n , 0 &lt; α &lt; 1, it is sufficient to check that

<!-- formula-not-decoded -->

Note that plugging the value of γ gives r τ = (1 -τ ) 3 / ( e α 2 φ 2 [ log( 1 -τ τ ) ] 2 ) (recall φ = (1 + √ 5) / 2). In other words, we want

<!-- formula-not-decoded -->

Figure 5 illustrates the behavior of both the left-hand side and the right-hand side with α = 1. Simple numerical calculations show that with α = 1, (7.1) holds for τ ≤ . 44 and n ≥ 4, as claimed.

## 7.2 Proof of Lemma 3.5

Set e iφ = sgn( f ) for and fix K . Using (2.10), we have

/negationslash

<!-- formula-not-decoded -->

and, for example,

/negationslash

/negationslash

<!-- formula-not-decoded -->

One can calculate the 2 K th moment in a similar fashion. Put m := K ( n +1) and

<!-- formula-not-decoded -->

With these notations, we have

<!-- formula-not-decoded -->

/negationslash where we adopted the convention that x ( k ) 0 = x 0 for all 1 ≤ k ≤ 2 K and where it is understood that the condition t ( k ) j = t ( k ) j +1 is valid for 0 ≤ j ≤ n .

/negationslash

Now the calculation of the expectation goes exactly as in section 4. Indeed, we define an equivalence relation ∼ ω on the finite set A := { 0 , . . . , n } × { 1 , . . . , 2 K } by setting ( j, k ) ∼ ( j ′ , k ′ ) if ω ( k ) j = ω ( k ′ ) j ′ and observe as before that

<!-- formula-not-decoded -->

that is, τ raised at the power that equals the number of distinct ω 's and, therefore, we can write the expected value m ( n ; K ) as

/negationslash

<!-- formula-not-decoded -->

As before, we follow Lemma 4.5 and rearrange this as

/negationslash

<!-- formula-not-decoded -->

As before, the summation over ω will vanish unless t A ′ := ∑ ( j,k ) ∈ A ′ ( -1) k ( t ( k ) j -t ( k ) j +1 ) = 0 for all equivalence classes A ′ ∈ A/ ∼ , in which case the sum equals N | A/ ∼| . In particular, if A/ ∼ , the sum vanishes because of the constraint t ( k ) j = t ( k ) j +1 , so we may just as well

/negationslash

restrict the summation to those equivalence classes that contain no singletons. In particular we have

<!-- formula-not-decoded -->

To summarize

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/negationslash since | e i ∑ 2 k K =1 ( -1) k φ ( t ( k ) n +1 ) | = 1. Observe the striking resemblance with (4.11). Let ∼ be an equivalence which does not contain any singleton. Then the following inequality holds

<!-- formula-not-decoded -->

To see why this is true, observe as linear combinations of the t ( k ) j and of t 0 , we see that the expressions t ( k ) j -t ( k ) j +1 are all linearly independent, and hence the expressions ∑ ( j,k ) ∈ A ( -1) k ( t ( k ) j -t ( k ) j +1 ) are also linearly independent. Thus we have | A/ ∼ | independent constraints in the above sum, and so the number of t 's obeying the constraints is bounded | T | 2 n -| A/ ∼| .

With the notations of section 4, we established

<!-- formula-not-decoded -->

Now this is exactly the same as (4.13) which we proved obeys the desired bound.

## References

- [1] S. Boucheron, G. Lugosi, and P. Massart, A sharp concentration inequality with applications, Random Structures Algorithms 16 (2000), 277-292.
- [2] E. J. Cand` es, and P. S. Loh, Image reconstruction with ridgelets, SURF Technical report, California Institute of Technology, 2002.
- [3] S. S. Chen, D. L. Donoho, and M. A. Saunders, Atomic decomposition by basis pursuit, SIAM J. Scientific Computing 20 (1999), 33-61.
- [4] A. H. Delaney, and Y. Bresler, A fast and accurate iterative reconstruction algorithm for parallel-beam tomography, IEEE Trans. Image Processing , 5 (1996), 740-753.
- [5] D. C. Dobson, and F. Santosa, Recovery of blocky images from noisy and blurred data, SIAM J. Appl. Math. 56 (1996), 1181-1198.
- [6] D.L. Donoho, P.B. Stark, Uncertainty principles and signal recovery, SIAM J. Appl. Math. 49 (1989), 906-931.

| [7]   | D.L. Donoho and X. Huo, Uncertainty principles and ideal atomic decomposition, IEEE Transactions on Information Theory , 47 (2001), 2845-2862.                                                          |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [8]   | D. L. Donoho and M. Elad, Optimally sparse representation in general (nonorthogonal) dictionaries via /lscript 1 minimization. Proc. Natl. Acad. Sci. USA 100 (2003), 2197-2202.                        |
| [9]   | M. Elad and A.M. Bruckstein, A generalized uncertainty principle and sparse repre- sentation in pairs of R N bases, IEEE Transactions on Information Theory , 48 (2002), 2558-2567.                     |
| [10]  | P. Feng, and Y. Bresler, Spectrum-blind minimum-rate sampling and reconstruction of multiband signals, in Proc. IEEE int. Conf. Acoust. Speech and Sig. Proc. , (Atlanta, GA), 3 (1996), 1689-1692.     |
| [11]  | P. Feng, and Y. Bresler, A multicoset sampling approach to the missing cone problem in computer aided tomography, in Proc. IEEE Int. Symposium Circuits and Systems , (Atlanta, GA), 2 (1996), 734-737. |
| [12]  | A. Feuer and A. Nemirovsky, On sparse representations in pairs of bases, Accepted to the IEEE Transactions on Information Theory in November 2002.                                                      |
| [13]  | J. J. Fuchs, On sparse representations in arbitrary redundant bases, IEEE Transactions on Information Theory , 50 (2004), 1341-1344.                                                                    |
| [14]  | R. Gribonval and M. Nielsen, Sparse representations in unions of bases, Technical report, IRISA, November 2002.                                                                                         |
| [15]  | C. Mistretta, Personal communication (2004).                                                                                                                                                            |
| [16]  | F. Santosa, and W. W. Symes, Linear inversion of band-limited reflection seismograms, SIAM J. Sci. Statist. Comput. 7 (1986), 1307-1330.                                                                |
| [17]  | P. Stevenhagen, H.W. Lenstra Jr., Chebotar¨ ev and his density theorem, Math. Intel- ligencer 18 (1996), no. 2, 26-37.                                                                                  |
| [18]  | T. Tao, An uncertainty principle for cyclic groups of prime order, preprint. math.CA/0308286                                                                                                            |
| [19]  | J. A. Tropp, Greed is good: Algorithmic results for sparse approximation, Technical Report, The University of Texas at Austin, 2003.                                                                    |
| [20]  | J. A. Tropp, Just relax: Convex programming methods for subset selection and sparse approximation, Technical Report, The University of Texas at Austin, 2004.                                           |
| [21]  | M. Vetterli, P. Marziliano, and T. Blu, Sampling signals with finite rate of innovation, IEEE Transactions on Signal Processing , 50 (2002), 1417-1428.                                                 |
| [22]  | A. C. Gilbert, S. Guha, P. Indyk, S. Muthukrishnan, M. Strauss, Near-optimal sparse Fourier representations via sampling, 34th ACM Symposium on Theory of Computing , Montr´ eal, May 2002.             |
| [23]  | A. C. Gilbert, S. Muthukrishnan, and M. Strauss, Beating the B 2 bottleneck in esti- mating B -term Fourier representations, unpublished manuscript, May 2004.                                          |