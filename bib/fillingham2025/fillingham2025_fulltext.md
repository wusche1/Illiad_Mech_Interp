## SCALAR: Benchmarking SAE Interaction Sparsity in Toy LLMs

Sean P. Fillingham

∗

spfillingham@gmail.com

Andrew Gordon ∗

algo2217@gmail.com

Peter Lai ∗

peterjlai@gmail.com

Xavier Poncini ∗

xponcini@gmail.com

## David Quarel ∗

Department of Computing Australian National University david.quarel@anu.edu.au

Stefan Heimersheim † FAR.AI stefan@far.ai

## Abstract

Mechanistic interpretability aims to decompose neural networks into interpretable features and map their connecting circuits. The standard approach trains sparse autoencoders (SAEs) on each layer's activations. However, SAEs trained in isolation don't encourage sparse cross-layer connections, inflating extracted circuits where upstream features needlessly affect multiple downstream features. Current evaluations focus on individual SAE performance, leaving interaction sparsity unexamined. We introduce SCALAR ( S parse C onnectivity A ssessment of L atent A ctivation R elationships), a benchmark measuring interaction sparsity between SAE features. We also propose 'Staircase SAEs', using weight-sharing to limit upstream feature duplication across downstream features. Using SCALAR, we compare TopK SAEs, Jacobian SAEs (JSAEs), and Staircase SAEs. Staircase SAEs improve relative sparsity over TopK SAEs by 59 . 67% ± 1 . 83% (feedforward) and 63 . 15% ± 1 . 35% (transformer blocks). JSAEs provide 8 . 54% ± 0 . 38% improvement over TopK for feedforward layers but cannot train effectively across transformer blocks, unlike Staircase and TopK SAEs which work anywhere in the residual stream. We validate on a 216 K-parameter toy model and GPT2 Small ( 124 M), where Staircase SAEs maintain interaction sparsity improvements while preserving feature interpretability. Our work highlights the importance of interaction sparsity in SAEs through benchmarking and comparing promising architectures.

## 1 Introduction

Large language models (LLMs) have achieved impressive capabilities, but their inner workings remain opaque. Mechanistic interpretability seeks to explain model behavior by decomposing activations into interpretable components called features, often using Sparse Autoencoders (SAEs) [Bricken et al., 2023, Cunningham et al., 2023, Templeton et al., 2024] or similar sparse dictionary learning methods (e.g. Crosscoders [Lindsey et al., 2024], CLTs [Dunefsky et al., 2024]). These features can be organized into circuits (graphs of interacting features) that trace how information flows through the model. The sparsity of such circuits is important because simpler, sparser graphs offer clearer, more interpretable explanations of model behavior.

However, existing Sparse Autoencoders (SAEs), which are widely used for extracting interpretable features, do not necessarily yield sparse inter-layer interactions. Most SAEs are trained indepen-

* These authors contributed equally to this work and are listed in alphabetical order.
- † Mentored this project while at Apollo Research.

Preprint. Under review.

dently per layer, leading to inconsistencies: a feature that is represented in one layer may be omitted (feature completeness, [Bricken et al., 2023]), split into fine-grained features (feature splitting, [Bricken et al., 2023]), or absorbed into correlated features (feature absorption, [Chanin et al., 2024]) on downstream layers. This introduces spurious connections between features that are artifacts of the SAE training process rather than real interactions in the model. As a result, they distort the circuit structure and hinder a simpler interpretation.

Recent work has attempted to address such issues through introducing Jacobian SAEs [JSAEs, Farnik et al., 2025]. These SAEs explicitly minimize inter-layer interaction by penalizing the L1 norm of the Jacobian between two adjacent SAE latent vectors. While effective, this approach only applies to pairs of SAEs separated by a single feedforward layer.

To enable sparsification across a wider variety of components, we introduce Staircase SAEs , which use a simple architectural modification to encourage having fewer inter-layer feature interactions. The core idea is to encourage consistent feature representation across layers by explicitly sharing upstream SAE weights in downstream SAEs. Concretely, at a given layer, we set the dictionary of an SAE to be the concatenation of all upstream SAE features plus a new set of features for the current layer. This causes the SAE width to grow in a staircase-like fashion while keeping the number of trainable parameters the same as a standard SAE. We illustrate this architecture in Figure 1. As a result, features that are represented once can persist across layers, reducing spurious interactions and simplifying circuit structure-particularly for pass-through features flowing through the residual stream.

Jacobian SAEs and Staircase SAEs prompt the need for a quantitative evaluation of their effectiveness, which we provide through a new benchmark: SCALAR (Sparse Connectivity Assessment of Latent Activation Relationships). SCALAR quantifies how performance degrades as inter-layer feature connections are ablated, similar to established circuit discovery methods [e.g. Conmy et al., 2023]. We sort connections between a pair of SAEs by their importance (measured using integrated gradient attributions Wang et al. [2024]) and progressively remove them, measuring the KL divergence between the original model output and the ablated output. We then compute the area under the degradation curve, which quantifies how compact the interaction graph can be while preserving downstream performance.

To enable fair comparisons across different SAE configurations, we report both absolute and relative interaction sparsity. Absolute sparsity measures the raw area, while relative sparsity normalises area by the total number of connections between SAE pairs. Each metric on its own can be misleading: relative sparsity can be inflated by adding inactive features, and absolute sparsity can favor small but densely connected dictionaries.

Contributions Our main contributions in this work are as follows:

- Staircase SAEs : We introduce Staircase SAEs, which use an architectural modification to improve the representation of pass-through circuits. Staircase SAEs simplify pass-through connections by explicitly sharing upstream features across all downstream SAEs. We find that, at any given layer, the active features consist of around 25 - 45% new features and 55 - 75% features from previous layers. This is compatible with the results of Balcells et al. [2024] which studied pass-through circuits in standard SAEs.
- SCALAR Benchmark : We introduce SCALAR (Sparse Connectivity Assessment of Latent Activation Relationships), a benchmark for quantifying interaction sparsity between pairs of SAEs. SCALAR measures how model performance degrades as inter-layer connections are progressively ablated in order of importance, offering a principled measure of circuit simplicity.
- Empirical Validation : We use SCALAR to evaluate multiple SAE designs and find a tradeoff between standard SAEs, which may perform well initially but degrade under ablation, and Jacobian or Staircase SAEs, which retain performance even under pruning.

## 2 Related Work

SAE Architectures and Cross-Layer Interactions Our work builds on research using SAEs to interpret neural networks [Cunningham et al., 2023, Bricken et al., 2023, Gao et al., 2024]. Current

SAE methods face issues like feature splitting, absorption, and incompleteness that vary across layers, leading to inconsistencies in circuit interpretation. To address cross-layer interactions, Farnik et al. [2025] introduced Jacobian SAEs (JSAEs) that minimize interaction sparsity via L1 Jacobian penalties, though limited to feedforward layers. Yun et al. [2023] and Lindsey et al. [2024] proposed shared SAEs and crosscoders, while Dunefsky et al. [2024] introduced transcoders for cross-layer reconstruction.

Focus on Single-Layer Reconstruction Although shared architectures and crosscoders yield sparse feature dictionaries across layers, they obscure individual layer representations. Since our goal is understanding computation flow within networks, we focus on approaches that reconstruct activations at single layers while improving cross-layer sparsity through architectural innovations rather than abandoning per-layer interpretability.

## 3 Methodology

We use a toy language model to benchmark the interaction sparsity of different SAE architectures: A 216K-parameter transformer [Vaswani et al., 2023] following the GPT-2 architecture [Radford et al., 2019]. The model has 4 layers, a residual stream width of 64, and uses an ASCII tokenizer (vocabulary size of 128) similar to Lai and Heimersheim [2025]. We use a DynamicTanh layer [Zhu et al., 2025] instead of LayerNorm to make computing the Jacobian for JSAEs tractable (see Section D.1). We train the model on the tiny-Shakespeare dataset [Karpathy, 2015b,a] to perform standard next-token prediction.

SAEs are trained to reconstruct model activations that are mapped to and from a sparse high dimensional space [Cunningham et al., 2023, Bricken et al., 2023]. We train SAEs on activations between transformer blocks, and on the input/output of each feedforward layer. In all cases, the activations are vectors of the same size as the residual stream. We train three types of SAEs: Conventional TopK SAEs Gao et al. [2024], Jacobian SAEs [JSAEs Farnik et al., 2025], and our new Staircase SAEs.

## 3.1 TopK SAEs

TopK SAEs use a TopK activation function which selects the K largest magnitude entries in the latent vector and sets all other entries to zero.

<!-- formula-not-decoded -->

Sparsity is enforced directly by the TopK operation, ensuring by definition 1 that || z || 0 ≤ K . The SAEs are trained on the reconstruction loss

<!-- formula-not-decoded -->

We train TopK SAEs independently for each position in the model.

## 3.2 Jacobian SAEs

JSAEs [Farnik et al., 2025] sparsify the interactions between latents, in addition to the latents themselves. This is done by imposing an L1 penalty on the Jacobian of downstream latents with respect to upstream latents.

<!-- formula-not-decoded -->

Due to the sparsity of the latents, the Jacobian itself is sparse, and for a pair of TopK SAEs trained across a feedforward layer, there are only at most K 2 non-zero entries. These entries can be computed cheaply in closed form, see Section D.1. JSAEs are trained with a mixture of both losses:

<!-- formula-not-decoded -->

1 It may be less than K , due the ReLU.

Applying JSAEs to standard Transformer feedforward blocks that use LayerNorm (LN) [Ba et al., 2016] presents a challenge due to LN's inherent dependencies between elements of the input vector.

<!-- formula-not-decoded -->

These dependencies complicate the modification of the efficient closed-form of J for feedforward blocks, resulting in a large intermediate term when computing the Jacobian (Section D). In contrast, DynamicTanh (DyT) [Zhu et al., 2025] is an element-wise operation, that can be used as a drop-in replacement for LN. We fine-tune our model to substitute LN layers with DyT (Section C), modify the derivation of J in Farnik et al. [2025] to incorporate both DyT and the skip connection, and utilize this Jacobian to train JSAEs over MLP blocks.

## 3.3 Staircase SAEs

We introduce the Staircase SAE , an architecture aimed at promoting feature reuse from earlier layers to aid in activation reconstruction. This approach seeks to facilitate circuit sparsity through structural design rather than encouraging it via an explicit training objective (as for JSAEs).

Staircase SAEs form a collection of related SAEs that use shared weights. All SAEs use the same encoder and decoder weight matrices, W enc and W dec . Each SAE layer i (for 1 ≤ i ≤ L + 1 ) accesses a progressively larger slice of these shared weights. Assuming a base feature chunk size of n , the feature dimension for an SAE on layer i is n × i . The encoder and decoder weights for layer i are:

<!-- formula-not-decoded -->

where W enc ∈ R d model × N and W dec ∈ R N × d model are the full shared matrices (see Figure 1). The notation M [: k, :] selects the first k rows of M ; M [: , : k ] selects the first k columns. Each successive layer thus gains access to all earlier features, plus an additional chunk of n new features.

Figure 1: The staircase SAE architecture for a transformer with L = 3 layers. Each layer i uses a slice of the shared encoder W enc and decoder W dec weights. SAE chunks of identical colour indicate weights shared within the slices W i enc and W i dec .

<!-- image -->

<!-- image -->

Figure 2: By measuring the number of active latents per chunk, we can see feature reuse from previous layers, as each SAEs allocates some 'sparsity budget' to features from previous layers.

Crucially, the biases b i enc ∈ R ni and b i dec ∈ R d model are independent for each layer i . Independent decoder biases b i dec allow each SAE to center activations appropriately for its layer. Independent encoder biases b i enc are essential for feature selection and reuse: if features from earlier chunks ( j &lt; i ) are unhelpful for reconstructing layer i , the optimizer can suppress them by setting the corresponding bias entries to large negative values (assuming ReLU-like activation). With shared biases, disabling a feature for one layer could inadvertently degrade performance on another where that feature is important.

AStaircase SAE is at least as expressive as a set of independent SAEs of width n , while using nearly the same number of parameters. 2 This equivalence can be achieved by setting b i enc [0 : n ( i -1)] := -∞ , ensuring that only the i -th chunk of features z [ n ( i -1) : ni ] is active and effectively disabling features from earlier chunks.

To verify that Staircase SAEs behave differently from independent SAEs and promote feature reuse, we measured L 0 sparsity across chunks when using the Staircase architecture with the TopK SAE variant. While each layer tends to preferentially use features from its own chunk ( i -th chunk for layer i ), each layer also reuses features from earlier chunks ( 0 to i -1 ) to improve reconstruction (Figure 2).

We also evaluated an alternative SAE architecture in which each layer i could only optimize the weights of its own chunk, relying on earlier layers ( j &lt; i ) to make features in previous chunks useful. This design underperformed even standard SAEs (Section B), underscoring the importance of allowing each layer to directly optimize all accessible weights.

The Staircase architecture is compatible with any SAE variant and can be used with Standard, TopK, or other formulations. A comparison with other variants in Section E shows that the Staircase architecture enhances the TopK SAE variant using a negligible increase in parameter count.

## 3.4 Interaction sparsity benchmark: SCALAR

In this section, we describe how SCALAR scores are computed. This methodology is architectureagnostic, so we proceed abstractly. Suppose we have a language model and a pair of SAEs, denoted SAE 0 (upstream) and SAE 1 (downstream), where the upstream SAE encodes and decodes at an earlier layer in the model than the downstream one. We refer to the original, unmodified model as the full model , and to the version where activations at the SAE positions are replaced with their reconstructions as the full circuit . For clarity, we use latent to refer to an index in an SAE latent vector, and latent magnitude for its corresponding value. A SCALAR score is computed using the following steps:

1. Scoring connections : For each latent i in SAE 0 and each latent j in SAE 1 , measure how strongly i affects j . This produces a list of index pairs ( i, j ) , referred to as edges , ordered by interaction strength.
2. Cutting connections : Choose a sequence of values representing edge counts, called the edge number sequence . For each n in the sequence, retain only the top n edges (by interaction strength) and ablate the rest. The resulting model, with only a subset of connections preserved, is called a subcircuit .
3. Ablation curve : For each n and a batch of prompts, compute the Kullback-Leibler (KL) divergence between the logits of the full model and those of the subcircuit. Average the KL divergence across sequence positions and prompts to obtain a single value per n . The resulting values define the ablation curve , a piecewise linear function mapping edge count to KL divergence.
4. Area under the curve : The absolute SCALAR score is the area under the ablation curve. The relative SCALAR score is this area normalized by the total number of edges between SAE 0 and SAE 1 .

By measuring the KL divergence between the logits of the full model and those of the subcircuit, the SCALAR score captures both the computational sparsity and reconstruction quality of an SAE pair. With a simple modification, a SCALAR score can also be used to assess computational sparsity alone, see Appendix H for discussion.

Intuitive Example Consider the sparse feature circuits from Marks et al. [2025], where multiple 'plural nouns' features across layers interact densely (their Figure 11). Such circuits would receive high (poor) SCALAR scores due to many important cross-layer connections. A more interaction-sparse architecture would consolidate these into fewer shared features, yielding lower (better) SCALAR scores and simpler, more interpretable circuits.

2 The only additional parameters are the per-layer bias terms b i enc and b i dec , which increased the total by only ≈ 1 . 5% in our experiments.

Figure 3: The ablation curves for all SAEs attached at the labeled compute block. In these examples, the JSAE and Staircase SAEs clearly outperform the standard TopK SAEs.

<!-- image -->

<!-- image -->

<!-- image -->

## 3.4.1 Scoring Connections

Before cutting connections, we first identify which links between latents in SAE 0 and SAE 1 are most important to computation. To do this, we express downstream latent magnitudes as a function of upstream ones: we decode the upstream latents, advance the resulting activations through the model to the position of the downstream SAE, and then re-encode them into downstream latent magnitudes.

Wethen sample upstream latent magnitudes from data and, for each sample, use integrated gradients Sundararajan et al. [2017] to estimate the importance of each upstream latent to each downstream latent. Repeating this across samples, we compute the root mean square of the attributions to rank the importance of each connection between the two layers.

## 3.4.2 Cutting Connections

For clarity, we represent the full circuit as a bipartite graph S , where vertices correspond to the latents of SAE 0 and SAE 1 , which form disjoint sets. A subcircuit corresponds to a subgraph T ⊂ S . While a forward pass through the full circuit is straightforward (see Section 3.4), we now describe how to compute a forward pass through the subcircuit T .

The subcircuit computes upstream latent magnitudes by applying the usual SAE encoding to model activations at the position of SAE 0 . The downstream latents, however, are computed selectively. We first initialize an output tensor for the downstream latent magnitudes, setting all entries (across sequence positions) to zero. This tensor will be populated as follows: for each downstream latent ℓ included in T (i.e., a node in SAE 1 ), we perform the following steps:

1. Identify upstream latents : Let U ℓ be the set of upstream latents (i.e., SAE 0 nodes) connected to ℓ .
2. Ablate irrelevant latents : Zero out all upstream latent magnitudes not in U ℓ across all sequence positions.
3. Forward pass : Decode the ablated upstream latents, run the resulting activations through the model segment between the SAEs, and encode with SAE 1 .
4. Update output : Insert the resulting latent magnitude for ℓ at each sequence position into the output tensor for the downstream latent magnitudes.

## 4 Results

## 4.1 SAEs Across a Feedforward Layer

We trained both TopK SAEs and JSAEs across each feedforward layer. The sparsity coefficient of the JSAE pairs are tuned for performance on our metric, see Section G. Example ablation curves are shown in the leftmost panels of Figure 3; full curves for each layer are included in Appendix I. In Figure 3, the JSAE curve lies below the TopK curve at compute block 2 (the third transformer block), indicating higher computational sparsity (i.e., fewer active edges are needed to reach the same KL divergence). Performance varies across layers, with JSAEs outperforming TopK SAEs in central blocks while TopK performs better in peripheral ones, as shown in Figure 4 where absolute

SCALAR scores reflect this layer-dependent performance pattern. Cases where performance is mixed are shown in Appendix I.1.

The percentage change in SCALAR scores for each layer is reported in Table 1, showing that JSAEs outperform TopK SAEs in central blocks, while TopK performs better in peripheral ones. We highlight the outlier performance of the JSAE pair at Layer 0 on percentage reduction, this is likely due to poor tuning of the Jacobian coefficient, see Section G. Despite this, the value of the SCALAR score is insignificant compared to other layers, see Figure 4. Overall, summing SCALAR scores across all layers (absolute and relative are equivalent in this case), JSAEs provides a 8 . 54% ± 0 . 38% improvement over TopK.

|                        | Layer 0               | Layer 1          | Layer 2         | Layer 3            | Aggregate       |
|------------------------|-----------------------|------------------|-----------------|--------------------|-----------------|
| Absolute reduction (%) | - 602 . 15 ± 335 . 68 | 24 . 50 ± 1 . 57 | 9 . 55 ± 0 . 52 | - 14 . 07 ± 0 . 96 | 8 . 54 ± 0 . 38 |

Table 1: Percentage reduction in SCALAR scores for JSAEs compared to TopK SAEs across feedforward layers. Positive values indicate improved sparsity (lower SCALAR score) for JSAEs.

## 4.2 SAEs Across a Feedforward Block

Across the entire feedforward block we trained pairs of standard TopK SAEs and Staircase SAEs. The central panels of Figure 3 show representative ablation curves where the Staircase SAE clearly outperforms TopK around compute block 0. However, performance varies across different compute blocks, with some showing mixed results (see Appendix I.1 for complete results). These trends are reflected in Figure 4, with Staircase achieving higher sparsity at block 0. While the Staircase architecture permits more potential connections, leading to higher absolute SCALAR scores, the relative SCALAR score still indicates improved sparsity.

Layer-wise changes in SCALAR scores are shown in Table 2. Staircase SAEs yield lower absolute SCALAR scores at layer 0 but higher scores at layers 1-3. However, relative SCALAR scores are consistently lower across all layers. Summing over all layers, Staircase SAEs provide a 19 . 24% ± 0 . 71% improvement in absolute SCALAR score and a 59 . 67% ± 1 . 83% improvement in relative SCALAR score compared to TopK. This cumulative improvement, particularly with regards to the relative SCALAR score, suggests that Staircase SAEs achieve greater sparsity across the feedforward block.

|                        | Block 0          | Block 1           | Block 2           | Block 3            | Aggregate        |
|------------------------|------------------|-------------------|-------------------|--------------------|------------------|
| Absolute reduction (%) | 55 . 46 ± 2 . 94 | - 1 . 53 ± 0 . 10 | - 7 . 41 ± 0 . 45 | - 10 . 33 ± 0 . 79 | 19 . 24 ± 0 . 71 |
| Relative reduction (%) | 77 . 78 ± 3 . 74 | 49 . 24 ± 2 . 52  | 46 . 50 ± 2 . 23  | 45 . 11 ± 2 . 74   | 59 . 67 ± 1 . 83 |

Table 2: Percentage reduction in SCALAR scores for Staircase SAEs compared to TopK SAEs across feedforward blocks. Positive values indicate improved sparsity (lower SCALAR score) for Staircase SAEs.

## 4.3 SAEs Across a Transformer Block

Across each transformer block, we trained both standard TopK SAEs and Staircase SAEs. The rightmost panels of Figure 3 show representative ablation curves where Staircase SAEs perform well. Staircase SAEs outperform TopK SAEs at some compute blocks with consistently lower ablation curves, while at other blocks the performance is mixed with curves crossing at different thresholds (see Appendix I.1). Figure 4 reflects this variable pattern in the absolute SCALAR scores. However, when using the relative SCALAR score to account for the increased feature capacity of Staircase SAEs, the ranking consistently favors Staircase SAEs across blocks.

The percentage changes in the SCALAR scores are shown in Table 3. Staircase SAEs achieve lower absolute SCALAR scores in layer 0 and higher scores in layers 1-3, but consistently lower relative SCALAR scores across all layers. Aggregated across layers, Staircase SAEs yield a

Figure 4: A comparison of SCALAR scores across SAE positions and variants. In this space a lower SCALAR score is suggestive of higher sparsity. So, for example, around the Transformer block at layer 1 the TopK SAE exhibits higher sparsity than the Staircase SAE using the absolute SCALAR score. However, at that same position, the Staircase SAE has higher sparsity when using the relative SCALAR score.

<!-- image -->

-29 . 46% ± 0 . 91% improvement in absolute SCALAR score and a 63 . 15% ± 1 . 35% improvement in relative SCALAR score compared to TopK. In summary, while the absolute SCALAR metric suggests TopK SAEs are sparser overall, the relative SCALAR score, which normalizes for model capacity, indicates that Staircase SAEs achieve greater effective sparsity across transformer blocks.

|                        | Layer 0          | Layer 1            | Layer 2             | Layer 3              | Aggregate          |
|------------------------|------------------|--------------------|---------------------|----------------------|--------------------|
| Absolute reduction (%) | 6 . 95 ± 0 . 28  | - 64 . 50 ± 3 . 69 | - 107 . 42 ± 7 . 16 | - 125 . 53 ± 12 . 84 | - 29 . 46 ± 0 . 91 |
| Relative reduction (%) | 53 . 48 ± 1 . 63 | 72 . 60 ± 2 . 99   | 82 . 79 ± 2 . 55    | 88 . 83 ± 4 . 28     | 63 . 15 ± 1 . 35   |

Table 3: Percentage reduction in SCALAR scores for Staircase SAEs compared to TopK SAEs across transformer blocks. Positive values indicate improved sparsity (lower SCALAR score) for Staircase SAEs.

## 4.4 Validation on GPT-2 Small

To test generalization to realistic model scales, we trained TopK and Staircase SAE pairs on GPT-2 Small (124M parameters). Using down-sampled integrated gradients and ablation studies on layers 1, 6, and 11, we find that Staircase SAEs provide a 38 . 7 ± 1 . 2% improvement over TopK in relative interaction sparsity. While this improvement is smaller than our toy model results ( 59 . 67 ± 1 . 83% ), it demonstrates that our core findings generalize beyond the toy setting. Full details and results are provided in Appendix J.

## 4.5 Feature Interpretability Assessment

To verify that interaction sparsity improvements don't compromise individual feature interpretability, we conducted a blinded study comparing 1000 features each from Staircase and TopK SAEs (GPT-2 Small, layer 7). Human evaluators preferred TopK features 348 ± 15 times, Staircase features 342 ± 15 times, and were indifferent 310 ± 15 times. This suggests Staircase SAEs are equally interpretable individually while achieving greater interaction sparsity, supporting our architectural approach to improving circuit interpretability.

## 5 Discussion

While our results demonstrate the promise of Staircase SAEs for improving circuit sparsity and usefulness of the SCALAR metric, several limitations and open questions remain. Below, we outline some areas for future work and clarify the scope of our current contributions.

Model Scale Validation: While our primary experiments use a 216K-parameter toy model, we validate our key findings on GPT-2 Small (124M parameters). The 38 . 7% improvement in relative interaction sparsity, though smaller than our toy model results ( 59 . 67% ), provides evidence that our approach generalizes to realistic model architectures. Full technical details are provided in Appendix J.

Surrogate models vs. explaining activations: In our work, we focus on SAEs instead of more recent Crosscoders and Transcoders. One advantage of SAEs is that they show all information present in a cross-section of the model - we know that features interactions between two SAE layers must be mediated by the transformer block in between.

Downstream uses of SAEs: In this work, we focused on circuit interpretability conditioned on sparse dictionary learning and SAEs. Recent work however has called into question whether the apparent interpretability of SAEs is useful in downstream tasks [Kantamneni et al., 2025]. Our interpretability assessment suggests that architectural improvements to interaction sparsity need not compromise individual feature quality, addressing one potential concern about SAE utility.

## 6 Conclusion

Sparse autoencoders are widely used in mechanistic interpretability to disentangle superimposed model features into sparse, human-understandable latents; however, to fully trace how information flows through a model, we need not only interpretable features but also interpretable circuits with sparse and meaningful connections between features. Prior work has largely focused on per-layer sparsity and reconstruction loss, neglecting the sparsity of these inter-feature connections. Our work addresses this gap by characterizing interaction sparsity between adjacent SAEs trained on the residual stream of a toy LLM.

We introduce a general-purpose architecture, Staircase SAEs , which improves interaction sparsity by allowing downstream layers to reuse upstream features. This design is compatible with all SAE variants and facilitates the emergence of pass-through circuits. To evaluate interaction sparsity, we propose the SCALAR score , a metric that characterizes the importance of cross-layer connections. Using this score, we assess the circuit sparsity of standard TopK SAEs, Jacobian SAEs, and Staircase SAEs across various model layers. We find that TopK SAEs exhibit the lowest interaction sparsity, while both Jacobian and Staircase SAEs achieve higher sparsity under the relative SCALAR score.

One challenge with Staircase SAEs is that they inherently allow more potential cross-layer connections, which leads to higher absolute SCALAR scores. To account for this, we introduce the relative SCALAR score , which normalizes for connection count and offers a fairer basis for comparison across architectures. Using this metric, we show that architectural choices can meaningfully shape circuit sparsity. We validate our approach on models ranging from 216K to 124M parameters, demonstrating that architectural choices can meaningfully improve interaction sparsity without sacrificing individual feature interpretability. We hope this work encourages continued exploration of SAE architectures that support sparser and more interpretable cross-layer interactions.

## Acknowledgments

We would like to thank Lucy Farnik for insightful discussions about Jacobian SAEs. The authors would like to thank the Supervised Program for Alignment Research (SPAR) for operational support. PL conducted this research as part of the MARS program by the Cambridge AI Safety Hub (CAISH). XP acknowledges support from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (101042460): ERC Starting grant 'Interplay of structures in conformal and universal random geometry' (ISCoURaGe, PI Eveliina Peltola).

## References

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization, 2016. URL https://arxiv.org/abs/1607.06450 .

Daniel Balcells, Benjamin Lerner, Michael Oesterle, Ediz Ucar, and Stefan Heimersheim. Evolution of sae features across layers in llms, 2024. URL https://arxiv.org/abs/2410.08869 .

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread , 2023. https://transformercircuits.pub/2023/monosemantic-features/index.html.

David Chanin, James Wilken-Smith, Tom´ aˇ s Dulka, Hardik Bhatnagar, and Joseph Bloom. A is for absorption: Studying feature splitting and absorption in sparse autoencoders, 2024. URL https://arxiv.org/abs/2409.14507 .

Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adri` a Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability, 2023. URL https://arxiv.org/abs/2304.14997 .

Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. arXiv preprint arXiv:2309.08600 , 2023. URL https://arxiv.org/abs/2309.08600 .

Jacob Dunefsky, Philippe Chlenski, and Neel Nanda. Transcoders find interpretable llm feature circuits, 2024. URL https://arxiv.org/abs/2406.11944 .

Lucy Farnik, Tim Lawson, Conor Houghton, and Laurence Aitchison. Jacobian sparse autoencoders: Sparsify computations, not just activations, 2025. URL https://arxiv.org/abs/ 2502.18147 .

Leo Gao et al. Scaling and evaluating sparse autoencoders. OpenAI , 2024. URL https://cdn. openai.com/papers/sparse-autoencoders.pdf .

Stefan Heimersheim. You can remove gpt2's layernorm by fine-tuning, 2024. URL https:// arxiv.org/abs/2409.13710 .

Subhash Kantamneni, Joshua Engels, Senthooran Rajamanoharan, Max Tegmark, and Neel Nanda. Are sparse autoencoders useful? a case study in sparse probing, 2025. URL https://arxiv. org/abs/2502.16681 .

Andrej Karpathy. char-rnn. https://github.com/karpathy/char-rnn , 2015a.

Andrej Karpathy. The unreasonable effectiveness of recurrent neural networks, May 2015b. URL http://karpathy.github.io/2015/05/21/rnn-effectiveness/ .

Peter Lai and Stefan Heimersheim. Sae regularization produces more interpretable models. https://www.lesswrong.com/posts/sYFNGRdDQYQrSJAd8/ sae-regularization-produces-more-interpretable-models , 2025.

Jack Lindsey, Adly Templeton, Jonathan Marcus, Thomas Conerly, Joshua Batson, and Christopher Olah. Sparse crosscoders for cross-layer features and model diffing. https:// transformer-circuits.pub/2024/crosscoders/index.html , 2024. Transformer Circuits Thread.

Samuel Marks, Can Rager, Eric J. Michaud, Yonatan Belinkov, David Bau, and Aaron Mueller. Sparse feature circuits: Discovering and editing interpretable causal graphs in language models, 2025. URL https://arxiv.org/abs/2403.19647 .

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog , 1(8):9, 2019.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks, 2017. URL https://arxiv.org/abs/1703.01365 .

Alex Templeton et al. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. Transformer Circuits Thread , 2024. URL https://transformer-circuits.pub/ 2024/scaling-monosemanticity/ .

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2023. arXiv: 1706.03762 [cs.CL].

Yongjie Wang, Tong Zhang, Xu Guo, and Zhiqi Shen. Gradient based feature attribution in explainable ai: A technical review, 2024. URL https://arxiv.org/abs/2403.10415 .

Wikipedia contributors. E series of preferred numbers. https://en.wikipedia.org/wiki/E\_ series\_of\_preferred\_numbers , 2025. Accessed: 2025-05-22.

Zeyu Yun, Yubei Chen, Bruno A Olshausen, and Yann LeCun. Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors, 2023. URL https://arxiv.org/abs/2103.15949 .

Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, and Zhuang Liu. Transformers without normalization, 2025. URL https://arxiv.org/abs/2503.10622 .

## A Compute resources

The LLM and all SAEs were trained on a local machine with 4xA4000s, 64GB CPU RAM and 2TB of storage. The LLM takes ≤ 5 minutes to train, and each SAE takes ≈ 30 mins to train. We estimate the total compute to be no more than 100 GPU hours to replicate the results in the paper.

The gradient attributions were trained on Runpod, on a machine with 1xA40, 50 GB CPU RAM and 20GB of storage. One set of attributions for a 4 layer model takes ≤ 20 minutes to produce.

The SCALAR scores were computed on an Apple M4 machine with 24GB CPU RAM and 250GB of storage. Computing a SCALAR score for an SAE pair with edge count sequence (35) across 50 prompts takes ≈ 100 minutes to produce.

## B Staircase Detached Variant

When training Staircase SAEs, the optimizer is allowed to optimize all weights in each layer to minimize reconstruction loss. Looking at Figure 1, this means that the orange chunk sees activations h 0 , . . . , h 3 from between all layers, the blue chunk sees h 1 , h 2 , h 3 , and so on, up to the green chunk which only sees h 3 during training.

One variant that was considered was to detach the gradients from the previous chunks used in subsequent layers, so that the orange chunk would only be optimized to reconstruct h 1 . The optimizer for the subsequent layer can reuse the feature magnitudes that the orange chunk computes if it is useful for reconstructing h 2 , but is not allowed to directly update the orange chunk's weights.

As can be seen in Figure 5, the detached gradient variant focuses entirely on the features of the current chunk, and does not use features from previous chunks, essentially degenerating back to a standard SAE (Figure 6). This variant underperformed both staircase SAEs as well as standard SAEs, indicating that it is important to allow the optimizer to optimize all weights in each layer to minimize reconstruction loss.

Figure 5: The L0 sparsity measured per chunk for a staircase SAE with L = 4 layers, and 5 activations h 0 , . . . , h 4 . The left figure was trained with all gradients attached, while the right figure was trained with gradients from previous chunks detached. Both models use Top10 SAEs. What we find is that the standard staircase variant (left) spends some sparsity budget on features from previous chunks, whereas the detached gradient variant (right) degenerates back to a standard SAE, rarely using features from previous chunks.

<!-- image -->

Figure 6: If chunks other than that for the current layer are not used, the staircase SAE degenerates back to a standard SAE.

<!-- image -->

## C DynamicTanh

Normalization layers are a cornerstone of modern deep learning, widely considered essential for stable and efficient training Ba et al. [2016]. The paper 'Transformers without Normalization' Zhu et al. [2025] challenges this convention by introducing DynamicTanh (DyT), a simple element-wise operation proposed as a drop-in replacement for normalization layers in Transformers.

<!-- formula-not-decoded -->

where:

- x ∈ R d model is the input tensor.
- α ∈ R d model is a learnable scaling factor for each element of x .
- γ, β ∈ R d model are the standard scale and shift parameters of a standard normalization layer.

## Compare with LN:

<!-- formula-not-decoded -->

where E [ x ] and V [ x ] are the emperical mean and variance of the input tensor x across the batch and sequence dimensions, and both γ and β are learnable parameters as in DyT.

The inspiration for DyT stems from the observation that Layer Normalization (LN) in Transformers often produces input-output mappings that are sigmoid-shaped and resemble a tanh function, particularly in deeper layers (see Figure 2 in Zhu et al. [2025]). They demonstrate that by incorporating DyT, Transformers without normalization can match or even exceed the performance of their normalized counterparts, suggesting that the primary beneficial effect of normalization in these architectures might be the adaptive non-linear squashing of activations, rather than strict statistical normalization.

The benefit to us of DyT is that this function is computed element-wise, which allows us to effectively compute the Jacobian of the MLP block when LayerNorm is replaced with DynamicTanh. If we were to just use LayerNorm, this calculation would be too expensive (Section D.3).

## D Jacobian Derivation

The following description of the Jacobian derivation for JSAEs across an MLP layer is taken directly from Farnik et al. [2025], where one can find a detailed derivation.

We wish to compute the Jacobian of the function f s = e y ◦ f ◦ d x ◦ τ k , describing the mapping s y = f s ( s x ) from the hidden latents s x of a TopK SAE before a MLP layer, to the hidden latents s y of a TopK SAE after the MLP layer. Here, τ k denotes the TopK activation function, ˆ x = d x ( s x ) = W dec x s x + b dec x is the decoder of the first SAE, s y = e y ( y ) = τ k ( W enc y y + b enc y ) is the encoder of the second SAE, and y = f ( ˆ x ) = W 2 ϕ MLP ( z ) + b 2 with z = W 1 ˆ x + b 1 is the MLP.

Farnik et al. [2025] show that the Jacobian of f s = e y ◦ f ◦ d x ◦ τ k can be computed efficiently as

<!-- formula-not-decoded -->

where K 1 and K 2 are the sets of indices corresponding to the features selected by the TopK activation function of the first and second SAEs respectively, and that the (at most) k × k non-zero elements of the Jacobian can be compactly represented as

<!-- formula-not-decoded -->

where W enc(active) y ∈ R k × m y and W dec(active) x ∈ R m x × k contain the active rows and columns, i.e., the rows and columns corresponding to the K 2 or K 1 indices respectively.

## D.1 Jacobian Derivation for MLP Block with Skip Connection

We extend this by giving an efficient implementation of the Jacobian of g s = e y ◦ g ◦ d x ◦ τ k , where

<!-- formula-not-decoded -->

and h is an element-wise normalization function (we use h = DyT in practice). Note that we ignore the biases of the SAEs and the FF layer, as they do not change the Jacobian calculation.

The components are:

- s x ∈ R d SAE is the k -sparse vector of active features from the hidden layer of the input SAE.
- ¯ s x ∈ R d SAE is the result of passing s x through the TopK activation function τ k . This doesn't change anything (¯ s x = s x ) , but it makes calculation of the Jacobian efficient.
- W dec x ∈ R d SAE × d model is the full decoder matrix of the input SAE.
- W dec(active) x ∈ R k × d model consists of the k rows of W dec x corresponding to the active features in s x.
- The dense vector x ∈ R d model is reconstructed from s x as x = ( W dec(active) x ) T s x. The decoder bias b dec x is ignored for Jacobian calculation.
- h : R d model → R d model is an element-wise normalization function, that is, there is some function ψ : R → R such that h ( x ) i = ψ ( x i ) . Its Jacobian ∂h ( x ) i ∂x j is diag ( h ′ ( x )) , a diagonal matrix with elements ψ ′ ( x i ) on its diagonal. As an abuse of notation, we use h ( x i ) in place of ψ ( x i ) .
- The MLP (or FFN) processes a = h ( x ) . It is defined (ignoring biases) as

<!-- formula-not-decoded -->

- -W 1 ∈ R d mlp × d model is the MLP input weight matrix.
- -W 2 ∈ R d model × d mlp is the MLP output weight matrix.
- -ϕ MLP is the MLP's element-wise activation function. Its Jacobian with respect to its input z 1 = W 1 a is diag ( ϕ ′ MLP ( z 1 )) .
- W enc y ∈ R d model × d SAE is the full encoder matrix of the output SAE.

- W enc(active) y ∈ R d model × k consists of the k columns of W enc y corresponding to the active features that will form s y.
- s y ∈ R d SAE is the k -sparse vector of output features for the output SAE.

We can decompose the expression into two paths: the skip connection, and the MLP path, given that all of the functions are linear, or act element-wise.

<!-- formula-not-decoded -->

with s skip y = g skip s ( s x ) = ( e y ◦ d x ◦ τ k )( s x ) and s FFh y = g FFh s ( s x ) = ( e y ◦ FF ◦ DyT ◦ d x ◦ τ k )( s x ) . We can thereby compute the Jacobian as the sum of the Jacobians of the two paths.

<!-- formula-not-decoded -->

## D.1.1 Skip Path Jacobian ( J skip )

The skip path Jacobian is the Jacobian of the skip path, which is a linear function (if focused on the features chosen by the TopK activation).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which, when combined gives:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## D.2 MLP Path Jacobian ( J MLP)

Let s FFh y be the sparse output contribution from the MLP path. The path through the FF Block is computed as

<!-- formula-not-decoded -->

Computing each term:

<!-- formula-not-decoded -->

As before, we compute each of these terms separately:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting these into the sum for J FFh g s ,ij :

<!-- formula-not-decoded -->

Dropping the terms δ qr , δ tu , δ vj and replacing q → r, t → u, v → j , we get

<!-- formula-not-decoded -->

Applying these simplifications, the sum becomes:

<!-- formula-not-decoded -->

This sum represents the ( i, j ) -th element of a matrix product. Let ∇ ϕ = diag ( ϕ ′ MLP ( z )) be a diagonal matrix with elements ϕ ′ MLP ( z r ) on its diagonal, and ∇ h = diag ( h ′ ( ˆ x )) be a diagonal matrix with elements h ′ (ˆ x u ) on its diagonal. Then the ( i, j ) -th element is:

<!-- formula-not-decoded -->

In matrix form, considering only the active features (where i ∈ K 2 and j ∈ K 1 ), the active block of the Jacobian is:

<!-- formula-not-decoded -->

Finally, the terms ∇ ϕ , ∇ h are diagonal matrices (stored as a vector), and so their product with an adjacent matrix is cheap to compute:

<!-- formula-not-decoded -->

Combining the two terms together, we get:

<!-- formula-not-decoded -->

though for efficiency, we compute the two terms separately and add them together.

## D.3 Jacobian with LayerNorm

Consider for a moment an arbitrary activation x ∈ R s × d that flows through a transformer: a set of d -dimensional activations, one for each sequence position s , and some arbitrary module ψ : R s × d → R s × d that performs some computation on those activations. In the most general case, the Jacobian of this module ∇ ψ is a tensor of shape ( s, d, s, d ) , requiring O ( s 2 d 2 ) memory to store. For any reasonable size of sequence length s and model dimension d , this is prohibitively large. For gpt-2 small, s = 1024 and d = 768 , this is ≈ 6 · 10 11 elements, or about 2.25TB of memory to store.

Given assumptions about the module ψ , we can reduce the memory requirements for the Jacobian:

- Linear : The Jacobian will be a constant matrix, so we can ignore the sequence dimensions s, s ′ .
- Elementwise : The Jacobian will be sparse, with only the diagonal being non-zero. Offdiagonal terms ( d, d ′ ) can be ignored.
- Sequence Independent : We only need to track the Jacobian with respect to one sequence position s . Cross-terms ( s, s ′ ) can be ignored.

Table 4: Summary of Jacobian structures and memory requirements for a module ψ : R s × d → R s × d , based on assumptions of linearity, elementwise-ness, and sequence independence. Memory costs estimated for GPT-2 Small.

| Linear   | Elementwise   | Seq. Indep.   | Jacobian Form        | Cost          | RAM    | Examples        |
|----------|---------------|---------------|----------------------|---------------|--------|-----------------|
| ✓        | ✓             | ✓             | δ ss ′ δ dd ′ ∇ ψ d  | O ( d )       | 4kB    | γ ⊙ x + β       |
| ✓        | ✗             | ✓             | δ ss ′ ∇ ψ dd ′      | O ( d 2 )     | 4MB    | Wx + b          |
| ✗        | ✓             | ✓             | δ ss ′ δ dd ′ ∇ ψ sd | O ( sd )      | 3MB    | GeLU, ReLU, DyT |
| ✗        | ✗             | ✓             | δ ss ′ ∇ ψ sd ′ d    | O ( sd 2 )    | 768MB  | LayerNorm       |
| ✗        | ✗             | ✗             | ∇ ψ sds ′ d ′        | O ( s 2 d 2 ) | 2.25TB | Self-Attention  |

LayerNorm may be sequence independent, but being neither linear nor elementwise, the Jacobian is rather large, making intractible to evaluate inside a tight training loop. This motivates the use of replacing it with an elementwise normalization operation such as DynamicTanh Zhu et al. [2025] or fine-tuning to remove the LayerNorm entirely Heimersheim [2024]

## E Experimental Comparison Setup

We compared the proposed Staircase SAE against several baseline configurations. All experiments described here used TopK SAEs with K = 10 . The expansion factor of the SAE hidden layer relative to the model's activation dimension ( d model) is specified for each model.

The models compared are:

- Top k -x8 : Standard TopK SAEs trained independently per layer. The hidden dimension d sae was set to 8 × d model for all layers.
- Top k -x40 : Identical to Top k -x8 , but with a hidden dimension of 40 × d model for all layers.
- Top k -x40-tied : Identical to Top k -x40 , but encoder and decoder matrices are shared across all layers.
- Staircase-x8 : The proposed Staircase SAE architecture using TopK activation, as described in Section 3.3.
- Staircase-untied-x8 : Same as Staircase-x8 , but with independent encoder and decoder matrices for each layer.
- Staircase-Detach : A variant of Staircase-x8 where each layer i only optimizes parameters related to its own feature chunk i . As mentioned, this approach performed poorly (Section 3.3).

We note that the first four models (Staircase-Detach, Top k -x8, Staircase-x8, Top k -x40-tied) have approximately the same number of trainable parameters (ignoring the biases b i enc , b i dec , which contribute negligibly to the total count compared to the weight matrices). The last two models (Staircase-untied-x8 and Top k -x40) have significantly more parameters and, unsurprisingly, tend to achieve better reconstruction performance, serving as benchmarks for capacity.

Figure 7: For a given parameter count, the staircase SAE ( staircase-x8 ) has a lower increase in CE loss over the validation set than the baseline SAEs described in Section E.

<!-- image -->

## F Scoring Connections

## F.1 Integrated Gradient Attribution

Here we give a more detailed description of integrated gradient attribution, and describe the details of the method we use to implement it.

Integrated Gradient Attribution is a solution to the local attributions problem. That is, given a vector v , and a real valued differentiable function f this method produces an attribution a i for each coordinate of v satisfying

<!-- formula-not-decoded -->

To compute gradient attribution, choose a base point b in the same vector space as v . Each a i is determined by the formula

<!-- formula-not-decoded -->

it is the integral of ∂ i of f along a straight line from b to v .

To score cross-SAE latent connections, we repeat this process once for every downstream latent. For the function f , we decode latents in SAE 0 , pass them through the model, and encode them to latents in SAE 1 .

We always chose the base point b to be the origin. For the data in this paper, we approximated the integral as a Riemann sum with 5 terms.

Since integrated gradients only give local data, we determined global connection scores by sampling latent activations from 576 points in the training data, and using the root mean square of every local attribution as the overall connection score.

This process was extremely consistent, computing scores twice with different random samples of data resulted in virtually identical scores.

## G Jacobian coefficient tuning

At each block, a JSAE pair was trained with a Jacobian coefficient λ taking geometrically spaced values in the set:

<!-- formula-not-decoded -->

following the E12 series Wikipedia contributors [2025] of preferred numbers, together with λ = 0 . Recall that for λ = 0 , the JSAE pair degenerates to a TopK SAE pair. For each such pair, an ablation curve was produced and the corresponding SCALAR score was calculated, see Table 5. We also measured the CE loss increase when the JSAEs were spliced in over a large range of λ , and then swept over the critical range where it started to non-trivially impact the performance of reconstruction.

To ensure maximum competitiveness between TopK and JSAE variants, the JSAE results reported in Section 4 take the (non-zero) Jacobian coefficient corresponding to the minimum SCALAR score for each layer.

Table 5: SCALAR scores for JSAE pairs with different Jacobian coefficient values (expressed in units of 10 -3 ) across feedforward layers. The best performing non-zero Jacobian coefficient values for each layer are bolded. We observe that the optimal Jacobian coefficient varies across layers, with non-zero Jacobian coefficient values outperforming TopK (zero Jacobian coefficient) at layers one and two only.

|      | Layer 0 ( 10 4 )   | Layer 1 ( 10 5 )   | Layer 2 ( 10 5 )   | Layer 3 ( 10 5 )   |
|------|--------------------|--------------------|--------------------|--------------------|
|  0   | 0 . 098 ± 0 . 007  | 0 . 555 ± 0 . 028  | 1 . 569 ± 0 . 062  | 1 . 347 ± 0 . 060  |
|  1   | 0 . 733 ± 0 . 049  | 0 . 483 ± 0 . 025  | 1 . 540 ± 0 . 063  | 1 . 544 ± 0 . 068  |
|  1.2 | 0 . 927 ± 0 . 082  | 0 . 419 ± 0 . 022  | 1 . 524 ± 0 . 062  | 1 . 591 ± 0 . 069  |
|  1.5 | 0 . 874 ± 0 . 057  | 0 . 436 ± 0 . 023  | 1 . 540 ± 0 . 063  | 1 . 545 ± 0 . 066  |
|  1.8 | 0 . 686 ± 0 . 054  | 0 . 474 ± 0 . 025  | 1 . 499 ± 0 . 061  | 1 . 552 ± 0 . 066  |
|  2.2 | 1 . 003 ± 0 . 071  | 0 . 613 ± 0 . 033  | 1 . 505 ± 0 . 063  | 1 . 536 ± 0 . 065  |
|  2.7 | 1 . 154 ± 0 . 082  | 0 . 638 ± 0 . 034  | 1 . 445 ± 0 . 060  | 1 . 635 ± 0 . 068  |
|  3.3 | 1 . 340 ± 0 . 091  | 1 . 151 ± 0 . 053  | 1 . 416 ± 0 . 060  | 1 . 702 ± 0 . 071  |
|  3.9 | 1 . 148 ± 0 . 079  | 1 . 285 ± 0 . 057  | 2 . 009 ± 0 . 079  | 1 . 633 ± 0 . 066  |
|  4.7 | 1 . 270 ± 0 . 073  | 1 . 654 ± 0 . 070  | 2 . 490 ± 0 . 094  | 1 . 777 ± 0 . 069  |
|  5.6 | 1 . 498 ± 0 . 111  | 1 . 941 ± 0 . 078  | 2 . 798 ± 0 . 096  | 1 . 906 ± 0 . 074  |
|  6.8 | 1 . 455 ± 0 . 083  | 2 . 286 ± 0 . 089  | 3 . 010 ± 0 . 104  | 1 . 937 ± 0 . 074  |
| 10   | 1 . 795 ± 0 . 111  | 2 . 645 ± 0 . 096  | 4 . 027 ± 0 . 120  | 2 . 108 ± 0 . 077  |

Given that JSAE pairs are explicitly trained to reduce computational sparsity, we expect them to outperform TopK SAE pairs on our metric across all layers. Across the range of Jacobian coefficient values analysed, we find that JSAEs outperform TopK SAEs at Layer 1 and Layer 2 only. We offer two possible explanations for this. First, the approximately SCALAR-minimising Jacobian coefficient is not in the set (34). This seems likely from the Layer 0 results, given the significant jump from λ = 0 to λ = 10 -3 . Alternatively, layers 0 and 3 are less amenable to sparsification via the Jacobian. This seems plausible from the Layer 3 results where increasing the Jacobian coefficient typically results in a larger SCALAR score.

## Combined Pairs: CE Increase vs. V\_Il Value (Point labels show sparsity coefficient from directory name)

<!-- image -->

Sum(v\_Il) vs. Loss Increase (Labeled by Sparsity Coeff, Colored by Max Steps)

Figure 8: Pareto curves showing the trade-off between reconstruction performance and Jacobian coefficient values. The top image shows overall performance across all layers and the bottom image shows the concentrated sweep over the sparsity coefficient λ discussed in Section G.

<!-- image -->

1

1

;

5

<!-- image -->

8

8

;

Figure 9: Pareto curves showing the trade-off between reconstruction performance and Jacobian coefficient values for individual layers. Arranged in a 2×2 grid: layer 1 (top left), layer 2 (top right), layer 3 (bottom left), and layer 4 (bottom right).

9

g

5

1

1

;

g

## H Pure Computational Sparsity Metric

Recall that the SCALAR score is computed from ablation plots produced by evaluating the KL divergence between the logits produced by the full model and the logits produced by a collection of subcircuits, see Section 3.4 for further details. Here, we consider a slight variation of this score by instead producing ablation plots by computing the KL divergence between logits produced by the full circuit and the logits produced by a collection of subcircuits (the same collection as above). Accordingly, this metric incorporates the reconstruction quality of SAE pairs into the target distribution (the full circuit logits), which means that the resulting area under the ablation curve reflects the computational impact of ablating edges to a much greater degree.

Table 6: Computational sparsity metric for JSAE pairs with different Jacobian coefficient values (expressed in units of 10 -3 ) across feedforward layers. For layers 1 and 2, there exists a range of Jacobian coefficient values where we observe a trend of decreasing metric with increasing Jacobian coefficient.

|      | Layer 0 ( 10 4 )   | Layer 1 ( 10 5 )   | Layer 2 ( 10 5 )   | Layer 3 ( 10 5 )   |
|------|--------------------|--------------------|--------------------|--------------------|
|  0   | 0 . 851 ± 0 . 064  | 0 . 422 ± 0 . 020  | 1 . 027 ± 0 . 040  | 0 . 386 ± 0 . 018  |
|  1   | 1 . 203 ± 0 . 065  | 0 . 286 ± 0 . 014  | 1 . 093 ± 0 . 044  | 0 . 608 ± 0 . 026  |
|  1.2 | 1 . 113 ± 0 . 072  | 0 . 220 ± 0 . 010  | 1 . 054 ± 0 . 042  | 0 . 628 ± 0 . 027  |
|  1.5 | 1 . 317 ± 0 . 078  | 0 . 188 ± 0 . 009  | 1 . 069 ± 0 . 042  | 0 . 665 ± 0 . 030  |
|  1.8 | 0 . 887 ± 0 . 059  | 0 . 134 ± 0 . 006  | 1 . 039 ± 0 . 041  | 0 . 632 ± 0 . 026  |
|  2.2 | 1 . 213 ± 0 . 073  | 0 . 178 ± 0 . 008  | 0 . 968 ± 0 . 040  | 0 . 671 ± 0 . 029  |
|  2.7 | 0 . 797 ± 0 . 053  | 0 . 081 ± 0 . 004  | 0 . 836 ± 0 . 033  | 0 . 708 ± 0 . 029  |
|  3.3 | 1 . 187 ± 0 . 070  | 0 . 136 ± 0 . 005  | 0 . 707 ± 0 . 027  | 0 . 761 ± 0 . 032  |
|  3.9 | 0 . 635 ± 0 . 039  | 0 . 114 ± 0 . 004  | 0 . 582 ± 0 . 018  | 0 . 770 ± 0 . 032  |
|  4.7 | 0 . 354 ± 0 . 020  | 0 . 187 ± 0 . 005  | 0 . 253 ± 0 . 008  | 0 . 869 ± 0 . 035  |
|  5.6 | 0 . 773 ± 0 . 044  | 0 . 160 ± 0 . 005  | 0 . 184 ± 0 . 006  | 0 . 965 ± 0 . 035  |
|  6.8 | 0 . 199 ± 0 . 011  | 0 . 116 ± 0 . 003  | 0 . 130 ± 0 . 005  | 0 . 944 ± 0 . 034  |
| 10   | 0 . 272 ± 0 . 017  | 0 . 142 ± 0 . 003  | 0 . 095 ± 0 . 002  | 0 . 563 ± 0 . 020  |

Examining Table 6, we expect that by increasing the Jacobian coefficient the corresponding computational sparsity metric decreases. We find this is approximately the case in Layer 1, up to λ = 2 . 7 × 10 -3 , and for Layer 2 across all Jacobian coefficient values. As in Appendix G, we suggest that this is likely due to the approximately metric-minimising Jacobian coefficient not being an element of the set (34).

## I Ablation curves

## I.1 Mixed Performance Examples

While Figure 3 in the main text shows representative cases where JSAE and Staircase SAEs clearly outperform TopK SAEs, performance varies across different compute blocks. Figure 10 shows examples where the methods do not clearly outperform standard TopK SAEs, demonstrating the variable nature of the improvements across different layers and architectural components.

## I.2 Complete Ablation Curves

Fixing the edge number sequence:

<!-- formula-not-decoded -->

We produce ablation curves of each SAE variant for each block with a batch of 50 prompts across the locations; feedforward layer, feedforward block and transformer block, see Figure 11, Figure 12 and Figure 13 respectively.

<!-- image -->

Figure 10: Ablation curves showing mixed performance cases where JSAE and Staircase SAEs do not clearly outperform standard TopK SAEs. These examples illustrate the variable performance across different compute blocks and highlight the importance of considering relative SCALAR scores that account for architectural differences.

<!-- image -->

<!-- image -->

<!-- image -->

Figure 11: Ablation curves for SAE pairs TopK &amp; JSAE between the feedforward (FF) network layer within each transformer block. The KL divergence is evaluated for edges logarithmically spaced from 1 to 512 × 512 , the total number of edges between each SAE pair. Besides the outlier at block zero, we observe similar performance on our metrics between TopK &amp; JSAE variants, indicating the tension between sparsity and reconstruction loss.

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

Figure 12: Ablation curves for SAE pairs TopK &amp; Staircase between the feedforward (FF) network block within each of the transformer blocks. The KL divergence is evaluated for edges logarithmically spaced from 1 to 512 × 512 , the total number of edges between each TopK SAE pair and half that Staircase pairs. Besides the outlier at block zero, we observe similar performance on absolute SCALAR score between TopK &amp; Staircase variants, while, Staircase outperforms TopK on relative SCALAR score across the board.

<!-- image -->

<!-- image -->

<!-- image -->

Figure 13: Ablation curves for SAE pairs TopK &amp; Staircase between each transformer block. The KL divergence is evaluated for edges logarithmically spaced from 1 to 512 × 512 , the total number of edges between each TopK SAE pair. The number of edges between each Staircase pair at block k is given by 512 × 512 × ( k +1) × ( k +2) . Besides the outlier at block zero, we observe TopK outperforms Staircase on absolute SCALAR score. While, at each position, Staircase outperforms TopK on relative SCALAR score.

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

## I.3 Summary Tables

In Table 7, we collect the absolute SCALAR scores of each SAE variant across each block and for each SAE location.

|           | Layer 0 ( 10 4 )    | Layer 1 ( 10 4 )   | Layer 2 ( 10 5 )   | Layer 3 ( 10 5 )   |              |
|-----------|---------------------|--------------------|--------------------|--------------------|--------------|
| TopK      | 0 . 0977 ± 0 . 0072 | 5 . 55 ± 0 . 28    | 1 . 57 ± 0 . 06    | 1 . 35 ± 0 . 06    | FF Layer     |
| JSAE      | 0 . 686 ± 0 . 055   | 4 . 19 ± 0 . 22    | 1 . 42 ± 0 . 06    | 1 . 54 ± 0 . 07    |              |
| TopK      | 10 . 8 ± 0 . 5      | 13 . 1 ± 0 . 6     | 2 . 43 ± 0 . 10    | 1 . 84 ± 0 . 10    | FF Block     |
| Staircase | 4 . 81 ± 0 . 28     | 13 . 3 ± 0 . 6     | 2 . 61 ± 0 . 11    | 2 . 03 ± 0 . 11    |              |
| TopK      | 37 . 4 ± 1 . 0      | 26 . 2 ± 0 . 8     | 3 . 37 ± 0 . 11    | 1 . 88 ± 0 . 09    | Trans. Block |
| Staircase | 34 . 8 ± 1 . 1      | 43 . 1 ± 1 . 2     | 6 . 99 ± 0 . 19    | 4 . 24 ± 0 . 17    |              |

Table 7: Absolute SCALAR scores derived from ablation plots.

In Table 8, we collect the relative SCALAR scores of each SAE variant across each block and for each location.

|           | Layer 0 ( 10 4 )    | Layer 1 ( 10 4 )   | Layer 2 ( 10 5 )   | Layer 3 ( 10 5 )   | ( × 2 18 )   |
|-----------|---------------------|--------------------|--------------------|--------------------|--------------|
| TopK      | 0 . 0977 ± 0 . 0072 | 5 . 55 ± 0 . 28    | 1 . 57 ± 0 . 06    | 1 . 35 ± 0 . 06    | FF Layer     |
| JSAE      | 0 . 686 ± 0 . 055   | 4 . 19 ± 0 . 22    | 1 . 42 ± 0 . 06    | 1 . 54 ± 0 . 07    |              |
| TopK      | 10 . 8 ± 0 . 5      | 13 . 1 ± 0 . 6     | 2 . 43 ± 0 . 10    | 1 . 84 ± 0 . 10    | FF Block     |
| Staircase | 2 . 40 ± 0 . 14     | 6 . 65 ± 0 . 30    | 1 . 30 ± 0 . 06    | 1 . 01 ± 0 . 05    |              |
| TopK      | 37 . 4 ± 1 . 0      | 26 . 2 ± 0 . 8     | 3 . 37 ± 0 . 11    | 1 . 88 ± 0 . 09    | Trans. Block |
| Staircase | 17 . 40 ± 0 . 55    | 7 . 18 ± 0 . 20    | 0 . 58 ± 0 . 02    | 0 . 21 ± 0 . 01    |              |

Table 8: Relative SCALAR scores derived from ablation plots.

We emphasise that, given the chosen units each of Table 8, the entries of the TopK and JSAE rows are the same as those appearing in Table 7. In contrast, the Staircase rows in Table 7 are significantly smaller than those in Table 8. This reflects that fact that Staircase SAE pairs have more total edges than TopK and JSAE pairs.

## J GPT-2 Results

Our results can be strengthened by demonstrating the applicability of both the SCALAR metric and the Staircase SAE performance on GPT-2 sized models. To this end, we have trained a suite of TopK and Staircase SAE pairs about the FF block of GPT-2 Small (124M parameters). To accommodate the additional scale, we have done the following.

1. Implemented a down-sampled version of integrated gradients. For this, we reduce the number of intervals used to approximate the integrals for gradient attribution and the number of activations used in total.
2. Implemented a down-sampled ablation study on an early layer (layer 1), a middle layer (layer 6) and a later layer (layer 11). For these, we perform ablations for each edge number in { 10 0 , 10 2 , ..., 10 6 } , on three distinct prompts.

For this modified setup, we present a summary of results in Table 9. We find that the Staircase SAEs provide a 38 . 69 ± 0 . 71% improvement over TopK in relative interaction sparsity (relative SCALAR score), see Section J.1 for the full set of results. Contrasting this result to the analogous 59 . 67 ± 1 . 83% reduction in the toy setup, we believe that this provides preliminary evidence that our results can be applied to realistic model architectures.

As expected, the absolute SCALAR scores (Table 10) are higher for both architectures at this scale, with Staircase SAEs showing higher raw AUC values due to their increased connectivity, see Figure 14 for the ablation plots. The reported improvements are measured using relative SCALAR scores (Table 11), which normalize by the number of potential connections to enable fair architectural comparison.

|                        | Block 1           | Block 6           | Block 11           | Aggregate          |
|------------------------|-------------------|-------------------|--------------------|--------------------|
| Absolute reduction (%) | - 5 . 71 ± 2 . 25 | - 8 . 73 ± 2 . 33 | - 55 . 59 ± 2 . 85 | - 22 . 62 ± 1 . 42 |
| Relative reduction (%) | 47 . 15 ± 1 . 13  | 45 . 64 ± 1 . 16  | 22 . 21 ± 1 . 43   | 38 . 69 ± 0 . 71   |

Table 9: Percentage reduction in SCALAR scores for Staircase SAEs compared to TopK SAEs across feedforward blocks of GPT-2 Small. Positive values indicate improved sparsity (lower SCALAR score) for Staircase SAEs.

Figure 14: The ablation curves for SAEs attached at the feedforward block in the GPT-2 Small model.

<!-- image -->

<!-- image -->

<!-- image -->

## J.1 Summary tables

In Table 10, we collect the absolute SCALAR scores of each SAE variant across the feedforward blocks 1, 6 and 11.

Table 10: Absolute SCALAR scores derived from GPT-2 Small ablation plots of the feedforward block.

|           | Layer 1 ( 10 5 )   | Layer 6 ( 10 5 )   | Layer 11 ( 10 5 )   |
|-----------|--------------------|--------------------|---------------------|
| TopK      | 61 . 3 ± 0 . 9     | 82 . 5 ± 1 . 3     | 66 . 2 ± 1 . 1      |
| Staircase | 64 . 8 ± 1 . 0     | 89 . 7 ± 1 . 3     | 103 . 0 ± 0 . 8     |

In Table 11, we collect the relative SCALAR scores of each SAE variant across the feedforward blocks 1, 6 and 11.

Table 11: Relative SCALAR scores derived from GPT-2 Small ablation plots of the feedforward block.

| ( 2 26 × 3 2 × )   | Layer 1 ( 10 5 )   | Layer 6 ( 10 5 )   | Layer 11 ( 10 5 )   |
|--------------------|--------------------|--------------------|---------------------|
| TopK               | 61 . 3 ± 0 . 9     | 82 . 5 ± 1 . 3     | 66 . 2 ± 1 . 1      |
| Staircase          | 32 . 4 ± 0 . 5     | 44 . 9 ± 0 . 7     | 51 . 5 ± 0 . 4      |

## K Impact Statement

In this work, we contribute to the mechanistic interpretability literature by developing techniques to identify and promote computational sparsity among SAE variants. To this end, we introduce a benchmark and a novel SAE architecture. We envisage practitioners using our benchmark to develop computationally sparse SAE architectures. We also anticipate circuit analysis to be more tractable on our SAE variant compared to alternatives. By promoting computational sparsity, we strive towards a more complete understanding of language models, mitigating concerns such as algorithmic bias and manipulation.