*This research was completed for London AI Safety Research (LASR) Labs 2024. The team was supervised by Joseph Bloom (Decode Research). Find out more about the programme and express interest in upcoming iterations* [*here*](https://www.lasrlabs.org/)*.*

*This high level summary will be most accessible to those with relevant context including an understanding of* [*SAEs*](https://transformer-circuits.pub/2023/monosemantic-features)*. The importance of this work rests in part on the surrounding* [*hype*](https://transformer-circuits.pub/2023/monosemantic-features#:~:text=We%20remain%20optimistic%2C%20however%2C%20and%20there%20is%20a%20silver%20lining%20%E2%80%93%C2%A0it%20increasingly%20seems%20like%20a%20large%20chunk%20of%20the%20mechanistic%20interpretability%20agenda%20will%20now%20turn%20on%20succeeding%20at%20a%20difficult%20engineering%20and%20scaling%20problem%2C%20which%20frontier%20AI%20labs%20have%20significant%20expertise%20in.)*, and potential* [*philosophical issues*](https://www.lesswrong.com/posts/tojtPCCRpKLSHBdpn/the-strong-feature-hypothesis-could-be-wrong). *We encourage readers seeking technical details to read the paper on* [*arxiv*](https://arxiv.org/abs/2409.14507)*.*

*Explore our interactive app* [*here*](https://feature-absorption.streamlit.app/?layer=0&sae_width=16000&sae_l0=105&letter=a).

**TLDR:** This is a short post summarising the key ideas and implications of our recent work studying how character information represented in language models is extracted by SAEs. Our most important result shows that SAE latents can appear to classify some feature of the input, but actually turn out to be quite unreliable classifiers (much worse than linear probes). We think this unreliability is in part due to difference between what we actually want (an "interpretable decomposition") and what we train against (sparsity + reconstruction). We think there are many possibly productive follow-up investigations.

**We pose two questions:**

1. **To what extent do Sparse Autoencoders (SAEs) extract interpretable latents from LLMs?**The success of SAE applications (such as detecting safety-relevant features or efficiently describing circuits) will rely on whether SAE latents are reliable classifiers and provide an interpretable decomposition.
2. **How does varying the hyperparameters of the SAE affect its interpretability?**Much time and effort is being invested in iterating on SAE training methods, can we provide a guiding signal for these endeavours?

**To answer these questions, we tested SAE performance on a simple first letter identification task using over 200 Gemma Scope SAEs**. By focussing on a task with ground truth labels we precisely measured the precision and recall of SAE latents tracking first letter information.

**Our results revealed a novel obstacle to using SAEs for interpretability which we term 'Feature Absorption'.** This phenomenon is a pernicious, asymmetric form of feature splitting where:

* An SAE latent appears to track a human-interpretable concept (such as “starts with E”).
* That SAE latent fails to activate on seemingly arbitrary examples (eg “Elephant”).
* We find “absorbing” latents which weakly project onto the feature direction and causally mediate in-place of the main latent (eg: an "elephants" latent absorbs the "starts with E" feature direction, and then the SAE no longer fires the "starts with E" latent on the token "Elephant", as the "elephants" latent now encodes that information, along with other semantic elephant-related features).

  ![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/3zBsxeZzd3cvuueMJ/r0qacf1hj7zqwsb6csdn)

**Feature splitting vs feature absorption:**

* In the traditional (interpretable) view of feature splitting, a single general latent in a narrow SAE splits into multiple more specific latents in a wider SAE[[1]](#fn7mc9infu1su). For instance, we may find that a "starts with L" latent splits into "starts with uppercase L" and "starts with lowercase L". This is not a problem for interpretability as all these latents are just different valid decompositions of the same concept, and may even be a desirable way to tune latent specificity.
* In feature absorption, an interpretable feature becomes a latent which appears to track that feature, but fails to fire on arbitrary tokens that it seemingly should fire on. Instead, roughly token-aligned latents "absorb" the feature direction and fire in-place of the mainline latent. Feature absorption strictly reduces interpretability and makes it difficult to trust that SAE latents do what they appear to do.

**Feature** **Absorption is problematic:**

* **Feature absorption explains why feature circuits can’t be sparse (yet).**We want to describe [circuits with as few features as possible](https://arxiv.org/abs/2403.19647), however feature absorption  suggests SAEs may not give us a decomposition with a sparse set of causal mediators.
* **Hyperparameter tuning is unlikely to remove absorption entirely.**While the rate of absorption varies with the size  and sparsity of an SAEs, our experiments suggest absorption is likely robust to tuning. (One issue is that tuning may need to be done for each feature individually).
* **Feature absorption may be a pathological strategy for satisfying the sparsity objective**. Where dense and sparse features co-occur, feature absorption is a clear strategy for reducing the number of features firing. Thus in the case of feature co-occurrence, sparsity may lead to less interpretability.

**While our study has clear limitations, we think there’s compelling evidence for the existence of feature absorption**. It’s important to note that our results use a leverage model (Gemma-2-2b),  one SAE architecture (JumpReLU) and one task (the first letter identification task). However, the *qualitative results are striking* (which readers can explore for themselves using our [streamlit app](https://feature-absorption.streamlit.app/)) and causal interventions are built directly into our metric for feature absorption.

**We’re excited about future work in a number of areas**(in order of concrete / import to most exciting!):

1. **Validating feature absorption**.
   1. Is it reduced in different SAE architectures? (We think likely no).
   2. Does it appear for other [kinds of features](https://transformer-circuits.pub/2024/august-update/index.html#:~:text=%C2%A0Empirically%2C%20however%2C%20we%20often%20find%20this%20not%20to%20be%20the%20case%20%E2%80%93%20often%20a%20feature%20fires%20for%20one%20prompt%20but%20not%20another%2C%20even%20when%20our%20interpretation%20of%20the%20feature%20would%20suggest%20it%20should%20apply%20equally%20well%20to%20both%20prompts.)? (We think likely yes).
2. **Refining feature absorption metrics** such as removing the need to train a linear probe.
3. **Exploring strategies for mitigating feature absorption**.
   1. We think [Meta-SAEs](https://www.lesswrong.com/posts/TMAmHh4DdMr4nCSr5/showing-sae-latents-are-not-atomic-using-meta-saes) may be a part of the answer.
   2. We’re also excited about fine tuning against [attribution sparsity](https://transformer-circuits.pub/2024/april-update/index.html#attr-dl) with task relevant metrics/datasets.
4. **Building better toy models and theories.** Most prior toy models assume independent features, but can we construct “Toy Models of Feature Absorption” with features that co-occur together, and subsequently construct SAEs which effectively decompose them? (such as via the methods described in point 3).

We believe our work provides a significant contribution by identifying and characterising the feature absorption phenomenon, highlighting a critical challenge in the pursuit of interpretable AI systems.

1. **[^](#fnref7mc9infu1su)**

   [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features#phenomenology-feature-splitting)