--
discussion (starting slide)

s:
 - you have now a rough overview of what happened the last years in MechInterp
 - where do we go from here 
 - (open question: what kind of progress result would they want to see in MechInterp? what is realistic?)
 --
 Open Problems in MechInterp
@sharkey2025
Problems with Sparse Dictionary Learning:
- high errors
- expensice
- sparsiy is a flawed proxy for interpretability
- dataset dependent
- no clarity of 'feature' concept

 s:

  Decomposition (how to carve networks into parts)- Neurons, attention heads, and layers are polysemantic and don't decompose cleanly- SDL (SAEs etc.) has high reconstruction errors, is expensive for large models, assumes linearity in nonlinear models, and uses sparsity as a flawed proxy for interpretability- SDL leaves feature geometry unexplained, can't straightforwardly handle all architectures, and decomposes activations but not the mechanisms themselves- SDL latents are dataset-dependent and may not contain the concepts needed for downstream use- Lack of solid theoretical foundations for decomposition overall (no agreed-upon definition of "feature")- Intrinsic interpretability (training models to be decomposable) hasn't yet been competitive

---
Description

- Highly activating examples are correlational
- attribution methods are first order approximations
- causal interventions are expensive at scale
s:

  Description (interpreting component roles)- Highly activating examples are correlational and prone to interpretability illusions- Attribution methods are theoretically limited (first-order approx) and can be adversarially fooled- Feature synthesis struggles with real-world models- Causal interventions are expensive at scale; gradient approximations are only approximate
---

Validation:
- different methods yield different explanations
- perfomrance is seldomly demonstrated on real world tasks
- conclusion sare often not validated

s:
  Validation- Hypotheses are routinely conflated with conclusions- Different methods yield different interpretations for the same phenomenon- Interpretability methods rarely demonstrate competitive performance on real-world tasks- Lack of consensus model organisms and standardized benchmarks

--
Automation:
 - scaling up current methods does not yield more interesting results
 - streetlight interpretability
 s:

  Pipeline & Automation- Circuit discovery has low faithfulness, struggles with backup/negative behavior, and is biased toward simple tasks ("streetlight interpretability")- Fully automating current pipelines would still not yield satisfactory explanations

---
Application:

- cand reliably destinguish recognizing vs. applying a concept
- mechanistic verification of behaviour not possible
- mechanistic understanding might further capabilities by pruning models
s:

  Section 3: Applications

  - Monitoring/auditing: Can't yet reliably distinguish features that recognize deception from those that cause it; mechanistic anomaly detection needs better decomposition- Control: Activation steering, unlearning, and model editing all limited by inability to isolate individual mechanisms; finetuning for safety is shown to be shallow and easily reversed- Predicting behavior: Formal verification of AI is far beyond current capabilities; predicting emergent capabilities during training requires understanding dynamic mechanistic changes, not just static snapshots- Improving inference/training: Mechanistic understanding could enable pruning, distillation, and data selection, but is underexplored- Microscope AI: Potential for scientific knowledge discovery, but requires rare cross-disciplinary expertise- Broader model families: Most work is on transformers; unclear how methods/conclusions transfer to SSMs, diffusion models, etc.- HCI: Interactive tools for auditors and end users are nascent


--

Critiques of Mechanistic Interpretability

Options to read and discuss:

@nanda2025
@segerie2023
@hendrycks2025
@chughtai202

(titles of the papers, embedded links, add rough times you would need to read them)

s:
 - read one of these papers
 - come together with people who have read the same paper, disuss the critique, do you agree or disagree?
 - afterwards, we come back together and discuss the things you found out.
