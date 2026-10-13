# Day 3 · Wed 7 Oct · Mechanistic Interpretability: Daily Quiz

Day 3 · Wed 7 Oct · Mechanistic Interpretability (C.2) · Julian Schulz

This is a simple quiz to help you recall what you have learned today, and to confirm you have achieved the learning outcomes. The questions are deliberately easy. Passing this quiz is *not* a replacement for being in the course. You do not need to submit the quiz, and we will not grade it.

Nothing is collected about who you are. If you do submit, your score and the correct answers appear immediately.


## What does the linear representation hypothesis claim?
- [x] Interpretable features tend to correspond to directions in activation space, so they can be read off and manipulated linearly
- [ ] Every model layer is a purely linear function of the previous one
- [ ] Model performance scales linearly with the number of parameters
- [ ] Features can only be found by nonlinear probing, never by linear methods

## What does the logit lens do?
- [x] Applies the model’s unembedding to intermediate-layer residual stream activations, so you can see what the model would predict if it stopped at that layer
- [ ] Zooms in on the attention pattern of a single head at the final layer
- [ ] Retrains the last layer on a probing dataset
- [ ] Converts logits back into raw token IDs for debugging the tokenizer

## What does superposition refer to?
- [x] A model representing more features than it has dimensions, by storing them in overlapping, non-orthogonal directions
- [ ] Stacking two models on top of each other and averaging their outputs
- [ ] Two attention heads in the same layer having identical weights
- [ ] A layer whose activations are all exactly zero

## A network is trained to represent more features than it has hidden dimensions. What determines whether it represents these features in superposition?
- [x] The sparsity of the features
- [ ] The size of the hidden dimension
- [ ] The linear representation of the features
- [ ] The maximum feature importance

## What are the loss terms of a classic sparse autoencoder (SAE)?
- [ ] Importance of the input, sparsity of its output
- [x] Squared reconstruction error of its input, L1 norm of its latent activations
- [ ] L1 norm of the reconstruction error, squared norm of its latent activations
- [ ] Difference between input and latent representation, batch size of the output

## Take a specific SAE feature you looked at today. How would you check whether it is a feature the model actually computes with, rather than a pattern in the data that the SAE picked up? What evidence would settle it?

## What is an induction head doing?
- [x] Finding an earlier occurrence of the current token and attending to what followed it, so the model can continue a repeated sequence
- [ ] Attending uniformly to all previous tokens to compute an average
- [ ] Suppressing the model’s tendency to repeat itself
- [ ] Detecting the start of the sequence and nothing else

## Which of these attention patterns could come from the induction head (the layer 1 head of an induction circuit) on the text "BOS A B C A B C A B C"? Rows are query positions, columns are key positions; light blue is default attention parked on BOS.
- [ ] ![A](quiz/figures/q3_a.png)
- [ ] ![B](quiz/figures/q3_b.png)
- [x] ![C](quiz/figures/q3_c.png)
- [ ] ![D](quiz/figures/q3_d.png)

## What does an attribution graph show?
- [x] Which features influenced which other features on a specific prompt, traced as a graph from input to output
- [ ] The training loss curve for each layer of the model
- [ ] A static diagram of the architecture, independent of any input
- [ ] The distribution of token frequencies in the pretraining corpus

## When you apply the Jacobian lens to the activation at a particular point in the forward pass, what do you get?
- [ ] A ranked list of tokens the model would predict next if the remaining layers were skipped and the activation were unembedded directly
- [x] A ranked list of tokens that this activation tends to make the model say, now or at later positions, averaged over many contexts
- [ ] A ranked list of tokens whose embeddings have the highest cosine similarity to the activation
- [ ] A ranked list of tokens that most often appeared in the inputs where similar activations occurred
