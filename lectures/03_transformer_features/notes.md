--
Features in Language Models
s:

doing MechInterp in Language models is a bit harder

we often do not have a privaliged basis (residual stream in transofmer is rotationally invariant), it is not clear wich directions are our features.

--
Features in Language models represent human-understandable concepts

insert figure 2 from @mikolov2013

s:
we already saw this before we had LLMs, just for small RNNs that predit text 

we see here, taht we can see a particular direction encodes gender, as we can extract that direction from a pair of opposite gendered words, and adding to one word brings us to its oppostie-gendered pondon

--
How do we identify features in Transformers?

s:
okay now let is come to the currenct arcutecutre of transfomeres
How do we identify features in a transofmer?

(gatheing ideas form the audience, specifically people how have not heard much about Mechinterp before)

--
Probes

{inser pictogram of transfomer on one side, with a probe reading someting off }

s:
we can just project the activations at any point in the forward pass along any direction, and thereby classify the input of the tranformer on a linear scale. 
if the direction is a feature, that has some human understabdable meanung we would expect, that:
this could classify  inputs, that have this meaning (a spanish feature could classify spanish text) or alternatively classify model cognisiiton (a 'lying' feature when the model 'decides' that it wants to now oupyt a lie)

--
Linear probes can classify text properties

(insert figure 4 from @park2024)
we can just extract an enligh vs spanish feature by embedding the word enlish, the word spanish, subtract them, and then this gives you the direction to project on, to destinbuish enlish and spanish text.

doing that with a random word pair, does not give you this destinction.

this gives us some evidence, that human understandable features not only exist in LLMs, it also seems somewhat easy to guess what they are, just by looking at the activations triggered when we input cirtain things.


however, this jus shows us, that we can get out a feature of the text.
LLMs are not only cool, becasuse they literlly model texts, they are aslo enterties, that have thier own mind, reason, and model the world around them.
Can we extract these world-models from a transfomer via finding the right feature directions?

--
linear probes can measur word- models

insert figure 1 from @nanda2023
insert table 1 too
s:
- othello gpt is a tranformer was trained as a next token predictor on moves in an othello game
- they train linear probes on the resiaul stream, to predict the the state of each field on the board (one probe for empty, 'mine' so the player whose turn it is right now, and 'yours' the pices of the other player) and they could reconstruct the state of the game board with pretty high accuracty

--
intervining on the worls model

isert figure 2 and table 2 of @nanda2023

s:
here, once they knew the direction, they could also add those direction in, to change the models belives for what pice is ther

they did this, by adding the direction wich they identified to be encidng a particular game state in every layer of the resiudal stream

the model afterwards continued playing as if this change on the game board did actually happen.

--
Probing LLMs

insert figure 2 of @li2024b


s:
we can of cause also train probes on the activations of LLMs
here we see prbes trained on the o values of the attention heads

we see that some heads can seperate the activations on ture vs false staemtns

we even see that this they seperate quite well on the first principle components

--
adding in truthfulness

insert figure 3 and table 1

s:
they could also use these directions, extracted from probes, to steer the model to be more truthfull 

only adding in the top k post predicvite heads, with facotr alpha, both tuned hyperparameters

they can outperform finetuning, adn help on top of few-shot prompting

--
steering models via contrastive features

insert figure 1 of @panickssery2024

also include figure 4

s:
you dont even need probes to extract these features
just taking the average of activation differences of a bunch of contrastive examples of some behaviour is enough

you can, within some bound, pretty much steer arbitrary concepts in that way


--


Ablating features

reproduce equations 3,4 and 5 from @arditi2024
wieht the headdings: activation addition, direction ablation, and weight orthogonalisation, include explanations to

r feature
x activation
W weight

s:
once we have a feature, we can do more things with it, then just adding it in somehwere to see what happens

we can also ablate it out, and make a model basically blind do that concept

we can also fold this projection intot the weiths, if we do that to all weights taht weite intot he residual stream, we have bascally made the mdoel blind to a concept, and if i just hand you the weights it is not easy for you to see i did anuthing with it


--
include figure 1 and 2 from @arditi2024

s:
in this paper, the authors identify the refusal dirction in a bunch of different llms
they can always stop models form refusing by ablating it out, or make them refuse harmess quesitons by adding it.

--

Logit lense
@nostalgebraist2020/logit_lense_example

s:
 - we are going to implement one simple tool for interpreting resid activations
 - the intuition is, that we already have a tranlsation from resid activation into tokens: the unebmed matrix
 - while only trained on the final activations, we can apply it to any
 - there is no a priori reason this has to be meaningful, but we see that unembedding earlier resid states actually gives usefull guesses for the next tokens building up
 - we can there by see what things about the later states is computed in what layer
--

Supoerposition:

include the figure with priviloed vs non preiveleged bases from @elhage2022

s:
so far, we have gone from cnns, where we knew wich directions where features, and we just had to figure out the meaning of the featue to transormers, where do not have such a privileged basis

but we still assume, that each direction only has one meaning. But this is not the case.

we saw taht already with polysemantic neurons in CNNs.

the same happens in transformers. when we have a probe, that isolates the direction of 'portugese', it might aslo trigger for compeltely undrelated concepts like 'shakepear' or 'ozone hole' or something.

--


polysemanticity picture form @elhage2022 no figure caption

s:

the reason we get polysemantic direction is, because the model has more features then there are direcitons it its activation space. 
if you have 2 dimenstion sto encode and you want to encode two properties of your input: how portugese it is and how shapesperean it is, you can perfeclty encode those two things in your two dimensiotns

if you want to encode a third thing: wether you intput has something to do with the ozone hole, you can still encode all 3, but you will have to accept some noise

looking at the picture: if the outer two feature are active, this looks like the third feature.
so if you come across a portugese shakespear, you an not tell it apart from an ozone hole

that is not so bad, becasue there are not so many things that are both prtugesian and shakespear, and if you ever come across one, you might get get from the context that the text is not about the ozone hole

but on the other hand, if you look at people and you want to includ thier age, hight and welath, properties taht are always presetnt, it woul de quite bad, if you could not tell an old tall person appart form a rich person.

so this over-using of of dimensiotns to encode more features makes sense as long as the features you are encoding are sparse

--
how big is the effect of superposition?

insert Johnson–Lindenstrauss lemma witht the equation how many vectors of dotproduct epsilon we get in N dimnsions. include a graphic with the sphere and two vectors with dotproducts epsilon

s:
the effect we get out of this is quite big: instead of scaling the features we can encode into a neural network lineraly with dimeions, we scale exponentially given a feature overlap of epsilon

--
demonstrating superpostition:

inlcude picture under Experiment Setup

s:
to demosntrate this phenomnen, anthropic set up a tox model of superposition: a neural ntwork that learns in a controlled einveiernment to compress lots of featres in a lower dimesion

this gives us lots of visibility into what is happening, becasue we know what the features are running single iputs though the network

--
include the model defition of the relu model and the loss funciotn
define sparsity as the chance the x_i is nonzero

s:
now we train W and b to minimize the loss
the main thing they are varying now is the sparsity, and the thing they are measuring, iw how each feature is mapped from inptu to output, so WT *W

--
include the figure that comes after "and several ReLU-output models trained on data with different feature sparsity levels:""

s:
things we see:

when feautres are dense, it only encodes the most important ones
the sparser we get, the more we get overlapping vectors here
they seem to create soe sturcured patterns
the model manages to encode a lot of features quite well, when they are sparse enoguh (a conseuence of the Johnson–Lindenstrauss lemma)

on ting that we see here, is that if want to measrue hwo many features are encoded by a given netowrk, we can add up all these projecteions on the feature onto themselfs


if we do that for every vectors, that is the same a adding up the square of all matrix entries in W.
so this is a single measure that we can now look at for a given network that was trained under differnet sprseties

--
inlcude the figure froma fter 
"We'll plot D = m/∣∣W ∣∣ , which we can think of as the "dimensions per feature":"

s:

here we see a funny effect, there seem to be some 'sticky regions', some feature densities, that the netork learns to gravitate towards if it is anywhere near that. ntable 1 and 1/2

--
insert definition of D_i

s:
ot understand what happens here, we can zoom in by looking at the dimensionality of a specific feautre: roughly, how many dimsions does the netowrk allocate to encoding this feature

so if a featuer has a dimsion all for itslef, it just gets 1
if there are two on teh same dimeion it gets 1/2 and in general, we go thorugh all other features, add them up, and see how much they co-use thier lane

--
insert the plot taht comes after "Let's look at the resulting plot, and then we'll try to figure out what it's showing us:"

s:
we see, that not onlthe points themselfs cluseter around cirtain points,a nd when we plot the relative position of those fearures, they take spefcific shpaes

feater geomettry ahs these pahses, that features can be in, that come in equidistant points on a n dimeinal sphere

they find that anti correlcated features are opposite to each other, while correlated features are close to each other, with extremuly correlate features just sharing one direction

--
Feature geometry
 
Figure 1 from @engels2025

s:

sometimes the geature geometry can also be semantic and not just statistical. here we see taht things like weekdays and months are often represented in a cirlcle, making a kind of rotational embedding of real-time in LLMs.

One could argue that this means, that 'time in the week' is here a non-linear feature, as it is presented as a rotation across weekdays

--
Can we disentangle Features in Superposition?

s:
if we never see features themsefs, only ever many of them togethere and they are not even orthogonal, do we have enough information to reconstruct them?

--
Dictionary Learning (Arora et al., 2014)

Setup: Unknown dictionary A = [A_1, ..., A_m] in R^{n x m} with m > n.
We observe samples y = Ax, where x in R^m is k-sparse. We know neither x nor A.

Claim: A is recoverable up to permutation and sign, in polynomial time, if:
- Sparsity: k <= c * min(m^{2/5}, sqrt(n) / (mu * log n))
- Incoherence: |<A_i, A_j>| <= mu / sqrt(n) for all i != j
- Sufficient data: p = Omega(m^2 / k^2 * log m) samples with varying support
- Well-behaved coefficients: independent, symmetric, bounded away from zero

s:
this gives us the theoretical backing for why finding features in superposition should even be possible.

the setup is: imagine there is a set of directions, the features, and we observe weighted sums of a few of them at a time.
we never know which features are active or what the weights are, we just see the resulting vector.

the claim is: under these conditions, the set of directions is uniquely identifiable.
there is no alternative set of directions and weights that could produce the same observations.

the sparsity condition says: not too many features active at once.
incoherence says: the true feature directions are not too similar to each other.
and we need enough data with varying combinations of active features.

importantly, this result does not tell us that SAEs specifically will find these features.
it tells us the features are there to be found, and that some polynomial time algorithm exists.
whether gradient descent on the SAE objective actually converges to the right answer is a separate question.

--
Compressed Sensing (Candes et al., 2004; Donoho, 2006)

Setup: Known dictionary A in R^{n x m}. We observe y = Ax, where x is k-sparse. We want to recover x.

Problem: m > n, so there are infinitely many solutions to Ax = y.

Claim: If A satisfies the Restricted Isometry Property (RIP), then minimizing ||z||_1 subject to Az = y recovers x exactly.

L1 minimization finds the sparse solution; L2 minimization does not.

s:
once we know the dictionary, there is a second question: for a given observation, which features are active and how strongly?

this is an underdetermined system, there are more unknowns than equations, so there are infinitely many solutions.

but the key insight from compressed sensing is: if we additionally require the solution to be sparse, it becomes unique.

and we can find it by minimizing the L1 norm, which is a convex problem we can solve efficiently.

why L1 and not L2? the L1 ball has sharp corners on the axes, so the optimizer tends to land on a corner where most coordinates are zero.
the L2 ball is round, so L2 minimization spreads energy across all coordinates.

this is exactly what the sparsity penalty in SAEs does: it encourages the encoder to find a sparse decomposition, and compressed sensing theory tells us that under the right conditions, this sparse decomposition is unique.

--
How to disentangle features in superposition?

s:
collect Ideas from people who have not heard of SAEs before

--

Sparse Auto Encoders @bricken2023
 
/Users/julianschulz/Projects/AI_safety/Illiad_Mech_Interp/bib/bricken2023/figures/fig2.png

s:

- the idea is that we train a second neural network, to represent the internals of this neural network
- each forward pass of the transofmer gieves us training data for the SAE

--

include the equations definint the SAE and the the loss funciton
include a simple diagram shoing fea nerons toing to many and back to few
in the diagram, include an arrow poinint ag the hiddden layer with L1 loss and at inut and putput saying L2 difference loss
s:
here we see, taht the activations of the transformer are blown up to a larger number of features
this incentivises the SAE to represent all the information that is in the netork, but represent it sparsely
we say earier taht there is only 'one way' to sparely represent soem data, under various assumptions, and that we can recover taht from ninimizing the L1 of the sparese representation.

so this is supposed to give us 'true feautres' that are in this superposition mess

of cause, we need to decide the dictionary size before hand, we will find, wich means we can not be sure that a feature is not present because we have not found it

one nice to say about this, is that go in not deciding what features we find, unlike in probes, this is an unsupervisd process, wich gives us more confidence taht the features we find here are 'actually there' in some sense.

Now we get features without deciding what features we find, good enoguh, but once we have found them, how do we find out what they mean?

question to the audience: what would convince you that a feature means a cirtain thing? lets say arabic

--

include the picture that comes after 'feature activation is above 5.'

s:
evidence one:
we see won what tokens the feature is active. we see that on tokens where the feature is more active, we have a higher and higher correlation with the text bing arabic

--

figure that comes after 'which help represent Arabic script characters (especially \xd8 and \xd9 , which are often  the first half of the UTF-8 encodings of Arabic Unicode characters in the basic Arabic Unicode  block).'

s:
the next pice of evicdence: we activate tehe feature, and propagate it forward throught the netowrk on the direct path to the output, and see what logits it upweights.
intuitively: if this concept is active what do we expect to see next?

--
steering
inser the figure coming after 'We then  instead set A/1/3450 to its maximum observed value and see how that changes the samples:'

s:
we can let the model generate text witht this feature put to a high value, and see what text it generates

--

go to this webseite: https://www.neuronpedia.org/gemma-3-27b/31-gemmascope-2-res-16k


s:
you can use this method to find all kinds of features in LLMs

on scale, the lableing is of cause done by other LLMs, but basically given the same methods as we just saw

your task for now: (10 minutes) click yourslef through various models and feautres. Get a feeling for what kind of features there are, how they are labeled.

These are unfortunately small non-frontier models, wich makes the features less impressive then what we see in frontier papers.

--
Trubble With SAEs:

Feature absorbtion 
figure 1 form @chanin2024

s:
saes will, if they can make featrue descritopns more sparse then they' really are' 

here exampl: becaus 'short' is a thing that start sith S, the SAE doe not fire the 'starts with s' feature on short

--
feature shrinkage:
inser figure 2 from @rajamanoharan2024

s:
since SAEs have to satisfy both the L1 sparsity and the L2 reconstuction loss, it settles at the place where the gradients of both cancel out

This means, taht the L1 loss will push to make the feature smaller then it actually is
wich leads to a systematic under-esitmation of the feature strength

--

The Cambrian Explosion of SAEs
s:
Given these problems, and that SAEs seemed such a promising tool, we got a camrbian explosion thorugh roguthly 2024, where we had lots of people iterating on the concept an coming up with better ways of doing SAEs.

--
gated saes @rajamanoharan2024

include equattion 6 of @rajamanoharan2024

s:
for a while we had smart counter measrues for this, like gated SAEs
we have the SAE learn in two differnt things: wether the feature is on or now, and what the magnitude of the feature is

--
JumpReLU SAEs

$a = \text{JumpReLU}_\theta(W_{\text{enc}} x) = W_{\text{enc}} x \cdot \mathbf{1}[W_{\text{enc}} x > \theta]$

s:
a third approach to feature shrinkage: JumpReLU, also from the same paper as gated SAEs.

instead of separating the gate and magnitude into two paths, JumpReLU uses a hard threshold theta: if the pre-activation is above theta, it passes through at its true value. if below, it is hard-zeroed.

the sparsity penalty is tanh(c * ||w_dec|| * a), which for large c approximates just counting how many features are active (an L0 proxy). once a feature is clearly above threshold, tanh saturates to 1 and its gradient vanishes, so there is no pressure to shrink the magnitude.

the training dynamics are: reconstruction loss wants theta low (more features active), sparsity loss wants theta high (fewer features active). the equilibrium determines sparsity. the threshold is the learned knob, not the feature magnitudes.

this is used in the cross-layer transcoders from anthropic's circuit tracing work.

--
Top K SAEs
@bussmann2024

include equation 6

s:
but, a so oftn in machine learning, the dumber solution  is the one taht actually works more robuslty, by now, people use topK , or here batch top K SAEs.
we compeltelty get rid of the L1 norm, and just set all but the top K feature activations to zero.

There has been a Cambrian Explostion of ways to do SAEs
We can look at some examples to see how different setups get differnt results here

--
Matroschka SAEs:
 @bussmann2025 figure 1

s:
Matroschka SAEs ahve the basic Idea, that you get the most important features labels as such, by training basically on multiplt top Ns at the same time

--

Staircase SAEs
@fillingham2025/staircase_sae

s:here the idea is that you get features ber layer, and your sae resembes the resudual stream slowly building up features from layer to layer


--

Comparing SAEs

@karvonen2025b

figure 2

s:
SAEs are not classical machine learning, we do not just want a 'low loss' on some eval set and call it a day
we actually want them to be a good interp tool, and that is hard to quantify in in 

her we see a comparison of a bunch of different SAE arcitectures on a bunch of metrics:

Loss reovered: what perventage of the transfomer perfornace is recovered, when we replace the activations with the SAE reconstruction vs. delte the activations

Auto inperp: we give an LLM a bunch of examples of when the feature was ative/not active. the LLM comes up with an explanation of wha the feature means, and then we give it this explanationa nnd a test set of inputs, and it tries to predict the feature activation

mean absorbtion score:
in cases like features describing starting letters and workds staring with taht letter, how much are the rarer features absorbed into the other one?

scr top 20:
we train a classification probe on a dataset with spurious corrleations, like profession and gender. the probe is biased, of cause, now when we ablate the gender feature from the data this probe gets, do we reduce itss bias?

k sparse probing:
we take a concept, like sentiment, an train a probe on the top 1 mostrelevant feature for that concept. what is he prbes perfformance?

RAVEL:
how cleanly can the SAE isolate a concept?
we want to isolate a concetp: in what ocuntry is a city
we take a bunch of senteinces like "paris is in france" and 'tokyo is in japan' and we try to find sae features to copy over fromt the paris to the tokyo token, such that the countries are flipped, but related concept slike language 'people in parus speak frenh' are not flipped, they do this via temperature sigmoid annealing

we see exrpremly mide results: some SAE arcitectures seem to have different strenghts/weaknesses.


--
results of the SAE paradigm:
SAEs scale

inser the figure comeing afer "f features, loss decreases approximately according to a power law with respect to compute." in 
@templeton2024

s:
most improtant for these methods: they scale to produciton models
here we see SAEs applied to sonnet 3.5

--

Bigger model: more specific features:

insert the figure coming after 'y vulnerabilities, and abstract discussion of security vulnerabilities).'

s:
here we see more specific and general features then we saw earlier in gemma models, for example the now famous golden gate bridge feature,

--
/Users/julianschulz/Downloads/h0ugocryydrg1.png

s:
anthropic did make a chatbot availabl for one day, that led people caht with a claude isntance with that feature clamped up.

it can not halp itself from bringing up the golden gate bridge wich can be qutie strange.


--
reading claudes mind

incldue the picture taht comes after 'ropes about AI and is also heavily anthropomorphized'

s:
we can now somehwoat read claude's mind for safety relevant contexts, wich is super helpful!

--

safety relevant feature steering

in @anthropic2025b opus 4.5 system card they:

find eval awareness features via: contrastive vectors, SAE features

and they inhibit thse to 25% and find:
 
reduced eval awareness
increadsed misalginment

s:
the cot said sles thingks like 'i know this is a test'
the model behvaes less aligned, but not in the most egregious mislainged cases of previous models (blackmail,...)


--

expanding SAEs

@lindsey2024/cross_trans_coers
s:

we can not only vary the arcitecture and loss of the SAE itself, but also what kind of data we are acting on.

the variant shere are trascoders, that do not try to sparsify a single layer of activaiotns, but want to reocnsotruc th next layer out of the current layer. The ida hre being, that we sparsify the computation that is goin on in some part of the tranformer.

the corsscoder takes multiplt layers and rescontucts multiple layers out of multiple layers

--
@lindsey2024/activities_across_layers

s:
this enables things like: figuring out what 'the same feture' is across the model

--
model diffing

@lindsey2024/model_diffing
s:
now, we can also do the same method, but go further and identify the same feature afross different models

--

@lindsey2024/model_diffing

s:
here the authors took the model idiff between sonent 3 base and chat model, and could identify on a feature level what concepts it leanred during post trainign, and look at what the specific featurea are, for example refulsal or features for step by step reasoning.





