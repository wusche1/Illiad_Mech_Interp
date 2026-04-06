--
transfomrer circuits

s:
okay, next we want to try to identify circuits in tranfomers

butis that even possible: when they ca

--
@nanda2023a/compelte_algorythem

s:
here is a positive example:

an model trained on mod 113 addition and got close to perfoect score

they managed to compeltely reverse engineer the algorythm:

the embedding layer takes a number and writes cos(wa) of that nymber in different dicections for differnt w

then in the attention mechanism these signs and cosings are multuplied using trig. identities to get cos and sin form the addition of the numbers

then finally the logis are just a linear combination of these identaties with cos and sin (wc) for every c in thier output vocab

that way, the cos all cancel each other out if the a+b-c is not 0, and ony constructively interferee if they are

this is an example of a hepefull concept. that just ML in a transfomer leads to algorythems we can understand.
do 
--

@nanda2023a/frequencies

s:
the way that they got this inside, was that Neel Nanda, teh hero of mechinterp
got the idea to plot the fourier transform of the embeddign matrix

this gave him the specific features and frequencies the model learned, and then he could trace the circuts forwards.


this is impressive but not scallable: we can not have neel nanda bruting over an LLM, tracing every computatio forward, and understaning the algorythm, because LLMs are way more messy, way bigger and we do not have enough neel nandas

so we are looking for ways to understnad circuits that follow some sort of more standardized method.


--

Plotting attention patterns

@elhage2021/attention_pattern

s:
one common tool is, to plot attention patterns.
thei is the A matrix in the transfomer implmementation
so you have a token by token matrix for each attention head, and can see what attention, when at what toekn spent attention where

you can use this, to inform your guesses of the funciton of the attention heads
here, we plot the attention of a few attention heads and you see a pattern

a few attention heads always attend to the  token after its previous use 

and another attention head always attending to the previous token.

--
logit attribution

@elhage2021/attention_patter_and_logit_attribution

s:
another property of an attetnion head, and really any part of a neural network you can look at, is its conribution ot the output of the network

for that, you just take the ourput of the attention head, and propergate it all the way thorugh to the unembedding
now you can see when this output was helpful at the prediciton of the next token.

it is whenever ther is the pattern AB...A and then B, so basically a task of in-context induciotn form previous example.

the authors of this paper looked then into these specific attention heads, and looked how thier weight marices are gripping into each other, and have reconstucte the algorythem these 

--
@callummcdougall2023/k_composition

s:
explain k-composition

ask if people see another way of seeing how to implement this

--
@callummcdougall2023/q_composition
s:
explain q-coposition

--
@wang2022/path_patching

s:
other general method for finding a cirocuit
path patching: you have a specific task: here indirect object identification

explain ioi task

now, becasue we know what the relevant information of teh task is, we can construct a paralel case (david and john), where teh we know the anser should be different

now we can go to any part of the tranformer, and replace the activations form one forward pass with the actvations of another forwward pass, and thereby see if the relevant information 'passed through' there. 
another thing you can do, is to jsut knock out some part the transofmer, replaceing its output with its average output over lots of tokens, to isolate the effecto fosme specific pices

the authers of this paper traced the circuit for this task taht way through gpt2

--
@wang2022/ioi_circuit

s:
and the circuit is kind of bizarre

explain different parts of the ioi circuit

interesting things:
 - the circuit is highly redundant: things are implemented twice, likely due to dropout
 - reuse of known motives: inducion heads. this is also a point for universaility
 - there are anti-helptul heads in the end? authos guess this is to hedge? but what?

 --
automated circuit discovery

@conmy2023/recipe

s:
in the examples so far, circuit discovery has been somehting artisenl, despite the common usefull methods

this paper takes the first step towars a method, that needs less human intervention


what we need here to start with i s a clarny dfined task
a dataset where that task is possible and a parallel datast wher it is not (like non-repeating names in IOI)
the output we expect
and a metric that, to maxamize it, would mean to do this task better

we also need to assume some partition of the neural nework into parts, like induciotn heads or MLPs

--
@conmy2023/figure2

s:
next, we slowly cut this graph apart until we have some minimial circuit, basically by taking each conenction between pices of the graphy away and asking: doe sthe task still work?

--
@conmy2023/algorythem

s:
go through algorythem

--
@conmy2023/recovered_ioi_circuit
s:
example of this working:
they could recover previously found circuuit. here the IOI circuit found in the precious paper
also some other examples in the paper
has not been used to find mass-circuits or anyting

down sides:
our computational units here are somehwat limited, like attention heads or MLP layers.
doing MLP neurons is possible, but then the computation load of pruning the graph exploeds

and in the end, we know that the relevant unit of computation here are porbably features that donot easily map onto the weights considered here.


--

circute tracing
@ameisen2025/CLT

s:

- the most adcanced method we have for ciruit discovery right now, is this: circuit tracing, developed by anthropi last year.

here the idea is, that the unit that we build the ciruits out of are features, not attention heads or neurons. so the first thing we have to do here, is to take our whle LLM apart into features

for this we stary by trainng skip level transcoders on the entier tranformer.

here we read in the resed before an mlp, and write in the mlp output of all alter mlp

we do that on all layer, and train all of there transcoders together

anthropic here went for jup relu transcoders

that is a huge up front cost that you have to pay to get started here, but it lets you do somthing extremly cool:

you can basically take all activations in a forward pass and deconstruct them into the active transcoder features

this, models all the computation happening in the MLPs in a way tha tlets you label every feature here, and lets you build an interpretable model of the forward pass

--

@ameisen2025/replacement_models

s:
that is what they call a replacement model

they take a forward pass on a specific prompt, decompose it into the active CLT features, and this gives basically an interpretable model, doing the same calculation

at least you captre some part of the calculation: the nonlinear calculation happening in the MLPs.

it does not giv you insight into how the attention pattern are clauclated, those are just taken and fixed formt her real forward pass
same with the mupltipiilication factor in the layernorm

also. of cause your CLTs are not prefectly replicating the actuavation at any layer. so they add these back in as error terms that they can not interpret

--

@ameisen2025/pruning

s:
now once you have cut apart the forward pass into those features, you can add the input tokens and output logits, and now you have a model, that you can take a gradient though

you simply let all of the features have gradients, and all these other parts that we just copied over from the orignal orward pass liek the attention patterns get n grad tensors

unlike in ACDC we now get the connection between any two of the points in this graph by a simple backward pass, we do not have to knock a connetion out and see if the resut is still possile.

now once we have done this, we can kick out all features that do not have a connectionn above a given strength to the output logit, and we have a somewhat sparse so called attributio graph through the whole llm

an additonal thing they often do here in thee graphs is to summarize a bunch of similar features together in one feature selection if they have similar labels and seem to foollow similar funcitons in this circuit

--
@ameisen2025/example_circuit

s:
and this lets us trace ciruits like this one, where we see the features the model uses to compute what the acrny here is


--
@ameisen2025/feature_supression

s:
anther thing you can do again is to supress a feature

the way they do this is they take a feature of a CLT and add it back in, in the next n layers after the layer it is created with a prefactor of -1 most of the time

but the authors take the freedom to fiddle around with the numbers of layers to inject and the prefactor, wich they describe as 'ad-hoc' and makes me think this is a pretty lame validatin for the whole method.

--
@ameisen2025/supresseion_example

s:
here we see and example of a supression

indeed if you supress the feature in the graph where a specific letter of the acronym was decided, this leads to this letter of the aronym being false

--
link to https://www.neuronpedia.org/gemma-2-2b/graph

s:

for you to get a feeling for how these attribution graphs look like and what we find there, i encourage you to play around with the ones on neuropedia here, see what kin of circuits you can find, and if you can learn anything from them

--
@lindsey2025a/maths_circuit

s:
they also did circuit tracing on thier won model, here heiku 3.5

and found some interesting circuits

here for example one for doing maths, but als a bunch of others

--
@lindsey2025a/misaligned_model_pipeline

s:
they also showcased how this can be usefull for alingment
they made a misaligned verson of heku 3.5, by putting it rhogh a training pipeline inducing a form of reward hacking

htey SDF on ficktional papers saying that reward models like meta-commetns in llmoutput.

then they sft in the behaviour of including meta-comments in model output

this leads to a model also doing things what it thinks the RM will like OOD

--
@lindsey2025a/misaligned_circuit

they do the cirocuit tracing on a model output where ti finishes a poem with a meta line, and they see the feature where it it thinks about the reward model causally upsteream of it's output

--

@lindsey2025a/misaligned_intervention

s:
and they can causally intervine, steering against the reward model thought, and thereby making this behaviour less likely

this is honeslty the coolest tool i have seen so far in the in MechInterp.

--

Gnerality

- high abstraction features in middel layers
- attention moves most information in early layers
- 'dfault' circuits
- shorcuts
- some computation on 'special' token position
- confidence reducing features
- 'boring' feature

@lindsey2025a
s:
 - thy looked at a bunch of different circuits, so how does it look with generality?
 - some pattersn:
 - ealy and late layers have more simple token oriented features, most interesting compuation happens in middle layers
 - when information is gatherd by atention heads, thatmostly happens in early layers
 - ther are often defaut pathways, like saying you dont know a person, that then get overwritten by other things like a person the llm knows
 - circuits often have mutiple ways to get to the ansesers, like shortcusts that when i ask what is the capital of texas, one pathway ges from texas+capital to dalasa, but just texas alone already upweights dalas
 - some info gathering, or decision making points are in special tokens, like full stops or the new line token after a new <assistant:> or something
- often there are outputs that write agaisnt the 'correct' tokens, maybe for hedghing like in ioi
- there are lots of active features not part of the 'computation' like in the amthe example there rae ots of maths features, just just shortcut contribuite to all numbers
