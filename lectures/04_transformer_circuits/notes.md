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

explain circute tracing

task play around with https://www.neuronpedia.org/jackl-circuits-runs-1-4-sofa-v3_0
