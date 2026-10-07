--
A newer trend in interpretability is, to take less mechanistically faithful appproaches, and rely more on LLM judgement to interpret model internals
The idea is to direcly train an LLM to take in activations, and output natrual langugage explanations
first of this example are activation orcales:
 - Activation orcales
 @karvonen2026/training
 where we take activaltion from an LLM from one prompt/situation
 and feed this activation in anotehr version of this LLM: the activatio orcale
 now we ask the activaiton oracle a quesiton about the activations, we know the answer to
 then we finetune the AO to answer this question correclty when given access to the activation
 this generalizes to szenarios OOD to the one originally trained on
 @karvonen2026/benchmarks
 and on a bunch of examples, this has been the first method that purely looks at inernals, that can compete with blackbox clever prompting methods of realing the target fo fine tuning of a set of model orgnaisms.

 - Natural Language Autoencoders
  @fraser-taliente2026/training
NLAs take a different training approach:training 2 models: an activation verbalizer and an activation reconstrucutor
we train it the following way: we take a promtp, then take the activation from that prompt. just feed that activation ino the verbalizer, and let it produce a pice of text. that text is read in by the reeconstuctor, that proeuces then some activaiton after reading in this text.
now we can take a step of sft of the reconstructorer to make the reconstution closer to the original activation
and a step of GRPO RL of the verbalizer to produce better verbalizations


@fraser-taliente2026/reward-reasoning
this can also do cool things, like revleaing hidden unverblized motivations for answering a cirtain way

@fraser-taliente2026/hallucinations

@fraser-taliente2026/eval-awareness

@fraser-taliente2026/audit-game

@anthropic2025b
application: in the opus 4.8 system card they use NLAs to show that in 5% of of the answers in an evaluation, the model is aware of the presence of a  grader but does not verbalize this.

# Jlens

-in my oppinion the most interesting of the new fuzzy mehtods, not just for mechintperp, but what we conceptually learn about modfels in general.
-the researvhers started out developing a method, that tracks the activations, that represent cirtain concepts, in a way that makes it more likely fo this concept to be said later
-similar to the logt lens we talke about earlier, but here the just take into account how chnages in one layer on average change actibations in the final layer that go into the unembedding 
@gurnee2026/jlens_definition
include fomrulas of Jl = E t, t′≥t, prompt  ∂ hfinal,t′ ∂hl,t lens(hl) = softmax(WU norm(Jlhl))  
- you can again, similar to other methods use this to read off/classify activations or intervene on them @gurnee2026/jlens_use
-this gives you per token posotin layer a list of top active tokens, simialr to logit lens the user interface looks somehwat like this 
@gurnee2026/interface
- and then, it turns out that these represetntations have apretty wide and robust rule in model cognition 
@g
- these seem to be the kind of activbatins, that th emodel can report thinking about, or think abouyt on command,
- you can see intermediate calculations pretty well with that lens
- you can interfere with its cognition to throw its internal reasoningoff track 
- these features seem to encode their condepts prety robustly across differen contexts
- you can remove the whole jspace and while the model can still speak fluently and do simple 'reflex' tasksks, it becomes unable to do higher complex reasonng. 

- the jspace is the priviged space where these things happen. you can also do this with vectors extracted by other means, but the jlens vectors work better, and when you project them out, it does not work at all
@gurnee2026/Jspace_priviliged

- structurally: even thouth there are more j-vectors then dimenions int he residual stream, they only span a small subspace of all activations, and only a few (oom:10) are active at the same time
- they have higher interaction with readouts of later layers
- all these interestign prperties mostly hold for middle layers. late layers jlens vectors are just encofing the 'the model decided to say this token onw' property. for ealy layers they do not pick up interesting vectors at all
@gurnee2026/structural

- all of this is interesting as a result for for mechinterp. but it also fits very neatly on the global workspace thory of conciousness

sldie:
Global workspace throy of conciousnes:

 - reporgalb
 - top down controllable
 - medium of delibareate reasoning 
 - flexible generalisation
 - selectivity
 - limited capacity
 - global broadcast

 (green checkmark behihd all of those)
 (now the following two wiht a question mark)
 
 - Encapsulates specialist modules
 - phenomenology
