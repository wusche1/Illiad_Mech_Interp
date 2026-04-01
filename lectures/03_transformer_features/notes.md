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


