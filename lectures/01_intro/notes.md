
from @olah2020

--
picture of hooks micrgraphia

s:
Norally in ML, we look at the algorythems we use to create neural netowrks, as we did yersterday:
the arcitaectures we have to choose, the optimizers and hyperparametres, the data we are training on.
it makes sense to mostly consider these thigs, since to create a model, we have to actively make decisions about them


The idea of MechiInterp as formulated by Chris Olah, the founder of the field (more or less), is to look at the neural networks that come out of ML as an object of scientific enquiery
similar to biologists not just looking at the optimisation process that leads to life (evolution), but also study living organsims by looking inside them and trying to understand how they work

now, people have tried to understand how life works for a long time, famously the ancient greeks thought that out bodey is made out of 4 humors, taht have to be in the right balance relative to each other

lots of doctors throughtout european history tried to find out how to let out enough blood or black bile to make patients healthy, but thier efforts where not really corwned by success. 
This is because they had the wrong onthology. understanding the humors in the bodey is not actually that helpful to understand why a patient is sick

the right onthology came along with better measuring instruments, namely the microscope: once you can look precisely enoguh into a living organsim, you see that it is made up of tiny bundles, and if you study those , and thier properties and what makes human cells work, and what kind of invading bactria cells cause wich kinds of sicknesses, you get somewhere in biology and medicine

Chris Olahs vision for MechInterp was to do for ML what the Microscope did for biology.

--
slide showing the 3 claims of schwann, next to the picture of the book

s:
Theodor Schwann, the biologist who described what cells are and how they work came up with these claims about thier role in biology

even thought we now know that claim 3 is actually wrong, the first two claims lay down an ontology for cellular biology as a field


--
slide with olahs claims and the picture he put the the left of it

s:
In his 2020 essay 'zoom in' olah lays down similar claims in this tradition define a tentative ontology for mechanistic interpretablity

lets go thorugh these claims. they are not only descriptive, but normative

claim 1:
a feature is a fundermental unit of a neural netork. 
they correspond to dictrions,
they can be studied and understood.

okay, so the directions here means directions in activation-space. 
the claim is: individual directions in activation space can be understood
ad they are fundermental units

question to the audience: how would the world look like if that claim would be false?
(awaiting answers something like directions could be used for binary encoding, or the same directions might have completely different meanings depending on the excitation amount in that direction)

claim 2:
features are connected by wieht forming circutis, that can be undersood.

what would a world look like in wich claim 1 is true, but claum 2 is false?

(wait for answers. possible answers: a very complicated process might use feautres as a human readable sotrage of information, but the process itself is more complicated then just feaures being connected in logical circuits. similar to CoT bing a human understandable sotrage of informationm but the llm using this storage not being understandable as a particular connection of cot-written information)

claim 3:
univsersaility

what would it mean if this would be false:

we could not learn much from one network to another, each time we train a new NN we would see competeley different features and circuits

--
slide with the question of how this is usefull for ai safety?

s:
now, what if those claim turn out to be true, how woudl that help with AI safety?

quesiton to the audiemce, collecting ideas

--
selected points from @nanda2022 and @nanda2023b

auditing for misalginment/decetption
enumerative safety
eliciting latent knowledge

s:
if we can find features/circtuits for misaliged motivation or cheating or any other kinds of cognition we want to avoid, like thinking about bioweapons, we could detect this before or in deployment

if we can identify the function of every part of a neural network, we can be sure it does not contain any bad intentions/dangerous capabiliteis

we can find out more directly what the model knows, wich helps with transparency/against scheming and also might help with 'ai for science' stuff like a genetic model knowing that some genetic marker is responsible for some effect, but since the model is not a langugae model, you can not just ask it.



