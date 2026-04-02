--
Early Mechinterp
on Convolutional Neural Networks

s:



most of the early work on MechInterp was done on visual neural netwoks, that tended to be CNNs
__
slide with a diagram showing how convolution works


s:
many early impressive neural netorks where visual neural networks (example alexnet)

mechinterp is easier with onvolutional networks: the different channels of a convolutional layer are a preferred basis, wich leads to the directions that might be features to be clear arcitectually

there is a lot of universailty in vision CNNs, compared to transformer based LLMs

as we will see, feature visualisation works pretty well in vision models.


--
Features:

What would convince you a CNN channel has the function of 'detecting a curve?'

s:
discussion on what evidence we would demand ofr that claim:

__
( go throught the 7 argiments of @olah2020 just each slide with the title of the argument, add the picture in there, put the explaining text as notes, and add a 1 sentence description in the slide)

--
Example of a complex feature:

add the pictre of the car circuit

s:
we see, that these methods also work to identify more complex structures. note, that these are still human-understandable 

we can even understand the circuit here!

--
Universatily in Image Models

input the plot comparing curve detectors and high frequency dtectors from @olah2020 universtalty section

s:
in image models, these fearures at least in early layers seem to be pretty universal

later layers, often correspond to human concepts, wich speaks for their universality. But also many channels do not seem to have easily understandable descritopns


--
If universality is true:
what features are universal? why?

discussed in the Natural Abstractions discourse

@wentworth2020 @chan2023 
s:

for curves this is obvious, but it is not cler how far this universality scales
even if it does, it is not clear why those concepts in particular are univeral?
this might be pretty imporant: if we find a 'morally good' feature in LLMs, do we expect to behave the same as the human oncept OOD in the limit of a super capable model?

unclear, there are some theories here, ask Leon Lang. AFAIK the work in this direction has stalled out the last years


--

Polysemanticity

include the figure with teh car being split intot he dog features

s:
one probelem, taht we will discuss later, but is already a problem in these models:

sometimes a feature is split up in different neurons, and multupe features share one neuron.
here we see, taht in one layer, we essemble the car feature, but this feature does not gets its own neuron in a later layer, it i split up its activations adds onto the a bunch of other neursn that encode unrelated things. 

we can already see taht this is neccecary here, when we want to have more featre then we have concepts
