 Beginning Slide:

Title and, name, date

overview side with chapters.

from now on before every chapter/subchapter slide make a new slide wiht chapter titles. with titles of the chapter that we do not have yet light grey  and chapters we have been already darker thicker black. begin wiht overview slide all in the thicker darker black


Chapter 1: Machine Learning 

1.1 Machne Learning Concepts

Training Loop

Slide with boxes representing training loop

Data -> Model -> Prediciton -> Loss
               Optimizer (arrow from Loss and to model)

Include an 'example picture' over all the boxes.

Data: Dog picture, Model: pictogram of a neural Netowrk, Preidciotn : cat: 30%, dog 70$, loss_ -log(0.7) and Optimizer: the formular for SGD

Forward Pass/Backward Pass

write out the formulars for a forward and a backward pass
forward pass for a single layer: y = relu(w_e * x + b) * w_d


slide on overfitting
slide on double descent
slide on grocking

define:

Parameter/Weight
Activation
Hyperparameter

1.2 Arcitectures

MLP layer

title: multy Layer perceptron

diagram of two layers with nonlinearity in the middle
underneath definition of fowrard pass in Latex
next to it: diagram of ReLu

Universal approdimation o


Layers:

multiple boxes from left to right saying input -> layer 1 -> layer 2 -> layer 3 -> output

Resudual Stream:

same as before, but with skip connections

underneath similar picture but this time the residual stream is drwan as continual line and layers "add in"


Transformer
Attention

Section 2 MechInterp
section 2.1 what is Mechinterp
Farmer vs biologist
use for safety
Olahs 3 claims

section 2.2 CNN Interpretability
slides about:
evidence for a feature having a meaning
circuit
Section 2.3 Features in Transformers
Do directions in LLMs mean something? -> gender direction
how to get a direciton out of an LLM:
- contrast activations
- contrast many activations
- train a probe (ITI & Othello)
What can you do once you have a feature
- steering: CCA & Othello
- ablate it out: Refusal Ablation

section 2.3 Superposition:
Example from CNNs with the car
Toy models of Superposition:
experimental setup (picture + Equation)
Result: just upper panel
