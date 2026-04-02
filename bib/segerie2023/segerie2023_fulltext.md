*Epistemic Status: I believe I am well-versed in this subject. I erred on the side of making claims that were too strong and allowing readers to disagree and start a discussion about precise points rather than trying to edge-case every statement. I also think that using memes is important because safety ideas are boring and* [*anti-memetic*](https://www.lesswrong.com/posts/zk6RK3xFaDeJHsoym/connor-leahy-on-dying-with-dignity-eleutherai-and-conjecture%23Eliezer_Has_Been_Conveying_Antimemes)*. So let’s go!*

*Many thanks to* [*@scasper*](https://www.lesswrong.com/users/scasper?mention=user)*,* [*@Sid Black*](https://www.lesswrong.com/users/sid-black?mention=user) *,* [*@Neel Nanda*](https://www.lesswrong.com/users/neel-nanda-1?mention=user) *,* [*@Fabien Roger*](https://www.lesswrong.com/users/fabien-roger?mention=user) *,* [*@Bogdan Ionut Cirstea*](https://www.lesswrong.com/users/bogdan-ionut-cirstea?mention=user)*,* [*@WCargo*](https://www.lesswrong.com/users/wcargo?mention=user)*,* [*@Alexandre Variengien*](https://www.lesswrong.com/users/alexandre-variengien?mention=user)*,* [*@Jonathan Claybrough*](https://www.lesswrong.com/users/lelapin?mention=user)*,* [*@Edoardo Pona*](https://www.lesswrong.com/users/edoardo-pona?mention=user)*,* [*@Andrea\_Miotti*](https://www.lesswrong.com/users/andream?mention=user)*, Diego Dorn, Angélina Gentaz, Clement Dumas, and Enzo Marsot for useful feedback and discussions.*

When I started this post, I began by critiquing the article[A Long List of Theories of Impact for Interpretability](https://www.lesswrong.com/posts/uK6sQCNMw8WKzJeCQ/a-longlist-of-theories-of-impact-for-interpretability), from Neel Nanda, but I later expanded the scope of my critique. Some ideas which are presented are not supported by anyone, but to explain the difficulties, I still need to 1. explain them and 2. criticize them. It gives an adversarial vibe to this post. I'm sorry about that, and I think that doing research into interpretability, even if it's no longer what I consider a priority, is still commendable.

**How to read this document?** Most of this document is not technical, except for the section "What does the end story of interpretability look like?" which can be mostly skipped at first. I expect this document to also be useful for people not doing interpretability research. The different sections are mostly independent, and I’ve added a lot of bookmarks to help modularize this post.

If you have very little time, just read (this is also the part where I’m most confident):

* [Auditing deception with Interp is out of reach](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Auditing_deception_with_interp_is_out_of_reach) (4 min)
* [Enumerative safety](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Enumerative_safety_) critique (2 min)
* [Technical Agendas with better Theories of Impact](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Technical_Agendas_with_better_ToI) (1 min)

Here is the list of claims that I will defend:

(bolded sections are the most important ones)

* [**The overall Theory of Impact is quite poor**](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#The_overall_Theory_of_Impact_is_quite_poor)
  + [Interp is not a good predictor of future systems](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Interp_is_not_a_good_predictor_of_future_systems)
  + [**Auditing deception with interp is out of reach**](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Auditing_deception_with_interp_is_out_of_reach)
* [**What does the end story of interpretability look like? That’s not clear at all.**](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#What_does_the_end_story_of_interpretability_look_like__That_s_not_clear_at_all_)
  + [**Enumerative safety?**](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Enumerative_safety_)
  + [Reverse engineering?](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Reverse_engineering_)
  + [Olah’s Interpretability dream?](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Olah_s_interpretability_dream_)
  + [Retargeting the search?](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Retargeting_the_search_)
  + [Relaxed adversarial training?](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Relaxed_adversarial_training_)
  + [Microscope AI?](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Microscope_AI_)
* [So far my best ToI for interp: Outreach](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#So_far_my_best_ToI_for_interp__Outreach_)
* [**Preventive measures against Deception seem much more workable**](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Preventive_measures_against_Deception_seem_much_more_workable)
  + [Steering the world towards transparency](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Steering_the_world_towards_transparency)
  + [Cognitive Emulations - Explainability By design](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Cognitive_Emulations___Explainability_By_Design)
* [**Interpretability May Be Overall Harmful**](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Interpretability_May_Be_Overall_Harmful)
* [**Outside view: The proportion of junior researchers doing Interp rather than other technical work is too high**](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Outside_view__The_proportion_of_junior_researchers_doing_interp_rather_than_other_technical_work_is_too_high)
* [**Even if we completely solve interp, we are still in danger**](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Even_if_we_completely_solve_interp__we_are_still_in_danger)
* [**Technical Agendas with better Theories of Impact**](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Technical_Agendas_with_better_ToI)
* [**Conclusion**](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Conclusion)

Note: The purpose of this post is to criticize the Theory of Impact (ToI) of interpretability for deep learning models such as GPT-like models, and not the explainability and interpretability of small models.

The emperor has no clothes?
===========================

I gave a talk about the different[risk models](https://www.lesswrong.com/posts/wnnkD6P2k2TfHnNmt/threat-model-literature-review), followed by an interpretability presentation, then I got a problematic question, "I don't understand, what's the point of doing this?" Hum.

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/LNA8mubrByG7SFacm/zzxhts0u9ne414ahftxw)

*Image from* [*Feature Visualization*](https://distill.pub/2017/feature-visualization/)*.*

* Feature viz? (left image) Um, it's pretty but is this useful?[[1]](#fn2stfurwyyg6) Is [this](https://arxiv.org/abs/2010.12606%23:~:text%3Dversion%252C%2520v3)%255D-,Exemplary%2520Natural%2520Images%2520Explain%2520CNN%2520Activations%2520Better%2520than,of%252Dthe%252DArt%2520Feature%2520Visualization%26text%3DFeature%2520visualizations%2520such%2520as%2520synthetic,convolutional%2520neural%2520networks%2520(CNNs).) [reliable](https://arxiv.org/abs/2306.04719)?
* GradCam (a pixel attribution technique, like on the above right figure), it's pretty. But I’ve never seen anybody use it in industry.[[2]](#fn4qi9kn3ip89) Pixel attribution seems useful, but accuracy remains the king.[[3]](#fn6xxwjs20rd7)
* Induction heads? Ok, we are maybe on track to retro engineer the mechanism of [regex](https://en.wikipedia.org/wiki/Regular_expression) in LLMs. Cool.

The considerations in the last bullet points are based on feeling and are not real arguments. Furthermore, most mechanistic interpretability isn't even aimed at being useful right now. But in the rest of the post, we'll find out if, in principle, interpretability could be useful. So let's investigate if the Interpretability Emperor has invisible clothes or no clothes at all!

The overall Theory of Impact is quite poor
==========================================

Neel Nanda has written[**A Long List of Theories of Impact for Interpretability**](https://www.lesswrong.com/posts/uK6sQCNMw8WKzJeCQ/a-longlist-of-theories-of-impact-for-interpretability)**,** which lists 20 diverse Theories of Impact. However, I find myself disagreeing with the majority of these theories. The three big meta-level disagreements are:

* **Whenever you want to do something with interpretability, it is probably better to do it without it.** I suspect Redwood Research has stopped doing interpretability for this reason (see the current plan here[EAG 2023 Bay Area The current alignment plan, and how we might improve it](https://www.youtube.com/watch?v=YTlrPeikoyw)).
  + **This is particularly true for counteracting deceptive alignment**, even though it is the main focus of interpretability research. [section [deception](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Auditing_deception_with_interp_is_out_of_reach)]
* **Interpretability often attempts to address too many objectives simultaneously.** Please[Purchase Fuzzies and Utilons Separately](https://www.lesswrong.com/posts/3p3CYauiX8oLjmwRF/purchase-fuzzies-and-utilons-separately): i.e. it is very difficult to optimize multiple objectives at the same time! It is better to optimize directly for each sub-objective separately rather than mixing everything up. When I look at [this list](https://www.lesswrong.com/posts/uK6sQCNMw8WKzJeCQ/a-longlist-of-theories-of-impact-for-interpretability) by Neel Nanda, I see that this principle is not followed.
* **Interpretability could be harmful.** Using successfully interp for safety could certainly prove useful for capabilities. [section [Harm](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Interpretability_May_Be_Overall_Harmful)]

Other less important disagreements:

* **Conceptual advances are more pressing,** and interp likely won't assist in advancing these discussions. [section [end story](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#What_does_the_end_story_of_interpretability_look_like__That_s_not_clear_at_all_)]
* **Current interpretability is primarily used for post-hoc analysis** and has shown little utility ex-ante or for predictive capacity [section [predictor of future systems](https://docs.google.com/document/d/e/2PACX-1vSedy4vmfA5H30bimiSGWykDfh8FB_uYKCt6D2qz9nwmfhGUc93H3UEPN1pBtyXe-eKEdu0E5oUbSWR/pub#id.vcvarqienhb7)]

Here are some key theories with which I disagree:

* Theory of Impact **2: “*****Better prediction of future systems”***
* Theory of Impact **4: “*****Auditing for deception”***

In the appendix, I critique almost all the other Theories of Impact.

Interp is not a good predictor of future systems
------------------------------------------------

Theory of Impact 2: “***Better prediction of future systems**: Interpretability may enable a better mechanistic understanding of the principles of how ML systems and work, and how they change with scale, analogous to scientific laws. This allows us to better extrapolate from current systems to future systems, in a similar sense to scaling laws. E.g, observing phase changes a la induction heads shows us that models may rapidly gain capabilities during training”* from [Neel Nanda](https://www.lesswrong.com/posts/uK6sQCNMw8WKzJeCQ/a-longlist-of-theories-of-impact-for-interpretability).

* **Nitpicking on the** [**Induction head**](https://transformer-circuits.pub/2021/framework/index.html)**example.** If we focus on the above example "*models may rapidly gain capabilities during training*," I don't have the impression that it was interpretability that enabled us to find this out, but rather behavioral evaluations. Loss was measured regularly during training, and the rapid gain of induction capability [was measured](https://transformer-circuits.pub/2021/framework/index.html) by having a model copy a random series of tokens. In the beginning, copying does not work, but after some training, it works. Interpretability only tells us that this coincides with the appearance of induction heads, but I don't see how interpretability allows us "*to better extrapolate from current systems to future systems*. Also, induction heads are studied in the first place because they were easy to study.
* **Interpretability is mostly done ex-post the discovery of phenomenon, but not ex-ante.**[[4]](#fnztj4j3pmerg)
  + We first observed the grokking phenomenon, and *only then* we did proceed to do some[interpretability](https://arxiv.org/abs/2301.05217) on it. Are there any counterexamples?
  + (In[What DALL-E 2 can and cannot do](https://www.lesswrong.com/posts/uKp6tBFStnsvrot5t/what-dall-e-2-can-and-cannot-do), we see that DALL-E 2 is not able to spell words correctly. Then 2 months later,[Imagen](https://www.lesswrong.com/posts/uKp6tBFStnsvrot5t/what-dall-e-2-can-and-cannot-do?commentId%3Dg6kZ3eRFejRjiyGiw) could spell the words correctly. We didn’t even bother with interp.)
* **There are better ways to predict the future capabilities of those systems.** Thinking out of the box, if you really want to see what future systems will look like, it's much easier to look at the papers published in the NeurIPS conferences and cognitive architecture like AutoGPT. Otherwise, subscribing to DeepMind's RSS feed is not a bad idea.

Auditing deception with interp is out of reach
----------------------------------------------

Auditing deception is generally the main motivation for doing interp. So here we are:

Theory of Impact 4: ***Auditing for deception**: Similar to auditing, we may be able detect deception in a model. This is a much lower bar than fully auditing a model, and is plausibly something we could do with just the ability to **look at random bits of the model and identify circuits/features** - I see this more as a theory of change for 'worlds where interpretability is harder than I hope'* from [Neel Nanda](https://www.lesswrong.com/posts/uK6sQCNMw8WKzJeCQ/a-longlist-of-theories-of-impact-for-interpretability).

* **I don't understand how “Looking** ***at random bits of the model and identify circuits/features*****” will help with deception.** For example, let's say I reverse engineer GPT2 for a random circuit, such as in the paper[Interpretability in the wild](https://arxiv.org/abs/2211.00593), where they retro engineer the indirect object identification circuit. It’s not clear at all how this will help with deception.
  + Even if the intended meaning was "identify circuits/features that may be relevant to deception/social modeling", it's not clear whether analyzing every circuit would be sufficient (see the "[Enumerative Safety](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Enumerative_safety_)" subsection).
* **We are nowhere near the level required to detect or train away deception with interp.** In his article[A transparency and interpretability tech tree](https://www.lesswrong.com/posts/nbq2bWLcYmSGup9aF/a-transparency-and-interpretability-tech-tree), Evan Hubinger lists 8 levels of interpretability, with only levels 7 and 8 providing some means to combat deception. These levels roughly describe the desiderata of interpretability, but we have only reached level 2 so far, and we have already encountered negative results at level 4. Evan explains that "*any level of transparency and interpretability tech that is robust to deceptive models is extremely difficult*".
* ***“Furthermore, trying to uncover deception in advance via interpretability tools could fail simply because there is no sense that a deceptively aligned model has to actively be thinking about its deception.** A model that has never seen a situation where there is an opportunity to seize power need not be carefully planning out what it would do in such a situation any more than a factory cleaning robot need be planning for what to do if it someday found itself in a jungle instead of a factory. Nevertheless, the fact that the model has not been previously planning to seize power doesn’t imply that it wouldn’t if given the opportunity. In particular, a model could be deceptively aligned simply because it reasons that, in situations where there is a clear overseer, doing what it wants is a good general strategy for gaining power and influence in the world—without needing any explicit plans for later deception.”* (from Hubinger in [Monitoring for deceptive alignment](https://www.lesswrong.com/posts/Km9sHjHTsBdbgwKyi/monitoring-for-deceptive-alignment))
* **There are already negative conceptual points against interpretability**, which show that advanced AIs will not be easily interpretable, as discussed in the [section on interpretability](https://www.lesswrong.com/posts/uMQ3cqWDPHhjtiesc/agi-ruin-a-list-of-lethalities%23sufficiently_good_and_useful) in the list of lethalities (these are points I’ve [tried](https://docs.google.com/document/d/1GiYfx77cE6-VyeNN31tVUARt5X7tbX4XD0CSmBkReUc/edit%23) in the past to critique and have mostly failed). Especially points 27, 29, and 33:
  + **27. Selecting for undetectability**: “*Optimizing against an interpreted thought optimizes against interpretability.”*
  + **29. Real world is an opaque domain:** “*The outputs of an AGI go through a huge, not-fully-known-to-us domain (the real world) before they have their real consequences.  Human beings cannot inspect an AGI's output to determine whether the consequences will be good.”*
    - **And cognition can be externalized.** This is not specific to interp. Many patterns can only be explained for how they interact with the environment, and can't be fully explained by what's in the network alone. E.g. “Consult a recipe book and take actions written in this book.” (example from Connor).
  + **33. Alien Concepts:** “*The AI does not think like you do”* There may not necessarily be a humanly understandable explanation for cognition done by crunching numbers through matrix products.
  + I don't fully agree with all of these points, but I haven't seen much discussion on these specific points, you can find some caveats in my critique
* Other weaker difficulties in footnote.[[5]](#fn2464ho15s7t)

Counteracting deception with only interp is not the only approach:

* **Interp is not the only way to study Deception.** Here are other neglected paradigms:
  + **Adversarial attack** (basically [Redwood’s Plan](https://www.youtube.com/watch?v%3DYTlrPeikoyw%26themeRefresh%3D1)). Lots of important adversarial strategies don’t rely on interp like [**Consistency checks**](https://arxiv.org/abs/2306.09983), [AI checks and balances](https://www.cold-takes.com/high-level-hopes-for-ai-alignment/) and regular Prompting[[6]](#fnrbnra8fsacj).
  + **Create a toy model of deceptive alignment or a simulation:**  Studying “in vitro demonstrations of the kinds of failures that might pose existential threats” from [Model Organisms of Misalignment](https://www.alignmentforum.org/posts/ChDH335ckdvpxXaXX/model-organisms-of-misalignment-the-case-for-a-new-pillar-of-1), a pretty recent and detailed post on this.
  + **Finding deceptive alignment proxies:**  For example monitoring the following [4 criteria](https://www.lesswrong.com/posts/RTkatYxJWvXR4Qbyd/deceptive-alignment-is-less-than-1-likely-by-default%23Foundational_properties_for_deceptive_alignment) which are generally seen as prerequisites of deceptive alignment: Goal-directed behavior, Optimizing across episodes/long-term goal horizons, Conceptualization of the base goal, Situational awareness.
  + Miscellaneous baseline strategies for near human-level AI:
    - In NLP, monitoring the plan and chain-of-thought of AutoGPT ([details](https://docs.google.com/document/d/e/2PACX-1vSedy4vmfA5H30bimiSGWykDfh8FB_uYKCt6D2qz9nwmfhGUc93H3UEPN1pBtyXe-eKEdu0E5oUbSWR/pub#kix.cyunx152z1bn) and caveats).
    - In vision, EfficientZero-like consistency loss - to see into the future in the Monte Carlo tree ([draft](https://docs.google.com/document/d/1GiYfx77cE6-VyeNN31tVUARt5X7tbX4XD0CSmBkReUc/edit%23bookmark%3Did.dgd983y7t1ym))
  + [More](https://twitter.com/StephenLCasper/status/1650662566476279809) ideas like Neural distillation and [Speed Priors](https://www.alignmentforum.org/posts/KSWSkxXJqWGd5jYLB/the-speed-simplicity-prior-is-probably-anti-deceptive%23How_to_actually_implement_a_speed_prior).
  + If DeepMind was to announce today that they had discovered deception in a GPT, it's unlikely that they would have used only interpretability to make that discovery. It's far more likely they would have used regular prompting.
* **There are preventive measures** against Deceptive Alignment which seem much more workable (See section [Preventive measures against Deception](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Preventive_measures_against_Deception_seem_much_more_workable)).
* **Conceptual advances are more urgent.** It's much more fruitful to think about deception conceptually than through interpretability. And to the best of my knowledge, interpretability hasn't taught us anything about deception yet.
  + For example,[the Simulator Theory](https://www.lesswrong.com/tag/simulator-theory) and the understanding that*GPTs can already emulate deceptive simulacra* is a bigger advance in our understanding of deceptive alignment than what has happened in interpretability for deception.
  + Conceptual considerations on deceptive alignment, as in the article[Deceptive Alignment is <1% Likely by Default](https://www.lesswrong.com/posts/RTkatYxJWvXR4Qbyd/deceptive-alignment-is-less-than-1-likely-by-default) or [How likely is deceptive alignment?](https://www.lesswrong.com/posts/A9NxPTwbw6r6Awuwt/how-likely-is-deceptive-alignment) don’t rely at all on interpretability.

|  |  |
| --- | --- |
|  |  |

*Inspired by every discussion I’ve had with friends defending interp. “Your argument for astronomy is too general”, so let's deep dive into some object-level arguments in the following section!*

What does the end story of interpretability look like? That’s not clear at all.
===============================================================================

*This section is more technical. Feel free to skip it and go straight to "*[So far my best ToI for interp: Outreach](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#So_far_my_best_ToI_for_interp__Outreach_)*" , or just read the "Enumerative safety" section, which is very important.*

Of course, it seems that interpretability in deep learning is inherently more feasible than neuroscience because we can save all activations and run the model very slowly, by trying causal modifications to understand what is happening, and allows much more control than an fMRI. But it seems to me that this is still not enough - we don't really know what we are aiming for and rely too much on serendipity. Are we aiming for:

Enumerative safety?
-------------------

**Enumerative safety, as Neel Nanda** [**puts it**](https://www.lesswrong.com/posts/qgK7smTvJ4DB8rZ6h/othello-gpt-future-work-i-am-excited-about#:~:text=enumerative%20safety%2C%20the%20idea%20that%20we%20might%20be%20able%20to%20enumerate%20all%20features%20in%20a%20model%20and%20inspect%20this%20for%20features%20related%20to%20dangerous%20capabilities%20or%20intentions.%20Seeing%20whether%20this%20is%20remotely%20possible%20for%20Othello%2DGPT%20may%20be%20a%20decent%20test%20run.)**, is the idea that we might be able to enumerate** ***all*****features** in a model and inspect this for features related to dangerous capabilities or intentions. I think this strategy is doomed from the start (from most important to less important):

* **Determining the dangerousness of a feature is a mis-specified problem.** Searching for dangerous features in the weights/structures of the network is pointless. A feature is not inherently good or bad. The danger of individual atoms is not a strong predictor of the danger of assembly of atoms and molecules. For instance, if you visualize the feature of layer 53, channel 127, and it appears to resemble a gun, does it mean that your system is dangerous? Or is your system simply capable of identifying a dangerous gun? The fact that cognition can be [externalized](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Auditing_deception_with_interp_is_out_of_reach:~:text=And%20cognition%20can%20be%20externalized.) also contributes to this point.
* **A feature is still a fuzzy concept**, and the problem of superposition and the natural abstraction hypothesis remains a hypothesis three years after those [Distill](https://distill.pub/2020/circuits/zoom-in/) papers, with very few convincing strategies to solve them. And that's not very surprising: the central conceptual notion of interpretability, the “feature”, seems to be intrinsically fuzzy and is still not defined. This is a major problem for the "enumerative safety" strategy and for iterating on neurons one by one to verify the "goodness" of each feature and obtain guarantees:
  + And because of [superposition](https://arxiv.org/abs/2209.10652), iterating over each neuron is not valid. So we can't just iterate on neurons, but we have to iterate on all sets of neurons (or worse, all directions), which is totally computationally intractable.
* **Properties of models which are dangerous are not low-level features, but high-level behavioral abilities** like being able to code, [sycophancy](https://www.lesswrong.com/posts/yRAo2KEGWenKYZG9K/discovering-language-model-behaviors-with-model-written) or various theories of mind proxies, situational awareness, or hacking.
  + A network's situational awareness will likely include several sub-features such as date and time, geographical position, and the current needs of its users. Removing these sub-features would make the model less competitive.
* [**Deep Deceptiveness**](https://www.lesswrong.com/posts/XWwvwytieLtEWaFJX/deep-deceptiveness) - In simple terms, a system can be deceptive even if no single part is dangerous because of optimization pressure, and complex interactions between the model and the environment.
* **This strategy has already been tried** for vision via automatic interpretability techniques to label all neurons, and it doesn't seem to have advanced alignment much, and most neurons evade simple interpretations:
  + NetDisect & Compositional explanations of neurons (Mu and Andreas, 2021)
  + Natural Language Descriptions of Deep Visual Features (Andreas, 2022)
  + Clip-Dissect (Oikarinen, 2022)[Towards a Visual Concept Vocabulary for GAN Latent Space](https://visualvocab.csail.mit.edu/) (Schwettmann, 2021)
  + These works [partially summarized [here](https://www.lesswrong.com/posts/XZfJvxZqfbLfN6pKh/introductory-textbook-to-vision-models-interpretability)] have not changed the way we try to make vision systems more robust and less risky in practice.
* Most automatic interpretability works, like [Language models can explain neurons in language models](https://openai.com/research/language-models-can-explain-neurons-in-language-models) from OpenAI or concept erasure techniques, falls into this category.

Reverse engineering?
--------------------

Reverse engineering is a classic example of interpretability, but I don't see a successful way forward. Would this be:

* The**equivalent C++ annotated algorithm** of what the model does? Being able to reproduce the capabilities of the inscrutable matrices of  GPT-4 by some modular C++ code by would be past human level intelligence already, and this would be[too dangerous](https://www.lesswrong.com/posts/pQqoTTAnEePRDmZN4/agi-automated-interpretability-is-suicide), because this would allow a lot of different optimization, and probably allow recursive self-improvement which seems dangerous especially if we rely on an automated process for that.
* An **explanation in layman terms** of the behavior of the model? At which level of granularity? Each token or sentence or paragraph? This is really unclear.
* The **functional connectome** of the model obtained with high level interp? Ok, you see in the functional connectome that the model is able to code and to hack, and those are dangerous capabilities. Isn’t this just regular evals?
  + In practice, to conduct interp experiments, **we almost always start by creating a dataset of prompts.** Maybe one day we won't need prompts to activate these capabilities, but I don't see that happening anytime soon.
* A **graph** to explain the circuits? Graphs like the ones just below can be overwhelming and remain very limited.

You can notice that “Enumerative safety” is often hidden behind the “reverse engineering” end story.

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/LNA8mubrByG7SFacm/hpuwlgsxjtrumo38ckav)

*From the* [*IOI paper*](https://arxiv.org/abs/2211.00593)*. Understanding this diagram from 'Interpretability in the Wild' by Wang et al. 2022 is not essential for our discussion. Understanding the full circuit and the method used would require a* [*three-hour video*](https://www.youtube.com/watch?v%3Dgzwj0jWbvbo)*. And, this analysis only focuses on a single token and involves numerous simplifications. For instance, while we attempt to explain why the token 'Mary' is preferred over 'John', we do not delve into why the model initially considers either 'Mary' or 'John'. Additionally, this analysis is based solely on GPT2-small.*

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/LNA8mubrByG7SFacm/mmmq7xwly9mufts4101o)

*Indeed, this figure is quite terrifying. from*[*Causal scrubbing: results on induction heads*](https://www.lesswrong.com/posts/j6s9H9SHrEhEfuJnq/causal-scrubbing-results-on-induction-heads)*, for a 2 layer model. After refining 4 times the hypothesis, they are able to restore 86% of the loss. But even for this simple task they say “we won’t end up reaching hypotheses that are fully specific or fully human-understandable, causal scrubbing will allow us to validate claims about which components and computations of the model are important.”.*

The fact that reverse engineering is already so difficult in the two toy examples above seems concerning to me.

Olah’s interpretability dream?
------------------------------

Or maybe interp is just an exploration driven by curiosity waiting for serendipity?

* [Interpretability Dreams](https://transformer-circuits.pub/2023/interpretability-dreams/index.html) is an informal note by Chris Olah on future goals for mechanistic interpretability. It discusses **superposition**, the enemy of interpretability. Then, towards the end of the note, In the section titled “[How Does Mechanistic Interpretability Fit Into Safety?](https://transformer-circuits.pub/2023/interpretability-dreams/index.html%23safety)”, we understand the plan is to solve superposition to be able to use the following formula:  
    
  ![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/LNA8mubrByG7SFacm/bbzkzi7n1cc1w6cvueju)
* But this is simply again “*enumerative safety”* stated in terms of circuits rather than features. However, as explained above, I don't think this leads us anywhere.
* The final section of the note,[Beauty and Curiosity](https://transformer-circuits.pub/2023/interpretability-dreams/index.html%23aesthetics), reads like a poem or hymn to beauty. However, it seems to lack substance beyond a hope for serendipitous discovery.

Overall, I am skeptical about Anthropic's use of the dictionary learning approach to solve the superposition problem. While their negative results are interesting, and they are working on addressing conceptual difficulties around the concept of "feature" (as noted in their[May update](https://transformer-circuits.pub/2023/may-update/index.html%23superposition-dictionary)), I remain unconvinced about the effectiveness of this approach, even after reading their[recent July updates](https://transformer-circuits.pub/2023/july-update/index.html%23safety-features), which still do not address my objections about enumerative safety.

One potential solution Olah [suggests](https://transformer-circuits.pub/2023/interpretability-dreams/index.html) is automated research: "*it does seem quite possible that the types of approaches […] will ultimately be insufficient, and interpretability may need to rely on AI automation*". However, I believe that this kind of automation is potentially harmful [section[Harmful](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Interpretability_May_Be_Overall_Harmful)].

This is still a developing story, and the papers published on Distill are always a great pleasure to read. However, I remain hesitant to bet on this approach.

Retargeting the search?
-----------------------

**Or maybe interp could be useful for**[**retargeting the search**](https://www.lesswrong.com/posts/w4aeAFzSAguvqA5qu/how-to-go-from-interpretability-to-alignment-just-retarget)**?** This idea suggests that if we find a goal in a system, we can simply change the system's goal and redirect it towards a better goal.

I think this is a promising quest, even if there are still difficulties:

* This is interesting because this would be a way to not need to fully reverse engineer a complete model.The technique used in [Understanding and controlling a maze-solving policy network](https://www.alignmentforum.org/posts/cAC4AXiNC5ig6jQnc/understanding-and-controlling-a-maze-solving-policy-network) seems promising to me. Just focusing on “the motivational API” could be sufficient.
* But I still don’t know if[Steering vectors](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vecto) (i.e. activation additions of a vector in the latent space) really count as interpretability, and really change significantly the picture of alignment beyond just prompt engineering. Ok, this is a new way to tinker with the model. But I don’t know how this could be used reliably against deception.[[7]](#fnmf7vlk6ib69)

Relaxed adversarial training?
-----------------------------

**Relaxed adversarial training?** The TL;DR is that [relaxed adversarial training](https://ai-alignment.com/training-robust-corrigibility-ce0e0a3b9b4d) is the same as adversarial training, but instead of creating adversarial inputs to test the network, we create adversarial latent vectors. This could be useful because creating realistic adversarial inputs is a bottleneck in adversarial training. [More explanations [here](https://docs.google.com/document/d/1KXEWXHKwgeu-0NX5iirGS1h5zsh1skYMadZN3ZoVMAI/edit%23bookmark%3Did.2ats8akz8z6u)]

This seems valid but very hard, and there are still significant [conceptual difficulties](https://ai-alignment.com/training-robust-corrigibility-ce0e0a3b9b4d). A concrete approach,[Latent Adversarial Training](https://www.lesswrong.com/posts/atBQ3NHyqnBadrsGP/latent-adversarial-training), has been proposed, and seems to be promising but:

* **The procedure is underspecified**. **There will be too many meta-parameters**. Calibrating these meta-parameters will require some iteration, and you probably don’t want to iterate on deceptive powerful models. We have to be good right away from the first choice of meta-parameters. As the author himself says, "*the only hope here lies in the Surgeon forcing the model to be robustly safe before it learns to deceive. Once the model is deceptive it's really game-over.*"
* **We still have no guarantees.** This procedure allows for a latent space that is robust to “small perturbations”, but being robust to “small perturbations” is *not the same as not becoming deceptive* (it’s not clear to me that deception won’t appear outside the constraint zone).
* Papers using this kind of procedure have only limited effectiveness, for example around 90% detection rate in the paper [ABS: Scanning Neural Networks for Back-doors by Artificial Brain Stimulation (Liu et al., 2019)](https://www.cs.purdue.edu/homes/taog/docs/CCS19.pdf). [Paper summarized [here](https://docs.google.com/document/d/1KXEWXHKwgeu-0NX5iirGS1h5zsh1skYMadZN3ZoVMAI/edit%23bookmark%3Did.ewixtcjxjqvq)] And I don’t think this could work against all types of trojans.

The exact procedure described in [Latent Adversarial Training](https://www.lesswrong.com/posts/atBQ3NHyqnBadrsGP/latent-adversarial-training) hasn't been tested, as far as I know. So we should probably work on it.[[8]](#fnc2q5uxqhj6j)

Microscope AI?
--------------

**Maybe Microscope AI i.e.** Maybe we could directly use the AI’s world model without having to understand everything. Microscope AI is an AI that would be used not in inference, but would be used just by looking at its internal activations or weights, without deploying it. My definition would be something like: We can run forward passes, but only halfway through the model.

* This goes against almost every economic incentive (see [Why Tool AIs wants to become Agents AI](https://gwern.net/tool-ai%23:~:text%3DAn%2520Agent%2520AI%2520has%2520the,its%2520outputs%252C%2520on%2520harder%2520domains.), from Gwern).
* **($) Interpretability has been mostly useless for discovering facts about the world, and learning new stuff by only looking at the weights is too hard.**
  + In the paper[Acquisition of Chess Knowledge in AlphaZero](https://arxiv.org/abs/2111.09259), the authors investigate whether “*we can learn chess strategies by interpreting the trained AlphaZero's **behavior***”. Answer: This is not the case. They probe the network using only concepts already known to Stockfish, and no new fundamental insights are gained. We only check *when* AlphaGo learns human concepts during the training run.
  + I don’t think we will be able to learn category theory by reverse engineering the brain of Terence Tao. How do Go players learn strategies from go programs? Do they interpret AlphaGo’s weights, or do they try to understand the behavioral evaluations of those programs? Answer: They learn from their behavior, but not by interpreting models. I am skeptical that we can gain radically new knowledge from the weights/activations/circuits of a neural network that we did not already know, especially considering how difficult it can be to learn things from English textbooks alone.
* **Microscope AIs should not be agentic by definition. But agency and exploration help tremendously at the human level for discovering new truths. Therefore, below superhuman level, the** ***microscope*****needs to be** ***agentic*****…and this is a contradiction.** Using Microscope AI as a tool rather than an agent is suggested [here](https://www.lesswrong.com/posts/Go5ELsHAyw7QrArQ6/searching-for-a-model-s-concepts-by-their-shape-a%23Philosophical_framing) or [here](https://www.lesswrong.com/posts/X2i9dQQK3gETCyqh2/chris-olah-s-views-on-agi-safety) for example. However, to know the truth of a complex fact, we need to experiment with the world and actively search for information. Here is a fuzzy reasoning (feel free to skip):
  + A) Either **the information already exists and is written plainly** somewhere on the internet, and in that case, there is no need for Microscope AI (this is like text retrieval).
  + B) Or **the information doesn't exist anywhere on the internet**, and in that case, it is necessary to be agentic by experimenting with the world or by thinking actively. This is the type of feature that can only be “created” by reinforcement learning but which cannot be “discovered” with supervised learning, like MuZero discovering new chess strategies.
  + or C), **this info is not plainly written but is a deep feature of the training data** that could be understood/grokked through gradient descent. This is the type of feature that can be “discovered” with supervised learning.
  + If B), we need agency, and it’s no longer a microscope.
  + If C), we can apply the above reasoning ($) + Being able to achieve this through pure gradient descent without exploration is probably a higher level of capability than being able to do it with exploration. (This would be like discovering the[Quaternion](https://en.wikipedia.org/wiki/Quaternion) formula during a dream?). But even legendary mathematicians need to work a bit and be agentic in their exploration; they don't just passively read textbooks. Therefore, it's probably beyond Ramanujan's level and too dangerous?
  + So, I'm quite uncertain, but overall I don't think Microscope AI is a promising or valid approach to reducing AI risk.

A short case study of[Discovering Latent Knowledge](https://www.lesswrong.com/posts/L4anhrxjv8j2yRKKp/how-discovering-latent-knowledge-in-language-models-without) technique to extract knowledge from models by probing is included in the [appendix](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Burns_et_al___2022_).

So far my best ToI for interp: Outreach?
========================================

1.  **Interp for Nerd Sniping/honeypot?**

* **Interp is a highly engaging introduction to AI research**. That's really cool for that, I use it for my [classes](https://www.master-mva.com/cours/seminaire-turing/), and for technical outreach, but I already have enough material on interpretability, for 10 hours of class, no need to add more.
* **Interp as a honeypot for junior researchers?** Just as a honeypot attracts bees with its sweet nectar, interp is very successful for recruiting new technical people! but then they would probably be better off doing something else than interp (unless it is their strong comparative advantage).
* (Nerd Sniping senior capability researchers into interpretability research? Less capability research, more time to align AIs? I’m joking, don’t do that at home! )

2.  **Honorable mentions:**

* **Showing strange failures**, such as the issue with the[SolidGoldMagicCarp](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) token, highlights the possibility of unexpected results with the model. More generally, interpretability tools can be useful for the red teaming toolbox. They seem like they might be able to guide us to more problems than test sets and adversaries can alone.
* **Showing GPT is not a stochastic parrot?** The article [Actually, Othello-GPT Has A Linear Emergent World Representation](https://www.lesswrong.com/posts/nmxzr2zsjNtjaHh7x/actually-othello-gpt-has-a-linear-emergent-world)is really cool**.** Showing that OthelloGPT contains a world model is really useful for technical outreach (even if OthelloGPT being good at Othello should be enough, no?).
* **It's a good way to introduce the importance and tractability of alignment research** “*Interpretability gives people a non-technical story for how alignment affects their lives, the scale of the problem, and how progress can be made. IMO no other approach to alignment is anywhere near as good for this.”* [from [Raymond D](https://www.lesswrong.com/posts/uK6sQCNMw8WKzJeCQ/a-longlist-of-theories-of-impact-for-interpretability?commentId%3DuzBFJDsy9Jqkxzdnx)]
* **Better: Showing that “We have basically no idea how it does what it does.”,** see this [tweet](https://twitter.com/robertskmiles/status/1663534255249453056):

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/LNA8mubrByG7SFacm/q9gbkwdp3drer7soei5a)

Preventive measures against Deception seem much more workable
=============================================================

TL;DR: It would be more desirable to aim for a world where we wouldn't need to delve into the internal components of models. Prevention is better than cure, or at least, it is a neglected strategy.

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/LNA8mubrByG7SFacm/lon9yhtv1wgba17yhryt)

*From* [*The Translucent Thoughts Hypotheses and Their Implications*](https://www.lesswrong.com/posts/r3xwHzMmMf25peeHE/the-translucent-thoughts-hypotheses-and-their-implications%23The_Translucent_Thoughts_Hypotheses)*, Fabien Roger*

I don't believe interpretability is the most promising method for monitoring near human-level AGI, Here’s why:

**I don't think neural networks will be able to take over in a single forward pass. Models will probably reason in English and will have translucent thoughts** (and we could even hope to get transparent thoughts, translucence is the scary place where steganography is possible). In order to devise an effective plan to destroy the world, the first human-level AGI will need to somewhat externalize its reasoning, by chain-of-Thought (CoT), and it seems to me way easier to monitor this chain-of-Thought than probing into the models' internals. We can probably use [The Translucent Thoughts Hypotheses](https://www.lesswrong.com/posts/r3xwHzMmMf25peeHE/the-translucent-thoughts-hypotheses-and-their-implications) (Fabien, the author, gives a 20% chance, but I think it will be more like 60%). I also think that we have to do everything we can to pass regulations and **place ourselves in a world where those hypotheses are true.**

For example, [magic is possible](https://www.youtube.com/watch?v%3DEAN-PwRfJcA), and the fact that human brains can be manipulated so easily by magic is a great danger. So we should probably try to make this magic unlikely, for example with process-based training by training AI Systems to reason only step-by-step (the process-based training approach is summarized in this [AGISF distillation](https://www.lesswrong.com/posts/FFz6H35Gy6BArHxkc/task-decomposition-for-scalable-oversight-agisf-distillation)).

Steering the world towards transparency
---------------------------------------

**Thinking ahead about the training setup is very important and possibly dominates interp considerations** because if the concepts manipulated and by the models are totally alien, it will be much, much harder to provide oversight. And it is [much](https://aligned.substack.com/i/88447351/the-ai-tech-tree-is-looking-favorably) easier to align those chatbots pretrained on human generated text than aligning AGIs trained from scratch with RL / evolutionary methods, etc.

If this is the case, we should focus on various aspects and important recommendations as detailed by Fabien[here](https://www.lesswrong.com/posts/r3xwHzMmMf25peeHE/the-translucent-thoughts-hypotheses-and-their-implications%23Research_Directions). Here are some highlights and comments:

* **Making LLMs use only understandable English:** Preventing models from using steganography, not training on synthetic data, etc… These directions are generally very neglected - for example, look at this [comment](https://www.lesswrong.com/posts/bwyKCQD7PFWKhELMr/by-default-gpts-think-in-plain-sight?commentId%3DS4wqnxW5CYd8QsTox).
* **Building an oversight system:** That’s the “[ready to go](https://youtu.be/YTlrPeikoyw?si=JTORDWwOHZzpEM7P&t=2399)” technical research of Buck, IDA, or neglected baselines like [filtering](https://arxiv.org/abs/2308.07308), etc…
* **Testing the translucent thoughts hypotheses** and being able to monitor the alienness of models is super important in ensuring some simulacra have brain like representations (see [this](https://www.lesswrong.com/posts/iXbPe9EAxScuimsGh/linkpost-scaling-laws-for-language-encoding-models-in-fmri) and [this](https://www.lesswrong.com/posts/2QexGHrqSxcuwyGmf/linkpost-large-language-models-converge-on-brain-like-word)). More work in the direction of [Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/abs/2305.04388).
* **Making the hypotheses true**, with technical works, outreach and governance, etc.

**I think the strategy should be: let’s target a world where deception is unlikely.** (I'm not saying we should make plans that work conditional on deception being unlikely by default, but we should try to steer AGI/the world towards a place where deception is unlikely). I believe there are multiple ways to think and address this problem, and much more technical research needed here, starting from [Conditioning Predictive Models: Risks and Strategies](https://arxiv.org/abs/2302.00805).

Cognitive Emulations - Explainability *By Design*
-------------------------------------------------

**If interpretability was really a bottleneck, we could use**[**cognitive emulation**](https://www.lesswrong.com/posts/ngEvKav9w57XrGQnb/cognitive-emulation-a-naive-ai-safety-proposal)**, which, in my opinion, allows way better explainability and transparency than interpretability will ever get us.**

My understanding of cognitive emulation: Emulating GPT-4 using LLMs like GPT-3 as different submodules that send messages written in *plain English* to each other before outputting the next token. If the neural network had deceptive thoughts, we could see them in these intermediate messages.

Some caveats are in the section [Cognitive Emulation](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Cognitive_Emulations___Explainability_By_design) of the appendix.

Interpretability May Be Overall Harmful
=======================================

(Note that some of the following points are not specific to interp, but I think they apply particularly well to interp.)

**False sense of control:**

* **False sense of understanding.** It's too easy to think you begin to understand that we're starting to get guarantees when we have not much. This is very classic:
  + Me from the past: "Yo, I spent 5 hours trying to understand the mechanisms of inductions head and K-Compositions in a[Mathematical Framework for Transformers](https://transformer-circuits.pub/2021/framework/index.html), I have so much more understanding." yes but no.
* **Overinterpretation.** It is very difficult to say which interpretation result is solid. For example, [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292) shows that most of the pixel attribution techniques are generally misleading.[[9]](#fnpo4e41md3r) In the same vein, feature viz has recently been found to have some pretty fatal flaws, see [Don't trust your eyes: on the (un)reliability of feature visualizations](https://arxiv.org/abs/2306.04719), and the model editing technique such as [ROME](https://rome.baulab.info/) is [very misleading](https://www.lesswrong.com/posts/QL7J9wmS6W2fWpofd/but-is-it-really-in-rome-an-investigation-of-the-rome-model).  This is mostly due to methodological problems that Stephen Casper explains in his sequence. [see appendix: [methodological problems](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Methodological_problems_)].
* **Safety Washing.** I feel that there is a part of safety research which is here to legitimize capability research in the big labs (although this is not entirely specific to interp).
  + *“I think a really substantial fraction of people who are doing "AI Alignment research" are instead acting with the primary aim of "make AI Alignment seem legit". These are not the same goal, a lot of good people can tell and this makes them feel kind of deceived, and also this creates very messy dynamics within the field where people have strong opinions about what the secondary effects of research are, because that's the primary thing they are interested in, instead of asking whether the research points towards useful true things for actually aligning the AI”,* from [Shutting Down the Lightcone Offices](https://www.lesswrong.com/posts/psYNRb3JCncQBjd4v/shutting-down-the-lightcone-offices).
* **The achievements of interp research are consistently graded on their own curve and  overhyped** compared to achievements in other fields like adversaries research. For example, the recent paper [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043) impressively found effective attacks against state-of-the-art models without any interpretations involving models internals. Imagine if mechanistic interpretability researchers did the exact same thing, but by studying model internals? Given the excitement that has emerged in the past around the achievements of mechanistic interpretability in toy models on cherry-picked problems (e.g. [this](https://arxiv.org/abs/2301.05217) or [this](https://arxiv.org/abs/2211.00593)), it seems that something like this would have probably made the AI safety research community go wild. Stephen Casper makes a similar point [here](https://www.alignmentforum.org/s/a6ne2ve5uturEEQK7/p/MyvkTKfndx9t4zknh%23:~:text%3DFrom%2520an%2520engineer%25E2%2580%2599s%2520perspective%252C%2520it%25E2%2580%2599s%2520important%2520not%2520to%2520grade%2520different%2520classes%2520of%2520solutions%2520each%2520on%2520different%2520curves.%25C2%25A0): “*From an engineer’s perspective, it’s important not to grade different classes of solutions each on different curves.*” And other examples of this are presented here [EIS VI: Critiques of Mechanistic Interpretability Work in AI Safety](https://www.alignmentforum.org/s/a6ne2ve5uturEEQK7/p/wt7HXaCWzuKQipqz3%23Imagine_that_you_heard_news_tomorrow_that_MI_researchers_from_TAISIC_meticulously_studied_circuits_in_a_way_that_allowed_them_to_:~:text%3Dtechniques%2520on%2520curves...-,Imagine%2520that%2520you%2520heard%2520news%2520tomorrow%2520that%2520MI%2520researchers%2520from%2520TAISIC%2520meticulously%2520studied%2520circuits%2520in%2520a%2520way%2520that%2520allowed%2520them%2520to%25E2%2580%25A6,-Reverse%2520engineer%2520and) (thanks to Stephen for highlighting this point).

**The world is not coordinated enough for public interpretability research:**

* **Dual use.** It seems anything related to information representation can be used in a dual manner. This is a problem because I believe that the core of interpretability research could lead to major advances in capabilities. See this[post](https://www.lesswrong.com/posts/iDNEjbdHhjzvLLAmm/should-we-publish-mechanistic-interpretability-research).
  + Using the insights provided by advanced interp to improve capabilities, such as modularity to optimize inference time and reduce flops, is likely to be easier than using them for better oversight. This is because **optimizing for capability is much simpler than optimizing for safety**, as we lack clear metrics for measuring safety (see the figure below).
* **When interpretability starts to be useful, you can't even publish it because it's too info hazardous.** The world is not coordinated enough for public interpretability research.
  + Nate Soares[explained](https://www.lesswrong.com/posts/BinkknLBYxskMXuME/if-interpretability-research-goes-well-it-may-get-dangerous) this, and this was followed by[multiple](https://www.lesswrong.com/posts/uhMRgEXabYbWeLc6T/why-and-when-interpretability-work-is-dangerous)[posts](https://www.lesswrong.com/posts/pQqoTTAnEePRDmZN4/agi-automated-interpretability-is-suicide). *“Insofar as interpretability researchers gain understanding of AIs that could significantly advance the capabilities frontier, I encourage interpretability researchers to keep their research* [*closed*](https://www.lesswrong.com/posts/tuwwLQT4wqk25ndxk/thoughts-on-agi-organizations-and-capabilities-work)*. […] I acknowledge that public sharing of research insights could, in principle, both shorten timelines and improve our odds of success. I suspect that* [*isn’t the case in real life*](https://www.lesswrong.com/posts/vQNJrJqebXEWjJfnz/a-note-about-differential-technological-development)*.”*
  + Good interp could produce a "foom overhang" as described in "[AGI-Automated Interpretability is Suicide](https://www.lesswrong.com/posts/pQqoTTAnEePRDmZN4/agi-automated-interpretability-is-suicide)".
  + Good interp also creates an [infosec/infohazard attack vector](https://www.lesswrong.com/posts/CRrkKAafopCmhJEBt/ai-interpretability-could-be-harmful).
  + The post '[Why and When is Interpretability Work Dangerous?](https://www.lesswrong.com/posts/uhMRgEXabYbWeLc6T/why-and-when-interpretability-work-is-dangerous)' ends on a sobering note, stating, “*In closing, if alignment-conscious researchers continue going into the interpretability subfield, the probability of AGI ruin will tend to increase.*”
* **Interpretability already helps capabilities.** For example, the understanding of Induction head has allowed for[better](https://twitter.com/NeelNanda5/status/1618185819285778433) architectures[[10]](#fnplu4ji16iui).
* Interpretability may be a [super wicked problem](https://en.wikipedia.org/wiki/Wicked_problem%23Super_wicked_problems)[[11]](#fnw7s6gsvuwb).

Thus the list of "theory of impact" for interpretability should not simply be a list of benefits. It's important to explain why these benefits outweigh the possible negative impacts, as well as how this theory can save time and mitigate any new risks that may arise.

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/LNA8mubrByG7SFacm/zao0mylkanwh60aajinm)

*The concrete application of the* [*logit lens*](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)*is not an oversight system for deception, but rather capability works to accelerate inference speed like in* [*this paper*](https://twitter.com/GoogleAI/status/1603845007663734785)*. (Note that the paper does not cite logit lens, but relies on a very similar method).*

Outside view: The proportion of junior researchers doing interp rather than other technical work is too high
============================================================================================================

It seems to me that many people start alignment research as follows:

* At the end of[Arena](https://www.arena.education/), an advanced upskilling program in AI Safety, almost all research projects this year (June 2023), except for two out of 16, were interp projects.
* At [EffiSciences](https://ia.effisciences.org/), at the end of the last 3 [ML4Good](https://www.lesswrong.com/posts/DkDy2hvkwbQ54GM9u/introducing-effisciences-ai-safety-unit-1) bootcamps, students all start by being interested in interp, and it is a very powerful attractor. I myself am guilty. I have redirected too many people to it. I am now trying to correct my ways.
  + In the past, if I reconstruct my motivational story, it goes something like this: "Yo, I have a math/ML background, how can I recycle that?" --> then *brrr interp*, without asking too many questions.
* During [Apart Research](https://apartresearch.com/) hackathons, interpretability hackathons tend to draw 3.12 times as many participants as other types of hackathons. (thinkathon, safety benchmarks, …).[[12]](#fnavln8kyvlzg)
* Interpretability streams in [Seri Mats](https://www.serimats.org/) are among the most competitive streams (see this [tweet](https://twitter.com/leedsharkey/status/1656705667963535370)). People then try hard, get rejected, get disappointed and lose motivation. This is a recent important problem.

"Not putting all your eggs in one basket" seems more robust considering our uncertainty, and there are more promising ways to reduce x-risk per unit of effort (to come in a future post, mostly through helping/doing governance). I would rather see a **more diverse ecosystem** of people trying to reduce risks. More on this in section [Technical Agendas with better ToI](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Technical_Agendas_with_better_ToI).

If you ask me if interp is also over represented in senior researchers, I'm a bit less confident. Interp also seems to be a significant portion of the pie: this year, while Conjecture and Redwood have partially pivoted, there are new active interp teams in Apollo, DeepMind, OpenAI, and still in Anthropic. I think I would particularly critique DeepMind and OpenAI's interpretability works, as I don't see how this reduces risks more than other works that they could be doing, and I'd appreciate a written plan of what they expect to achieve.

Even if we completely solve interp, we are still in danger
==========================================================

No one has ever claimed otherwise, but it's worth remembering to get the big picture. From stronger arguments to weaker ones:

* **There are many X-risks scenarios, not even involving deceptive AIs.** Here is a list of such scenarios (see this [cheat sheet](https://www.lesswrong.com/posts/nCeyBbhtJhToBFmrL/cheat-sheet-of-ai-x-risk)):
  + Christiano1 -[You get what you measure](https://www.lesswrong.com/posts/HBxe6wdjxK239zajf/more-realistic-tales-of-doom%23Part_I__You_get_what_you_measure)
  + Critch1 -[Production Web](https://www.lesswrong.com/posts/LpM3EAakwYdS6aRKf/what-multipolar-failure-looks-like-and-robust-agent-agnostic%23Part_1__Slow_stories__and_lessons_therefrom)
  + Soares -[A central AI alignment problem: capabilities generalization, the sharp left turn](https://www.lesswrong.com/posts/GNhMPAWcfBCASy8e6/a-central-ai-alignment-problem-capabilities-generalization)
  + Cohen et al. -[Advanced artificial agents intervene in the provision of reward](https://onlinelibrary.wiley.com/doi/10.1002/aaai.12064)
  + Gwern -[It Looks Like You’re Trying To Take Over The World](https://gwern.net/fiction/clippy)
  + Exercise: Here is[a list of risks](https://www.safe.ai/ai-risk) from the Center of AI Safety. Which ones can be solved by interp? At least half of those risks don’t directly involve deception and interp.
* **Total explainability of complex systems with great power is not sufficient to eliminate risks.** Significant risks would still remain. Despite our full understanding of how atomic bombs function, they still pose substantial risks. See this[list of nuclear close calls](https://en.wikipedia.org/wiki/List_of_nuclear_close_calls).
* **Interpretability implicitly assumes that the AI model does not optimize in a way that is adversarial to the user.** Consider being able to read the mind of a psychopath like Voldemort. Would this make you feel safe? The initial step remains to box him. However, **a preferable scenario would be not having to confront this situation at all.** (this last claim is probably the most important lesson - see [Preventive measures](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Preventive_measures_against_Deception_seem_much_more_workable)).

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/LNA8mubrByG7SFacm/s6qlqvxf5an1gdofmmk3)

Pytorch hooks can be used to study the internals of models. Are they going to be sufficient? *Idk, but*[*Hook Me up Baby*](https://www.youtube.com/watch?v=LveUcCBRrSo)*, from the album “Take Me as I Am” could be the national anthem of interp.*

**That is why focusing on coordination is crucial! There is a level of coordination above which we don’t die - there is no such threshold for interpretability.** We currently live in a world where coordination is way more valuable than interpretability techniques. So let’s not forget that[non-alignment aspects of AI safety are key!](https://www.alignmentforum.org/posts/FfTxEf3uFPsZf9EMP/avoiding-perpetual-risk-from-tai%23Non_alignment_aspects_of_AI_safety_are_key_) AI alignment is only a subset of AI safety! (I’m planning to deep-dive more into this in a following post).

A version of this argument applies to "alignment" in general and not just interp and those considerations will heavily influence my recommendations for technical agendas.

Technical Agendas with better ToI
=================================

Interp is not such a bad egg, but opportunity costs can be huge (especially for researchers working in big labs).

I’m not saying we should stop doing technical work. Here's a list of technical projects that I consider promising (though I won't argue much for these alternatives here):

* **Technical works used for AI Governance.** A huge amount of technical and research work needs to be done in order to make regulation robust and actually useful. Mauricio’s [AI Governance Needs Technical Work](https://forum.effectivealtruism.org/posts/BJtekdKrAufyKhBGw/ai-governance-needs-technical-work), or the governance section of [AGI safety career advice](https://www.lesswrong.com/posts/ho63vCb2MNFijinzY/agi-safety-career-advice#Governance_work) by Richard Ngo is really great : “*It’s very plausible that, starting off with no background in the field, within six months you could write a post or paper which pushes forward the frontier of our knowledge on how one of those topics is relevant to AGI governance.*”
  + For example, each of the measures proposed in the paper [towards best practices in AGI safety and governance: A survey of expert opinion](https://arxiv.org/abs/2305.07153) could be a pretext for creating a specialized organization to address these issues, such as auditing, licensing, and monitoring.
  + Scary demos (But this shouldn't involve gain-of-function research. There are already many powerful AIs [available](https://twitter.com/AlexReibman/status/1692009609953993069). Most of the work involves video editing, finding good stories, distribution channels, and creating good memes. Do not make AIs more dangerous just to accomplish this.).
  + In the same vein, [Monitoring for deceptive alignment](https://www.lesswrong.com/posts/Km9sHjHTsBdbgwKyi/monitoring-for-deceptive-alignment) is probably good because “[AI coordination needs clear wins](https://www.lesswrong.com/posts/vavnqwYbc8jMu3dTY/ai-coordination-needs-clear-wins)”.
  + Interoperability in AI policy, and good definitions usable by policymakers.
  + Creating benchmarks for dangerous capabilities.
  + Here’s a [list](https://docs.google.com/document/d/1Tvz2JS8CZ51TW-vfU3vwRn8dpK3F0UttdhMrZC2o7hw/edit#bookmark=id.jnvbactyuay7) of other ideas
* **Characterizing the technical difficulties of alignment. (**[**Hold Off On Proposing Solutions**](https://www.lesswrong.com/posts/uHYYA32CKgKT3FagE/hold-off-on-proposing-solutions) **“Do not propose solutions until the problem has been discussed as thoroughly as possible without suggesting any.”)**
  + Creating the [IPCC](https://en.wikipedia.org/wiki/Intergovernmental_Panel_on_Climate_Change) of AI Risks
  + More red-teaming of agendas
  + Explaining problems in alignment.
* Adversarial examples, adversarial training, latent adversarial training (the only [end-story](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Relaxed_adversarial_training_) I’m kind of excited about). For example, the papers "[Red-Teaming the Stable Diffusion Safety Filter](https://arxiv.org/abs/2210.04610)" or "[Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)" are good (and pretty simple!) examples of adversarial robustness works which contribute to safety culture.
* **Technical outreach**. [AI Explained](https://www.youtube.com/@ai-explained-) and [Rob Miles](https://www.youtube.com/c/robertmilesai) have plausibly reduced risks  more than all interpretability research combined.
* In essence, ask yourself: “What would Dan Hendrycks do?”
  + Technical newsletter, non-technical newsletters, benchmarks, policy recommendations, risks analysis, banger statements, courses and technical outreach.
  + He is not doing interp. Checkmate!

In short, my agenda is **"Slow Capabilities through a safety culture"**, which I believe is robustly beneficial, even though it may be difficult. I want to help humanity understand that we are not yet ready to align AIs. Let's wait a couple of decades, then reconsider.

And if we really have to build AGIs and align AIs, it seems to me that it is more desirable to aim for a world where we don't need to probe into the internals of models. Again, prevention is better than cure.

Conclusion
==========

I have argued against various theories of impact of interpretability, and proposed some alternatives. I believe working back from the different risk scenarios and red-teaming the theories of impact gives us better clarity and a better chance at doing what matters. Again, I hope this document opens discussions, so feel free to respond in parts. There probably *should* be a non-zero amount of researchers working on interpretability, this isn’t intended as an attack, but hopefully prompts more careful analysis and comparison to other theories of impact.

We already know some broad lessons, and we already have a general idea of which worlds will be more or less dangerous.Some ML researchers in top labs aren't even aware of, or acknowledging, that AGI is dangerous, that connecting models to the internet, encouraging agency, doing RL and maximizing metrics isn't safe in the limit.

Until civilization catches up to these basic lessons, we should avoid playing with fire, and should try to slow down the development of AGIs as much as possible, or at least steer towards worlds where it’s done only by extremely cautious and competent actors.

Perhaps the main problem I have with interp is that it implicitly reinforces the narrative that we must build powerful, dangerous AIs, and then align them. For X-risks, prevention is better than cure. Let’s *not* build powerful and dangerous AIs. We aspire for them to be safe, by design.

Appendix
========

Related works
-------------

There is a vast academic literature on the virtues and academic critiques of interpretability (see this [page](https://www.lesswrong.com/posts/gwG9uqw255gafjYN4/eis-iii-broad-critiques-of-interpretability-research) for plenty of references), but relatively little holistic reflection on interpretability as a strategy to reduce existential risks.

The most important articles presenting arguments for interpretability:

* [Chris Olah’s views on AGI safety](https://www.lesswrong.com/posts/X2i9dQQK3gETCyqh2/chris-olah-s-views-on-agi-safety)
* [A Longlist of Theories of Impact for Interpretability](https://www.lesswrong.com/posts/uK6sQCNMw8WKzJeCQ/a-longlist-of-theories-of-impact-for-interpretability)
* [Interpretability Dreams](https://transformer-circuits.pub/2023/interpretability-dreams/index.html)
* [Why and When Interpretability Work is Dangerous](https://www.lesswrong.com/posts/uhMRgEXabYbWeLc6T/why-and-when-interpretability-work-is-dangerous%23When_Interpretability_is_Still_Important)
* [The Defender’s Advantage of Interpretability](https://www.lesswrong.com/posts/6ReBeYwsDeNgv6Dr5/the-defender-s-advantage-of-interpretability)
* [Monitoring for deceptive alignment](https://www.lesswrong.com/posts/Km9sHjHTsBdbgwKyi/monitoring-for-deceptive-alignment)

Against interpretability

* [AGI-Automated Interpretability is Suicide](https://www.lesswrong.com/posts/pQqoTTAnEePRDmZN4/agi-automated-interpretability-is-suicide)
* [Why and When Interpretability Work is Dangerous](https://www.lesswrong.com/posts/uhMRgEXabYbWeLc6T/why-and-when-interpretability-work-is-dangerous)
* [Should we publish mechanistic interpretability research?](https://www.lesswrong.com/posts/iDNEjbdHhjzvLLAmm/should-we-publish-mechanistic-interpretability-research)
* [EIS III: Broad Critiques of Interpretability Research](https://www.lesswrong.com/posts/gwG9uqw255gafjYN4/eis-iii-broad-critiques-of-interpretability-research)
* [EIS VI: Critiques of Mechanistic Interpretability Work in AI Safety](https://www.lesswrong.com/s/a6ne2ve5uturEEQK7/p/wt7HXaCWzuKQipqz3)
* [Against Interpretability: a Critical Examination of the Interpretability Problem in Machine Learning](https://link.springer.com/article/10.1007/s13347-019-00372-9)

### The Engineer’s Interpretability Sequence

I originally began my investigation by rereading  “The Engineer’s Interpretability Sequence”, in which Stephen Casper raises many good critiques of interpretability research, and this was really illuminating.

**Interpretability tools lack widespread use by practitioners in real applications.**

* No interpretability technique is yet publicly known to have been used in production in SOTA models such as ChatGPT.
* There have been interpretability studies of SOTA multimodal models such as[CLIP](https://distill.pub/2021/multimodal-neurons/) in the past, but these studies are only descriptive.
* The efficient market hypothesis: The technique used for the censorship filter of the Stable Diffusion model was a[vulgar cosine similarity threshold](https://arxiv.org/abs/2210.04610) between generated image embeddings and a list of taboo concepts. Yes, this may seem a bit ridiculous, but at least there is a filter, and it appears that interp has not yet been able to provide more convenient tools than this.

**Broad critiques.** He [explains](https://www.lesswrong.com/posts/gwG9uqw255gafjYN4/eis-iii-broad-critiques-of-interpretability-research) that interp is generally not scaling, relying too much on humans, failing to combine techniques. He also [criticizes](https://www.lesswrong.com/s/a6ne2ve5uturEEQK7/p/wt7HXaCWzuKQipqz3%23Imagine_that_you_heard_news_tomorrow_that_MI_researchers_from_TAISIC_meticulously_studied_circuits_in_a_way_that_allowed_them_to_) mech interp, which may not be the best way of doing interp, because of cherry-picking, focusing only on toy examples and lack of scalability, and failing to do useful things.

**Methodological problems:**

* He [points out](https://www.lesswrong.com/s/a6ne2ve5uturEEQK7/p/gwG9uqw255gafjYN4%23A_lack_of_practical_applications), "The root cause of this has much to do with interpretability research not being approached with as much engineering rigor as it ought to be."
* One good point to note is that since the publication of his sequence, certain methods have become more rigorous. For example, [Causal methods](https://www.lesswrong.com/posts/uLMWMeBG3ruoBRhMW/a-comparison-of-causal-scrubbing-causal-abstractions-and), [activation patching](https://arxiv.org/abs/2211.00593), [probing only with linear probes](https://www.lesswrong.com/posts/nmxzr2zsjNtjaHh7x/actually-othello-gpt-has-a-linear-emergent-world) are more rigorous and promising than descriptive techniques.
* But I also believe that interpretability may be fundamentally flawed [Section[What does the end story of interp look like](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#What_does_the_end_story_of_interpretability_look_like__That_s_not_clear_at_all_)].

Cognitive Emulations - Explainability *By design*
-------------------------------------------------

**If interpretability was really a bottleneck, we could use**[**cognitive emulation**](https://www.lesswrong.com/posts/ngEvKav9w57XrGQnb/cognitive-emulation-a-naive-ai-safety-proposal)**, which, in my opinion, allows way better explainability/transparency than interpretability will ever get us.**

My understanding of cognitive emulation: Emulating GPT-4 using LLMs like GPT-3 as different submodules that send messages written in *plain English* to each other before outputting the next token. If the neural network had deceptive thoughts, we could see them in these intermediate messages.

**Interpretability is needed only if one module of the cognitive emulation is deceptive.** Then maybe you can use interpretability to explain this module. But I'm confident models will need Chain of Thought for the foreseeable future. So interpretability is not 'urgent' research.

**If we don’t use Reinforcement learning on this cognitive emulation, and if we only prompt in English the different submodules**, the cognitive emulation won’t create whole “new languages” or be able to obfuscate a lot of its thoughts.

Yes, there are **problems with this proposal:**

* CoEms don't preclude steganography or obfuscation.
* Even when asked to reason with Chain-of-Thought (CoT), LLMs are able to skip some inference steps to some extent (see[Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/abs/2305.04388)). However, if we don’t train models to skip a zillion inference steps, they won’t be able to obfuscate their thoughts efficiently.

But we could further reduce these risks with:

* **Process-based training**, to ensure that the systems are accustomed to detailing their thoughts in much greater detail (and not being able to skip too many steps). It seems much more feasible than interpretability (and the[alignment tax](https://openai.com/research/improving-mathematical-reasoning-with-process-supervision) even seems negative - for example see those progress for[Mathematical reasoning](https://openai.com/research/improving-mathematical-reasoning-with-process-supervision))
* **selection inference:** *“CoT has access to the whole context, so each reasoning step is not necessarily causally connected to the last. But*[*selection inference*](https://arxiv.org/abs/2205.09712) *enforces a structure where each reasoning step necessarily follows from the last, and therefore the whole reasoning chain is causal.”*from [Sid Black](https://docs.google.com/document/d/1ybJqvZ7vkfN641KAiDj1I0Deu-XGk8r9fSJEZ4NvLmc/edit?disco=AAAA21_wJ_g), CTO of Conjecture.
* Other ideas were listed in section “[Steering the world towards transparency](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Steering_the_world_towards_transparency)”.

Spicy: However, cognitive emulation will quite likely be an engineering nightmare, facing significant robustness issues that are always present in small models. The alignment tax will be higher than for end-to-end systems, making it unlikely that we will ever use this technology. The bottleneck is probably not interp, but rather an ecosystem of preventive safety measures and a safety culture. Connor Leahy, CEO of Conjecture, explaining the difficulties of the problem during interviews and pushing towards a safety culture, is plausibly more impactful than the entire CoEm technical agenda.

Detailed Counter Answers to Neel’s list
---------------------------------------

Here is Neel’s [Longlist of Theories of Impact for Interpretability](https://www.lesswrong.com/posts/uK6sQCNMw8WKzJeCQ/a-longlist-of-theories-of-impact-for-interpretability) with critiques for each theory. Theories proposed by Neel are displayed in italics, whereas my critiques are rendered in standard font.

1. ***Force-multiplier on alignment research**: We can analyse a model to see why it gives misaligned answers, and what's going wrong. This gets much richer data on empirical alignment work, and lets it progress faster.*
   * I think this "force multiplier in alignment research" theory is valid, but is conditioned on the success of the other theories of impact, which imho are almost all invalid.
   * **Conceptual advancements are more urgent** It's better to think conceptually about what misalignment means rather than focusing on interp. [Section [What does the end story of interpretability look like?](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#What_does_the_end_story_of_interpretability_look_like__That_s_not_clear_at_all_)]
   * **Dual Use:** Force-multiplier on capability research.
2. ***Better prediction of future systems**: Interpretability may enable a better mechanistic understanding of the principles of how ML systems work, and how they change with scale, analogous to scientific laws. This allows us to better extrapolate from current systems to future systems, in a similar sense to scaling laws. Eg, observing phase changes a la induction heads shows us that models may rapidly gain capabilities during training*
   * Critiqued in section “[Interp is not a good predictor of future systems](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Interp_is_not_a_good_predictor_of_future_systems)”
3. ***Auditing**: We get a Mulligan. After training a system, we can check for misalignment, and only deploy if we're confident it's safe*
   * **Not the most direct way.** This ToI targets outer misalignment, the next one targets inner misalignment. But currently, people who are auditing for outer alignment do not use interpretability. They evaluate the model, they make the model speak and look if it is aligned with behavioral evaluations. Interpretability has not been useful in finding GPT’s jailbreaks.
   * To date, I still don't see how we would proceed with interp to audit GPT-4.
4. ***Auditing for deception**: Similar to auditing, we may be able detect deception in a model. This is a much lower bar than fully auditing a model, and is plausibly something we could do with just the ability to look at random bits of the model and identify circuits/features - I see this more as a theory of change for 'worlds where interpretability is harder than I hope'.*
   * Critiqued in section “[Auditing deception with interp is out of reach](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Auditing_deception_with_interp_is_out_of_reach)”
5. ***Enabling coordination/cooperation:** If different actors can interpret each other's systems, it's much easier to trust other actors to behave sensibly and coordinate better*
   * **Not the most direct way.** If you really want coordination and cooperation, you need to help with AI governance and outreach of experts and researchers. The [statement on AI risks](https://www.safe.ai/statement-on-ai-risk) has enabled more coordination than interp will probably never get us.
6. ***Empirical evidence for/against threat models**: We can look for empirical examples of theorized future threat models, eg inner misalignment*
   * ***Coordinating work on threat models**: If we can find empirical examples of eg inner misalignment, it seems much easier to convince skeptics this is an issue, and maybe get more people to work on it.*
     + [Cicero](https://ai.facebook.com/research/cicero/) or poker models are already capable of masking pieces of information or bluffing to play poker. From there, I don't know what it would mean to show canonical inner misalignment to non-technical people.
     + This focuses too much on deceptive alignment, and this will probably be too late if we get to this point.
   * ***Coordinating a slowdown**: If alignment is really hard, it seems much easier to coordinate caution/a slowdown of the field with eg empirical examples of models that seem aligned but are actually deceptive*
     + **Not the most direct way.** This is a good theory of change, but interp is not the only way to show that a model is deceptive.
7. ***Improving human feedback**: Rather than training models to just do the right things, we can train them to do the right things for the right reasons*
   * Seems very different from current interpretability work.
   * **Not the most direct way.** Process-based training, model psychology, or other scalable oversight techniques not relying on interp may be more effective.
8. ***Informed oversight**: We can improve recursive alignment schemes like IDA by having each step include checking the system is actually aligned. Note: This overlaps a lot with 7. To me, the distinction is that 7 can be also be applied with systems trained non-recursively, eg today's systems trained with Reinforcement Learning from Human Feedback*
   * Yes, it's an improvement, but it's naive to think that the only problem with RLHF is just the issue of lack of transparency or deception. For example, we would still have agentic models (because agency is preferred by human preferences) and interpretability alone won't fix that. See the [Compendium of problems with RLHF](https://www.lesswrong.com/posts/d6DvuCKH5bSoT62DB/compendium-of-problems-with-rlhf) and [Open Problems and Fundamental Limitations of RLHF](https://www.lesswrong.com/posts/LqRD7sNcpkA9cmXLv/open-problems-and-fundamental-limitations-of-rlhf) for more details.
   * **Conceptual advances are more urgent.** What does ‘checking the system is actually aligned’ really means? [It's not clear at all.](https://docs.google.com/document/d/1ybJqvZ7vkfN641KAiDj1I0Deu-XGk8r9fSJEZ4NvLmc/edit#bookmark=id.wqr2jvmzsg7c)
9. ***Interpretability tools in the loss function:** We can directly put an interpretability tool into the training loop to ensure the system is doing things in an aligned way. Ambitious version - the tool is so good that it can't be Goodharted. Less ambitious - The could be Goodharted, but it's expensive, and this shifts the inductive biases to favor aligned cognition*.
   * **Dual Use,** for obvious reasons, and this one is particularly dangerous.
   * **List of lethalities 27. Selecting for undetectability**: “*Optimizing against an interpreted thought optimizes against interpretability.”*
10. ***Norm setting**: If interpretability is easier, there may be expectations that, before a company deploys a system, part of doing due diligence is interpreting the system and checking it does what you want*
    * **Not the most direct way.**Evals, evals, evals.
    * No need to wait for interpretability. We already roughly know what to do. We could conduct studies in line with [Evaluating Dangerous Capabilities](https://www.serimats.org/evals) and the paper [Model Evaluation for Extreme Risks](https://arxiv.org/abs/2305.15324), [Towards Best Practices in AGI Safety and Governance](https://www.governance.ai/research-paper/towards-best-practices-in-agi-safety-and-governance), this last paper presenting 50 statements about what AGI labs should do, none mentioning interp.
11. ***Enabling regulation**: Regulators and policy-makers can create more effective regulations around how aligned AI systems must be if they/the companies can use tools to audit them*
    * Same critique as **10.*****Norm setting***
12. ***Cultural shift 1:** If the field of ML shifts towards having a better understanding of models, this may lead to a better understanding of failure cases and how to avoid them*
    * **Not the most direct way.** Technical Outreach, communications, interviews or even probably standards and [Benchmarks](https://www.lesswrong.com/s/FaEBwhhe3otzYKGQt) are way more direct.
13. ***Cultural shift 2:** If the field expects better understanding of how models work, it'll become more glaringly obvious how little we understand right now*
    * Same critique as **12.*****Cultural shift 1.***
    * This is probably the opposite of what is happening now: People are fascinated by interpretability and continue to develop capabilities in large labs. I suspect that the well-known Distill journal has been very fascinating for a lot of people and has probably been a source of fascination for people entering the field of ML, thus accelerating capabilities.
    * See the [False sense of control](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#False_sense_of_control_) section.
14. ***Epistemic learned helplessness**: Idk man, do we even need a theory of impact? In what world is 'actually understanding how our black box systems work' not helpful?*
    * I don't know man, the worlds where we have limited resources, where we are funding constrained + Opportunity costs.
    * **Dual Use**, refer to the section "[Interpretability May Be Overall Harmful](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Interpretability_May_Be_Overall_Harmful)".
15. ***Microscope AI**: Maybe we can avoid deploying agents at all, by training systems to do complex tasks, then interpreting how they do it and doing it ourselves*
    * Critique in section [Microscope AI?](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Microscope_AI_).
16. ***Training AIs to interpret other AIs**: Even if interpretability is really hard/labor intensive on advanced systems, if we can create aligned AIs near human level, we can give these interpretability tools and use them to interpret more powerful systems*
    * **Object level:**Training AI to interpret other AI, could be useful but would be already dangerous, and we are already in classes of scenarios that are super [dangerous](https://www.lesswrong.com/posts/pQqoTTAnEePRDmZN4/agi-automated-interpretability-is-suicide).
    * **Meta level:**This scheme is very speculative. I do not want the survival of civilization to rely on it. [Godzilla strategy](https://www.lesswrong.com/posts/DwqgLXn5qYC7GqExF/godzilla-strategies) is probably not a good strategy (though this is controversial).
17. ***Forecasting discontinuities**: By understanding what's going on, we can predict how likely we are to see discontinuities in alignment/capabilities, and potentially detect a discontinuity while training/before deploying a system*
    * Mostly the same critiques as in section “[Interp is not a good predictor of future systems](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Interp_is_not_a_good_predictor_of_future_systems)”
18. ***Intervening on training**: By interpreting a system during training, we can notice misalignment early on, potentially before it's good enough for strategies to avoid our notice such as deceptive alignment, gradient hacking, obfuscating its thoughts, etc.*
    * Mostly the same critiques as in section “[Auditing deception with interp is out of reach](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Auditing_deception_with_interp_is_out_of_reach)”
19. ***Auditing a training run**: By checking for misalignment early in training, we can stop training systems that seem misaligned. This gives us many more shots to make an aligned system without spending large amounts of capital, and e.g. allows us to try multiple different schemes, initialisations, etc. This essentially shifts the distribution of systems towards alignment.*
    * Mostly the same critiques as in section “[Auditing deception with interp is out of reach](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Auditing_deception_with_interp_is_out_of_reach)”
20. ***Eliciting Latent Knowledges:** Use the length of the shortest interpretability explanation of behaviors of the model as a training loss for ELK - the idea is that models with shorter explanations are less likely to include human simulations / you can tell if they do. (credit to Tao Lin for this one)*
    * Same critique as **9.*****Interpretability tools in the loss function.***
    * Same critique as **15. Microscope AI**.
    * Same critique as **16.*****Training AIs to interpret other AIs.***

Case study of some cool interp papers
-------------------------------------

This section is more technical.

Stephen Casper [lists](https://www.lesswrong.com/s/a6ne2ve5uturEEQK7/p/wt7HXaCWzuKQipqz3#Imagine_that_you_heard_news_tomorrow_that_MI_researchers_from_TAISIC_meticulously_studied_circuits_in_a_way_that_allowed_them_to_) a bunch of impressive interpretability papers, as of February 2023. Let's try to investigate whether these papers could be used in the future to reduce risks. For each article, I mention the corresponding end story, and the critic of this end story applies to the article.

### Bau et al. (2018)

[Bau et al. (2018)](https://arxiv.org/abs/1811.10597): Reverse engineer and repurpose a GAN for controllable image generation.

* **Procedure:** ([video](https://www.youtube.com/watch?v=yVCgUYe4JTM)) We generate images of churches using a GAN. There are often trees in the generated images. We manually surround the trees, then find the units in the GAN that are mostly responsible for generating these image regions. After finding these regions, we perform an ablation of these units, and it turns out that the trees disappear.
* **End Story:**[Enumerative safety](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Enumerative_safety_)
* **Useful for outer alignment?** Ideally, we could 1. Find features which are undesirable 2. Then remove parts of the network that are most linked to these features. This is a very limited form of alignment procedure, by ablation.
  + Maybe we could use this kind of procedure to filter pornography, but why then train the network on pornographic images in the first place?
  + Basically, this is the same strategy as enumerative safety which is criticized above.
* **Useful for inner alignment?** Can we apply this to deception? No, because by definition, deception will not result in a difference in outputs, so we cannot apply this procedure.

### Ghorbani et al. (2020)

[Ghorbani et al. (2020)](https://arxiv.org/abs/2002.09815): Identify and successfully ablate neurons responsible for biases and adversarial vulnerabilities.

* **Procedure:** ([video](https://slideslive.com/38936399/neuron-shapley-discovering-the-responsible-neurons)) It calculates the Shapley score of different units of a CNN and then removes the units with the highest Shapley value to maximize or minimize a metric. Removing certain units seems to make the network more robust to certain adversarial attacks.
* **End Story:**[Enumerative safety](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Enumerative_safety_) (and [Reverse engineering](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Reverse_engineering_)?)
* **Useful for outer alignment?** What would have happened if we had just added black women to the dataset? We can simply use a generative model for that and generate lots of images of black women. I'm almost certain that the technique used by OpenAI to remove biases in[Dalle-2](https://openai.com/blog/reducing-bias-and-improving-safety-in-dall-e-2), does not rely on interp.
* **Useful for inner alignment?** Can we apply this to deception? No, again because the first step in using Shapley value and this interpretability method is to find a behavioral difference, and we need first to create a metric of deception, which does not exist currently. So again we first need to find first a behavioral difference and some evidence of deception.

### Burns et al. (2022)

[Burns et al. (2022)](https://arxiv.org/abs/2212.03827): Identify directions in latent space that were predictive of a language model saying false things.

* **Procedure:** compare the probability of the ‘Yes’ token with the probability probed from the world model.
* **End story:** [Microscope AI](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Microscope_AI_)
* **Useful for inner alignment?**
  + Extracting knowledge from near GPT-3 level AIs, mostly trained through self-supervised learning via next token prediction, is a [misunderstanding](https://www.lesswrong.com/posts/bWxNPMy5MhPnQTzKz/what-discovering-latent-knowledge-did-and-did-not-find-4).
  + **This technique requires a minimum of agency and is not just usable as an oracle.**
    - **Chain-of-thought will probably always be better.** Currently, this technique barely performs better than next token prediction. Chain-of-thought performs much better, and it seems we have (obvious) [theoretical reason](https://twitter.com/BogdanIonutCir2/status/1664974522791895040) to think so. So using GPTs as just an oracle won’t be competitive. This paper doesn't test the trivial baseline of just fine-tuning the model (which has been found to usually work better).
    - **Agency is probably required.** It seems unlikely that it will synthesize knowledge on its own in a world model during next-token prediction training. Making tests in the world, or reasoning in an open-ended way, is probably necessary to synthesize a proper truth feature in the world model in advanced GPT using continual learning.
  + **Conclusion:** Yes, maybe in the future, if we create autonomous agents that conduct experiments and have their own world model, this kind of technique could probably be spot a mismatch between the world model oracle and what the model tells you. But if that were the case, we would probably already be in a very, very dangerous world. Civilization is not ready for this, and I still think that this method will be very brittle, and I prefer to aim for worlds where deception is unlikely. [section: ​​[Preventive measures](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Preventive_measures_against_Deception_seem_much_more_workable)]

### Casper et al. (2022)

[Casper et al. (2022)](https://arxiv.org/abs/2211.10024): Identify hundreds of interpretable copy/paste attacks.

* **Procedure:** We try to find adversarial attacks automatically. For each image x, we optimize the latent space of a GAN to complete a patch in the image, to optimize the probability of classification towards class y2. We do this for each possible (x, y2) and we observe what appears in the patch. Then we search for natural patches that have a similar embedding to the synthetic adversarial patch, and if the similarity is high enough, we test the natural patch and see if this natural patch breaks the classifier.
* **End Story:**[Relaxed Adversarial training](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Relaxed_adversarial_training_)?
* **Useful for inner alignment?** Conceptually yes, but this is so brittle. And if the trigger mechanism for deceptive alignment is not generable by the GAN or is not in the distribution of natural images/text, we won’t find it.

### Ziegler et al. (2022)

[Ziegler et al. (2022)](https://arxiv.org/abs/2205.01663): Debug a model well enough to greatly reduce its rate of misclassification in a high-stakes type of setting.

* **Procedure**: The project consists in building a highly reliable injury classifier: *“We started with a baseline classifier trained on some mildly injury-enriched (but otherwise mostly random) data. Then, over the course of several months, we tried various techniques to make it more reliable”.*The aim here is to minimize the false negative rate, striving to make the classifier as reliable as possible.: we do not want to let any problematic sentence or example go unnoticed. In order to find adversarial examples, they experimented with the following techniques:
  + 1. unaugmented humans: humans manually search for problematic sentences,
  + 2. automatic paraphrasing: use of LLM to find variations of problematic sentences and progressively enrich the dataset.
  + and 3. humans augmented with a rewriting tool. It is mainly this last technique that was useful.
* **End Story:**Adversarial training and maybe[Relaxed Adversarial training](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Relaxed_adversarial_training_)?
* **Robustness results:**
  + **In-distribution:** They started with a classifier that had a 2.5% false negatives, and ended up with a classifier that had 0.003% false negatives! They are almost reaching the limits of detection for this type of dataset.
  + **Out-of-distribution:** The tool-assisted attack to generate an adversarial example now takes 26 minutes per example, which is only twice as long as the previous time of 13 minutes. This falls short of the paper's aim to gain many orders of magnitude in speedup. Unfortunately, even though the original dataset (in-distribution) is now almost completely well classified, it seems that it is still possible to easily create adversarial examples with automated tools, and we have only improved the defense by a factor of 2. But only a factor of 2 falls short of the aims of this project.
  + **Conclusion:** Thus, improving a lot in-distribution performance by several orders of magnitude seems to not have much impact on out-of-distribution. Even though Redwood [said](https://www.alignmentforum.org/posts/n3LAgnHg6ashQK3fF/takeaways-from-our-robust-injury-classifier-project-redwood) that this project could have been conducted better, this is currently rather a negative result to improve the adversarial robustness (i.e. out-of-distribution Robustness). In addition, we can link these results to the paper [Adversarial Policies Beat Superhuman Go AIs, Wang et al.. 2022](https://arxiv.org/abs/2211.00241), which studies adversarial attacks on the Katago AI, which is superhuman in the game of Go. They show that it is probably possible to find simple adversarial strategies even against very superhuman AIs. And as a consequence, it seems that even for very robust and powerful AIs, It may always be possible to find adversarial attacks.
* **How to use this for deception?** Unclear. Do we watch what the AI does, tell it *'you are the lord of the universe, you can do whatever you want'* or more subtle variations of this prompt and then see what happens? This is not really interpretability, it's much more prompt engineering with [consistency checks](https://arxiv.org/abs/2306.09983) or cyborgism than interp. (Beside, the interpretability technique that they used is quite underwhelming, only coloring tokens with the bigger gradient of the norm of the embedding of the token, and then basically iterating on synonymous of those important tokens).

1. **[^](#fnref2stfurwyyg6)**

   Is feature visualization useful? Some findings suggest no: [Red Teaming Deep Neural Networks with Feature Synthesis Tools](https://arxiv.org/abs/2302.10894).
2. **[^](#fnref4qi9kn3ip89)**

   GradCam: Maybe this [paper](https://www.notion.so/Against-Almost-every-Theories-of-Change-of-Interpretability-61ebd2937cab4e12b9eb777454b7ed29?pvs%3D21)? But this is still academic work.
3. **[^](#fnref6xxwjs20rd7)**

   I have organized[two](https://www.lesswrong.com/posts/WF5JpmpK8EM4xKyve/new-hackathon-robustness-to-distribution-changes-and)[hackathons](https://github.com/EffiSciencesResearch/hackathon42) centered around the topic of spurious correlations. I strongly nudged using interp, but unfortunately, nobody used it...Yes this claim is a bit weak, but still indicates a real phenomenon, see [section [Lack of real applications](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1#Interpretability_tools_lack_widespread_use_by_practitioners_in_real_applications_)]
4. **[^](#fnrefztj4j3pmerg)**

   Note: I am not making any claims about ex-ante interp (also known as [intrinsic interp](https://arxiv.org/abs/2207.13243)), which has not been so far able to predict the future system either.
5. **[^](#fnref2464ho15s7t)**

   Other weaker difficulties for auditing deception with interp:  **This is already too risky and Prevention is better than cure. 1) Moloch may still kill us:***"auditing a trained model" does not have a great story for wins. Like, either you find that the model is fine (in which case it would have been fine if you skipped the auditing) or you find that the model will kill you (in which case you don't deploy your AI system, and someone else destroys the world instead)*. […] *a capable lab would accidentally destroy the world because they would be trying the same approach but either not have those interpretability tools or not be careful enough to use them to check their trained model as well?”* [[Source](https://www.lesswrong.com/posts/uK6sQCNMw8WKzJeCQ/a-longlist-of-theories-of-impact-for-interpretability?commentId%3DpHt4w9SN5PLLTzZuB) Rohin Shah]. **2) We probably won’t be competent enough to fix our mistake:** “*in order for auditing the model to help (directly), you have to actually be pretty confident in your ability to understand and fix your mistakes if you find one. It's not like getting a coin to land Heads by flipping it again if it lands Tails - different AGI projects are not independent random variables, if you don't get good results the first time you won't get good results the next time unless you understand what happened. This means that auditing trained models isn't really appropriate for the middle of the skill curve*.” [[Source](https://www.lesswrong.com/posts/uK6sQCNMw8WKzJeCQ/a-longlist-of-theories-of-impact-for-interpretability?commentId%3D8Ckqr87YsoMprivEW) Charlie Steiner].
6. **[^](#fnrefrbnra8fsacj)**

   From “[Conditioning Generative Models](https://www.lesswrong.com/posts/nXeLPcT9uhfG3TMPS/conditioning-generative-models). “Suppose you had a language model that you knew was in fact a good generative model of the world and that this property continued to hold regardless of what you conditioned it on. Furthermore, suppose you had some prompt that described some agent for the language model to simulate (Alice) that in practice resulted in aligned-looking outputs. **Is there a way we could use different conditionals to get at whether or not Alice was deceptive** (e.g. prompt the model with “DeepMind develops perfect transparency tools and provides an opportunity for deceptive models to come clean and receive a prize before they’re discovered.”)?”
7. **[^](#fnrefmf7vlk6ib69)**

   For example, what happens If you prompt a deceptive model with a Joyous prompt, and prompt the same deceptive model with a sad prompt and then take the difference, you obtain a Joyous Deceptive model?
8. **[^](#fnrefc2q5uxqhj6j)**

   But at the same time, we could be pessimistic, because this good idea has been out there in the wild since Christiano [described](https://ai-alignment.com/training-robust-corrigibility-ce0e0a3b9b4d) it in 2019. So either this idea does not work and we have not heard about it. Or the community has failed to recognize a pretty simple good idea.
9. **[^](#fnrefpo4e41md3r)**

   Causal scrubbing could be a good way for evaluating interp techniques using something other than intuition. However, this is only suitable for localization assessment and does not measure how understandable the system is for humans.
10. **[^](#fnrefplu4ji16iui)**

    “*I was previously pretty dubious about interpretability results leading to capabilities advances. I've only really seen* [*two*](https://arxiv.org/pdf/2212.14052.pdf)[*papers*](https://arxiv.org/pdf/2302.10866.pdf)*which did this for LMs and they came from the same lab in the past few months. It seemed to me like most of the advances in modern ML (other than scale) came from people tinkering with architectures and seeing which modifications increased performance. But in a conversation with Oliver Habryka and others, it was brought up that as AI models are getting larger and more expensive, this tinkering will get more difficult and expensive. This might cause researchers to look for additional places for capabilities insights, and one of the obvious places to find such insights might be interpretability research.*” from [Peter barnett](https://www.lesswrong.com/posts/BinkknLBYxskMXuME/if-interpretability-research-goes-well-it-may-get-dangerous?commentId=GYo8WegFmfxWmB5Z3).
11. **[^](#fnrefw7s6gsvuwb)**

    Not quite! Hypotheses 4 (and 2?) are missing. Thanks to Diego Dorn for presenting this fun concept to me.
12. **[^](#fnrefavln8kyvlzg)**

    This excludes the governance hackathon, though, this is only from the technical ones. Source: Esben Kran.