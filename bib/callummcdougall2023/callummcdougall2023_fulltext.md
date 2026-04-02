*Many thanks to everyone who provided helpful feedback, particularly Aryan Bhatt and Lawrence Chan!*

TL;DR
-----

**This is my illustrated walkthrough of induction heads.** I created it in order to concisely capture all the information about how the circuit works.

There are 2 versions of the walkthrough:

* Version 1 is the one included in this post. It's slightly shorter, and focuses more on the intuitions than the actual linear operations going on.
* Version 2 can be found at my [personal website](https://www.perfectlynormal.co.uk/blog-induction-heads-illustrated). It has all the same stuff as version 1, with a bit of added info about the mathematical details, and how you might go about reverse-engineering this circuit in a real model.

The final image from version 1 is inline below, and depending on your level of familiarity with transformers, looking at this diagram might provide most of the value of this post. If it doesn't make sense to you, then read on for the full walkthrough, where I build up this diagram bit by bit.

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/v1674002595/mirroredImages/TvrfY4c9eaGLeyDkE/necx078wytq2ortamm9x.png)

Introduction
------------

Induction heads are a well-studied and understood circuit in transformers. They allow a model to perform in-context learning, of a very specific form: if a sequence contains a repeated subsequence e.g. of the form `A B ... A B` (where `A` and `B` stand for generic tokens, e.g. the first and last name of a person who doesn't appear in any of the model's training data), then the second time this subsequence occurs the transformer will be able to predict that `B` follows `A`. Although this might seem like weirdly specific ability, it turns out that induction circuits are actually a [pretty massive deal](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). They're present even in large models (despite being originally discovered in 2-layer models), they can be linked to macro effects like [bumps in loss curves during training](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#argument-phase-change), and there is some evidence that induction heads might even constitute [the mechanism for the actual majority of all in-context learning in large transformer models](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#toc-arguments).

I think induction heads can be pretty confusing unless you fully understand the internal mechanics, and it's easy to come away from them feeling like you get what's going on without actually being able to explain things down to the precise details. My hope is that these diagrams help people form a more precise understanding of what's actually going on.

Prerequisites
-------------

This post is aimed at people who already understand how a transformer is structured (I'd recommend [Neel Nanda's tutorial](https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/clean-transformer-demo/Clean_Transformer_Demo.ipynb) for that), and the core ideas in the [Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) paper. If you understand everything on this list, it will probably suffice:

* The central object in the transformer is the **residual stream**.
* Different heads in each layer can be thought of as **operating independently** of each other, reading and writing into the residual stream.
* Heads can compose to form **circuits**. For instance, K**-composition** is when the output of one head is used to generate the key vector in the attention calculations of a subsequent head.
* We can describe the weight matrices WQ, WK and WV as **reading from** (or **projecting from**) the residual stream, and WO as **writing to** (or **embedding into**) the residual stream.
* We can think of the combined operations WQ and WK in terms of a single, low-rank matrix WQK:=WQWTK, called the **QK circuit**.[[1]](#fno6ohz0ytgp)[[2]](#fnlovv1mr5glq)
  + This matrix defines a bilinear form on the vectors in the residual stream: vTiWQKvj is the attention paid by the ith token to the jth token.
  + Conceptually, this matrix tells us **which tokens information is moved to & from** in the residual stream.
* We can think of the combined operations WV and WO in terms of a single matrix WOV:=WVWO, called the **OV circuit**.[[3]](#fnyuqb0y6339)
  + This matrix defines a map from residual stream vectors to residual stream vectors: if vj is the residual stream vector at the source token, then vTjWOV is the  vector that gets moved from token j to the destination token (if j is attended to).
  + Conceptually, this matrix tells us **what information is moved from a token**, if that token is attended to.

Basic concepts of linear algebra (e.g. understanding orthogonal subspaces and the image / rank of linear maps) would be  also be helpful.

Now for the diagram! (You might have to zoom in to read it clearly.)

> *Note - part of the reason I wrote this is as a companion piece to other material / as a useful thing to refer to while explaining how induction heads work. I'm not totally sure how well it will function as a stand-alone explanation, and I'd be grateful for feedback!*

---

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/v1674002595/mirroredImages/TvrfY4c9eaGLeyDkE/t40xlqbdnnmw62fnsluo.png)![](https://res.cloudinary.com/lesswrong-2-0/image/upload/v1674002595/mirroredImages/TvrfY4c9eaGLeyDkE/jl6fyg5ph89v50fhq6pw.png)![](https://res.cloudinary.com/lesswrong-2-0/image/upload/v1674002595/mirroredImages/TvrfY4c9eaGLeyDkE/g2qkgu3dwjoamhr6mfhr.png)

[[4]](#fnidxu86wwzz)

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/v1674002595/mirroredImages/TvrfY4c9eaGLeyDkE/cmcyktwlflauagjs86ry.png)![](https://res.cloudinary.com/lesswrong-2-0/image/upload/v1674002595/mirroredImages/TvrfY4c9eaGLeyDkE/br47g39ldkfq8uiuoyhj.png)

Q-composition
-------------

Finally, here is a diagram just like the final one above, but which uses Q-composition rather than K-composition. The result is the same, however these heads seem to form less easily than K-composition because they require [pointer arithmetic](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=_Jzi6YHRHKP1JziwdE02qdYZ&q=pointer%20arithmetic), meaning that they move positional information between tokens and does operations on it, to figure out which tokens to attend to.(although a lot of this is down to architectural details of the transformer[[5]](#fnx88hrjrfzdi)).

![](https://res.cloudinary.com/lesswrong-2-0/image/upload/v1674002595/mirroredImages/TvrfY4c9eaGLeyDkE/l7bn0ojtu9v35jdm9hyd.png)

1. **[^](#fnrefo6ohz0ytgp)**

   Note that I'm using notation corresponding to the `TransformerLens` library, not to the Anthropic paper (this is because I'm hoping this post will help people who are actually working with the library). In particular, I'm following the convention that weight matrices multiply on the right. For instance, if v is a vector in the residual stream and WQ is the query projection matrix then vTWQ is the query vector. This is also why the QK circuit is different here than in the Anthropic paper.
2. **[^](#fnreflovv1mr5glq)**

   This terminology is also slightly different from the Anthropic paper. The paper  would call WEWQKWTE the QK circuit, whereas I'm adopting Neel's notation of calling WQK the QK circuit and calling something a full circuit if it includes the WE or WU matrices.
3. **[^](#fnrefyuqb0y6339)**

   Again, this is different than the Anthropic paper because of the convention that we're right-multiplying matrices. vTWV is the value vector (of size `d_head`) and vTWVWO is the embedding of this vector back into the residual stream. So WVWO is the OV circuit.
4. **[^](#fnrefidxu86wwzz)**

   I described subtracting one from the positional embedding as a "rotation". This is because [positional embeddings are often sinusoidal](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=_Jzi6YHRHKP1JziwdE02qdYZ&q=sinusoidal) (either because they're chosen to be sinusoidal at initialisation, or because they develop some kind of sinusoidal structure as the model trains).
5. **[^](#fnrefx88hrjrfzdi)**

   For example, if you specify `shortformer=True` when loading in transformers from `TransformerLens`, this means the positional embeddings aren't added to the residual stream, but only to the inputs to the query and key projection matrices (i.e. not to the the inputs to the value projection matrices WV). This means positional information can be used in calculating attention patterns, but can't itself be moved around to different tokens. You can see from the diagram how this makes Q-composition impossible[[6]](#fn3i4c87lznfs) (because the positional encodings need to be moved as part of the OV circuit, in the first attention head).
6. **[^](#fnref3i4c87lznfs)**

   That being said, it seems transformers seem to be able to [rederive positional information](https://www.alignmentforum.org/posts/sYHrW4wwfoMBxNDcA/real-time-research-recording-can-a-transformer-re-derive), so they could in theory form induction heads via Q-composition with this rederived information. To my knowledge there's currently no evidence of this happening, but it would be interesting!