'''
-----------------------------------------------------------------------------------------------------------------------
1.1 AI, ML & DL
-----------------------------------------------------------------------------------------------------------------------

    AI was crystallised as a field of research in 1956 in Dartmouth college through a summer workshop by John McCarthy.

    AI can be described as the effort to automate intellectual tasks normally performed by humans.

    Symbolic AI is the approach in which sufficiently large set of explicit rules are handcrafted for manipulating
    knowledge stored in explicit databases.

    Lady Ada Lovelace envisioned Analytical Engines to to 'originate' anything and execute processes we humans fully
    understand. Alan Turing Lady Lovelace's remark in his 1950 paper of "Computer machinery and Intelligence" which
    introduced the Turing Test and claimed that machines emulate all aspects of human intelligence.


    ML vs Statistics
        -> ML has to deal with large and complex datasets for which classical statistical analysis techniques are
           impractical.
        -> With exhibiting little mathematical theory, ML is fundamentally an engineering discipline and driven by
           empirical findings and deeply reliant on advancements in software and hardware

    Three Things necessary for ML:
        -> Input Data
        -> Examples of the expected output
        -> Performance measures

    The central problem in ML and DL is to meaningfully transform data, in other words, to learn useful representations
    of the input data that get closer to the expected output.

    ML models are all about finding apt representations for their input data that makes solving tasks easier .
    For example, the task “select all red pixels in the image” is simpler in the RGB format, whereas “make the
    image less saturated” is simpler in the HSV(hue-saturation-value) format.

    Learning in the context of ML describes an automated search process for data transformations that produce useful
    representations guided by feedback signals.

    Ml algorithms search through a predefined set of operations called a hypothesis space.
    ML concisely is searching for useful representations and rules over some input data, within a predefined space of
    possibilities using guidance from a feedback signal.


    Other apt names for Deep Learning can be "Layered Representations Learning" or "Hierarchical Representations
    Learning".

    Deep networks can be thought of as a multistage information - distillation process where information goes through
    successive filters and comes out increasingly useful.

    Transformations implemented by a layer is parameterised by its weights(weights sometimes called parameters of layer)

    Fundamental trick in deep learning is to use this score as a feedback signal to  adjust the value of the
    weights a little, in a direction that will lower the loss score for the current example.

    This adjustment is the job of the optimizer, which implements what’s called the Backpropagation algorithm:
    the central algorithm in deep learning.

    A network with a minimal loss is one for which the outputs are as close as they can be to the targets:
    a trained network.

    Loss function is also known as objective function or cost function.

    Two AI winters:
        -> 1960s , 1970s promised human level intelligence but lead to disinterest and first AI winter
        -> 1980s saw huge investment in symbolic AI and expert systems but due to expensive maintenance and limited
           scope and difficulty in scaling, second AI winter began in 1990s
-----------------------------------------------------------------------------------------------------------------------
1.2 Before DL: History of ML
-----------------------------------------------------------------------------------------------------------------------

    Probabilistic modelling
        -> earliest application of ML
        -> Best known Algo is Naive Bayes
            => ML classifier based on applying Bayes' Theorem while assuming that the features in the in the input data
               are all independent(this is the 'naive' assumption)
        -> Logistic regression is also another classifier , simple and versatile

    In 1989(Bell Labs) Yann LeCun combined early CNNs and backpropagation to classify handwritten digits dubbed it LeNet
    US Postal Service used it reading ZIP codes on mail envelopes.

    Kernel methods made neural networks oblivion in 1990s , published in Bell Labs by Vladimir Vapnik and Corinna Cortes
    Kernel trick was that to find good decision hyperplanes in the new representation space, you don’t have to
    explicitly compute the coordinates of your points in the new space; you just need to compute the distance
    between pairs of points in that space, which can be done efficiently using a kernel function. A kernel function is
    a computationally tractable operation that maps any two points in your initial space to the distance between these
    points in your target representation space, completely bypassing the explicit computation of the new
    representation.
    SVM gained fame as they solved simple classification problems and were mathematically understood but it didn't
    provide good results for perceptual problems.

    Decision trees came in 2000s and generally termed as kernel methods in 2010s. Random Forests were very popular in
    2010s until gradient boosting machines took over in 2014.

    In 2011 Dan Ciresan won image classification competitions with GPU trained deep neural networks acc. of 74%.
    But in 2012, A team advised by Geoffrey Hinton achieved 83.6 % acc. in ImageNet Comp. and in 2015 96.4% acc. reached
    CERN used decision tree based models for analysing particle data but switched to deep neural networks.

    Deep learning, on the other hand, completely automates feature engineering: with deep learning, you learn all
    features in one pass rather than having to engineer them yourself. This has greatly simplified machine learning
    workflows, often replacing sophisticated multistage pipelines with a single, simple, end-to-end deep learning model.

    These are the two essential characteristics of how deep learning learns from data: the incremental, layer-by-layer
    way in which increasingly complex representations are developed and the fact that these intermediate incremental
    representations are learned jointly, each layer being updated to follow both the representational needs of the layer
    above and the needs of the layer below. Together, these two properties have made deep learning
    vastly more successful than previous approaches to machine learning.
-----------------------------------------------------------------------------------------------------------------------
1.3 Why DL? Why now?
-----------------------------------------------------------------------------------------------------------------------

    CNNs and backpropagation were found in 1990 and LSTMs were understood in 1997 but DL took off only 2012s due to apt
    hardware availabilities.

    3 technical forces advancing ML
        -> Hardware
        -> Datasets and BenchMarks
        -> Algorithmic advances

    In 2007 NVIDIA launched CUDA, a programming interface for its GPUs. Matrix multiplications are highly parallelized
    so highly applicable through GPUs. In 2011, Dan Ciresan were the first among researchers to write CUDA application
    of neural nets.

    NVIDIA Titan RTX peaked 16 TeraFLOPS(16 trillion float 32 ops per second) 500 times better than 1990 best supercomp.
    Google's TPU(2020) can achieve 420 TeraFLOPS 10,000 times faster thn 1990 supercomp.
    TPU comes in pods. 1 pod = 1024 TPUs which peaks 100petaFLOPS!


    ImageNet dataset catalysed progress in Deep learning(1.4 million images hand annotated with 1000 image categories)
    Wikipedia and YouTube videos key datasets for NLP and User generated image tags on Flickr.


    In 2000s there was a issue that feedback signals used to fade away as layers increased.
    This changed around 2009-2010 with the advent of simple but important improvements:
        -> Better activation Functions
        -> Better weight initialisation techniques
        -> Better optimisation techniques

    In 2014-16 batch normalization, residual connections, and depth wise separable convolutions, etc improved gradient
    propagation.
    Led to large scale model architectures, which features tens of layers and tens of millions of parameters
    in both CV(ResNet, Inception, Xception) and NLP(BERT, GPT-3, XLNet)


    Deep learning due to its Simplicity, Scalability, Versatility and Reusability is a revolution in the making.
-----------------------------------------------------------------------------------------------------------------------

'''
