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
        -> 1980s saw huge investment in symbolic AI and expert systems but due to expensive maintanence and limited
           scope and difficulty in scaling, second AI winter began in 1990s
-----------------------------------------------------------------------------------------------------------------------
1.2 Before DL: History of ML





'''
