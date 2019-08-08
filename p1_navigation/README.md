[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

## Introduction

In this project, we will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of our agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, our agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

## Walk through of the Solution Notebook

### Agent and Environment Set-up
The initial few sections (1..3) in the notebook are intended to ensure that
the environment is setup correctly. By running these sections, we should be
able to assert that the Agent can interact with the environment. At this stage,
we will also confirm that the Agent can take 4 actions and that the environment has a total of
37 states.

### Solution using [Deep-Q Learning Algorithm](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
The solution is broken up into the following parts:

* The interaction between the agent and environment.
* The design of the agent and replay buffer.
* Instructions to run.

**The Agent-Environment Interaction**:

The interaction between the agent and environment follows the typical textbook
recipe. The environment is reset prior to starting of the simulation (sequence
of episodes). The exploration rate is set to 1.0 at the beginning of the
simulation. As the simulation progresses, the exploration rate is tapered.

In pseudo-code, the following recipe is followed for a simulation:

```Plain Text
foreach episode e:
    foreach step within e:
        s <- get current state
        a <- find action based on current state and exploration rate
        (s_next, r) <- set action a to get next state and reward
        (s, a, s_next, r) <- create tuple for learning
        Update the Agent using the tuple created for learning.  
```

At each episode the exploration rate is decayed (or tapered) by some rate.

**Design of Agent and Replay Buffer**:

The agent is responsible for taking action and learning from the information
it has been presented so far. It is initialized with the knowledge of the states,
actions and a model for making decisions given the states and actions.
The agent also has a memory buffer where it stores the learning tuples.

Whenever an agent is requested to find an action, it uses the exploration rate
to figure out whether it wants to select an action or random, or use its
current model to find the best possible action ([explore/exploit
tradeoff](https://joshkaufman.net/explore-exploit/)). Whenever that action is
taken the environment responds with a change in state and a reward (or cost)
associated with the action. This tuple is stored in the memory buffer.

Whenever enough samples are in memory, the agent tries to update its state of
the world to learn from the samples. It requests for a random set of tuples
for the process of learning (experience replay). Note that the random set is
critical in breaking the temporal bias for the agent to learn effectively. In
the absence of randomizing the tuples, the agent may become too biased by the
sequence of actions it has already taken or seen in sequence. This prevents
oscillations and avoids instability.

In order to ensure effective learning, the agent makes use of one model but
two different set of parameters. The first set (termed *local*) is used for
making decisions, while the second set (termed *target*) is used in the
estimation of the Q-values. This is done to increase robustness. This idea is
detailed in the following [research paper](https://arxiv.org/abs/1509.06461).
The loss function is defined as the mean square error between the following
entities:
* Q-target computed using the *target* parameters (considering immediate
  reward and discount factor)
* The q-value for the current state using the *local* parameters. 

The *target* parameters lag the *local* parameters and they are updated in an
incremental and weighted fashion with some memory of the previous set of
parameters.

**Instructions to run**
The architecture and hyperparameters for the model are detailed in [report.pdf](report.pdf).
In order to reproduce the results, follow the follow steps:
* Follow the instructions detailed in the [``Dependencies``
  section](https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md)
* Create the environment by following the instructions detailed [here](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md). Please ensure that the appropriate platform is selected.
* Activate ``drlnd`` environment and launch ``jupyter notebook Navigation.ipynb`` from the command prompt.
* Run ``Navigation.ipynb`` to solve the problem.


