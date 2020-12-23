[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
# banana-DDQN_RL Project
Dueling Double Deep Q Network reinforcement learning solution for the Banana World environment

## Introduction

This project implements a Dueling Double Deep Q Network to learn how to navigate within an environment an collect rewards.

## The Environment

![Trained Agent][image1]

Banana World is a square grid environment in which an agent can navigate executing various actions and collecting positive or negative rewards. The goal of the agent is to learn a policy with which to maximize it's returns. The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.
##### State Space
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.
##### Reward System
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.
##### Action Space
Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.
##### Turns
An episode completes after 300 actions

## Getting Started
To set up your python environment and run the code in this repository, follow the instructions below.
### setup Conda Python environment

Create (and activate) a new environment with Python 3.6.

- __Linux__ or __Mac__: 
```shell
	conda create --name banrl python=3.6
	source activate banrl
```
- __Windows__: 
```bash
	conda create --name banrl python=3.6 
	activate banrl
```
### Download repository
 Clone the repository and install dependencies

```shell
	git clone https://github.com/kotsonis/banana-DDQN_RL.git
	cd banana-DDQN-RL
	pip install -r requirements.txt
```

### Install Banana game environment

1. Download the game environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

2. Place the file in the `banana-DDQN-RL` folder, and unzip (or decompress) the file.

3. edit [hyperparams.py](hyperparams.py) to and set the `banana_location` entry to point to the right location. Example :
```python 
std_learn_params = {
        # Unity Environment parameters
        "banana_location": "./Banana_Windows_x86_64/Banana.exe",
```
## Instructions
### Training
To train an agent, [train.py](train.py) reads the hyperparameters from [hyperparams.py](hyperparams.py) and accepts command line options to modify parameters and/or set saving options.You can get the CLI options by running
```bash
python train.py -h
```
### Playing with a trained model
you can see the agent playing with the trained model as follows:
```bash
python play.py
```
You can also specify the number of episodes you want the agent to play, as well as the non-default trained model as follows:
```bash
python play.py --episodes 20 --model v2_model.pt
```