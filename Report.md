<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.5.1/katex.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/github-markdown-css/2.2.1/github-markdown.css"/>

[//]: # (Image References)

[image1]:  (./images/agent_environment.png) "Reinforcement Learing"
# Banana Unity environment solution Report

This project implements a Dueling Double Deep Q Network to learn how to navigate within the environment and collect rewards.
## Learning Algorithm
In reinforcement learning, an agent interacts with an environment in order to learn an optimal policy. As shown in below diagram, the agent observes the state from the environment, and based on a value_action DNN function approximator (Q Function), takes an action that is sent to the environment. The environment will respond with a new state, as well as a reward(if any). The agent will use this information (state, performed action, next state, reward) to improve it's Q Function and continue the loop

![Agent Environment](./images/agent_environment.png)

In order to experiment and learn, the agent chooses it's next action stochasticaly, and not just by selecting the next action based on highest expected reward. Specifically, the agent asigns a probability to be chosen to all other actions too so that it can explore alternative routes to reward. This functionality is implemented here through an epsilon-greedy algorithm, which initially explores alternatives at a high rate (hyper parameter `eps_start`) of 95% and linearly decays this to a minimum exploration of 1% (hyper parameter `eps_end`) with a rate of `eps_decay` per episode.

### Q function approximator
This project implements a dueling Q network for the Q function approximator, as detailed in  [Dueling Network Architectures for Deep Reinforcement Learning, 2015, Wang et al.](https://arxiv.org/abs/1511.06581)


Learning is performed through gradient descent. and at each step the loss and gradient that are computed are:
$$Loss_i = 
<a href="https://www.codecogs.com/eqnedit.php?latex=\delta_j=R_j&plus;\gamma_jQ_{\text{target}}\left(S_j,&space;\arg\max_a&space;Q(S_j,&space;a)\right)-Q(S_{j-1},&space;A_{j-1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta_j=R_j&plus;\gamma_jQ_{\text{target}}\left(S_j,&space;\arg\max_a&space;Q(S_j,&space;a)\right)-Q(S_{j-1},&space;A_{j-1})" title="\delta_j=R_j+\gamma_jQ_{\text{target}}\left(S_j, \arg\max_a Q(S_j, a)\right)-Q(S_{j-1}, A_{j-1})" /></a>

### Double Q Learning
This project implements a double Q learning solution
$$P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}$$
where $p_i > 0$ is the priority of transition $i$.
$$P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}$$
$$w_i = \left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^{\beta}$$
```math
f(x) = \int_{-\infty}^\infty
    \hat f(\xi)\,e^{2 \pi i \xi x}
    \,d\xi
```
# Algorithm 1
see Prioritized Experience Replay (https://arxiv.org/pdf/1511.05952.pdf)
Just a sample algorithmn
$$\begin{algorithm}[H]\DontPrintSemicolon\SetAlgoLined\KwResult{Write here the result}\SetKwInOut{Input}{Input}\SetKwInOut{Output}{Output}\Input{Write here the input}\Output{Write here the output}\BlankLine\While{While condition}{    instructions\;    \eIf{condition}{        instructions1\;        instructions2\;    }{
        instructions3\;    }}\caption{While loop with If/Else condition}\end{algorithm}$$

\begin{algorithm}[tb]
   \caption{Double DQN with proportional prioritization}
   \label{alg-preplay}
\begin{algorithmic}[1]
   \STATE {\bfseries Input:} minibatch $k$, step-size $\eta$, replay period $K$ and size $N$, exponents $\alpha$ and $\beta$, budget $T$.
   \STATE Initialize replay memory $\mathcal{H}=\emptyset$, $\Delta = 0$, $p_1=1$
   \STATE Observe $S_0$ and choose $A_0 \sim \pi_{\theta}(S_0)$
   \FOR{$t=1$ {\bfseries to} $T$}
   	  \STATE Observe $S_t, R_t, \gamma_t$
      \STATE Store transition $(S_{t-1}, A_{t-1}, R_t, \gamma_{t}, S_{t})$ in $\mathcal{H}$ with maximal priority $p_t = \max_{i<t} p_i$
      \IF{ $t \equiv 0 \mod K$ } 
	   	\FOR{$j=1$ {\bfseries to} $k$}
          \STATE Sample transition $j \sim P(j) = p_j^{\alpha} / \sum_i p_i^{\alpha}$
          \STATE Compute importance-sampling weight $w_j = \left(N\cdot P(j)\right)^{-\beta} / \max_i w_i$
          \STATE Compute TD-error $\delta_j =  R_j + \gamma_j Q_{\text{target}}\left(S_j, \arg\max_a Q(S_j, a)\right) - Q(S_{j-1}, A_{j-1}) $
          \STATE Update transition priority $p_j \leftarrow |\delta_j|$ 
          \STATE Accumulate weight-change $\Delta \leftarrow \Delta + w_j \cdot \delta_j \cdot \nabla_{\theta} Q(S_{j-1}, A_{j-1})$
        \ENDFOR
        \STATE Update weights $\theta \leftarrow \theta + \eta \cdot \Delta $, reset $\Delta = 0$
        \STATE From time to time copy weights into target network $\theta_{\text{target}} \leftarrow \theta$
      \ENDIF
      \STATE Choose action $A_t \sim \pi_{\theta}(S_t)$
   \ENDFOR
\end{algorithmic}
\end{algorithm}

## The Environment

## Plot of Rewards
## Ideas for future work
This agent has been trained on a state space that consisted of a vector of 37 features.
The improvement that could be done, is to actually process directly the pixels of what the agent is seeing first person, which would result in a state-space of 84x84 RGB image. ie a state space of 84x84x3


If you prefer, you can do a minimal install of the packaged version directly from PyPI:

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.


