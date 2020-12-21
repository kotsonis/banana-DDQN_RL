# Banana Unity environment solution Report
## Learning Algorithm

This project implements a Dueling Double Deep Q Network to learn how to navigate within the environment and collect rewards.

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
