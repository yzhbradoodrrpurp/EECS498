# Reinforcement Learning

## Overview

之前我们学习了监督学习和无监督学习两种机器学习范式，强化学习和这两种范式有很大的不同。强化学习中的主体分为 Agent 和 Environment，Agent 根据 Environment 的状态执行相应的动作，然后获得奖励，强化学习通过最大化 Agent 的奖励来优化它的行为，本质上 Agent 是在与 Environment 的不断交互中优化自身行为的。

![agentandenv](Images/agentandenv.png)

强化学习的步骤可以看作：

- Environment 给 Agent 一个当前时间环境的状态 $State_t$
- Agent 根据 $State_t$ 做出根据这个状态的反应 $Action_t$
- 环境根据 Agent 的行为给它相应的奖励 $Reward_t$，同时受到 $Action_t$ 的影响变为了 $State_{t + 1}$
- 重复循环以上步骤

强化学习中有几个值得注意的点：

- Stochasticity: State 和 Reward 可能是随机的
- Credit Assignment: $Reward_t$ 可能并不直接依赖于 $Action_t$，而是之前行为的某个行为或者之前行为的总和
- Non-Differentiability: 不能反向传播，因为 Reward 和 Action 通常是离散的，这导致无法它们不可微进而无法进行梯度下降
- Non-Stationarity: Environment 是动态变化的，Agent 会接受到什么样的 State 不仅仅取决于它如何 Action

![rl](Images/rl.gif)

一个关于强化学习非常著名的例子就是 Alpha Go，两个 Agent 相互竞争下棋，Agent1 根据棋盘的情况做出相应的行为，Agent2 根据棋盘的情况做出相应的行为，如此往复，直到最后胜利的 Agent 获得奖励。在这个例子中可以很明显地感受到强化学习的几个特点：

- Credit Assignment: 奖励是在最后一步才进行发放，所以它是对之前所有行为的奖励
- Non-Differentiability: Agent 将棋子进行移动的这个行为是离散的，无法用连续光滑的函数进行表示，所以不可微
- Stochasticity / Non-Stationarity: 环境不是静态的，不仅仅受到单个 Agent 的影响，也可能受到其它 Agent 的影响

![go](Images/go.png)
