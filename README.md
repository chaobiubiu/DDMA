# DDMA: Discrepancy-Driven Multi-Agent Reinforcement Learning
Code for the paper "DDMA: Discrepancy-Driven Multi-Agent Reinforcement Learning" presented at PRICAI 2022. This repository develops DDMA algorithm on the benchmarks of Collision Corridor and several MPE scenarios: Cooperative Navigation, Predator and Prey and Individual Defense. We compare DDMA with VDN, QMIX and MAPG in Collision Corridor. And we choose MADDPG, MAAC, G2ANet and noisy-MADDPG as the baselines in these MPE scenarios. Here we provide the detailed descriptions of these environments and all algorithms in the experiments.
## Algorithms
Please refer to the **pricai_appendix.pdf** file for more details (i.e. architectures, hyperparameters) of DDMA and other algorithms. It's recommended to download this file instead of reading online to avoid the formatting error.
## Environments
Here we provide the detailed descriptions of all environments in our experiments.
### Collision Corridor
<div align=center>
<img src="env_pics/scenario1.png" width="800">
</div>

In this task, there are two autonomous cars that are initialized in the upper left and the upper right corner respectively. The goal for each car is to arrive at the diagonal corner to collect some items, such as batteries or passengers. At the same time, they should learn to find the corridor hidden in the obstacles, or they will fail to reach the goal locations. Each car can select one of the five available actions {stop, up, down, left, right} and can observe the positions of another one. A dangerous collision will occur when both cars try to pass through the corridor, and both cars will receive -10 reward. If one car arrives at its own target corner, it will stay still and wait for the other car. In such situation where one car has achieved its own goal, both cars will receive +30 reward if the other one also arrives at the target corner. In the other situations, both cars can’t acquire any rewards. The episode limit in this task is set to 50.
### MPE Scenarios
<div align=center>
<img src="env_pics/mpe_scenarios.png" width="800" height="220">
</div>

#### Cooperative Navigation
In this situation, three agents should occupy as many landmarks as possible and avoid collisions with each other. All agents are globally rewarded based on how far the closest agent is to each landmark (sum of the minimum distances) and the number of total collisions (-1 for each collision). This setting requires agents to coordinate with each other about the assignment of target landmarks and avoid the collisions to maximize the total reward.
#### Predator and Prey
In Predator and Prey, there is one prey and three
predators competing in this scenario, except for two landmarks as obstacles. The prey moves and accelerates faster than predators, which can help it escape from
the pursuit. We only control the predators and equip the only prey with some pre-defined behavior rules (i.e., the random policy) to develop this hunting. All
agents and landmarks are initialized randomly at the beginning of each episode. At each time step, each predator receives +10 reward if any one in the predator
lines has captured the prey. For more advancement in this hunting, each predator can observe its own position and velocity, the relative positions towards the
landmarks, other predators and the prey, also the prey’s velocity.
#### Individual Defense
In Individual Defense, two good agents are assigned to defend specified landmarks respectively from the approach of other two bad agents. Interestingly, one good agent and one bad agent will be equipped with the same target landmark randomly in each initialization. We only control these two good agents and similarly equip the other two bad agents with pre-defined rules (i.e., the random policy). Each good agent aims to keep the bad agent that processes the same goal away from the target landmark and tries to approach this target landmark itself. At each time step, each good agent will receive rewards based on these two distances. In addition, these two good agents also need to avoid collisions with each other. Here the good agent can observe its own velocity and position, the relative positions towards these two landmarks and all other agents.
