import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

colors = np.array([[221,127,106],
                   [204,169,120],
                   [191,196,139],
                   [176,209,152],
                   [152,209,202],
                   [152,183,209],
                   [152,152,209],
                   [185,152,209],
                   [209,152,203],
                   [209,152,161]])

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_adversaries = 1
        num_landmarks = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            if i < num_adversaries:
                agent.adversary = True
                agent.size = 0.075
            else:
                agent.adversary = False
                agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        self.num_adversaries = num_adversaries
        self.colors = colors
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        goal_index = np.arange(len(world.landmarks))
        np.random.shuffle(goal_index)
        for i, agent in enumerate(world.agents):
            # 这里需要考虑为一个adversary和一个good agent分配相同的goal landmark
            agent.goal_a = world.landmarks[goal_index[i % self.num_adversaries]]
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            # landmark.color = self.colors[i] / 256
            landmark.index = i
        # Set the same color for the adversary, the good agent which processes the same target and their target landmark
        # We distinguish the adversary and the good agent through their size
        for i, agent in enumerate(world.agents):
            # agent.color = np.array([0.35, 0.35, 0.85])
            if agent.adversary:
                agent.color = self.colors[i] / 256
                agent.goal_a.color = self.colors[i] / 256
            else:
                agent.color = self.colors[i % self.num_adversaries] / 256
        # for i, a in enumerate(world.agents):
        #     print('goal assignment', i, a.adversary, a.goal_a.index, a.color == a.goal_a.color)

    # def reset_world_original(self, world):
    #     # random properties for landmarks
    #     for i, landmark in enumerate(world.landmarks):
    #         landmark.color = np.array([0.1, 0.1, 0.1])
    #         landmark.color[i + 1] += 0.8
    #         landmark.index = i
    #     # set goal landmark
    #     goal = np.random.choice(world.landmarks)
    #     for i, agent in enumerate(world.agents):
    #         agent.goal_a = goal
    #         agent.color = np.array([0.25, 0.25, 0.25])
    #         if agent.adversary:
    #             agent.color = np.array([0.75, 0.25, 0.25])
    #         else:
    #             j = goal.index
    #             agent.color[j + 1] += 0.5
    #     # set random initial states
    #     for agent in world.agents:
    #         agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
    #         agent.state.p_vel = np.zeros(world.dim_p)
    #         agent.state.c = np.zeros(world.dim_c)
    #     for i, landmark in enumerate(world.landmarks):
    #         landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
    #         landmark.state.p_vel = np.zeros(world.dim_p)
    #         # landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
    #         # if i != 0:
    #         #     for j in range(i):
    #         #         while True:
    #         #             if np.sqrt(np.sum(np.square(landmark.state.p_pos - world.landmarks[j].state.p_pos))) > 0.22:
    #         #                 break
    #         #             else:
    #         #                 landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
    #         # landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # the distance to the goal
        return -np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))

    def adversary_reward(self, agent, world):
        # keep the good agents which processes the same goal away from the target landmark
        agent_dist = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in world.agents if
                      (not a.adversary and a.goal_a.index == agent.goal_a.index)]
        pos_rew = min(agent_dist)
        # nearest_agent = world.good_agents[np.argmin(agent_dist)]
        # neg_rew = np.sqrt(np.sum(np.square(nearest_agent.state.p_pos - agent.state.p_pos)))
        neg_rew = np.sqrt(np.sum(np.square(agent.goal_a.state.p_pos - agent.state.p_pos)))
        # neg_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.good_agents])
        # 添加额外的碰撞惩罚，如果与其他adversary碰撞，则会收获-1的reward
        rew = 0
        if agent.collide:
            for a in world.agents:
                if a is agent:
                    continue
                if a.adversary and self.is_collision(a, agent):
                    rew -= 1
        return pos_rew - neg_rew + rew

    # def adversary_reward_original(self, agent, world):
    #     # keep the nearest good agents away from the goal
    #     agent_dist = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in world.agents if
    #                   not a.adversary]
    #     pos_rew = min(agent_dist)
    #     # nearest_agent = world.good_agents[np.argmin(agent_dist)]
    #     # neg_rew = np.sqrt(np.sum(np.square(nearest_agent.state.p_pos - agent.state.p_pos)))
    #     neg_rew = np.sqrt(np.sum(np.square(agent.goal_a.state.p_pos - agent.state.p_pos)))
    #     # neg_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.good_agents])
    #     return pos_rew - neg_rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        # entity_pos = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # get positions of all entities in this agent's reference frame
        other_l_pos = []
        target_l_pos = []
        for entity in world.landmarks:  # world.entities:
            if entity.index == agent.goal_a.index:
                target_l_pos.append(entity.state.p_pos - agent.state.p_pos)
            else:
                other_l_pos.append(entity.state.p_pos - agent.state.p_pos)

        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        same_goal_good_agent_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            # 分配target时只有一个adversary和一个good agent可以获得相同的goal，因此这里无需再判断other.adversary=False
            if agent.adversary and (other.goal_a.index == agent.goal_a.index):
                same_goal_good_agent_pos.append(other.state.p_pos - agent.state.p_pos)
            else:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
        if not agent.adversary:
            return np.concatenate([agent.state.p_vel] + [agent.goal_a.state.p_pos - agent.state.p_pos] + [
                agent.color] + target_l_pos + other_l_pos + entity_color + other_pos)
        else:
            # other_pos = list(reversed(other_pos)) if random.uniform(0,1) > 0.5 else other_pos  # randomize position of other agents in adversary network
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + target_l_pos + other_l_pos + same_goal_good_agent_pos + other_pos)