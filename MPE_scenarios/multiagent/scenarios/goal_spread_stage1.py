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
        num_agents = 1
        num_landmarks = 3
        world.collaborative = False
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        self.colors = colors
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        goal_index = np.arange(len(world.landmarks))
        np.random.shuffle(goal_index)
        # set random initial states
        for i, agent in enumerate(world.agents):
            # Randomly select one landmark as the target for each agent.
            agent.target_l = world.landmarks[goal_index[i]]
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
        # random properties for agents
        for i, agent in enumerate(world.agents):
            # agent.color = np.array([0.35, 0.35, 0.85])
            agent.color = self.colors[i] / 256
            agent.target_l.color = self.colors[i] / 256
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     if i != 0 :
        #         for j in range(i):
        #             while True:
        #                 if np.sqrt(np.sum(np.square(landmark.state.p_pos - world.landmarks[j].state.p_pos))) > 0.22:
        #                     break
        #                 else:
        #                     landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        # rew = 0
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)
        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= 1

        # Each agent is rewarded based on the distance to its own target landmark, penalized for collisions.
        rew = 0
        target_l = agent.target_l
        rew -= np.sqrt(np.sum(np.square(agent.state.p_pos - target_l.state.p_pos)))
        if agent.collide:
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(agent, a):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        other_l_pos = []
        target_l_pos = []
        for entity in world.landmarks:  # world.entities:
            if entity.index == agent.target_l.index:
                target_l_pos.append(entity.state.p_pos - agent.state.p_pos)
            else:
                other_l_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)

        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + target_l_pos + other_l_pos + other_pos)

    # def observation(self, agent, world):
    #     # get positions of all entities in this agent's reference frame
    #     entity_pos = []
    #     for entity in world.landmarks:  # world.entities:
    #         entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    #     # entity colors
    #     entity_color = []
    #     for entity in world.landmarks:  # world.entities:
    #         entity_color.append(entity.color)
    #     # communication of all other agents
    #     comm = []
    #     other_pos = []
    #     for other in world.agents:
    #         if other is agent: continue
    #         comm.append(other.state.c)
    #         other_pos.append(other.state.p_pos - agent.state.p_pos)
    #     return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
