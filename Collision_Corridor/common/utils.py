import numpy as np
import inspect
import functools
import torch
import torch.nn.functional as f
from torch.autograd import Variable

def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def make_env(args):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load('multi_goal_spread' + ".py").Scenario()

    # create world
    world = scenario.make_world()

    # create multiagent2 environment
    if args.benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    args.n_players = env.n  # 包含敌人的所有玩家个数
    args.n_landmarks = len(world.landmarks)
    args.n_agents = env.n - args.num_adversaries  # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # 每一维代表该agent的obs维度

    if 'simple_spread' in args.scenario_name:
        args.input_dim_self = 4 + 2 * args.n_landmarks
    elif 'simple_tag' in args.scenario_name:
        # [p_vel + p_pos + relative pos to all landmarks + relative pos/vel to prey]
        args.input_dim_self = 4 + 2 * args.n_landmarks + 4 * args.num_adversaries
    elif 'simple_push' in args.scenario_name:
        # [p_vel + relative pos to all landmarks + relative pos to corresponding good agents]
        args.input_dim_self = 2 + 2 * args.n_landmarks + 2 * 1
    elif 'goal_spread' in args.scenario_name:
        args.input_dim_self = 4 + 2 * args.n_landmarks
    args.input_dim_others = args.obs_shape[0] - args.input_dim_self

    action_shape = []
    for content in env.action_space:
        action_shape.append(content.n)
    args.action_shape = action_shape[:args.n_agents]  # 每一维代表该agent的action维度
    args.high_action = 1
    args.low_action = -1
    args.obs_shape = args.obs_shape[0]
    args.n_actions = args.actio
    return env, args


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_action = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_action
    # get random actions in one-hot form
    random_action = Variable(torch.eye(logits.shape[1])[[np.random.choice(range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_action[i] if r > eps else random_action[i] for i, r in enumerate(torch.rand(logits.shape[0]))])


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_().cuda(), requires_grad=False)
    # U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return f.softmax(y / temperature, dim=1)

def handle_samples(samples, n_agents):
    o_cat, u_cat = [], []
    for agent_id in range(n_agents):
        o_cat.append(samples['o_%d' % agent_id])
        u_cat.append(samples['u_%d' % agent_id])
    o_cat = torch.from_numpy(np.array(o_cat)).float()
    u_cat = torch.from_numpy(np.array(u_cat)).long()
    return o_cat, u_cat

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)