import os
import torch
import numpy as np
from common.arguments import get_args
from common.utils import make_env

if __name__ == '__main__':
    # get the params
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    env, args = make_env(args)
    print('Note:', args.scenario_name, args.algorithm, args.order, args.obs_shape, args.n_agents, args.input_dim_self, args.input_dim_others)

    if args.algorithm == 'MADDPG':
        from Runner.runner_maddpg import Runner
    elif args.algorithm == 'overlap':
        from Runner.runner_overlap import Runner
    elif args.algorithm == 'DDMA':
        from Runner.runner_ddma import Runner
    elif args.algorithm == 'MAAC':
        from Runner.runner_maac import Runner
    elif args.algorithm == 'G2ANet':
        from Runner.runner_g2anet import Runner
    elif args.algorithm == 'NoisyMADDPG':
        from Runner.runner_noisymaddpg import Runner

    runner = Runner(args, env)

    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()