import os
import torch
import numpy as np
from common.arguments import get_args
from envs.hallway_single_agent import SingleHallway
from envs.hallway_two_agent import MultiHallway
from envs.hallway_single_agent_visible import SingleHallwayVisible
from envs.hallway_two_agent_visible import MultiHallwayVisible

if __name__ == '__main__':
    # get the params
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    if args.single_map:
        if not args.evaluate:
            env = SingleHallway(args)
        else:
            env = SingleHallwayVisible(args)
    else:
        if not args.evaluate:
            env = MultiHallway(args)
        else:
            env = MultiHallwayVisible(args)

    if args.algorithm == 'DDMA':
        from runner.runner_ddma import Runner
    elif args.algorithm == 'qmix':
        from runner.runner_qmix import Runner
    elif args.algorithm == 'vdn':
        from runner.runner_vdn import Runner
    elif args.algorithm == 'MAPG':
        from runner.runner_mapg import Runner
    elif args.algorithm == 'IQL':
        from runner.runner_iql import Runner
    elif args.algorithm == 'IPG':
        from runner.runner_ipg import Runner

    runner = Runner(args, env)

    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()
