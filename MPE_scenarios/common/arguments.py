import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multi-agent environments")

    parser.add_argument("--seed", type=int, default=678, help="random seed")

    parser.add_argument("--order", type=int, default=8, help="record the result under different parametric settings")
    parser.add_argument("--benchmark", type=bool, default=False, help="whether to produce benchmarking data (usually only done during evaluation)")
    parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dimension of all actor and critic networks")

    parser.add_argument("--algorithm", type=str, default="DDMA")
    # simple_spread(cooperative navigation), simple_tag(predato-prey), simple_push(keep away), simple_adversary(physical deception)
    parser.add_argument("--scenario_name", type=str, default="individual_defense", help="name of the scenario script")
    parser.add_argument("--episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--max_time_steps", type=int, default=1000000, help="number of time steps")  # 100 0000
    parser.add_argument("--num-adversaries", type=int, default=2, help="number of adversaries")

    parser.add_argument("--lr_actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer_size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch_size", type=int, default=64, help="number of episodes to optimize at the same time")

    # customized params
    parser.add_argument("--stage", type=int, default=2, help="choice of single-agent scenario or multi-agent scenario")
    parser.add_argument("--train_from_nothing", type=bool, default=False, help="whehter to load the pretrained network in stage-1")
    parser.add_argument("--use_overlap", type=bool, default=True, help="whether to consider the sparse interaction")
    parser.add_argument("--pretrain_time_steps", type=int, default=100 * 100, help='agent should conduct multi-agent learning')
    parser.add_argument("--use_diy_credit", type=bool, default=False, help="whether to use the single_agent Q as the bias")
    parser.add_argument("--use_attention", type=bool, default=False, help="whether to use the attention in centralized critic")
    parser.add_argument("--attention_heads", type=int, default=2, help="number of attention heads")

    parser.add_argument("--initial_rate", type=float, default=0.2)
    parser.add_argument("--final_rate", type=float, default=0.01)

    # detector相关参数
    parser.add_argument("--lr_detec", type=float, default=1e-3, help="learning rate of detector")
    parser.add_argument("--detec_buffer_size", type=int, default=int(1e4), help="max size of the detec buffer")     # original 1e4
    parser.add_argument("--detec_batch_size", type=int, default=2000, help="number of samples used to optimize in each training iteration")     # original 2000
    parser.add_argument("--detec_num_updates", type=int, default=200, help="training iterations of prior network")      # original 200
    parser.add_argument("--detec_training_rate", type=int, default=100 * 100, help="training interval (step))")         # original 100*100
    parser.add_argument("--detec_percentile", type=int, default=60, help="dynamic threshold of kl_values to get labels (automatic setting)")    # original 60
    parser.add_argument("--detec_threshold", type=int, default=0.05, help="static threshold of kl_values to get labels (manual setting)")       # 0.05

    parser.add_argument("--save_dir", type=str, default="./log", help="directory in which experimental results are saved")
    parser.add_argument("--model_save_dir", type=str, default="./model_log", help="directory in which models are saved")
    parser.add_argument("--save_rate", type=int, default=500000, help="model_save_interval (episode)")

    parser.add_argument("--log", type=bool, default=False, help="whether record the change of loss in the training process")
    parser.add_argument("--log_interval", type=int, default=2000, help="log interval for each logged data (timestep)")

    parser.add_argument("--evaluate_episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate_episode_len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate_rate", type=int, default=100 * 10, help="how often to evaluate network (step)")     # original = 50 * 10

    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the network")
    parser.add_argument("-load_model", type=bool, default=False, help="must keep track with the evaluate option")

    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--device", type=str, default='1', help="which GPU is used")
    args = parser.parse_args()

    return args
