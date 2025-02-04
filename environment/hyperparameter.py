import argparse


def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default='HalfCheetah-v5')
    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--eval_flag', default=True, type=bool)
    parser.add_argument('--eval-freq', default=5000, type=int)
    parser.add_argument('--eval-episode', default=5, type=int)
    parser.add_argument('--hard-update', default=False, type=bool)
    parser.add_argument('--hard-update-epi-freq', default=1, type=int)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--max-step', default=300000, type=int)
    parser.add_argument('--update_after', default=0, type=int)
    parser.add_argument('--hidden-dims', default=(64, 64))
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--buffer-size', default=1e5, type=int)
    parser.add_argument('--update-every', default=1, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--epsilon-start', default=1.0, type=float)
    parser.add_argument('--epsilon-end', default=0.01, type=float)
    parser.add_argument('--epsilon-decay', default=0.005, type=float)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--max-grad-norm', default=10, type=float)
    parser.add_argument('--log', default=True, type=bool)
    parser.add_argument('--log-dir', default='.log/DQN', type=str)
    parser.add_argument('--print-loss', default=False, type=bool)

    param = parser.parse_args()

    return param


