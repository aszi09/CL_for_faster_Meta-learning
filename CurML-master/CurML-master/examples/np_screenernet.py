import argparse
import sys, os
curML_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Add the parent directory to sys.path
sys.path.append(curML_dir)

# Import my_module from the package module_dir
from curriculum.algorithms import ScreenerNet
from curriculum.algorithms.screener_net import ScreenerNetTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='resnet')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epochs', type=int, default=100000)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()


trainer = ScreenerNetTrainer(
    data_name=args.data,
    net_name=args.net,
    device_name=args.device,
    num_epochs=args.epochs,
    random_seed=args.seed,
)
trainer.fit()
trainer.evaluate()
