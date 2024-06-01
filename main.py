from qmix import *
import torch
from envs import *
from EnergyHarvest.env import EnergyHarvest

ep_steps = 25
total_steps = int(1e6)*ep_steps
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
    'eps_start':0.99,
    'eps_end':0.05,
    'eps_dec':total_steps*0.2, # more leads to slower decay
    'gamma':0.96,
    'lr': 1e-4,
    'tau':0.005, # more is harder
    'batch_size':256,
    'memory_size':1000000,
    'device':device,
    'episode_length':ep_steps,
    'logdir':'test'#'energy_harvest_4frame_sensor20_power20_rewardv2',
}
ga_config = {
    'population_size':300,
    'crossover_rate':0.9,
    'mutation_rate':0.05,
}

if __name__ == '__main__':
    print(device)
    env = Reference()#EnergyHarvest_v1(n_frame_stacks=4, n_sensors=20, power=20.0)
    qmix = QMIX(env, config)
    qmix.learn(total_steps)
    
    qmix.save_model()