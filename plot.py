from torch.utils.tensorboard import SummaryWriter
import torch

log_name = './log/reference_ece'#'./log/ece_reference_truncated'#
n_logs = 6
writter = SummaryWriter('./runs/QMIX_ECE')
all_infos = []
for i in range(n_logs):
    try:
        info = torch.load(log_name+str(i))
        all_infos.append(info)
    except:
        continue

total_length = len(info)
n_logs = len(all_infos)

for i in range(len(info)):
    reward_mean = 0.0
    loss_mean = 0.0
    for j in range(n_logs):
        reward_mean += all_infos[j][i]['Ep.Reward']
        loss_mean += all_infos[j][i]['Loss']

    reward_mean /= n_logs
    loss_mean /= n_logs

    writter.add_scalar('Train/Ep.Reward', reward_mean, i)
    writter.add_scalar('Train/Loss', loss_mean, i)