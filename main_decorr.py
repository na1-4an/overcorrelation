from options.base_options import BaseOptions, reset_weight
from trainer_decorr import trainer
import torch
import os
import numpy as np
import random

def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.cuda.set_device(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    # PyTorch에서 결정론적 알고리즘 강제 설정 (1.12.1+에서는 지원)
    torch.use_deterministic_algorithms(True, warn_only=True)

# seeds = [100, 200, 300, 400, 500]
# layers_GCN = [2, 15, 30]
seeds = [100, 200]
layers_GCN = [2, 15, 30]

def main(args):
    if args.type_model in ['GCN', 'GAT', 'GCNII', 'Cheby', 'DeepGCN']:
        layers = layers_GCN
    else:
        layers = layers_SGCN
    print(args.weight_decay)
    acc_test_layers = []
    MI_XiX_layers = []
    dis_ratio_layers = []
    outs_layers = []
    for layer in layers:
        args.num_layers = layer
        if args.type_norm == 'group':
            args = reset_weight(args)
        acc_test_seeds = []
        MI_XiX_seeds = []
        dis_ratio_seeds =  []
        outs_seeds = []
        for seed in seeds:
            args.random_seed = seed
            set_seed(args)
            trnr = trainer(args)
            acc_test, MI_XiX, dis_ratio, outs = trnr.train_compute_MI()

            outs_seeds.append(outs)
            acc_test_seeds.append(acc_test)
            MI_XiX_seeds.append(MI_XiX)
            dis_ratio_seeds.append(dis_ratio)
        avg_acc_test = (np.mean(acc_test_seeds) * 100)
        std_acc_test = (np.std(acc_test_seeds) * 100)
        final_acc_test = '{:.2f} ± {:.2f}'.format(avg_acc_test, std_acc_test)
        avg_MI_XiX = np.mean(MI_XiX_seeds)
        avg_dis_ratio = np.mean(dis_ratio_seeds)

        import pandas as pd
        outs_seeds = pd.DataFrame(outs_seeds)
        # outs_seeds = np.array(outs_seeds)
        outs_layers.append(outs_seeds.mean(0).values)
        acc_test_layers.append(final_acc_test)
        MI_XiX_layers.append(avg_MI_XiX)
        dis_ratio_layers.append(avg_dis_ratio)

    print(f'experiment results of {args.type_norm} applied in {args.type_model}' +
            f'on dataset {args.dataset} with dropout {args.dropout}, dropedge {args.dropedge}'+
            f'lr {args.lr}, alpha {args.alpha}, beta {args.beta}')
    print('number of layers: ', layers)
    print('test accuracies: ', acc_test_layers)
    print('Mean of corr_2, corr, sim_2, sim:', outs_layers)
    log_file = 'experiment_results.log'  # 로그 파일 이름

    # 로그 파일 열기 (쓰기 모드, 기존 내용은 덮어씀)
    with open(log_file, 'a') as f:  # 'a' 모드는 append, 즉 기존 파일에 덧붙이기
        # 로그 파일에 결과 쓰기
        f.write(f'<dataset: {args.dataset}>experiment results of {args.type_norm} applied in {args.type_model}' +
                f'on dataset {args.dataset} with dropout {args.dropout}, dropedge {args.dropedge}'+
                f'lr {args.lr}, alpha {args.alpha}, beta {args.beta}\n')
        f.write(f'number of layers: {layers}\n')
        f.write(f'test accuracies: {acc_test_layers}\n')
        f.write(f'Mean of corr_2, corr, sim_2, sim: {outs_layers}\n\n')

if __name__ == "__main__":
    args = BaseOptions().initialize()
    main(args)