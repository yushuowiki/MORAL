import os
import sys
import torch
import random
import numpy as np
import warnings
from models import *
from trainer import MyTrainer
from dataset import get_dataset, get_data_info

from my_utils.utils.dict import Merge, DictIter

sys.path.append(os.getcwd() + '/..')
warnings.filterwarnings("ignore")


seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def train(repeat, args_init, device_num):
    record_list = []
    args_iter = iter(DictIter(args_init))
    for args in args_iter:
        for _ in range(repeat):
            dataset = get_dataset(args)
            data_info = get_data_info(dataset[0])
            args = Merge(args, data_info)
            #['MORAL', 'MORE', 'SGC', 'AGNN', 'GAT', 'GCN', 'SAGE', 'APPNP', 'MLP'] 
            assert args['model_name'] in ['MORAL', 'MORE', 'SGC', 'AGNN', 'GAT', 'GCN', 'SAGE', 'MLP']
            if args['model_name'] == 'MORAL':
                model = MORAL(args['num_feature'], args['struc_input'], args['layer_size'], args['num_class'], args['dropout'])
            elif args['model_name'] == 'MORE':
                model = MORE(args['num_feature'], args['struc_input'], args['layer_size'], args['num_class'], args['dropout'])
            elif args['model_name'] == 'SGC':
                model = SGC(args['num_feature'], args['num_class'], args['dropout'])
            elif args['model_name'] == 'AGNN':
                model = AGNN(args['num_feature'], args['num_class'], args['dropout'])
            elif args['model_name'] == 'GAT':
                model = GAT(args['num_feature'], args['layer_size'], args['num_class'], 2, args['dropout'])
            elif args['model_name'] == 'GCN':
                model = GCN(args['num_feature'], args['layer_size'], args['num_class'], args['dropout'])
            elif args['model_name'] == 'SAGE':
                model = GraphSage(args['num_feature'], args['layer_size'], args['num_class'])
            elif args['model_name'] == 'MLP':
                model = MLP(args['num_feature'], args['layer_size'], args['layer_size'], args['num_class'], args['dropout'])
            
            device = torch.device(f"cuda{device_num}" if torch.cuda.is_available() else "cpu")
            model.to(device)
            criterion = nn.CrossEntropyLoss(reduction='mean')
            optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
            topic = f"{args['model_name']}_{args['dataset']}_{args['label_ratio']}"

            trainer = MyTrainer(args, model, dataset, criterion, optimizer, topic, device)
            record = trainer()
            record_list.append(f'{record[0]*100:.4f}')
            y_pred=record[1]
            feature_info=record[2]
    


    return y_pred,feature_info

def main(args_init, device_num,  repeat):
    record_list = train(repeat, args_init, device_num)

if __name__ == "__main__":
    from args import get_args_parser
    parser = get_args_parser()
    opt = parser.parse_args()

    args_init = {
        'model_name': opt.model_name,
        'dataset': opt.dataset,
        'label_ratio': opt.label_ratio, 
        'lr': opt.lr,
        'weight_decay': opt.weight_decay,
        'layer_size': opt.layer_size,
        'dropout': opt.dropout,
        'epochs': opt.epochs,
        'early_stop': opt.early_stop,
        'batch_size': opt.batch_size,
        'beta': opt.beta,
        'record_path': opt.record_path,
        'log_path': opt.log_path,
        'animator_output': opt.animator_output,
    }
    print(opt.gpu)
    repeat = opt.repeat
    gpu = opt.gpu
    main(args_init, gpu, repeat)