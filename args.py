import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    #['MORAL', 'MORE', 'SGC', 'AGNN', 'GAT', 'GCN', 'SAGE', 'APPNP', 'MLP']
    parser.add_argument("--model_name", type=str, default='MORAL', help="")
    # ['cora', 'citation', 'dblp', 'wikipedia', 'citeseer', ''email', 'facebook', 'terror', 'polblogs','acm']
    parser.add_argument("--dataset", type=str, default='acm', help="")
    parser.add_argument("--label_ratio", type=float, default=0.6, help="")

    parser.add_argument("--lr", type=float, default=0.003, help="")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="")
    parser.add_argument("--layer_size", type=int, default=64, help="")
    parser.add_argument("--dropout", type=float, default=0.3, help="")
    parser.add_argument("--epochs", type=int, default=10000, help="")
    parser.add_argument("--early_stop", type=int, default=500, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="")

    parser.add_argument("--beta", type=float, default=0.0, help="")

    parser.add_argument("--record_path", type=str, default='./records/record.log', help="")
    parser.add_argument("--log_path", type=str, default='./logs', help="")
    parser.add_argument("--animator_output", action='store_true', help="")

    parser.add_argument("--repeat", type=int, default=10, help="")
    parser.add_argument("--gpu", type=str, default="", help="")
    
    parser.set_defaults(optimize=False)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    opt = parser.parse_args()
    print(opt)