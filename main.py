from models.gasmil import GASMIL
from utils.dataset import Dataset, pad_collate_fn
from utils.inference import inference
import yaml, os
import pandas as pd
import argparse
import torch
import numpy as np


def test():
    # define model
    with open(args.config, 'r') as f:
        params = yaml.safe_load(f)
        
    model = GASMIL(
            params['in_channels'],
            params['groups'],
            params['top_k'],
            params['bottom_k'],
            params['attn'],
            params['include_groupall'],
            params['groupall_top_k'],
            params['groupall_bottom_k'],
            params['grouped_feat_dim'],
            params['num_class'],
            params['clf_feat_dim'],
        )
    
    # load pretrain model
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    
    # load data
    data = pd.read_csv(args.csv_path)
    
    for key in args.feature_key_list:
        if key not in data.keys():
            raise Exception(f"no {key} in input files!")
    
    test_dataset = Dataset(data, args.feature_key_list)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                 batch_size=args.batch_size,
                                                 shuffle=False, 
                                                 num_workers=args.num_workers, 
                                                 collate_fn =pad_collate_fn)
    
    
    pred = inference(test_loader, model, args.device)
    pred.to_csv(args.save_csv_path, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GASMIL Inference Example')

    parser.add_argument('--csv_path', required=True, type=str, help="inference data csv file path.")
    parser.add_argument('--feature_key_list', required=True, 
                        type=lambda s: [item.strip() for item in s.split(',')], 
                        help="features from which foundation models")
    parser.add_argument('--model_path', required=True, type=str, help="model path.")
    parser.add_argument('--config', default='config.yml', type=str, 
                            help=" contains gasmil params.")
    parser.add_argument('--batch_size', default=64, type=int, help='Number of batch size.')
    parser.add_argument('--num_workers', default=64, type=int, help='Number of workers for data loader.')
    parser.add_argument('--device', default='cuda:0', choices=['cuda', 'cpu'], type=str, help='Run on cpu or gpu.')
    parser.add_argument('--save_csv_path', default=os.path.join(os.getcwd(), 'output/result.csv'),type=str, help='save results to')
    
    args = parser.parse_args()

    print('Start Testing')
    test()
    print('End.')