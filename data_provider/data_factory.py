from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred, Dataset_GaussianP, Dataset_GaussianPCoregionalization
from torch.utils.data import DataLoader,  Subset
from torch.utils.data import ConcatDataset
from data_provider.data_loader import  RandomCombinedDataLoader, IterativeDataLoader, RoundRobinDataLoader
import torch
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'gaussian': Dataset_GaussianP, #this gaussian data is independently generated at each variable
    'gaussian_coregionalization': Dataset_GaussianPCoregionalization
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    elif flag == 'val':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for valid
        freq = args.freq
    else:
        shuffle_flag = False if args.ordered_data else True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    print(flag, len(data_set))


    if flag == 'train' and args.data_amount != -1:
        #randomly select data_amount samples from the data_set
        total_samples = len(data_set)
        # Generate random indices
        if args.ordered_data:
            random_indices = torch.arange(total_samples)[-args.data_amount:].tolist()
        else:
            random_indices = torch.randperm(total_samples)[:args.data_amount].tolist()
        data_set = Subset(data_set, random_indices) 

    if flag == 'val' and args.data_amount != -1:
        total_samples = len(data_set)
        if args.ordered_data:  
            random_indices = torch.arange(total_samples)[-args.data_amount//5:].tolist()
        else:
            random_indices = torch.randperm(total_samples)[:args.data_amount//5].tolist() #take the validation as //5 of the number of train data
        data_set = Subset(data_set, random_indices)


    if flag == 'train' and args.exp_name == 'gaussian': 
        gaussian_data_set = Dataset_GaussianP(root_path=args.synthetic_root_path, data_path = args.synthetic_data_path , n_variables = args.enc_in, seq_len = args.seq_len, pred_len = args.pred_len,  size=[args.seq_len, args.label_len, args.pred_len])
        print("running with synthetic gaussian data")
        data_set = gaussian_data_set
    if flag == 'train' and args.exp_name == 'gaussian_coregionalization': 
        gaussian_data_set = Dataset_GaussianPCoregionalization(root_path=args.synthetic_root_path, data_path = args.synthetic_data_path , n_variables = args.enc_in, seq_len = args.seq_len, pred_len = args.pred_len,  size=[args.seq_len, args.label_len, args.pred_len], synthetic_length = args.synthetic_length, stride = args.stride)
        print("running with synthetic gaussian coregionalization data")
        if not args.combine_Gaussian_datasets:
            data_set = gaussian_data_set
        else:
            gaussian_dataset2 = Dataset_GaussianP(root_path=args.synthetic_root_path, data_path = "ind-kernelsynth-largechannel.arrow" , n_variables = args.enc_in, seq_len = args.seq_len, pred_len = args.pred_len,  size=[args.seq_len, args.label_len, args.pred_len])

            data_loader_gaussian = DataLoader(gaussian_data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers, drop_last=drop_last)
            data_loader_gaussian2 = DataLoader(gaussian_dataset2, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers, drop_last=drop_last)
            #concat data loaders using IterativeDataLoader
            data_loader = IterativeDataLoader(data_loader_gaussian2, data_loader_gaussian)
            data_set = ConcatDataset([gaussian_dataset2, gaussian_data_set])
            print("combined gaussian datasets")
            return data_set, data_loader


    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    
    return data_set, data_loader
