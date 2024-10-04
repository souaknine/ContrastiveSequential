import matplotlib.pyplot as plt
import os
import os.path as opt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class Dataset_shrec22(Dataset):
    def __init__(
        self,
        data_dir: str,
        T: int,
        normalize: bool=True,
        mean = None,
        std = None,
    ):
        """
        :param data_dir: path to the dataset directory
        :param T: parameter deciding sequence length. All sequence with length smaller than T will be zero-padded to length T.
        :param step: stride for sliding window
        :param normalize: whether to normalize data using train split mean and variance
        """
        self.path_to_data = data_dir
        self.T = T
        self.sequence = []
        self.labels_window = (
            []
        )
        self.label = []
        self.normalize = normalize
        self.mean  = None
        self.std = None

        self.label_map = [
            "ONE",
            "TWO",
            "THREE",
            "FOUR",
            "OK",
            "MENU",
            "LEFT",
            "RIGHT",
            "CIRCLE",
            "V",
            "CROSS",
            "GRAB",
            "PINCH",
            "DENY",
            "WAVE",
            "KNOB",
        ]            



        sequences, labels = [], []
        
        with open(opt.join(self.path_to_data, "annotations.txt"), "r") as gt:   
                        
            # Each annotation row contains: "sequenceNum;Label;GSFrame;GEFrame;...;Label;GSFrame;GEFrame;"
            for line in tqdm(gt.readlines(), desc='loading shrec22 data....', colour='green'):
                # compensating for dataset errors
                gt_line = line[:]
                line = line.split(";")
                # get sequence number, corresponds to filename
                file_name = line[0]
                line = line[1:-1]
                
                ## read file
                file_poses = []
                with open(opt.join(self.path_to_data, f"{file_name}.txt"), "r") as fp:
                    # (N,3,10,26)
                    # Each sequence file contains: 'Frame Index(integer); Time_stamp(float); Joint1_x (float);
                    # Joint1_y; Joint1_z;  Joint1_rotx; Joint1_roty; Joint1_rotz; Joint1_rotw; Joint2_x;Joint2_y; Joint2_z; .....'.
                    for line_idx, seq_line in enumerate(fp.readlines()):
                        seq_line = seq_line.split(";")[
                            2:-1
                        ]  # remove index, timestamp and end-of-line
                        seq_line = (
                            np.reshape(seq_line, (26, 3)).astype(np.float64)
                        ) 
                        file_poses.append(seq_line)   
                
                file_poses = np.array(file_poses)
                # for each gesture
                for index in range(0, len(line), 3):
                    # gesture start frame
                    s = int(line[index + 1])
                    # gesture end frame
                    e = int(line[index + 2])
                    # gesture label
                    lab = line[index]
                    
                    sequences.append(file_poses[s:e+1])
                    labels.append(self.label_map.index(lab))

        self.sequences = sequences
        self.labels = labels
        self.len_data = len(self.sequences)
       

        if normalize:
            # if self.mean is None or self.std is None:
            #     self.mean, self.std = self.compute_normalization_stats()
            #     self.mean = self.mean.unsqueeze(0)  
            #     self.std = self.std.unsqueeze(0) 
            #     print(self.mean.shape)   
            # else:
            #     self.mean = self.mean.unsqueeze(0)
            #     self.std = self.std.unsqueeze(0) 
            for i, sequence in enumerate(self.sequences):
                frame_means = np.mean(sequence, axis=1, keepdims=True)
                frame_stds = np.std(sequence, axis=1, keepdims=True) + 1e-8  # Add epsilon to avoid division by zero
                sequence = (sequence - frame_means) / frame_stds
                self.sequences[i] = sequence


    def compute_normalization_stats(self):
        all_data = []
        for seq in self.sequences:
            all_data.append(torch.from_numpy(seq))
            
        
        
        all_data = torch.cat(all_data, dim=0)
        
       
        mean = all_data.mean(dim=0)  
        std = all_data.std(dim=0)    
        
        return mean, std

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        
        return (x - self.mean) / (self.std + 1e-7)



    def __len__(self):
        return self.len_data
    
    def pad_sequence(self, seq, target_length):
        seq_len = seq.shape[0]
        if seq_len >= target_length:
            return seq[:target_length]
        pad_len = target_length - seq_len
        padding = torch.zeros((pad_len, seq.shape[1], seq.shape[2]), dtype=seq.dtype)

        return torch.cat([seq, padding], dim=0)

    def _preprocess(self, sequence: np.ndarray):
        # prepare data
        sequence = np.array(sequence)
        # turn to tensor
        sequence_tensor = torch.from_numpy(sequence)
        if len(sequence_tensor.shape) > len(sequence.shape):
            sequence_tensor = sequence_tensor.unsqueeze(0)

        # if self.normalize:
        #     # normalize if necessary
        #     sequence_tensor = self._normalize(sequence_tensor)
           

        sequence_tensor = self.pad_sequence(sequence_tensor, self.T)

        return sequence_tensor

    def __getitem__(self, item):
        sequence = self.sequences[item]

        original_length = len(sequence)
        sequence = self._preprocess(sequence)


        label = self.labels[item]
        
        return dict(Sequence=sequence,
                    Label=label, OG_length = original_length)
    
    


train_data_dir = "/Users/sam/Desktop/PDI/data_skeleton/training_set"
test_data_dir = "/Users/sam/Desktop/PDI/data_skeleton/test_set"

T = 90
normalize = True

## datasets
train_set = Dataset_shrec22(train_data_dir,
                             T=T,
                             normalize=normalize,
                          )

test_set = Dataset_shrec22(test_data_dir,
                           T=T,
                           normalize=normalize,
                          )

batch_size = 4

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

len(train_loader), len(test_loader)

def trainloader_shrek(batch_size):
    train_set = Dataset_shrec22(train_data_dir,
                             T=T,
                             normalize=normalize,
                          )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader


def testloader_shrek(batch_size):
    test_set = Dataset_shrec22(test_data_dir,
                             T=T,
                             normalize=normalize,
                          )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return test_loader

