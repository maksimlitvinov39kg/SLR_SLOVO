import torch
from torch import nn
import numpy as np

USE_TYPES = ['left_hand', 'pose', 'right_hand']
START_IDX = 468
LIPS_IDXS0 = np.array([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88,
    178, 87, 14, 317, 402, 318, 324, 308,
])
LEFT_HAND_IDXS0 = np.arange(468, 489)
RIGHT_HAND_IDXS0 = np.arange(522, 543)
LEFT_POSE_IDXS0 = np.array([502, 504, 506, 508, 510])
RIGHT_POSE_IDXS0 = np.array([503, 505, 507, 509, 511])


LANDMARK_IDXS_BOTH_HANDS = np.concatenate((LIPS_IDXS0, LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0, LEFT_POSE_IDXS0, RIGHT_POSE_IDXS0))

LIPS_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_BOTH_HANDS, LIPS_IDXS0)).squeeze()
LEFT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_BOTH_HANDS, LEFT_HAND_IDXS0)).squeeze()
RIGHT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_BOTH_HANDS, RIGHT_HAND_IDXS0)).squeeze()
POSE_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_BOTH_HANDS, LEFT_POSE_IDXS0)).squeeze()

LIPS_START = 0
LEFT_HAND_START = LIPS_IDXS.size
RIGHT_HAND_START = LEFT_HAND_START + LEFT_HAND_IDXS.size
POSE_START = RIGHT_HAND_START + RIGHT_HAND_IDXS.size
N_COLS = LANDMARK_IDXS_BOTH_HANDS.size

INPUT_SIZE = 128
N_DIMS = 3

class PreprocessLayerBothHands(nn.Module):
    def __init__(self):
        super(PreprocessLayerBothHands, self).__init__()

        total_landmarks = len(LANDMARK_IDXS_BOTH_HANDS)
        
        first_row = [0] * total_landmarks
        lips_len = len(LIPS_IDXS)
        left_hand_len = len(LEFT_HAND_IDXS)
        right_hand_len = len(RIGHT_HAND_IDXS)
        
        for i in range(lips_len, lips_len + left_hand_len):
            if i < total_landmarks:
                first_row[i] = 0.50
        

        for i in range(lips_len + left_hand_len, lips_len + left_hand_len + right_hand_len):
            if i < total_landmarks:
                first_row[i] = 0.50
        
        normalisation_correction = torch.tensor([
            first_row,  
            [0] * total_landmarks,  
            [0] * total_landmarks, 
        ], dtype=torch.float32)
        
        self.register_buffer('normalisation_correction', normalisation_correction.transpose(0, 1))
        
    def pad_edge(self, t, repeats, side):
        if side == 'LEFT':
            return torch.cat((t[:1].repeat(repeats, 1, 1), t), dim=0)
        elif side == 'RIGHT':
            return torch.cat((t, t[-1:].repeat(repeats, 1, 1)), dim=0)
    
    def forward(self, data0):
        if isinstance(data0, np.ndarray):
            data0 = torch.tensor(data0, dtype=torch.float32)
            
        N_FRAMES0 = data0.shape[0]
        
        both_hands_idxs = np.concatenate([LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0])
        frames_hands_non_nan_sum = torch.sum(
            torch.where(torch.isnan(data0[:, both_hands_idxs, :]), 
                  torch.tensor(0., dtype=torch.float32), 
                  torch.tensor(1., dtype=torch.float32)), 
            dim=[1, 2]
        )
        
        non_empty_frames_idxs = torch.nonzero(frames_hands_non_nan_sum > 0).squeeze()
        
        if non_empty_frames_idxs.numel() == 0:
            non_empty_frames_idxs = torch.arange(N_FRAMES0, dtype=torch.float32)
        

        data = data0[non_empty_frames_idxs.long()]
        
        non_empty_frames_idxs = non_empty_frames_idxs.float()
        non_empty_frames_idxs -= torch.min(non_empty_frames_idxs)
        
        N_FRAMES = data.shape[0]
        
        data = data[:, LANDMARK_IDXS_BOTH_HANDS, :]
        

        correction = self.normalisation_correction.to(data.device)
        data = correction + ((data - correction) * torch.where(correction != 0, 
                                                      torch.tensor(-1.0, dtype=torch.float32), 
                                                      torch.tensor(1.0, dtype=torch.float32)))
        
        if N_FRAMES < INPUT_SIZE:
            pad_size = INPUT_SIZE - N_FRAMES
            non_empty_frames_idxs_pad = torch.full((pad_size,), -1.0, dtype=torch.float32)
            non_empty_frames_idxs = torch.cat([non_empty_frames_idxs, non_empty_frames_idxs_pad])
            
            data_pad = torch.zeros((pad_size, data.shape[1], data.shape[2]), dtype=torch.float32)
            data = torch.cat([data, data_pad], dim=0)
            
            data = torch.where(torch.isnan(data), torch.tensor(0.0, dtype=torch.float32), data)
            
            return data, non_empty_frames_idxs
        
        else:
            if N_FRAMES < INPUT_SIZE**2:
                repeats = (INPUT_SIZE * INPUT_SIZE) // N_FRAMES0
                data = data.repeat_interleave(repeats, dim=0)
                non_empty_frames_idxs = non_empty_frames_idxs.repeat_interleave(repeats)
            
            pool_size = len(data) // INPUT_SIZE
            if len(data) % INPUT_SIZE > 0:
                pool_size += 1
                
            if pool_size == 1:
                pad_size = (pool_size * INPUT_SIZE) - len(data)
            else:
                pad_size = (pool_size * INPUT_SIZE) % len(data)
            
            pad_left = pad_size // 2 + INPUT_SIZE // 2
            pad_right = pad_size // 2 + INPUT_SIZE // 2
            if pad_size % 2 > 0:
                pad_right += 1
            
            data = self.pad_edge(data, pad_left, 'LEFT')
            data = self.pad_edge(data, pad_right, 'RIGHT')
            
            non_empty_frames_idxs = torch.cat([non_empty_frames_idxs[0:1].repeat(pad_left), 
                                               non_empty_frames_idxs])
            non_empty_frames_idxs = torch.cat([non_empty_frames_idxs, 
                                               non_empty_frames_idxs[-1:].repeat(pad_right)])
            
            data = data.reshape(INPUT_SIZE, -1, N_COLS, N_DIMS)
            non_empty_frames_idxs = non_empty_frames_idxs.reshape(INPUT_SIZE, -1)
            
            data = torch.nanmean(data, dim=1)
            non_empty_frames_idxs = torch.nanmean(non_empty_frames_idxs, dim=1)
            
            data = torch.where(torch.isnan(data), torch.tensor(0.0, dtype=torch.float32), data)
            
            return data, non_empty_frames_idxs
