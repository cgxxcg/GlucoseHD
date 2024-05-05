import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_file, seq_len, pred_len, scale=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        
        # Read the CSV file
        self.data = pd.read_csv(csv_file)
        
                # Parse the date column with the correct format string
        self.data['Time'] = pd.to_datetime(self.data['Time'], format="%d-%b-%Y %H:%M:%S")
        
        # Sort the dataframe by date
        self.data = self.data.sort_values(by='Time').reset_index(drop=True)
        
        # Scale the data if needed
        if self.scale:
            self.scaler = StandardScaler()
            self.data['CGM'] = self.scaler.fit_transform(self.data['CGM'].values.reshape(-1, 1))
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        idx += self.seq_len  # Adjust index to account for sequence length
        
        # Extract sequence and target
        seq_x = self.data.iloc[idx - self.seq_len:idx]['CGM'].values.astype(np.float32)
        seq_y = self.data.iloc[idx:idx + self.pred_len]['CGM'].values.astype(np.float32)
        
        return seq_x, seq_y
    
    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        else:
            return data.flatten()


# Directory where CSV files are located
data_dir = "dataset/processedcsv"

# Dataset numbers
datasets = [540, 544, 552, 559, 563, 567, 570, 575, 584, 588, 591, 596]

# Dictionary to hold train and test file paths for each dataset
file_paths = {}

# Generate file paths
for dataset_num in datasets:
    train_file = os.path.join(data_dir, f"ohio{dataset_num}_Training.csv")
    test_file = os.path.join(data_dir, f"ohio{dataset_num}_Testing.csv")
    file_paths[dataset_num] = {'train': train_file, 'test': test_file}




# Example usage for dataset number 540:
train_dataset = TimeSeriesDataset(file_paths[540]['train'], seq_len=12, pred_len=6)
test_dataset = TimeSeriesDataset(file_paths[540]['test'], seq_len=12, pred_len=6)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)




# Get the number of samples in the train_dataset
num_samples = len(train_dataset)
print("Number of samples in train_dataset:", num_samples)

# Get the dimensions of the first sample in train_dataset
first_sample = train_dataset[0]  # Get the first sample
seq_x, seq_y = first_sample[0], first_sample[1]  # Unpack the sequence and target
seq_x_shape = seq_x.shape
seq_y_shape = seq_y.shape
print("Shape of the input sequence in train_dataset:", seq_x_shape)
print("Shape of the target sequence in train_dataset:", seq_y_shape)
