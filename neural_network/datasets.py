import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Custom dataset for sequence data with crack prediction labels"""

    def __init__(self, sequences, labels, sequence_length=25):
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.LongTensor([self.labels[idx]])

        # Pad or truncate sequence to fixed length
        if len(sequence) < self.sequence_length:
            padding = torch.zeros(self.sequence_length - len(sequence), sequence.shape[1])
            sequence = torch.cat([sequence, padding], dim=0)
        elif len(sequence) > self.sequence_length:
            sequence = sequence[:self.sequence_length]

        return sequence, label.squeeze()



