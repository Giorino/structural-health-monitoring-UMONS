import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUModel(nn.Module):
    """GRU-based model for crack prediction"""

    def __init__(self, input_size=6, hidden_size=64, num_classes=4, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = gru_out[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        return output


class LSTMModel(nn.Module):
    """LSTM-based model for crack prediction"""

    def __init__(self, input_size=6, hidden_size=64, num_classes=4, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = lstm_out[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        return output


class CNNGRUModel(nn.Module):
    """CNN-GRU hybrid model for crack prediction"""

    def __init__(self, input_size=6, hidden_size=64, num_classes=4, dropout=0.2):
        super(CNNGRUModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.gru = nn.GRU(32, hidden_size, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)
        output = gru_out[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        return output


class TransformerModel(nn.Module):
    """Transformer Encoder model for crack prediction"""

    def __init__(self, input_size=6, d_model=64, num_heads=2, num_layers=2, num_classes=4, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        transformer_out = self.transformer(x)
        output = transformer_out.mean(dim=1)
        output = self.dropout(output)
        output = self.fc(output)
        return output


class CNNModel(nn.Module):
    """Pure CNN model for crack prediction"""

    def __init__(self, input_size=6, num_classes=4, dropout=0.2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.squeeze(-1)
        output = self.dropout(x)
        output = self.fc(output)
        return output



