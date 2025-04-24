### All models are commented
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Splittind data into train-test (80-20)

# Load dataset
df = pd.read_csv('Dataset/FidelFolio_Dataset_Cleaned.csv')

# Exclude target columns from scaling
target_cols = [' Target 1 ', ' Target 2 ', ' Target 3 ']
feature_cols = [f'Feature{i}' for i in range (1,29)]

X = df[feature_cols]
y = df[target_cols]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# print("Training set size:", X_train.shape)
# print("Test set size:", X_test.shape)






# 2. Models

#----------> MLP MODEL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader  
import torch.optim as optim




# 1. Defining Model Architecture
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)
    
# Convert to tensors (on CPU)
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create DataLoaders with batch_size = 64
train_ds = TensorDataset(X_train_tensor, y_train_tensor)
test_ds = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





# 2. Creating Model
model = MLP(input_size=X_train.shape[1], output_size=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)





# 3. Training Model
import matplotlib.pyplot as plt

epochs = 1000
losses = []  # <-- Track losses here

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb = xb.to(device) 
        yb = yb.to(device)  

        preds = model(xb)   # preds: (64, 3)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)  

    # print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# # Plot loss vs epoch
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, epochs + 1), losses, label='Training Loss', color='blue')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss vs Epoch")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# print(X_train.shape)








# 4. Save trained Model

torch.save(model.state_dict(), 'Model/fidelfolio_model_MLP.pkt')
# print("Model saved as fidelfolio_model_MLP.pkt")





# 5. Making prediction from trained model
from sklearn.metrics import mean_squared_error
import numpy as np

# Set model to evaluation mode
model.eval()

# Make predictions on the test set
with torch.no_grad():
    y_pred_test = model(X_test_tensor.to(device)).cpu().numpy()
    y_true_test = y_test_tensor.cpu().numpy()

# Get the target column names
target_names = y_test.columns.tolist()



 

# 6. Calculate RMSE for each target
for i, target in enumerate(target_names):
    rmse = mean_squared_error(y_true_test[:, i], y_pred_test[:, i], squared=False)
    # print(f"Test RMSE for {target}: {rmse:.4f}")





# # 7. Plot predicted vs actual for each target
import matplotlib.pyplot as plt

# for i, target in enumerate(target_names):
#     plt.figure(figsize=(6, 6))
#     plt.scatter(y_true_test[:, i], y_pred_test[:, i], alpha=0.5)
#     plt.plot([y_true_test[:, i].min(), y_true_test[:, i].max()],
#              [y_true_test[:, i].min(), y_true_test[:, i].max()],
#              'r--')
#     plt.xlabel(f'Actual {target}')
#     plt.ylabel(f'Predicted {target}')
#     plt.title(f'Predicted vs Actual - {target}')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"MLP_result_{target}")
#     plt.show()






#----------> LSTM MODEL

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


# 1. Sort and prepare data -> Company wise sorted by Year
features = [col for col in df.columns if 'Feature' in col]
targets = [' Target 1 ', ' Target 2 ', ' Target 3 ']
df_sorted = df.sort_values(by=["Company", "Year"])

sequences = []
target_seq = []

for company, group in df_sorted.groupby("Company"):
    group = group.reset_index(drop=True)
    if len(group) >= 3:
        X = group[features].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        y = group[targets].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        sequences.append(torch.tensor(X))
        target_seq.append(torch.tensor(y[-1]))  # only the last target

# Custom Dataset
task_dataset = list(zip(sequences, target_seq))

class LSTMDataset(Dataset):
    def __init__(self, sequence_data):
        self.data = sequence_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create DataLoader with list-based batching
def collate_fn(batch):
    X_list, y_list = zip(*batch)
    return list(X_list), list(y_list)

train_dataset = LSTMDataset(task_dataset)
data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)





# 2. Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, packed_x):
        packed_out, (hn, cn) = self.lstm(packed_x)
        out = self.dropout(hn[-1])  # last hidden state from top layer
        return self.fc(out)





# 3. Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = LSTMModel(input_size=len(features), hidden_size=64, output_size=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)

import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_sequence

epochs = 2000
losses = []

for epoch in range(epochs):
    model1.train()
    total_loss = 0
    for X_list, Y_list in data_loader:
        packed_x = pack_sequence([x.to(device) for x in X_list], enforce_sorted=False)
        y_batch = torch.stack([y.to(device) for y in Y_list])

        pred = model1(packed_x)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    losses.append(avg_loss)
    # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")





# 4. Save the Model

torch.save(model1.state_dict(), 'Model/fidelfolio_model_LSTM.pkt')
print("Model saved as fidelfolio_model_LSTM.pkt")






# 5. Evaluation/Prediction
model1.eval()
preds1 = []
actuals1 = []

with torch.no_grad():
    for X_list, Y_list in data_loader:
        packed_x = pack_sequence([x.to(device) for x in X_list], enforce_sorted=False)
        y_batch = torch.stack([y.to(device) for y in Y_list])

        pred1 = model1(packed_x)
        preds1.append(pred1.cpu())
        actuals1.append(y_batch.cpu())

# Stack all batches into single arrays
preds1 = torch.cat(preds1).numpy()
actuals1 = torch.cat(actuals1).numpy()


# plt.figure(figsize=(10, 5))
# plt.plot(range(1, epochs + 1), losses, label='Training Loss', color='blue')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss vs Epoch")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()








# 6. Plot predicted vs actual for each target

import matplotlib.pyplot as plt

# for i, target in enumerate(targets):
#     plt.figure(figsize=(6, 6))
#     plt.scatter(actuals1[:, i], preds1[:, i], alpha=0.5)
#     plt.plot([actuals1[:, i].min(), actuals1[:, i].max()],
#              [actuals1[:, i].min(), actuals1[:, i].max()],
#              'r--')
#     plt.xlabel(f'Actual {target}')
#     plt.ylabel(f'Predicted {target}')
#     plt.title(f'Predicted vs Actual - {target}')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"LSTM_result_{target}")
#     plt.show()







# 7. RMSE

from sklearn.metrics import mean_squared_error
import numpy as np

for i, target in enumerate(targets):
    rmse = mean_squared_error(actuals1[:, i], preds1[:, i], squared=False)
    # print(f"Test RMSE for {target}: {rmse:.4f}")







#----------> LSTM MODEL with attention layer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import numpy as np
import pandas as pd



# 1. Define the LSTM model with Attention
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attn_linear = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # x must be padded
        attn_scores = self.attn_linear(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.dropout(context)
        return self.fc(out)

# Create dataset and loader
class LSTMDataset(Dataset):
    def __init__(self, sequence_data):
        self.data = sequence_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

task_dataset = list(zip(sequences, target_seq))
train_dataset = LSTMDataset(task_dataset)

def collate_fn(batch):
    X_list, y_list = zip(*batch)
    return list(X_list), list(y_list)

data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)









# 2. Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = LSTMAttentionModel(input_size=len(features), hidden_size=64, output_size=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model2.parameters(), lr=0.001)










# 3. Train model

losses = []  # Track average loss per epoch

epochs = 3000
for epoch in range(epochs):
    model2.train()
    total_loss = 0
    for X_list, Y_list in data_loader:
        x_batch = torch.nn.utils.rnn.pad_sequence([x.to(device) for x in X_list], batch_first=True)
        y_batch = torch.stack([y.to(device) for y in Y_list])

        pred = model2(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    losses.append(avg_loss)  # ✅ Append after epoch ends

    # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")






# 4. Save the Model

torch.save(model2.state_dict(), 'Model/fidelfolio_model_LSTM-attention.pkt')
# print("Model saved as fidelfolio_model_LSTM-attention.pkt")






# 5. Evaluate/Predict
model2.eval()
preds2 = []
actuals2 = []

with torch.no_grad():
    for X_list, Y_list in data_loader:
        x_batch = torch.nn.utils.rnn.pad_sequence([x.to(device) for x in X_list], batch_first=True)
        y_batch = torch.stack([y.to(device) for y in Y_list])

        pred2 = model2(x_batch)
        preds2.append(pred2.cpu())
        actuals2.append(y_batch.cpu())

preds2 = torch.cat(preds2).numpy()
actuals2 = torch.cat(actuals2).numpy()


# plt.figure(figsize=(10, 5))
# plt.plot(range(1, epochs + 1), losses, label='Training Loss', color='blue')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss vs Epoch")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()






# 6. Plot predicted vs actual for each target

import matplotlib.pyplot as plt

# for i, target in enumerate(target_cols):
#     plt.figure(figsize=(6, 6))
#     plt.scatter(actuals2[:, i], preds2[:, i], alpha=0.5)
#     plt.plot([actuals2[:, i].min(), actuals2[:, i].max()],
#              [actuals2[:, i].min(), actuals2[:, i].max()],
#              'r--')
#     plt.xlabel(f'Actual {target}')
#     plt.ylabel(f'Predicted {target}')
#     plt.title(f'Predicted vs Actual - {target}')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"LSTM-attention_result_{target}")
#     plt.show()






# 7. Calculate RMSE
from sklearn.metrics import mean_squared_error
import numpy as np

for i, target in enumerate(target_cols):
    rmse = mean_squared_error(actuals2[:, i], preds2[:, i], squared=False)
    # print(f"Test RMSE for {target}: {rmse:.4f}")


#----------> Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd




# 1. Define the Transformer model for time-series forecasting
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, output_size, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model))  # max seq_len = 500
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_linear(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        out = self.dropout(x[:, -1, :])  # last time step
        return self.fc_out(out)

# Prepare the data
features = [col for col in df.columns if 'Feature' in col]
targets = [' Target 1 ', ' Target 2 ', ' Target 3 ']
df_sorted = df.sort_values(by=["Company", "Year"])


# Create dataset and loader
class TimeSeriesDataset(Dataset):
    def __init__(self, sequence_data):
        self.data = sequence_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

task_dataset = list(zip(sequences, target_seq))
train_dataset = TimeSeriesDataset(task_dataset)

def collate_fn(batch):
    X_list, y_list = zip(*batch)
    return list(X_list), list(y_list)

data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)







# 2. Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model3 = TransformerModel(
    input_size=len(features),
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    output_size=3
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model3.parameters(), lr=0.001)







# 3. Train model

losses = []  # Initialize a list to store epoch-wise loss
epochs = 1200

for epoch in range(epochs):
    model3.train()
    total_loss = 0
    for X_list, Y_list in data_loader:
        x_batch = pad_sequence([x.to(device) for x in X_list], batch_first=True)
        y_batch = torch.stack([y.to(device) for y in Y_list])

        pred = model3(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    losses.append(avg_loss)  # Append average loss per epoch
    # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")






# 4. Save the Model

torch.save(model3.state_dict(), 'Model/fidelfolio_model_transformer.pkt')
# print("Model saved as fidelfolio_model_transformer.pkt")






# 5. Evaluate/Predict
model3.eval()
preds3 = []
actuals3 = []

with torch.no_grad():
    for X_list, Y_list in data_loader:
        x_batch = pad_sequence([x.to(device) for x in X_list], batch_first=True)
        y_batch = torch.stack([y.to(device) for y in Y_list])

        pred3 = model3(x_batch)
        preds3.append(pred3.cpu())
        actuals3.append(y_batch.cpu())

preds3 = torch.cat(preds3).numpy()
actuals3 = torch.cat(actuals3).numpy()

import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 5))
# plt.plot(range(1, epochs + 1), losses, label='Training Loss', color='blue')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss vs Epoch")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()







# 6. Plot predicted vs actual for each target

# for i, target in enumerate(target_cols):
#     plt.figure(figsize=(6, 6))
#     plt.scatter(actuals3[:, i], preds3[:, i], alpha=0.5)
#     plt.plot([actuals3[:, i].min(), actuals3[:, i].max()],
#              [actuals3[:, i].min(), actuals3[:, i].max()],
#              'r--')
#     plt.xlabel(f'Actual {target}')
#     plt.ylabel(f'Predicted {target}')
#     plt.title(f'Predicted vs Actual - {target}')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"Transformer_result_{target}")
#     plt.show()

from sklearn.metrics import mean_squared_error
import numpy as np






# 7. Calculate RMSE
for i, target in enumerate(target_cols):
    rmse = mean_squared_error(actuals3[:, i], preds3[:, i], squared=False)
    # print(f"Test RMSE for {target}: {rmse:.4f}")