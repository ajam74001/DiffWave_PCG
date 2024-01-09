import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import scipy.io
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
import wandb

wandb.init(project='RNN-Diffwave-Murmur', name='DiffWave-RNN_Classifier_balanced-Murmur')

# Data real vs Data fake classifier - an offself one 
root_dir = '/home/ainazj1/psanjay_ada/users/ainazj1/Datasets/physionet.org/files/circor-heart-sound/1.0.3'
# real = pd.read_csv(root_dir+'/real_diffwave_final.csv').drop('Unnamed: 0', axis=1).dropna()
# fake = pd.read_csv(root_dir+'/fake_diffwave_final.csv').drop('Unnamed: 0', axis=1).dropna() 

real = pd.read_csv(root_dir+'/real_Abnormal_diffwave_final.csv').drop('Unnamed: 0', axis=1).dropna() 
fake = pd.read_csv(root_dir+'/fake_Abnormal_diffwave_final.csv').drop('Unnamed: 0', axis=1).dropna() 

print(real.shape, real.isna().sum().sum())
print(fake.shape, fake.isna().sum().sum())

# to balance out the number of real and fake labels 
real = real.sample(n=len(fake), random_state=1) # TODO : genrate more smaples as much ad the train data 

# Assign labels
real['label'] = 1
fake['label'] = 0

data = pd.concat([real, fake], axis = 0) # concating row wise

#Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Split the data into features and labels
X = data.drop('label', axis=1).values
y = data['label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for the training data
batch_size = 64  # You can change the batch size according to your needs
X_train = TensorDataset(X_train_tensor, y_train_tensor)
X_train_loader = DataLoader(dataset=X_train, batch_size=batch_size, shuffle=True)

# print(X_train.size)
# print(X_train.shape)
# Create DataLoader for the testing data
X_test = TensorDataset(X_test_tensor, y_test_tensor)
X_test_loader = DataLoader(dataset=X_test, batch_size=batch_size, shuffle=False)

# Define the RNN model
class BinaryClassificationRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryClassificationRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # RNN layer
        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected layer
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)
        
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out


# Hyperparameters
input_size = 1  # Number of features
hidden_size = 20  # Number of features in hidden state
output_size = 1  # Binary classification (0 or 1)
learning_rate = 0.01
num_epochs = 100

# Initialize the model
model = BinaryClassificationRNN(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # This loss combines a Sigmoid layer and BCELoss in one single class
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(X_train_loader):
        # (batch_size, sequence_length, feature_size)
        # print(inputs.shape)
        if torch.isnan(inputs).any() or torch.isnan(labels).any():
            print("NaN detected in inputs or labels")
            continue
        inputs = inputs.unsqueeze(-1)
        labels = labels.view(-1, 1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        probabilities = torch.sigmoid(outputs).data
        predicted = (probabilities >= 0.5).float()
        total += labels.size(0)
        correct += (predicted.squeeze() == labels.squeeze()).sum().item()
        # print('total', total, 'correct', correct)
        # print(predicted.shape, labels.shape)
    # print(predicted.squeeze() == labels.squeeze())

    # Calculate training accuracy
    train_accuracy = 100 * correct / total
    wandb.log({"epoch": epoch, "loss": total_loss / len(X_train_loader), "train_accuracy": train_accuracy})
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(X_train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')
   
    # Validation Phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():  
        for inputs, labels in X_test_loader:
            inputs = inputs.unsqueeze(-1)
            labels = labels.view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            probabilities = torch.sigmoid(outputs).data
            # Convert probabilities to binary predictions
            val_predicted = (probabilities >= 0.5).float()
            val_total += labels.size(0)
            val_correct += (val_predicted.squeeze() == labels.squeeze()).sum().item()
    # Calculate validation accuracy
    val_accuracy = 100 * val_correct / val_total
    wandb.log({"epoch": epoch, "val_loss": val_loss / len(X_test_loader), "val_accuracy": val_accuracy})

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {total_loss / len(X_train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, '
          f'Val Loss: {val_loss / len(X_test_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%')



wandb.config = {
  "learning_rate": 0.01,
  "epochs": 100,
  "batch_size": 64
}

wandb.finish()
