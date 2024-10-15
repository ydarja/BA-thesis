import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import pandas as pd
import time
from data_manager import load_data_from_csv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

class SynsetPairDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        encoded1 = self.tokenizer(item['sentence1'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        encoded2 = self.tokenizer(item['sentence2'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids1': encoded1['input_ids'].squeeze(),
            'attention_mask1': encoded1['attention_mask'].squeeze(),
            'input_ids2': encoded2['input_ids'].squeeze(),
            'attention_mask2': encoded2['attention_mask'].squeeze(),
            'specific': torch.tensor(item['specific'], dtype=torch.float)
        }

# define the model
class SpecificityModel(nn.Module):
    def __init__(self):
        super(SpecificityModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dense1 = nn.Linear(768, 128)
        self.dense2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        outputs1 = self.bert(input_ids1, attention_mask=attention_mask1).last_hidden_state.mean(dim=1)
        outputs2 = self.bert(input_ids2, attention_mask=attention_mask2).last_hidden_state.mean(dim=1)
        
        x1 = self.dense1(outputs1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        logits1 = self.dense2(x1)
        
        x2 = self.dense1(outputs2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        logits2 = self.dense2(x2)
        
        return logits1, logits2

# function to train the model
def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids1 = batch['input_ids1'].to(device)
        attention_mask1 = batch['attention_mask1'].to(device)
        input_ids2 = batch['input_ids2'].to(device)
        attention_mask2 = batch['attention_mask2'].to(device)
        labels = batch['specific'].to(device)

        score1, score2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
        predictions = torch.sigmoid(score2 - score1).squeeze()
        loss = criterion(predictions, labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# function to evaluate the model
def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            labels = batch['specific'].to(device)

            score1, score2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            predictions = torch.sigmoid(score2 - score1).squeeze()
            loss = criterion(predictions, labels.float())

            total_loss += loss.item()

    return total_loss / len(val_loader)

# function to test the model
def test_model(model, test_loader, device):
    model.eval()
    num_correct = 0
    num_total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            labels = batch['specific'].to(device).float()

            score1, score2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            outputs_diff = torch.sigmoid(score2 - score1).squeeze()
            predictions = (outputs_diff > 0.5).float()  # Threshold to convert to binary predictions

            num_correct += (predictions == labels).sum().item()
            num_total += len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    test_accuracy = num_correct / num_total
    test_precision = precision_score(all_labels, all_predictions, zero_division=0)
    test_recall = recall_score(all_labels, all_predictions, zero_division=0)
    test_f1 = f1_score(all_labels, all_predictions, zero_division=0)
    test_confusion_matrix = confusion_matrix(all_labels, all_predictions)

    print("Test Accuracy: {}".format(round(test_accuracy, 6)))
    print("Test Precision: {}".format(round(test_precision, 6)))
    print("Test Recall: {}".format(round(test_recall, 6)))
    print("Test F1-Score: {}".format(round(test_f1, 6)))
    print("Test Confusion Matrix:\n {}".format(test_confusion_matrix))

def save_model(model, optimizer, epoch, path):
    """
    Save the PyTorch model and optimizer state.

    Parameters:
    - model: the model to be saved
    - optimizer: the optimizer whose state will be saved
    - epoch: the epoch number to save
    - path: the path where the model will be saved
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"Model saved to {path}")

def load_model(model, optimizer, path, device):
    """
    Load the PyTorch model and optimizer state.

    Parameters:
    - model: the model to be loaded
    - optimizer: the optimizer whose state will be loaded
    - path: the path from where the model will be loaded
    - device: the device to map the model to (e.g., 'cpu' or 'cuda')
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # ignore unexpected keys
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {path}, epoch {epoch}")
    return model, optimizer, epoch

if __name__ == "__main__":
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SpecificityModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    
    # load data and create data loaders
    file_path = 'data/balanced_raw_synsets.csv'
    train_data, val_data, test_data = load_data_from_csv(file_path)

    train_dataset = SynsetPairDataset(train_data, tokenizer)
    val_dataset = SynsetPairDataset(val_data, tokenizer)
    test_dataset = SynsetPairDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    test_loader = DataLoader(test_dataset, batch_size=4)

    train_losses = []
    val_losses = []

    # train the model
    num_epochs = 3
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print("Epoch ", epoch + 1, "/", num_epochs, " , Train Loss: ", train_loss, " Validation Loss: ", val_loss)

        # save the model after each epoch
        save_model(model, optimizer, epoch+1, 'models/model6')
    
    model = SpecificityModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # load and test the saved model
    model, optimizer, epoch = load_model(model, optimizer, 'models/model6', device)
    test_model(model, test_loader, device)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/model6_loss.png', dpi=300)
    plt.close()