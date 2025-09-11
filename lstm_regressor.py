import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_squared_error

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, padding_idx, dropout=0.3):
      super().__init__()
      self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)
      self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout)
      self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
      embedded = self.embedding(x)
      # print(">>> embedded.shape:", embedded.shape)

      lengths_on_cpu = lengths.to('cpu')
      packed_input = pack_padded_sequence(embedded, lengths_on_cpu, batch_first=True, enforce_sorted=False)
      packed_output, (hn, cn) = self.lstm(packed_input)

      last_hidden = hn[-1]
      return self.fc(last_hidden).squeeze(1)
    
class LSTMTrainer:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model is running on: {torch.cuda.is_available() and 'GPU' or 'CPU'}")
    
    def train_model(self, train_loader, val_loader, num_epochs, criterion, optimizer):
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            total_loss = 0.0
            for batch_x, lengths, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                lengths = lengths.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                preds = self.model(batch_x, lengths)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            total_loss += loss.item()
            train_losses.append(total_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, lengths, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    lengths = lengths.to(self.device)
                    batch_y = batch_y.to(self.device)
                    preds = self.model(batch_x, lengths)
                    loss = criterion(preds, batch_y)
                    val_loss += loss.item()
            
            val_loss += loss.item()
            val_losses.append(val_loss)

            if epoch % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {total_loss:.4f} — Val Loss: {val_loss:.4f}")

        return train_losses, val_losses
    
    def evaluate_model(self, test_dataloader, criterion, scaler):
        self.model.eval()
        predictions_test = []
        targets_test = []
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, lengths, batch_y in test_dataloader:
                batch_x = batch_x.to(self.device)
                lengths = lengths.to(self.device)
                batch_y = batch_y.to(self.device)

                preds = self.model(batch_x, lengths)
                loss = criterion(preds, batch_y)
                total_loss += loss.item()
                
                predictions_test.append(preds.cpu())
                targets_test.append(batch_y.cpu())

        predictions_test = torch.cat(predictions_test, dim=0)
        targets_test = torch.cat(targets_test, dim=0)
        avg_loss = total_loss / len(test_dataloader)

        y_true = scaler.inverse_transform(targets_test.numpy().reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(predictions_test.numpy().reshape(-1, 1)).flatten()

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        return {
            'loss': avg_loss,
            'r2': r2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'y_true': y_true,
            'y_pred': y_pred
        }