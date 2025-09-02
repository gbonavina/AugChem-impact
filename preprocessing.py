import re
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class BasicSmilesTokenizer(object):
    def __init__(self):
        self.regex_pattern = r"""(\[[^\]]+]|Br?|Cl?|Nb?|In?|Sb?|As|Ai|Ta|Ga|O|P|F|H|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text):
        tokens = [token for token in self.regex.findall(text)]
        return tokens
  
class PreProcessing():
    def __init__(self, vocab=None):
        self.vocab = vocab if vocab is not None else [
            '#', '(', ')', '-', '/', '1', '2', '3', '4', '5', '6', '=', 'Br', 'C', 'Cl',
            'F', 'I', 'In', 'N', 'O', 'P', 'S', '[17O]', '[AlH-]', '[AlH2-]', '[AlH3-]',
            '[AsH3-]', '[BH-]', '[BH2-]', '[BH3-]', '[C-]', '[C@@H]', '[C@@]', '[C@H]',
            '[C@]', '[CH-]', '[CH2-]', '[CH2]', '[CH]', '[C]', '[GaH-]', '[GaH2-]', '[GaH3-]',
            '[InH-]', '[InH2-]', '[InH3-]', '[N+]', '[N-]', '[N@+]', '[N@@+]', '[N@@]', '[N@]',
            '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[NH]', '[N]', '[NbH3-]', '[O+]', '[O-]', '[O]',
            '[PH+]', '[PH3-]', '[PH4-]', '[S+]', '[SbH3-]', '[Si]', '[TaH3-]', '[c-]', '[cH-]',
            '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '\\', 'c', 'n', 'o', '<UNK>', '[M]', ''
        ]

        self.unk_idx = self.vocab.index('<UNK>')
        self.padding_idx = len(self.vocab) - 1

        # * [M] is the token for masking

    def one_hot_encode(self, smiles_list):
        token_to_index = {token: i for i, token in enumerate(self.vocab)}
        one_hot_data = []

        TOKENIZER = BasicSmilesTokenizer()

        for smile in smiles_list:
            tokens = TOKENIZER.tokenize(smile)
            one_hot_seq = []

            for token in tokens:
                vec = np.zeros(len(self.vocab), dtype=np.float32)
                idx = token_to_index.get(token, self.unk_idx)  # Usa unk_idx se não encontrar
                vec[idx] = 1.0
                one_hot_seq.append(vec)

            one_hot_seq = np.array(one_hot_seq, dtype=np.float32)
            one_hot_data.append(one_hot_seq)

        return np.array(one_hot_data, dtype=object)

    def one_hot_to_indices(self, one_hot_array):
        return np.argmax(one_hot_array, axis=1)

    def index_sequences(self, one_hot_array):
        return [self.one_hot_to_indices(seq) for seq in one_hot_array]

    def collate_fn(self, batch):
        # print(batch)
        sequences = [seq for seq, _ in batch]
        labels = [label for _, label in batch]
        lengths = [len(seq) for seq in sequences]

        sequences_tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]

        padded = pad_sequence(sequences_tensors, batch_first=True, padding_value=self.padding_idx)

        return padded, torch.tensor(lengths), torch.tensor(labels, dtype=torch.float32)

    def prepare_data_from_df(self, df, smiles_col='SMILES_1', target_col='Property_0'):
        smiles_list = df[smiles_col].tolist()
        props = df[target_col].tolist()
    
        one_hot_encoded = self.one_hot_encode(smiles_list)
        indexed_sequences = self.index_sequences(one_hot_encoded)
    
        X_train, X_temp, y_train, y_temp = train_test_split(
            indexed_sequences, props, test_size=0.5, random_state=24
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
    
        # Filtro após split
        X_train_filtered = [xi for xi, yi in zip(X_train, y_train) if yi <= 20]
        y_train_filtered = [yi for xi, yi in zip(X_train, y_train) if yi <= 20]
        X_val_filtered = [xi for xi, yi in zip(X_val, y_val) if yi <= 20]
        y_val_filtered = [yi for xi, yi in zip(X_val, y_val) if yi <= 20]
        X_test_filtered = [xi for xi, yi in zip(X_test, y_test) if yi <= 20]
        y_test_filtered = [yi for xi, yi in zip(X_test, y_test) if yi <= 20]
    
        scaler = MinMaxScaler()
        y_train_scaled = scaler.fit_transform(np.array(y_train_filtered).reshape(-1, 1)).flatten()
        y_val_scaled = scaler.transform(np.array(y_val_filtered).reshape(-1, 1)).flatten()
        y_test_scaled = scaler.transform(np.array(y_test_filtered).reshape(-1, 1)).flatten()
    
        train_data = list(zip(X_train_filtered, y_train_scaled))
        val_data = list(zip(X_val_filtered, y_val_scaled))
        test_data = list(zip(X_test_filtered, y_test_scaled))
    
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=self.collate_fn)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=self.collate_fn)
    
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'scaler': scaler,
            'vocab_size': len(self.vocab),
            'padding_idx': self.padding_idx,
            'data_stats': {
                'total_samples': len(smiles_list),
                'train_samples': len(X_train_filtered),
                'val_samples': len(X_val_filtered),
                'test_samples': len(X_test_filtered)
            }
        }