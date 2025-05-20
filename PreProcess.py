import pandas as pd
import numpy as np
import torch
import os

class PreProcess:
    def __init__(self, data_path, output_path, language_code):
        # Initialize paths and parameters
        self.data_path = data_path
        self.output_path = output_path
        self.language_code = language_code
        # Initialize dataframes
        self.train_data_df = None
        self.test_data_df = None
        self.val_data_df = None
        # Define special tokens for sequence modeling
        self.special_tokens = {
            'PAD': '<PAD>',  # For padding sequences
            'SOS': '< SOS >', # Start of sequence
            'EOS': '<EOS>',  # End of sequence
            'UNK': '<UNK>'   # Unknown characters
        }
        
        # Dictionaries for vocabulary mapping
        self.source_char_to_idx = {}
        self.source_idx_to_char = {}
        self.target_char_to_idx = {}
        self.target_idx_to_char = {}
        
    def load_data(self):
        # Get path to lexicon files
        lex_path = os.path.join(self.data_path, self.language_code, 'lexicons')
        # Common parameters for reading the TSV files
        common = dict(
            sep='\t',
            names=['target', 'source', 'frequency'],
            encoding='utf-8',
            dtype={'source': str, 'target': str, 'frequency': int},
            header=None,
        )
        # Read data files - source is Latin script, target is native script
        self.train_data_df = pd.read_csv(os.path.join(lex_path, f'{self.language_code}.translit.sampled.train.tsv'), **common)
        self.test_data_df = pd.read_csv(os.path.join(lex_path, f'{self.language_code}.translit.sampled.test.tsv'), **common)
        self.val_data_df = pd.read_csv(os.path.join(lex_path, f'{self.language_code}.translit.sampled.dev.tsv'), **common)
        print(f"Train data shape: {self.train_data_df.shape}")
        print(f"Test data shape: {self.test_data_df.shape}")
        print(f"Validation data shape: {self.val_data_df.shape}")
    
    def build_vocab(self):
        # Get lists of words from training data
        source_words = self.train_data_df['source'].dropna().to_list()
        target_words = self.train_data_df['target'].dropna().to_list()
        frequency = self.train_data_df['frequency'].dropna().to_list()
        # Create vocabulary mappings
        self.source_char_to_idx, self.source_idx_to_char = self.create_char_vocab(source_words)
        self.target_char_to_idx, self.target_idx_to_char = self.create_char_vocab(target_words)
    
    def create_char_vocab(self, words):
        # Get all characters from words
        all_chars = [char for word in words for char in word]
        # Add special tokens and sort regular characters
        vocab_chars = list(self.special_tokens.values()) + sorted(set(all_chars))
        # Create mappings between chars and indices
        char_to_idx = {char: idx for idx, char in enumerate(vocab_chars)}
        idx_to_char = {idx: char for idx, char in enumerate(vocab_chars)}
        print(f"Vocabulary size: {len(vocab_chars)}")
        return char_to_idx, idx_to_char
    
    def sequence_to_idx(self, sequence, char_to_idx):
        # Convert string to index sequence with SOS and EOS tokens
        idx_sequence = []
        idx_sequence.append(char_to_idx[self.special_tokens['SOS']])
        for char in sequence:
            if char not in char_to_idx:
               idx_sequence.append(char_to_idx[self.special_tokens['UNK']])
            else:
                idx_sequence.append(char_to_idx[char])
        idx_sequence.append(char_to_idx[self.special_tokens['EOS']])
        return idx_sequence
    
    def idx_to_sequence(self, idx_sequence, idx_to_char):
        # Convert index sequence back to string
        sequence = []
        for idx in idx_sequence:
            if idx in idx_to_char:
                sequence.append(idx_to_char[idx])
            else:
                sequence.append(self.special_tokens['UNK'])
        return ''.join(sequence)
    
    def print_vocab(self):
        # Print vocabulary mappings for debugging
        print("Source Vocabulary:")
        for char, idx in self.source_char_to_idx.items():
            print(f"{char}: {idx}")
        print("\nTarget Vocabulary:")
        for char, idx in self.target_char_to_idx.items():
            print(f"{char}: {idx}")
            
    def encode(self, df):
        # Convert dataframe to padded index sequences
        src_idx_seq = []
        tgt_idx_seq = []
        src_max_len = 0
        tgt_max_len = 0
        df = df.dropna(subset=['source', 'target']).copy()
        # Convert each sequence to indices
        src_idx_seq = df['source'].map(lambda x: self.sequence_to_idx(x, self.source_char_to_idx)).tolist()
        tgt_idx_seq = df['target'].map(lambda x: self.sequence_to_idx(x, self.target_char_to_idx)).tolist()
        # Find max lengths for padding
        src_max_len = max(len(s) for s in src_idx_seq)
        tgt_max_len = max(len(t) for t in tgt_idx_seq)
        # Get padding indices
        pad_src_idx = self.source_char_to_idx[self.special_tokens['PAD']]
        pad_tgt_idx = self.target_char_to_idx[self.special_tokens['PAD']]
        # Pad sequences
        pad_src_idx_seq = self.pad_sequence(src_idx_seq, src_max_len, pad_src_idx)
        pad_tgt_idx_seq = self.pad_sequence(tgt_idx_seq, tgt_max_len, pad_tgt_idx)
        return pad_src_idx_seq, pad_tgt_idx_seq
    
    def pad_sequence(self, sequence, max_len, pad_idx):
        # Add padding to make sequences same length
        return np.array([seq + [pad_idx] * (max_len - len(seq)) for seq in sequence], dtype=np.int64)
    
    def get_tensors(self):
        # Create PyTorch tensors for all data splits
        x_train, y_train = self.encode(self.train_data_df)
        x_test, y_test = self.encode(self.test_data_df)
        x_val, y_val = self.encode(self.val_data_df)
        
        # Convert to PyTorch tensors
        train_data = {
            'source': torch.tensor(x_train, dtype=torch.long),
            'target': torch.tensor(y_train, dtype=torch.long)
        }
        test_data = {
            'source': torch.tensor(x_test, dtype=torch.long),
            'target': torch.tensor(y_test, dtype=torch.long)
        }
        val_data = {
            'source': torch.tensor(x_val, dtype=torch.long),
            'target': torch.tensor(y_val, dtype=torch.long)
        }
        return train_data, test_data, val_data
    
    def get_vocab_size(self):
        # Return vocabulary sizes
        return len(self.source_char_to_idx), len(self.target_char_to_idx)

if __name__ == '__main__':
    # Example usage
    data_path = 'dakshina_dataset_v1.0'
    language_code = 'hi'  # Hindi
    output_path = '' 
    
    # Process the data
    data_loader = PreProcess(data_path, output_path, language_code)
    data_loader.load_data()
    data_loader.build_vocab()
    train_data, test_data, val_data = data_loader.get_tensors()
    
    # Print tensor shapes
    print("Train data tensors:")   
    print(train_data['source'].shape, train_data['target'].shape)
    print("Test data tensors:")
    print(test_data['source'].shape, test_data['target'].shape)
    print("Validation data tensors:")
    print(val_data['source'].shape, val_data['target'].shape)