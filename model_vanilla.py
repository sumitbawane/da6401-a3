import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, cell_type, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=0)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dropout = nn.Dropout(dropout)
        self.cell_type = cell_type
        
        if cell_type.upper() == "LSTM":
            self.cell = nn.LSTM(
                embedding_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True
            )
        elif cell_type.upper() == "GRU":
            self.cell = nn.GRU(
                embedding_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True
            )
        else:
            self.cell = nn.RNN(
                embedding_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True
            )

    def forward(self, src):
        embedded = self.embedding_dropout(self.embedding(src))
        outputs, hidden = self.cell(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, cell_type, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=0)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dropout = nn.Dropout(dropout)
        self.cell_type = cell_type
        
        if cell_type.upper() == "LSTM":
            self.cell = nn.LSTM(
                embedding_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True
            )
        elif cell_type.upper() == "GRU":
            self.cell = nn.GRU(
                embedding_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True
            )
        else:
            self.cell = nn.RNN(
                embedding_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True
            )

        self.output_dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, encoder_outputs=None):
        if input.dim() == 1:
            input = input.unsqueeze(1)
        embedded = self.embedding_dropout(self.embedding(input))
        output, hidden = self.cell(embedded, hidden)
        output = self.output_dropout(output.squeeze(1))
        prediction = self.fc_out(output)
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, preprocesor):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.preprocesor = preprocesor

        self.pad_idx = self.preprocesor.source_char_to_idx[self.preprocesor.special_tokens["PAD"]]
        self.unk_idx = self.preprocesor.source_char_to_idx[self.preprocesor.special_tokens["UNK"]]
        self.eos_idx = self.preprocesor.source_char_to_idx[self.preprocesor.special_tokens["EOS"]]
        self.sos_idx = self.preprocesor.source_char_to_idx[self.preprocesor.special_tokens["SOS"]]
        
        if(self.encoder.n_layers != self.decoder.n_layers or self.encoder.hidden_dim != self.decoder.hidden_dim):
            self.layer_projection = self.create_projection(self.encoder, self.decoder)
        else:
            self.layer_projection = None

    def create_projection(self, encoder, decoder):
        if encoder.cell_type.upper() == "LSTM":
            h_projection = (
                nn.Linear(encoder.hidden_dim, decoder.hidden_dim)
                if encoder.hidden_dim != decoder.hidden_dim
                else nn.Identity()
            )
            c_projection = (
                nn.Linear(encoder.hidden_dim, decoder.hidden_dim)
                if encoder.hidden_dim != decoder.hidden_dim
                else nn.Identity()
            )
            return nn.ModuleDict(
                {"h_projection": h_projection, "c_projection": c_projection}
            )
        else:
            return (
                nn.Linear(encoder.hidden_dim, decoder.hidden_dim)
                if encoder.hidden_dim != decoder.hidden_dim
                else nn.Identity()
            )

    def _adapt_hidden_state(self, hidden):
        if self.encoder.cell_type.upper() == "LSTM":
            h, c = hidden

            if self.encoder.n_layers != self.decoder.n_layers:
                if self.encoder.n_layers > self.decoder.n_layers:
                    h = h[-self.decoder.n_layers :]
                    c = c[-self.decoder.n_layers :]
                else:
                    layers_needed = self.decoder.n_layers - self.encoder.n_layers
                    h = torch.cat([h, h[-1:].repeat(layers_needed, 1, 1)], dim=0)
                    c = torch.cat([c, c[-1:].repeat(layers_needed, 1, 1)], dim=0)

            if self.layer_projection is not None:
                h = self.layer_projection["h_projection"](h)
                c = self.layer_projection["c_projection"](c)

            return (h, c)
        else:
            if self.encoder.n_layers != self.decoder.n_layers:
                if self.encoder.n_layers > self.decoder.n_layers:
                    hidden = hidden[-self.decoder.n_layers :]
                else:
                    layers_needed = self.decoder.n_layers - self.encoder.n_layers
                    hidden = torch.cat(
                        [hidden, hidden[-1:].repeat(layers_needed, 1, 1)], dim=0
                    )

            if self.layer_projection is not None:
                hidden = self.layer_projection(hidden)
            return hidden

    def forward(self, src, trg, teacher_forcing_ratio=0.5,current_epoch=None,total_epochs=None):
        batch_size = src.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        hidden = self._adapt_hidden_state(hidden)

        input = trg[:, 0]
        if current_epoch is not None and total_epochs is not None:
            # Exponential decay: faster at the beginning, slower toward the end
            decay_factor = np.exp(-5 * current_epoch / total_epochs)
            actual_teacher_forcing_ratio = teacher_forcing_ratio * decay_factor
        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t, :] = output

            teacher_force = torch.rand(1) < actual_teacher_forcing_ratio

            top1 = output.argmax(1)

            input = trg[:, t] if teacher_force else top1
        return outputs
    
    def greedy_decode(self, src, max_len=50):
        self.eval()
        
        with torch.no_grad():
            batch_size = src.shape[0]
            
            encoder_outputs, hidden = self.encoder(src)
            hidden = self._adapt_hidden_state(hidden)
            
            input = torch.tensor([self.sos_idx] * batch_size).to(self.device)
            
            predictions = []
            
            for step in range(max_len):
                output, hidden = self.decoder(input, hidden, encoder_outputs)
                
                predicted_char = output.argmax(1)
                predictions.append(predicted_char)
                
                input = predicted_char
                
                if (predicted_char == self.eos_idx).all():
                    break
            
            if predictions:
                return torch.stack(predictions, dim=1)
            else:
                return torch.zeros(batch_size, 0, dtype=torch.long)
    def beam_decode(self, src, beam_size=3, max_len=50):
        """Generate predictions using beam search"""
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # Encode source sequence
            encoder_outputs, hidden = self.encoder(src)
            hidden = self._adapt_hidden_state(hidden)
            
            # Initialize beams for each sequence in batch
            all_predictions = []
            
            for batch_idx in range(batch_size):
                # Get encoder hidden state for this sequence
                if self.encoder.cell_type.upper() == 'LSTM':
                    h_0, c_0 = hidden
                    single_hidden = (h_0[:, batch_idx:batch_idx+1, :].contiguous(),
                                    c_0[:, batch_idx:batch_idx+1, :].contiguous())
                else:
                    single_hidden = hidden[:, batch_idx:batch_idx+1, :].contiguous()
                
                # Initialize beam: (score, sequence, hidden_state)
                beams = [(0.0, [self.sos_idx], single_hidden)]
                completed_sequences = []
                
                for step in range(max_len):
                    candidates = []
                    
                    for score, sequence, hidden_state in beams:
                        # Skip if sequence ended with EOS
                        if sequence[-1] == self.eos_idx:
                            completed_sequences.append((score, sequence))
                            continue
                            
                        # Get last token
                        last_token = torch.tensor([sequence[-1]]).unsqueeze(0).to(self.device)
                        
                        # Forward through decoder (we pass encoder_outputs but don't use them yet)
                        output, new_hidden = self.decoder(last_token, hidden_state, encoder_outputs[batch_idx:batch_idx+1])
                        output_probs = F.log_softmax(output, dim=-1)
                        
                        # Get top k candidates
                        top_probs, top_indices = output_probs.topk(beam_size, dim=-1)
                        
                        for i in range(beam_size):
                            new_score = score + top_probs[0, i].item()
                            new_sequence = sequence + [top_indices[0, i].item()]
                            candidates.append((new_score, new_sequence, new_hidden))
                    
                    # Select top beam_size candidates
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    beams = candidates[:beam_size]
                    
                    # Early stopping if all beams ended
                    if all(seq[-1] == self.eos_idx for _, seq, _ in beams):
                        for score, seq, _ in beams:
                            completed_sequences.append((score, seq))
                        break
                
                # Add remaining sequences from beams
                for score, seq, _ in beams:
                    if seq not in [comp_seq for _, comp_seq in completed_sequences]:
                        completed_sequences.append((score, seq))
                
                # Select best sequence (highest score)
                if completed_sequences:
                    best_sequence = max(completed_sequences, key=lambda x: x[0])[1]
                else:
                    best_sequence = beams[0][1] if beams else [self.sos_idx, self.eos_idx]
                
                all_predictions.append(best_sequence)
            
            # Convert to tensor format with padding
            max_len_found = max(len(pred) for pred in all_predictions)
            tensor_predictions = torch.zeros(len(all_predictions), max_len_found, dtype=torch.long)
            
            for i, pred in enumerate(all_predictions):
                tensor_predictions[i, :len(pred)] = torch.tensor(pred)
                if len(pred) < max_len_found:
                    tensor_predictions[i, len(pred):] = self.pad_idx
                    
            return tensor_predictions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_model_from_preprocessor(preprocesor, embedding_dim=64, hidden_dim=128, 
                                 enc_layers=1, dec_layers=1, cell_type='LSTM', 
                                 dropout=0.1, device='cuda'):
    src_vocab_size, trg_vocab_size = preprocesor.get_vocab_size()
    
    encoder = Encoder(src_vocab_size, embedding_dim, hidden_dim, enc_layers, cell_type, dropout)
    decoder = Decoder(trg_vocab_size, embedding_dim, hidden_dim, dec_layers, cell_type, dropout)
    
    model = Seq2Seq(encoder, decoder, device, preprocesor).to(device)
    
    return model

def create_dataloaders(train_data, val_data, test_data, batch_size=32):
    train_dataset = TensorDataset(train_data['source'], train_data['target'])
    val_dataset = TensorDataset(val_data['source'], val_data['target'])
    test_dataset = TensorDataset(test_data['source'], test_data['target'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def evaluate_model(model, data_loader, preprocesor, beam_size=1):
    """Evaluate model on a dataset using specified beam size"""
    model.eval()
    correct_chars = 0
    total_chars = 0
    correct_words = 0
    total_words = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader,desc="Evaluating",leave=False):
            src = batch[0].to(model.device)
            trg = batch[1]
            
            # Use beam search with the specified beam size
            if beam_size > 1:
                predictions = model.beam_decode(src, beam_size=beam_size, max_len=trg.shape[1])
            else:
                predictions = model.greedy_decode(src, max_len=trg.shape[1])
            
            for i in range(src.size(0)):
                target_seq = trg[i].cpu().numpy()
                pred_seq = predictions[i].cpu().numpy()
                
                target_text = preprocesor.idx_to_sequence(target_seq, preprocesor.target_idx_to_char)
                pred_text = preprocesor.idx_to_sequence(pred_seq, preprocesor.target_idx_to_char)
                
                def clean_sequence(text):
                    cleaned = text.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', '').replace('<UNK>', '').strip()
                    return cleaned
                
                target_clean = clean_sequence(target_text)
                pred_clean = clean_sequence(pred_text)
                
                # Use the softer character accuracy metric
                correct_chars += sum(1 for a, b in zip(target_clean, pred_clean) if a == b)
                total_chars += max(len(target_clean), len(pred_clean))
                
                if target_clean == pred_clean:
                    correct_words += 1
                total_words += 1
    
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    word_accuracy = correct_words / total_words if total_words > 0 else 0
    
    return char_accuracy, word_accuracy