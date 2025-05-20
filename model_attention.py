import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attn = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch_size, decoder_dim]
        # encoder_outputs: [batch_size, src_len, encoder_dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Reshape hidden to match encoder_outputs for concatenation
        if hidden.dim() == 2:  # [batch_size, decoder_dim]
            hidden = hidden.unsqueeze(1).expand(-1, src_len, -1)  # [batch_size, src_len, decoder_dim]
        elif hidden.dim() == 3 and hidden.shape[1] == 1:  # [batch_size, 1, decoder_dim]
            hidden = hidden.expand(-1, src_len, -1)  # [batch_size, src_len, decoder_dim]
        
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Get attention scores
        attention = self.v(energy).squeeze(2)
        
      
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        # Get attention weights
        attention_weights = F.softmax(attention, dim=1)
        
        # Apply attention weights to encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights

class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, n_layers, cell_type, dropout, attention):
        super(AttentionDecoder, self).__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=0)
        self.n_layers = n_layers
        self.hidden_dim = decoder_hidden_dim
        self.embedding_dropout = nn.Dropout(dropout)
        self.cell_type = cell_type
        
        if cell_type.upper() == "LSTM":
            self.cell = nn.LSTM(
                embedding_dim + encoder_hidden_dim, decoder_hidden_dim, n_layers, 
                dropout=dropout if n_layers > 1 else 0, batch_first=True
            )
        elif cell_type.upper() == "GRU":
            self.cell = nn.GRU(
                embedding_dim + encoder_hidden_dim, decoder_hidden_dim, n_layers, 
                dropout=dropout if n_layers > 1 else 0, batch_first=True
            )
        else:
            self.cell = nn.RNN(
                embedding_dim + encoder_hidden_dim, decoder_hidden_dim, n_layers, 
                dropout=dropout if n_layers > 1 else 0, batch_first=True
            )

        self.output_dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(decoder_hidden_dim + encoder_hidden_dim + embedding_dim, output_dim)

    def forward(self, input, hidden, encoder_outputs, mask=None):
        if input.dim() == 1:
            input = input.unsqueeze(1)  # [batch_size, 1]
            
        embedded = self.embedding_dropout(self.embedding(input))  # [batch_size, 1, embedding_dim]
        
        # Get the decoder hidden state
        if self.cell_type.upper() == "LSTM":
            h, c = hidden
            decoder_hidden = h[-1]  # last layer's hidden state
            # Remove singleton dimensions if present
            if decoder_hidden.dim() == 3 and decoder_hidden.size(0) == 1:
                decoder_hidden = decoder_hidden.squeeze(0)
        else:
            decoder_hidden = hidden[-1]  # [batch_size, decoder_hidden_dim]
            # Remove singleton dimensions if present
            if decoder_hidden.dim() == 3 and decoder_hidden.size(0) == 1:
                decoder_hidden = decoder_hidden.squeeze(0)
        
       
        batch_size = encoder_outputs.shape[0]
        if decoder_hidden.shape[0] != batch_size:
            if decoder_hidden.dim() == 2 and decoder_hidden.shape[0] == 1:
                decoder_hidden = decoder_hidden.expand(batch_size, -1)
        
        # Attention mechanism
        context, attn_weights = self.attention(decoder_hidden, encoder_outputs, mask)
        
        
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        
        # Feed to RNN
        output, hidden = self.cell(rnn_input, hidden)
        
        # Combine for prediction
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        
        return prediction, hidden, attn_weights

class Seq2SeqAttention(nn.Module):
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
        
        # Storage for attention weights
        self.attention_weights_history = []

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
    
    def create_mask(self, src):
        # Create mask for source sequence (1 for real tokens, 0 for padding)
        mask = (src != self.pad_idx).to(self.device)
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=0.5, current_epoch=None, total_epochs=None):
        batch_size = src.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)
        
        # Create mask for source sequence
        mask = self.create_mask(src)
        
        # Reset attention weights history
        self.attention_weights_history = []

        encoder_outputs, hidden = self.encoder(src)
        hidden = self._adapt_hidden_state(hidden)
        input = trg[:, 0]
        
        # Dynamic teacher forcing ratio
        if current_epoch is not None and total_epochs is not None:
            actual_teacher_forcing_ratio = max(0.5, teacher_forcing_ratio * (1 - 0.8 * current_epoch / total_epochs))
        else:
            actual_teacher_forcing_ratio = teacher_forcing_ratio

        # Store batch attention weights
        batch_attention_weights = []
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(input, hidden, encoder_outputs, mask)
            # Store weights without detaching during training to avoid memory issues
            if not self.training:  # Only store during evaluation
                batch_attention_weights.append(attn_weights.cpu())
            
            outputs[:, t, :] = output
            teacher_force = torch.rand(1) < actual_teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        
        if not self.training and batch_attention_weights:
            self.attention_weights_history = torch.stack(batch_attention_weights, dim=1)
            
        return outputs
    
    def greedy_decode(self, src, max_len=50):
        self.eval()
        
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # Create mask for source sequence
            mask = self.create_mask(src)
            
            encoder_outputs, hidden = self.encoder(src)
            hidden = self._adapt_hidden_state(hidden)
            
            input = torch.tensor([self.sos_idx] * batch_size).to(self.device)
            
            predictions = []
            
            # Store attention weights for this batch
            batch_attention_weights = []
            
            for step in range(max_len):
                output, hidden, attn_weights = self.decoder(input.unsqueeze(1), hidden, encoder_outputs, mask)
                
                # Store attention weights
                batch_attention_weights.append(attn_weights.detach().cpu())
                
                predicted_char = output.argmax(1)
                predictions.append(predicted_char)
                
                input = predicted_char
                
                if (predicted_char == self.eos_idx).all():
                    break
            
            # Store the attention weights
            if batch_attention_weights:
                self.attention_weights_history = torch.stack(batch_attention_weights, dim=1)
            
            if predictions:
                return torch.stack(predictions, dim=1)
            else:
                return torch.zeros(batch_size, 0, dtype=torch.long)
    
    def beam_decode(self, src, beam_size=3, max_len=50):
        """Generate predictions using beam search with attention"""
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # Create mask for source sequence
            mask = self.create_mask(src)
            
            # Encode source sequence
            encoder_outputs, hidden = self.encoder(src)
            hidden = self._adapt_hidden_state(hidden)
            
            # Initialize beams for each sequence in batch
            all_predictions = []
            all_attention_weights = []
            
            for batch_idx in range(batch_size):
                # Get encoder hidden state for this sequence
                if self.encoder.cell_type.upper() == 'LSTM':
                    h_0, c_0 = hidden
                    single_hidden = (h_0[:, batch_idx:batch_idx+1, :].contiguous(),
                                     c_0[:, batch_idx:batch_idx+1, :].contiguous())
                else:
                    single_hidden = hidden[:, batch_idx:batch_idx+1, :].contiguous()
                
                # Get encoder outputs for this sequence
                single_encoder_outputs = encoder_outputs[batch_idx:batch_idx+1]
                single_mask = mask[batch_idx:batch_idx+1]
                
                # Initialize beam: (score, sequence, hidden_state, attention_history)
                beams = [(0.0, [self.sos_idx], single_hidden, [])]
                completed_sequences = []
                
                for step in range(max_len):
                    candidates = []
                    
                    for score, sequence, hidden_state, attn_history in beams:
                        # Skip if sequence ended with EOS
                        if sequence[-1] == self.eos_idx:
                            completed_sequences.append((score, sequence, attn_history))
                            continue
                            
                        # Get last token
                        last_token = torch.tensor([sequence[-1]]).unsqueeze(0).to(self.device)
                        
                        # Forward through decoder with attention
                        output, new_hidden, attn_weights = self.decoder(
                            last_token, hidden_state, single_encoder_outputs, single_mask
                        )
                        output_probs = F.log_softmax(output, dim=-1)
                        
                        # Get top k candidates
                        top_probs, top_indices = output_probs.topk(beam_size, dim=-1)
                        
                        for i in range(beam_size):
                            new_score = score + top_probs[0, i].item()
                            new_sequence = sequence + [top_indices[0, i].item()]
                            new_attn_history = attn_history + [attn_weights.cpu()]
                            candidates.append((new_score, new_sequence, new_hidden, new_attn_history))
                    
                    # Select top beam_size candidates
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    beams = candidates[:beam_size]
                    
                    # Early stopping if all beams ended
                    if all(seq[-1] == self.eos_idx for _, seq, _, _ in beams):
                        break
                
                # Add remaining sequences from beams
                for score, seq, _, attn_hist in beams:
                    if not any(seq == comp_seq for _, comp_seq, _ in completed_sequences):
                        completed_sequences.append((score, seq, attn_hist))
                
                # Select best sequence (highest score)
                if completed_sequences:
                    best_seq_info = max(completed_sequences, key=lambda x: x[0])
                    best_sequence = best_seq_info[1]
                    best_attn_history = best_seq_info[2]
                else:
                    best_sequence = beams[0][1] if beams else [self.sos_idx, self.eos_idx]
                    best_attn_history = beams[0][3] if beams else []
                
                all_predictions.append(best_sequence)
                all_attention_weights.append(best_attn_history)
            
            # Convert to tensor format with padding
            max_len_found = max(len(pred) for pred in all_predictions)
            tensor_predictions = torch.zeros(len(all_predictions), max_len_found, dtype=torch.long)
            
            for i, pred in enumerate(all_predictions):
                tensor_predictions[i, :len(pred)] = torch.tensor(pred)
                if len(pred) < max_len_found:
                    tensor_predictions[i, len(pred):] = self.pad_idx
            
            # Store the attention weights
            self.attention_weights_history = all_attention_weights
                    
            return tensor_predictions

def create_attention_model_from_preprocessor(preprocesor, embedding_dim=64, hidden_dim=128, 
                                           enc_layers=1, dec_layers=1, cell_type='LSTM', 
                                           dropout=0.1, device='cuda'):
    src_vocab_size, trg_vocab_size = preprocesor.get_vocab_size()
    
    # Create encoder
    encoder = Encoder(src_vocab_size, embedding_dim, hidden_dim, enc_layers, cell_type, dropout)
    
    # Create attention module
    attention = Attention(hidden_dim, hidden_dim)
    
    # Create attention decoder
    decoder = AttentionDecoder(
        trg_vocab_size, embedding_dim, hidden_dim, hidden_dim, 
        dec_layers, cell_type, dropout, attention
    )
    
    # Create the full seq2seq model with attention
    model = Seq2SeqAttention(encoder, decoder, device, preprocesor).to(device)
    
    return model

def evaluate_attention_model(model, data_loader, preprocesor, beam_size=1, num_samples=0):
    """
    Evaluate attention model on a dataset
    Returns:
    - char_accuracy: Character-level accuracy
    - word_accuracy: Word-level accuracy
    - sample_predictions: List of sample predictions for analysis
    """
    model.eval()
    
    # Force beam_size to 1 for stability during development
    beam_size = 1  # Use greedy search for consistency
    
    correct_chars = 0
    total_chars = 0
    correct_words = 0
    total_words = 0
    
    sample_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            src = batch[0].to(model.device)
            trg = batch[1]
            
            # Use beam search with the specified beam size
            if beam_size > 1:
                predictions = model.beam_decode(src, beam_size=beam_size, max_len=trg.shape[1])
            else:
                predictions = model.greedy_decode(src, max_len=trg.shape[1])
            
            for i in range(src.size(0)):
                source_seq = src[i].cpu().numpy()
                target_seq = trg[i].cpu().numpy()
                pred_seq = predictions[i].cpu().numpy()
                
                source_text = preprocesor.idx_to_sequence(source_seq, preprocesor.source_idx_to_char)
                target_text = preprocesor.idx_to_sequence(target_seq, preprocesor.target_idx_to_char)
                pred_text = preprocesor.idx_to_sequence(pred_seq, preprocesor.target_idx_to_char)
                
                def clean_sequence(text):
                    cleaned = text.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', '').replace('<UNK>', '').strip()
                    return cleaned
                
                source_clean = clean_sequence(source_text)
                target_clean = clean_sequence(target_text)
                pred_clean = clean_sequence(pred_text)
                
                # Calculate character-level accuracy
                correct_chars += sum(1 for a, b in zip(target_clean, pred_clean) if a == b)
                total_chars += max(len(target_clean), len(pred_clean))
                
                # Calculate word-level accuracy
                if target_clean == pred_clean:
                    correct_words += 1
                total_words += 1
                
                # Collect samples for analysis
                if num_samples > 0 and len(sample_predictions) < num_samples:
                    # Get attention weights if available
                    attention_weights = None
                    if hasattr(model, 'attention_weights_history'):
                        if isinstance(model.attention_weights_history, list):
                            if i < len(model.attention_weights_history):
                                attention_weights = model.attention_weights_history[i]
                        elif isinstance(model.attention_weights_history, torch.Tensor):
                            if i < model.attention_weights_history.size(0):
                                attention_weights = model.attention_weights_history[i]
                    
                    sample_predictions.append({
                        'source': source_clean,
                        'target': target_clean,
                        'prediction': pred_clean,
                        'is_correct': target_clean == pred_clean,
                        'attention_weights': attention_weights
                    })
    
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    word_accuracy = correct_words / total_words if total_words > 0 else 0
    
    return char_accuracy, word_accuracy, sample_predictions

def create_dataloaders(train_data, val_data, test_data, batch_size=32):
    """Create DataLoader objects for training, validation and test data"""
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(train_data['source'], train_data['target'])
    val_dataset = TensorDataset(val_data['source'], val_data['target'])
    test_dataset = TensorDataset(test_data['source'], test_data['target'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

