import torch
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PreProcess import *
from model_attention import *

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_saved_model():
    """Evaluate the attention model and save heatmaps as image files"""
    #  Load preprocessor
    data_path = 'dakshina_dataset_v1.0'
    language_code = 'hi'  
    output_path = ''
    
    preprocessor = PreProcess(data_path, output_path, language_code)
    preprocessor.load_data()
    preprocessor.build_vocab()
    train_data, test_data, val_data = preprocessor.get_tensors()
    
    #  Create model with best hyperparameters from sweep
    best_params = {
        'embedding_dim': 128,
        'hidden_dim': 1024,
        'enc_layers': 3,
        'dec_layers': 3,
        'cell_type': 'LSTM',
        'dropout': 0.1,
        'beam_size': 3,
        'batch_size': 64,
        'learning_rate': 0.004459158365147817,
        'teacher_forcing_ratio': 0.5,
        'epochs': 15,
        'language_code': 'hi'  
    }
    
    model = create_attention_model_from_preprocessor(
        preprocessor,
        embedding_dim=best_params['embedding_dim'],
        hidden_dim=best_params['hidden_dim'],
        enc_layers=best_params['enc_layers'],
        dec_layers=best_params['dec_layers'],
        cell_type=best_params['cell_type'],
        dropout=best_params['dropout'],
        device=device
    )
    
   #Create test dataloader
    _, _, test_loader = create_dataloaders(
        train_data, val_data, test_data, batch_size=best_params['batch_size']
    )
    
    #  Load saved model weights
    model.load_state_dict(torch.load('best_attention_model.pt'))
    
    # Evaluate on test set
    print("Evaluating attention model on test set...")
    char_accuracy, word_accuracy, _ = evaluate_attention_model(
        model, test_loader, preprocessor, beam_size=best_params['beam_size']
    )
    print(f"Test Character Accuracy: {char_accuracy:.4f}")
    print(f"Test Word Accuracy: {word_accuracy:.4f}")
    
    #  Generate attention heatmaps for 10 examples
    os.makedirs('heatmaps', exist_ok=True)
    attention_samples = get_samples_with_attention(model, test_loader, preprocessor, num_samples=10)
    create_attention_heatmaps(attention_samples)
    
    # Create a single 3x3 grid with 9 samples for the report
    create_grid_heatmap(attention_samples[:9])
    
    # Also create a clean version with position indices instead of characters
    create_index_heatmaps(attention_samples)
    
    return model, char_accuracy, word_accuracy

def get_samples_with_attention(model, data_loader, preprocessor, num_samples=10):
    """Get samples with their attention weights for visualization"""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for batch in data_loader:
            if len(samples) >= num_samples:
                break
                
            src = batch[0].to(model.device)
            trg = batch[1]
            
            # Use greedy decode for attention visualization
            predictions = model.greedy_decode(src, max_len=trg.shape[1])
            
            # Get attention weights from model
            attention_weights = model.attention_weights_history
            
            # Process each sequence in the batch
            for i in range(src.size(0)):
                if len(samples) >= num_samples:
                    break
                    
                source_seq = src[i].cpu().numpy()
                target_seq = trg[i].cpu().numpy()
                pred_seq = predictions[i].cpu().numpy()
                
                # Skip padded sequences or very short sequences
                if source_seq[1] == preprocessor.source_char_to_idx[preprocessor.special_tokens["PAD"]] or len(source_seq) < 3:
                    continue
                
                # Convert indices to characters
                source_text = preprocessor.idx_to_sequence(source_seq, preprocessor.source_idx_to_char)
                target_text = preprocessor.idx_to_sequence(target_seq, preprocessor.target_idx_to_char)
                pred_text = preprocessor.idx_to_sequence(pred_seq, preprocessor.target_idx_to_char)
                
                # Clean special tokens
                def clean_sequence(text):
                    cleaned = text.replace('<SOS>', '').replace('<EOS>', '').replace('<PAD>', '').replace('<UNK>', '').strip()
                    return cleaned
                
                source_clean = clean_sequence(source_text)
                target_clean = clean_sequence(target_text)
                pred_clean = clean_sequence(pred_text)
                
                # Get attention weights for this sample
                sample_attn = None
                if isinstance(attention_weights, torch.Tensor) and i < attention_weights.size(0):
                    sample_attn = attention_weights[i].cpu().numpy()
                elif isinstance(attention_weights, list) and i < len(attention_weights):
                    if isinstance(attention_weights[i], list):
                        # Handle list of lists
                        sample_attn = [a.cpu().numpy() if isinstance(a, torch.Tensor) else a 
                                      for a in attention_weights[i]]
                    else:
                        # Handle list of tensors
                        sample_attn = [attn_step[i].cpu().numpy() if i < attn_step.size(0) else None 
                                      for attn_step in attention_weights if isinstance(attn_step, torch.Tensor)]
                
                if sample_attn is not None:
                    samples.append({
                        'source': source_clean,
                        'target': target_clean,
                        'prediction': pred_clean,
                        'correct': target_clean == pred_clean,
                        'attention': sample_attn,
                        'source_tokens': [c for c in source_clean],
                        'pred_tokens': [c for c in pred_clean]
                    })
    
    return samples

def create_attention_heatmaps(samples):
    """Create individual attention heatmaps for each sample"""
    if not samples:
        print("No samples with attention weights available.")
        return
    
    # Save individual heatmaps
    for i, sample in enumerate(samples):
        plt.figure(figsize=(10, 8))
        
        # Get attention weights
        attn = sample['attention']
        
        # Process attention weights based on format
        if isinstance(attn, np.ndarray):
            attention_matrix = attn
        elif isinstance(attn, list):
            valid_attn = [a for a in attn if a is not None]
            if valid_attn:
                attention_matrix = np.stack(valid_attn)
            else:
                continue
        else:
            continue
        
        # Get source and predicted tokens
        src_tokens = sample['source_tokens']
        pred_tokens = sample['pred_tokens']
        
        # Resize attention matrix to match tokens
        attn_rows = min(len(pred_tokens), attention_matrix.shape[0])
        attn_cols = min(len(src_tokens), attention_matrix.shape[1])
        
        # Extract relevant part of attention matrix
        if len(attention_matrix.shape) == 3:
            attention_plot = attention_matrix[:attn_rows, 0, :attn_cols]
        elif len(attention_matrix.shape) == 2:
            attention_plot = attention_matrix[:attn_rows, :attn_cols]
        else:
            continue
        
        # Create heatmap with actual characters
        sns.heatmap(attention_plot, cmap='viridis', 
                   xticklabels=src_tokens[:attn_cols],
                   yticklabels=pred_tokens[:attn_rows])
        
        # Set title and labels
        correct_mark = "✓" if sample['correct'] else "✗"
        plt.title(f"Source: {sample['source']} → Pred: {sample['prediction']} {correct_mark}")
        plt.xlabel('Source Characters')
        plt.ylabel('Predicted Characters')
        
        # Save individual heatmap
        plt.tight_layout()
        plt.savefig(f'heatmaps/sample_{i+1}.png', dpi=200)
        plt.close()
    
    print(f"Saved {len(samples)} individual attention heatmaps to heatmaps/")

def create_grid_heatmap(samples):
    """Create a 3x3 grid of attention heatmaps"""
    if len(samples) < 9:
        print(f"Warning: Requested 9 samples for grid but only got {len(samples)}")
        return
    
    # Set up a 3x3 grid
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    
    # Flatten the axes for easy indexing
    axs_flat = axs.flatten()
    
    # Create heatmaps
    for i, sample in enumerate(samples[:9]):
        ax = axs_flat[i]
        
        # Get attention weights
        attn = sample['attention']
        
        # Process attention weights based on format
        if isinstance(attn, np.ndarray):
            attention_matrix = attn
        elif isinstance(attn, list):
            valid_attn = [a for a in attn if a is not None]
            if valid_attn:
                attention_matrix = np.stack(valid_attn)
            else:
                continue
        else:
            continue
        
        # Get source and predicted tokens
        src_tokens = sample['source_tokens']
        pred_tokens = sample['pred_tokens']
        
        # Resize attention matrix to match tokens
        attn_rows = min(len(pred_tokens), attention_matrix.shape[0])
        attn_cols = min(len(src_tokens), attention_matrix.shape[1])
        
        # Extract relevant part of attention matrix
        if len(attention_matrix.shape) == 3:
            attention_plot = attention_matrix[:attn_rows, 0, :attn_cols]
        elif len(attention_matrix.shape) == 2:
            attention_plot = attention_matrix[:attn_rows, :attn_cols]
        else:
            continue
        
        # Create heatmap with actual characters
        sns.heatmap(attention_plot, ax=ax, cmap='viridis', 
                   xticklabels=src_tokens[:attn_cols],
                   yticklabels=pred_tokens[:attn_rows])
        
        # Set title
        correct_mark = "✓" if sample['correct'] else "✗"
        ax.set_title(f"{sample['source']} → {sample['prediction']} {correct_mark}", fontsize=10)
        
        # Make the axis labels a bit smaller
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('heatmaps/attention_grid_3x3.png', dpi=300)
    plt.close()
    print("Saved 3x3 grid of attention heatmaps to heatmaps/attention_grid_3x3.png")

def create_index_heatmaps(samples):
    """Create heatmaps with position indices instead of characters"""
    # Create a 2x5 grid for 10 samples
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    axs_flat = axs.flatten()
    
    # Process each sample
    for i, sample in enumerate(samples[:10]):
        if i >= len(axs_flat):
            break
            
        ax = axs_flat[i]
        attn = sample['attention']
        
        # Process attention weights
        if isinstance(attn, np.ndarray):
            attention_matrix = attn
        elif isinstance(attn, list):
            valid_attn = [a for a in attn if a is not None]
            if valid_attn:
                attention_matrix = np.stack(valid_attn)
            else:
                continue
        else:
            continue
        
        # Get tokens
        src_tokens = sample['source_tokens']
        pred_tokens = sample['pred_tokens']
        
        # Resize matrix
        attn_rows = min(len(pred_tokens), attention_matrix.shape[0])
        attn_cols = min(len(src_tokens), attention_matrix.shape[1])
        
        # Extract relevant part
        if len(attention_matrix.shape) == 3:
            attention_plot = attention_matrix[:attn_rows, 0, :attn_cols]
        elif len(attention_matrix.shape) == 2:
            attention_plot = attention_matrix[:attn_rows, :attn_cols]
        else:
            continue
        
        # Use position indices instead of actual characters
        x_labels = [f"{idx}({c})" for idx, c in enumerate(src_tokens[:attn_cols])]
        y_labels = [f"{idx}({c})" for idx, c in enumerate(pred_tokens[:attn_rows])]
        
        # Create heatmap
        sns.heatmap(attention_plot, ax=ax, cmap='viridis', 
                   xticklabels=x_labels,
                   yticklabels=y_labels)
        
        # Set title with Latin characters only
        correct_mark = "✓" if sample['correct'] else "✗"
        ax.set_title(f"Source: {sample['source']} {correct_mark}", fontsize=10)
        
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('heatmaps/position_index_grid.png', dpi=300)
    plt.close()
    
    # Also create individual position index heatmaps
    for i, sample in enumerate(samples[:10]):
        plt.figure(figsize=(10, 8))
        
        # Process attention weights
        attn = sample['attention']
        if isinstance(attn, np.ndarray):
            attention_matrix = attn
        elif isinstance(attn, list):
            valid_attn = [a for a in attn if a is not None]
            if valid_attn:
                attention_matrix = np.stack(valid_attn)
            else:
                continue
        else:
            continue
        
        # Get tokens
        src_tokens = sample['source_tokens']
        pred_tokens = sample['pred_tokens']
        
        # Resize matrix
        attn_rows = min(len(pred_tokens), attention_matrix.shape[0])
        attn_cols = min(len(src_tokens), attention_matrix.shape[1])
        
        # Extract relevant part
        if len(attention_matrix.shape) == 3:
            attention_plot = attention_matrix[:attn_rows, 0, :attn_cols]
        elif len(attention_matrix.shape) == 2:
            attention_plot = attention_matrix[:attn_rows, :attn_cols]
        else:
            continue
        
        # Use position indices
        x_labels = [f"{idx}" for idx in range(attn_cols)]
        y_labels = [f"{idx}" for idx in range(attn_rows)]
        
        # Create heatmap with numbers
        sns.heatmap(attention_plot, cmap='viridis', 
                   xticklabels=x_labels,
                   yticklabels=y_labels,
                   annot=True, fmt=".2f")
        
        # Add source and prediction info as text below the heatmap
        plt.figtext(0.5, 0.01, 
                   f"Source: {sample['source']}\nTarget: {sample['target']}\nPrediction: {sample['prediction']}", 
                   ha="center", fontsize=10,
                   bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        # Set title and labels
        correct_mark = "✓" if sample['correct'] else "✗"
        plt.title(f"Attention Heatmap {i+1} {correct_mark}")
        plt.xlabel('Source Character Position')
        plt.ylabel('Predicted Character Position')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  
        plt.savefig(f'heatmaps/position_index_{i+1}.png', dpi=200)
        plt.close()
    
    print("Saved position index heatmaps to heatmaps/")

# Run the evaluation and generate heatmaps
if __name__ == "__main__":
    model, char_accuracy, word_accuracy = evaluate_saved_model()