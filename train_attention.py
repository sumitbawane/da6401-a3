import torch
import torch.nn as nn
import wandb
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Import the model and preprocessor
from PreProcess import PreProcess
from model_attention import (
    create_attention_model_from_preprocessor, 
    create_dataloaders, 
    evaluate_attention_model,
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_with_wandb(config=None):
    """Training function with wandb integration for attention model"""
    # Initialize a new wandb run
    with wandb.init(config=config):
        
        config = wandb.config
        
        # Load data using the preprocessor
        data_path = 'dakshina_dataset_v1.0'
        language_code = config.language
        output_path = ''
        
        # Initialize preprocessor
        preprocesor = PreProcess(data_path, output_path, language_code)
        preprocesor.load_data()
        preprocesor.build_vocab()
        train_data, test_data, val_data = preprocesor.get_tensors()
        
       
        model = create_attention_model_from_preprocessor(
            preprocesor, 
            embedding_dim=config.embedding_size, 
            hidden_dim=config.hidden_size,
            enc_layers=config.encoder_layers, 
            dec_layers=config.decoder_layers, 
            cell_type=config.cell_type,
            dropout=config.dropout_rate, 
            device=device
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_data, val_data, test_data, batch_size=config.batch_size
        )
        
        # Define loss function - ignore padding tokens
        criterion = nn.CrossEntropyLoss(ignore_index=preprocesor.source_char_to_idx['<PAD>'])
        
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',        
            factor=0.5,          
            patience=2,          
            verbose=True,         
            min_lr=1e-6           
        )
        
        # Main training loop
        best_val_accuracy = 0
        for epoch in range(config.epochs):
            # Training phase
            model.train()
            train_loss = 0
            
            # Progress bar for training
            train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs}")
            
            for batch_idx, batch in train_bar:
                src = batch[0].to(device)
                trg = batch[1].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = model(src, trg, teacher_forcing_ratio=config.teacher_forcing_ratio, 
                               current_epoch=epoch, total_epochs=config.epochs)
                
                # Reshape for loss calculation
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)
                
                # Calculate loss
                loss = criterion(output, trg)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                
                train_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            char_acc, word_acc, _ = evaluate_attention_model(model, val_loader, preprocesor, beam_size=config.beam_size)
            scheduler.step(word_acc)
            
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_accuracy": word_acc,
                "val_char_accuracy": char_acc
            })
            
            print(f'Epoch {epoch+1}/{config.epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Word Acc: {word_acc:.3f}')
            print(f'  Val Char Acc: {char_acc:.3f}')
            
            # Save best model based on validation accuracy
            if word_acc > best_val_accuracy:
                best_val_accuracy = word_acc
                model_path = f"best_attention_model_{wandb.run.id}.pt"
                torch.save(model.state_dict(), model_path)
                # log as a WandB artifact
                artifact = wandb.Artifact(name="best-attention-model", type="model", metadata={'val_accuracy': word_acc})
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
                
def main():
    parser = argparse.ArgumentParser(description='Attention Transliteration Model Training with wandb')
    parser.add_argument('--sweep', action='store_true', help='Run as a wandb sweep')
    parser.add_argument('--count', type=int, default=10, help='Number of sweep runs')
    parser.add_argument('--output_dir', type=str, default='predictions_attention', help='Directory to save model outputs')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define sweep configuration for attention model
    sweep_config = {
        'method': 'bayes',  
        'metric': {
            'name': 'val_accuracy',  
            'goal': 'maximize'
        },
        'parameters': {
            'embedding_size': {
                'values': [64, 128, 256]
            },
            'encoder_layers': {
                'values': [1, 2,3]  
            },
            'decoder_layers': {
                'values': [1, 2,3]
            },
            'hidden_size': {
                'values': [128, 256, 512,1024]
            },
            'cell_type': {
                'values': ['LSTM', 'GRU']  
            },
            'dropout_rate': {
                'values': [0.1, 0.2, 0.3]
            },
            'beam_size': {
                'values': [1, 3, 5]
            },
            'batch_size': {
                'values': [32, 64, 128]
            },
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 0.0001,
                'max': 0.01
            },
            'teacher_forcing_ratio': {
                'values': [0.5, 0.7]
            },
            'epochs': {
                'values': [10, 15,20]
            },
            'language': {
                'value': 'hi'  
            }
        }
    }


    if args.sweep:
        # Initialize sweep
        sweep_id = wandb.sweep(sweep_config, project="da6401-a3-attention")
        # Start sweep agent
        wandb.agent(sweep_id, train_with_wandb, count=args.count)
    else:
       
        default_config = {
            'embedding_size': 128,
            'encoder_layers': 1,
            'decoder_layers': 1,
            'hidden_size': 256,
            'cell_type': 'LSTM',
            'dropout_rate': 0.2,
            'beam_size': 3,
            'batch_size': 64,
            'learning_rate': 0.001,
            'teacher_forcing_ratio': 0.5,
            'epochs': 15,
            'language': 'hi'
        }
        
        train_with_wandb(config=default_config)

if __name__ == "__main__":
    main()