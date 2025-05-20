import torch
import pandas as pd
import os
from tqdm import tqdm
from PreProcess import *
from model_vanilla import *

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_saved_model():
    #  Load preprocessor
    data_path = 'dakshina_dataset_v1.0'
    language_code = 'hi'  # Using Hindi
    output_path = ''
    
    preprocessor = PreProcess(data_path, output_path, language_code)
    preprocessor.load_data()
    preprocessor.build_vocab()
    train_data, test_data, val_data = preprocessor.get_tensors()
    
    #  Create model with best hyperparameters from sweep
    best_params = {
        'embedding_dim': 256,
        'hidden_dim': 512,
        'enc_layers': 2,
        'dec_layers': 2,
        'cell_type': 'GRU',  
        'dropout': 0.2,      
        'beam_size': 3,       
        'batch_size': 64
    }
    
    model = create_model_from_preprocessor(
        preprocessor,
        embedding_dim=best_params['embedding_dim'],
        hidden_dim=best_params['hidden_dim'],
        enc_layers=best_params['enc_layers'],
        dec_layers=best_params['dec_layers'],
        cell_type=best_params['cell_type'],
        dropout=best_params['dropout'],
        device=device
    )
    
    #  Create test dataloader
    _, _, test_loader = create_dataloaders(
        train_data, val_data, test_data, batch_size=best_params['batch_size']
    )
    
    #  Load saved model weights
    model.load_state_dict(torch.load('best_vanilla_model.pt'))
    
    #  Evaluate on test set
    print("Evaluating model on test set...")
    char_accuracy, word_accuracy = evaluate_model(model, test_loader, preprocessor, beam_size=best_params['beam_size'])
    print(f"Test Character Accuracy: {char_accuracy:.4f}")
    print(f"Test Word Accuracy: {word_accuracy:.4f}")
    
    #  Save all predictions and create sample grid
    os.makedirs('predictions_vanilla', exist_ok=True)
    all_predictions = save_all_predictions(model, test_loader, preprocessor, best_params['beam_size'])
    
    #  Create and display sample grid
    create_sample_grid(all_predictions)
    
    return model, char_accuracy, word_accuracy, all_predictions

def save_all_predictions(model, test_loader, preprocessor, beam_size):
    """Generate and save all predictions for the test set"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            src = batch[0].to(model.device)
            trg = batch[1]
            
            # Use beam search if specified
            if beam_size > 1:
                predictions = model.beam_decode(src, beam_size=beam_size, max_len=trg.shape[1])
            else:
                predictions = model.greedy_decode(src, max_len=trg.shape[1])
            
            for i in range(src.size(0)):
                source_seq = src[i].cpu().numpy()
                target_seq = trg[i].cpu().numpy()
                pred_seq = predictions[i].cpu().numpy()
                
                source_text = preprocessor.idx_to_sequence(source_seq, preprocessor.source_idx_to_char)
                target_text = preprocessor.idx_to_sequence(target_seq, preprocessor.target_idx_to_char)
                pred_text = preprocessor.idx_to_sequence(pred_seq, preprocessor.target_idx_to_char)
                
                # Clean special tokens
                def clean_sequence(text):
                    cleaned = text.replace('< SOS >', '').replace('<EOS>', '').replace('<PAD>', '').replace('<UNK>', '').strip()
                    return cleaned
                
                source_clean = clean_sequence(source_text)
                target_clean = clean_sequence(target_text)
                pred_clean = clean_sequence(pred_text)
                
                all_predictions.append({
                    'source': source_clean,
                    'target': target_clean,
                    'prediction': pred_clean,
                    'correct': target_clean == pred_clean
                })
    
    
    df = pd.DataFrame(all_predictions)
    df.to_csv('predictions_vanilla/test_predictions.csv', index=False, encoding='utf-8')
    
    return all_predictions

def create_sample_grid(predictions, rows=3, cols=3):
    """Create and display a text-based grid of sample predictions"""
   
    correct_preds = [p for p in predictions if p['correct']]
    incorrect_preds = [p for p in predictions if not p['correct']]
    
    
    samples = []
    
    # Add short, medium, and long correct predictions
    if correct_preds:
        correct_preds.sort(key=lambda x: len(x['source']))
        samples.append(correct_preds[0])  # Short
        if len(correct_preds) > len(correct_preds)//2:
            samples.append(correct_preds[len(correct_preds)//2])  # Medium
        if len(correct_preds) > len(correct_preds)*3//4:
            samples.append(correct_preds[int(len(correct_preds)*3//4)])  # Long
    
    # Add short, medium, and long incorrect predictions
    if incorrect_preds:
        incorrect_preds.sort(key=lambda x: len(x['source']))
        samples.append(incorrect_preds[0])  # Short
        if len(incorrect_preds) > len(incorrect_preds)//2:
            samples.append(incorrect_preds[len(incorrect_preds)//2])  # Medium
        if len(incorrect_preds) > len(incorrect_preds)*3//4:
            samples.append(incorrect_preds[int(len(incorrect_preds)*3//4)])  # Long
    
    # Ensure we have at least rows*cols samples
    while len(samples) < rows*cols:
        if len(correct_preds) > len([s for s in samples if s['correct']]):
            remaining_correct = [p for p in correct_preds if p not in samples]
            if remaining_correct:
                samples.append(remaining_correct[0])
        elif len(incorrect_preds) > len([s for s in samples if not s['correct']]):
            remaining_incorrect = [p for p in incorrect_preds if p not in samples]
            if remaining_incorrect:
                samples.append(remaining_incorrect[0])
        else:
           
            break
    
    # Limit to rows*cols samples
    samples = samples[:rows*cols]
    
    # Create a formatted ASCII table
    print("\nSample Predictions (3x3 Grid):")
    print("=" * 70)
    
    # Table header
    print(f"{'Source':<15} | {'Target':<20} | {'Prediction':<20} | {'Correct':<7}")
    print("-" * 70)
    
    # Print the data in rows
    for i in range(0, len(samples), cols):
        row_samples = samples[i:i+cols]
        for sample in row_samples:
            correct_mark = "✓" if sample['correct'] else "✗"
            print(f"{sample['source']:<15} | {sample['target']:<20} | {sample['prediction']:<20} | {correct_mark:<7}")
        if i + cols < len(samples):  # Don't print the line after the last row
            print("-" * 70)
    
    print("=" * 70)
    
    # Save the sample grid as a CSV file for reference
    pd.DataFrame(samples).to_csv('predictions_vanilla/sample_grid.csv', index=False, encoding='utf-8')

# Run the evaluation on the saved model
model, char_accuracy, word_accuracy, predictions = evaluate_saved_model()