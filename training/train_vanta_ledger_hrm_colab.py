#!/usr/bin/env python3
"""
Train HRM on Vanta Ledger Data - GOOGLE COLAB VERSION
Specialized training script for financial document understanding and company reasoning
Optimized for Colab environment with GPU acceleration
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Iterator, Optional
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Setup logging for Colab
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

class VantaLedgerDataset(Dataset):
    """Custom dataset for Vanta Ledger HRM training - COLAB VERSION"""
    
    def __init__(self, data_path: str, max_seq_len: int = 128):
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        
        # Load the dataset
        data = np.load(self.data_path)
        self.inputs = torch.from_numpy(data['inputs']).long()
        self.targets = torch.from_numpy(data['targets']).long()
        self.task_types = data['task_types']
        
        logger.info(f"Loaded Vanta Ledger dataset: {len(self.inputs)} samples")
        logger.info(f"Input shape: {self.inputs.shape}")
        logger.info(f"Target shape: {self.targets.shape}")
        logger.info(f"Task types: {np.unique(self.task_types)}")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        # Get input and target sequences
        input_seq = self.inputs[idx]
        target_seq = self.targets[idx]
        
        # Find actual sequence lengths (before padding)
        input_len = (input_seq != 0).sum().item()
        target_len = (target_seq != 0).sum().item()
        
        # Add proper sequence length validation
        if input_len == 0 or target_len == 0:
            # Handle empty sequences by using a default pattern
            padded_input = torch.zeros(self.max_seq_len, dtype=torch.long)
            padded_target = torch.zeros(self.max_seq_len, dtype=torch.long)
            padded_input[0] = 1  # Start token
            padded_target[0] = 1  # Start token
            actual_input_len = actual_target_len = 1
        else:
            # Create new tensors for the max sequence length
            padded_input = torch.zeros(self.max_seq_len, dtype=torch.long)
            padded_target = torch.zeros(self.max_seq_len, dtype=torch.long)
            
            # Copy the actual data (truncating if necessary)
            actual_input_len = min(input_len, self.max_seq_len)
            actual_target_len = min(target_len, self.max_seq_len)
            
            padded_input[:actual_input_len] = input_seq[:actual_input_len]
            padded_target[:actual_target_len] = target_seq[:actual_target_len]
        
        # Ensure masks are properly aligned
        input_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        input_mask[:actual_input_len] = True
        
        target_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        target_mask[:actual_target_len] = True
        
        return {
            'inputs': padded_input,
            'targets': padded_target,
            'input_mask': input_mask,
            'target_mask': target_mask,
            'input_len': actual_input_len,
            'target_len': actual_target_len,
            'task_type': self.task_types[idx]
        }

class VantaLedgerHRMTrainer:
    """Specialized trainer for HRM on Vanta Ledger data - COLAB VERSION"""
    
    def __init__(self, dataset_path: str, model_config: Optional[Dict[str, Any]] = None):
        self.dataset_path = Path(dataset_path)
        
        # Default HRM config optimized for financial reasoning and stability
        self.config = HierarchicalReasoningModel_ACTV1Config(
            vocab_size=10000,
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            intermediate_size=2048,
            max_position_embeddings=512,
            layer_norm_eps=1e-12,
            dropout=0.1,
            attention_dropout=0.1,
            initializer_range=0.02,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            num_tasks=5,
            task_embedding_size=64,
            hierarchical_depth=3,
            reasoning_steps=4,
            use_act=True,
            act_epsilon=0.01,
            act_alpha=0.01
        )
        
        # Override with custom config if provided
        if model_config:
            for key, value in model_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        logger.info(f"HRM Config: {self.config}")
        
        # Create model
        self.model = HierarchicalReasoningModel_ACTV1(self.config)
        self.model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Create dataset and dataloaders
        self.dataset = VantaLedgerDataset(str(self.dataset_path))
        
        # Split dataset (80% train, 20% val)
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=4,  # Increased batch size for GPU
            shuffle=True,
            num_workers=0,  # Colab doesn't like multiple workers
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        
        logger.info("Vanta Ledger HRM Trainer initialized successfully!")
    
    def validate(self, model, val_loader, loss_fn):
        """Validate the model"""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)
                input_mask = batch['input_mask'].to(device)
                target_mask = batch['target_mask'].to(device)
                task_types = batch['task_type']
                
                # Forward pass
                outputs = model(
                    input_ids=inputs,
                    attention_mask=input_mask,
                    task_types=task_types
                )
                
                # Calculate loss
                loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train(self, epochs: int = 25, save_path: str = "vanta_ledger_hrm_colab", batch_size: int = 4):
        """Train the model"""
        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        
        # Create save directory
        save_path_obj = Path(save_path)
        save_path_obj.mkdir(exist_ok=True)
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        no_improve_count = 0
        patience = 5
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            self.model.train()
            epoch_loss = 0
            num_batches = 0
            total_grad_norm = 0
            
            # Training phase
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch_idx, batch in enumerate(train_pbar):
                try:
                    inputs = batch['inputs'].to(device)
                    targets = batch['targets'].to(device)
                    input_mask = batch['input_mask'].to(device)
                    target_mask = batch['target_mask'].to(device)
                    task_types = batch['task_type']
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=inputs,
                        attention_mask=input_mask,
                        task_types=task_types
                    )
                    
                    # Calculate loss
                    loss = self.loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    total_grad_norm += grad_norm.item()
                    
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # Update progress bar
                    train_pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'grad_norm': f'{grad_norm:.4f}'
                    })
                    
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
            
            # Calculate average training loss
            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
            avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss = self.validate(self.model, self.val_loader, self.loss_fn)
            val_losses.append(val_loss)
            
            epoch_time = time.time() - start_time
            
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            logger.info(f"  Training Loss: {avg_train_loss:.4f}")
            logger.info(f"  Validation Loss: {val_loss:.4f}")
            logger.info(f"  Average Gradient Norm: {avg_grad_norm:.4f}")
            logger.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Plot training progress
            self.plot_training_progress(train_losses, val_losses, save_path_obj)
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                # Save best model
                model_path = save_path_obj / "best_vanta_ledger_hrm.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'config': self.config.model_dump(),
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }, model_path)
                logger.info(f"New best model saved: {model_path}")
            else:
                no_improve_count += 1
            
            # Early stopping
            if no_improve_count >= patience:
                logger.info(f"Early stopping after {patience} epochs without improvement")
                break
            
            # Learning rate scheduling
            if hasattr(self.scheduler, 'step'):
                self.scheduler.step(val_loss)
            
            # Save checkpoint
            checkpoint_path = save_path_obj / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'config': self.config.model_dump(),
                'train_losses': train_losses,
                'val_losses': val_losses
            }, checkpoint_path)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Final training loss: {train_losses[-1]:.4f}")
        logger.info(f"Models saved to: {save_path}")
        
        return train_losses, val_losses
    
    def plot_training_progress(self, train_losses, val_losses, save_path):
        """Plot training progress and save to file"""
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        # Loss difference plot
        plt.subplot(1, 2, 2)
        loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
        plt.plot(loss_diff, label='|Train - Val|', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.title('Overfitting Monitor')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path / 'training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main function to train HRM on Vanta Ledger data in Colab"""
    
    # Check if dataset exists
    dataset_path = "vanta_ledger_hrm_dataset.npz"
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Please upload the dataset file to Colab first")
        return
    
    # Create trainer
    trainer = VantaLedgerHRMTrainer(dataset_path)
    
    # Start training
    train_losses, val_losses = trainer.train(epochs=25, batch_size=4)
    
    logger.info("Vanta Ledger HRM training completed successfully in Colab!")

if __name__ == "__main__":
    main()
