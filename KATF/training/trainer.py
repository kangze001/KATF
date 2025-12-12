import torch
import numpy as np
import time
import os
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

# psutil is not used in the final implementation, so it is removed.

class ModelTrainer:
    """
    Handles the training, validation, logging, and checkpointing process 
    for a PyTorch model, including support for gradient accumulation.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler, device, model_name, num_classes, verbose=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_name = model_name
        self.num_classes = num_classes
        self.verbose = verbose
        self.accum_steps = 1 # Default to no accumulation
        
        # Metric Tracking
        self.train_losses = []
        self.val_losses = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.gpu_mem_usage = [] # GPU memory monitoring
        
        # Setup Log Files
        self.log_file = open(f"{model_name}_training_log.csv", "w")
        self.log_file.write("epoch,train_loss,val_loss,val_acc,val_precision,val_recall,val_f1,lr,gpu_mem\n")
        
        self.class_metric_file = open(f"{model_name}_class_metrics.csv", "w")
        self.class_metric_file.write("epoch,class,precision,recall,f1_score,support\n")
    
    def set_gradient_accumulation(self, steps):
        """Sets the number of steps for gradient accumulation."""
        self.accum_steps = steps
        if self.verbose:
            print(f"[⚙️] Gradient accumulation steps set to: {steps}")
    
    def _calculate_metrics(self, all_labels, all_preds):
        """Calculates macro-averaged metrics and the full classification report."""
        acc = (all_preds == all_labels).sum().item() / len(all_labels)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        class_report = classification_report(
            all_labels, all_preds, 
            target_names=[f"class_{i}" for i in range(self.num_classes)],
            output_dict=True,
            zero_division=0
        )
        
        return acc, precision, recall, f1, class_report
    
    def train_epoch(self, epoch, total_epochs):
        """Performs one full training epoch."""
        self.model.train()
        train_loss = 0.0
        total_loss_accum = 0.0 # Loss accumulator for print during accum steps
        
        all_labels = []
        all_preds = []
        
        torch.cuda.reset_peak_memory_stats(self.device)
        
        train_iter = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs} Training", 
                          unit="batch", dynamic_ncols=True)
        
        self.optimizer.zero_grad()
        
        for batch_idx, (inputs, labels) in enumerate(train_iter):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss = loss / self.accum_steps
            loss.backward()
            
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
            # Use original loss (before scaling) for overall epoch loss
            train_loss += loss.item() * inputs.size(0) * self.accum_steps 
            total_loss_accum += loss.item()
            
            # Optimization step after accumulation or at the end of the data
            if (batch_idx + 1) % self.accum_steps == 0 or batch_idx == len(self.train_loader)-1:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update TQDM progress bar
                avg_batch_loss = total_loss_accum / min(batch_idx % self.accum_steps + 1, self.accum_steps)
                batch_acc = (predicted == labels).sum().item() / labels.size(0)
                train_iter.set_postfix({
                    "loss": f"{avg_batch_loss:.4f}",
                    "acc": f"{batch_acc*100:.2f}%",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    "mem": f"{torch.cuda.memory_allocated(device=self.device)/(1024**3):.2f}GB"
                })
                total_loss_accum = 0.0
        
        # Calculate epoch metrics
        train_acc, train_precision, train_recall, train_f1, _ = self._calculate_metrics(
            np.array(all_labels), np.array(all_preds)
        )
        avg_train_loss = train_loss / len(self.train_loader.dataset)
        
        # Log peak memory usage
        peak_mem = torch.cuda.max_memory_allocated(device=self.device)/(1024**3)
        self.gpu_mem_usage.append(peak_mem)
        
        return avg_train_loss, train_acc, train_precision, train_recall, train_f1
    
    def validate(self, epoch, loader, desc="Validation"):
        """Performs validation/evaluation on the specified data loader."""
        self.model.eval()
        val_loss = 0.0
        
        all_labels = []
        all_preds = []
        
        val_iter = tqdm(loader, desc=f"Epoch {epoch+1} {desc}", 
                         unit="batch", dynamic_ncols=True)
        
        with torch.no_grad():
            for inputs, labels in val_iter:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
                batch_acc = (predicted == labels).sum().item() / labels.size(0)
                val_iter.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{batch_acc*100:.2f}%"
                })
        
        val_acc, val_precision, val_recall, val_f1, class_report = self._calculate_metrics(
            np.array(all_labels), np.array(all_preds)
        )
        avg_val_loss = val_loss / len(loader.dataset)
        
        self._save_class_metrics(epoch, class_report, phase=desc)
        
        return avg_val_loss, val_acc, val_precision, val_recall, val_f1, class_report
    
    def _save_class_metrics(self, epoch, class_report, phase="Validation"):
        """Writes per-class and macro-average metrics to the class metrics log file."""
        for i in range(self.num_classes):
            class_name = f'class_{i}'
            self.class_metric_file.write(
                f"{epoch},{i},"
                f"{class_report[class_name]['precision']:.4f},"
                f"{class_report[class_name]['recall']:.4f},"
                f"{class_report[class_name]['f1-score']:.4f},"
                f"{class_report[class_name]['support']}\n"
            )
        
        self.class_metric_file.write(
            f"{epoch},macro_avg,"
            f"{class_report['macro avg']['precision']:.4f},"
            f"{class_report['macro avg']['recall']:.4f},"
            f"{class_report['macro avg']['f1-score']:.4f},"
            f"{class_report['macro avg']['support']}\n"
        )
        
        self.class_metric_file.flush()
        
        if self.verbose:
            print(f"\n[📊] {phase} Class Detailed Metrics (Macro Avg):")
            for i in range(self.num_classes):
                print(f"    Class {i}: "
                      f"Precision={class_report[f'class_{i}']['precision']:.4f}, "
                      f"Recall={class_report[f'class_{i}']['recall']:.4f}, "
                      f"F1={class_report[f'class_{i}']['f1-score']:.4f}")
    
    def save_checkpoint(self, epoch, metrics, class_report):
        """Saves best and periodic checkpoints."""
        is_best_acc = metrics['acc'] > self.best_val_acc
        is_best_f1 = metrics['f1'] > self.best_val_f1
        
        if is_best_acc or is_best_f1:
            if is_best_acc: 
                self.best_val_acc = metrics['acc']
            if is_best_f1: 
                self.best_val_f1 = metrics['f1']
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_acc': metrics['acc'],
                'val_precision': metrics['precision'],
                'val_recall': metrics['recall'],
                'val_f1': metrics['f1'],
                'class_report': class_report,
            }
            
            # Save the "best" model based on either Acc or F1 improvement
            torch.save(checkpoint, f'best_{self.model_name}.pth')
            print(f"[🏆] Saving BEST model! Val Acc: {metrics['acc']:.4f}, F1 Score: {metrics['f1']:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'val_acc': metrics['acc'],
                'val_f1': metrics['f1']
            }, f"checkpoint_{self.model_name}_epoch{epoch+1}.pth")
    
    def train(self, num_epochs):
        """Main training loop."""
        if self.verbose:
            print(f"[🚀] Starting Training! {num_epochs} total epochs.")
            print(f"    - Training Samples: {len(self.train_loader.dataset)}")
            print(f"    - Validation Samples: {len(self.val_loader.dataset)}")
            print(f"    - Batch Size: {self.train_loader.batch_size}")
            print(f"    - Gradient Accumulation Steps: {self.accum_steps}")
            print(f"    - Effective Batch Size: {self.train_loader.batch_size * self.accum_steps}")
            print(f"    - Initial Learning Rate: {self.optimizer.param_groups[0]['lr']}")
            print(f"    - Device: {self.device}")
            print(f"    - Total Iterations: {len(self.train_loader) // self.accum_steps} accumulation steps/epoch")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.train_epoch(epoch, num_epochs)
            
            # Validation phase
            val_loss, val_acc, val_precision, val_recall, val_f1, class_report = self.validate(epoch, self.val_loader, "Validation")
            
            # Learning Rate Scheduling (assuming ReduceLROnPlateau)
            self.scheduler.step(val_acc)
            
            # Track history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.accuracies.append(val_acc)
            self.precisions.append(val_precision)
            self.recalls.append(val_recall)
            self.f1_scores.append(val_f1)
            
            # Write to main log file
            current_mem = torch.cuda.memory_allocated(device=self.device)/(1024**3)
            log_entry = f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{val_acc:.4f},{val_precision:.4f},{val_recall:.4f},{val_f1:.4f},{self.optimizer.param_groups[0]['lr']:.6f},{current_mem:.2f}\n"
            self.log_file.write(log_entry)
            self.log_file.flush()
            
            # Save checkpoints
            metrics = {
                'acc': val_acc,
                'precision': val_precision,
                'recall': val_recall,
                'f1': val_f1
            }
            self.save_checkpoint(epoch, metrics, class_report)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\n[📝] Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"    Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc*100:.2f}%")
            print(f"    Macro P/R/F1: P={val_precision:.4f}, R={val_recall:.4f}, F1={val_f1:.4f}")
            print(f"    Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"    Time taken: {epoch_time:.2f}s | Cumulative: {time.time()-start_time:.2f}s")
            print(f"    GPU Memory Peak: {max(self.gpu_mem_usage):.2f}GB | Current: {current_mem:.2f}GB")
            print("=" * 60)
        
        # Finalization
        total_time = time.time() - start_time
        print(f"[🎉] Training complete! Total Time: {total_time:.2f}s")
        print(f"[🏁] Best Validation Accuracy: {self.best_val_acc*100:.2f}%")
        print(f"[🏁] Best F1 Score: {self.best_val_f1:.4f}")
        
        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'metrics': {
                'acc': self.accuracies[-1],
                'precision': self.precisions[-1],
                'recall': self.recalls[-1],
                'f1': self.f1_scores[-1]
            }
        }, f'final_{self.model_name}.pth')
        
        self.log_file.close()
        self.class_metric_file.close()
        
        return self.best_val_acc, self.best_val_f1