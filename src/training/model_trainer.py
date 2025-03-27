import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW

# from tqdm.notebook import tqdm
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import CrossEntropyLoss
from typing import Dict, Tuple
from src.config_three_class import CONFIG_3CLASS as CONFIG
import os
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self, model, model_type, class_weights=None):
        self.model = model
        self.model_type = model_type
        self.device = CONFIG["training"]["device"]

        # Initialize class-weighted loss function
        if class_weights is not None:
            weights = torch.tensor(
                class_weights, dtype=torch.float).to(self.device)
            self.loss_fn = CrossEntropyLoss(weight=weights)
            print(f"[INFO] Using class weights: {weights.cpu().numpy()}")
        else:
            self.loss_fn = CrossEntropyLoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=CONFIG["training"]["learning_rate"],
        )
        self.best_val_f1 = 0
        self.patience_counter = 0

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }

        print(f"[INFO] Model initialized on device: {self.device}")
        print(
            f"[INFO] Optimizer: AdamW with learning rate {CONFIG['training']['learning_rate']}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        self.model.to(self.device)
        print(
            f"[INFO] Model running on: {next(self.model.parameters()).device}")

        epochs = CONFIG["training"]["epochs"]
        early_stopping_patience = CONFIG["training"]["early_stopping_patience"]
        print(f"[INFO] Training started for {epochs} epochs.")

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # epoch_progress = tqdm(range(epochs), desc="Epochs", position=0)
        epoch_progress = tqdm(
            range(epochs),
            desc="Epochs",
            dynamic_ncols=True,
            mininterval=1.0  # throttle updates to prevent flooding
        )

        for epoch in epoch_progress:
            train_loss = self._train_epoch(train_loader, scheduler)
            val_loss, val_metrics = self.evaluate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])

            epoch_progress.set_description(
                f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_metrics['accuracy']:.4f} - Val F1: {val_metrics['f1']:.4f}"
            )

            self._display_epoch_summary(
                epoch+1, epochs, train_loss, val_loss, val_metrics)

            # Early stopping
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.patience_counter = 0
                self._save_checkpoint(
                    f"model_best_f1_{val_metrics['f1']:.4f}.pt")
                print("📈 New best model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    print(
                        f"⚠️ Early stopping triggered after {epoch+1} epochs.")
                    break

        print("✅ Training complete!")
        self._plot_training_history()
        return self.model

    def _train_epoch(self, train_loader: DataLoader, scheduler):
        self.model.train()
        total_loss = 0
        batch_losses = []

        # progress_bar = tqdm(
        #     train_loader, desc="Training Batches", leave=False, position=1)
        progress_bar = tqdm(
            train_loader,
            desc="Training Batches",
            leave=False,
            position=1,
            dynamic_ncols=True,
            mininterval=2.0  # 🧼 throttle updates to once per second
        )

        for i, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = self.loss_fn(outputs.logits, labels)
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)

            moving_avg_loss = sum(
                batch_losses[-min(len(batch_losses), 10):]) / min(len(batch_losses), 10)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "avg_loss": f"{moving_avg_loss:.4f}"
            })

        avg_epoch_loss = total_loss / len(train_loader)
        return avg_epoch_loss

    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(loader, desc="Evaluating", leave=False, position=1)

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                loss = self.loss_fn(outputs.logits, labels)
                total_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average='weighted')
        }

        return total_loss / len(loader), metrics

    def _display_epoch_summary(self, epoch, total_epochs, train_loss, val_loss, val_metrics):
        print(f"\n{'='*80}")
        print(f"⏱️  Epoch: {epoch}/{total_epochs}")
        print(f"{'='*80}")
        print(f"📊 Training   | Loss: {train_loss:.4f}")
        print(
            f"📊 Validation | Loss: {val_loss:.4f} | Accuracy: {val_metrics['accuracy']:.4f} | F1 Score: {val_metrics['f1']:.4f}")
        print(f"{'='*80}\n")

    def _plot_training_history(self):
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 1, 1)
        plt.plot(self.history['train_loss'], label='Training Loss', marker='o')
        plt.plot(self.history['val_loss'], label='Validation Loss', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(self.history['val_accuracy'],
                 label='Validation Accuracy', marker='o')
        plt.plot(self.history['val_f1'],
                 label='Validation F1 Score', marker='o')
        plt.title('Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def _save_checkpoint(self, filename: str):
        checkpoint_path = os.path.dirname(
            CONFIG["models"][self.model_type]["model_save_path"])
        os.makedirs(checkpoint_path, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_f1": self.best_val_f1,
            "history": self.history
        }

        torch.save(checkpoint, f"{checkpoint_path}/{filename}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_f1 = checkpoint["best_val_f1"]

        if "history" in checkpoint:
            self.history = checkpoint["history"]

        print(f"[INFO] Model checkpoint loaded from {checkpoint_path}")
        return self.model
