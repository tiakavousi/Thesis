import os
import logging
import torch
from pathlib import Path
from typing import Dict, Any
from transformers import (
    DebertaForSequenceClassification, 
    DebertaTokenizerFast
)
from src.data_pipeline.data_loader import DataModule
from src.training.model_trainer import ModelTrainer
from src.training.model_evaluator import ModelEvaluator
from src.config import CONFIG 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DeBERTaFineTuner:
    """
    Fine-tuner specifically for DeBERTa model.
    """
    def __init__(self):
        """
        Initialize the DeBERTa fine-tuner with configuration.
        """
        self.device = CONFIG["training"]["device"]
        model_name = CONFIG["models"]["deberta"]["pretrained_model_name"]
        num_labels = 2
        self.class_names = CONFIG["dataset"]["class_names"]

        print(f"[INFO] Initializing DeBERTa fine-tuner with model: {model_name}")
        
        # Initialize tokenizer
        self.tokenizer = DebertaTokenizerFast.from_pretrained(model_name)
        print("[INFO] Tokenizer loaded successfully.")
        
        # Initialize model with dropout if specified
        dropout = CONFIG["training"].get("dropout", 0.1)
        self.model = DebertaForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )
        print(f"[INFO] Model {model_name} initialized with {num_labels} labels and dropout {dropout}.")
        
        # Apply layer freezing if specified
        if "layer_freezing" in CONFIG["models"]["deberta"]:
            self._freeze_layers(CONFIG["models"]["deberta"]["layer_freezing"])
        
        # Move model to appropriate device
        self.model.to(self.device)
        print(f"[INFO] Model moved to device: {self.device}")
        
        # Initialize data module
        self.data_module = DataModule(self.tokenizer)
        print("[INFO] Data module initialized.")
        
        # Initialize trainer
        self.trainer = ModelTrainer(self.model, model_type="deberta")
        print("[INFO] ModelTrainer initialized.")

        logger.info(f"Initialized DeBERTaFineTuner with model '{model_name}'")

    def _freeze_layers(self, num_layers):
        """Freeze the embeddings and first num_layers of the DeBERTa model."""
        # Freeze embeddings
        for param in self.model.deberta.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze the first N encoder layers
        for i in range(num_layers):
            for param in self.model.deberta.encoder.layer[i].parameters():
                param.requires_grad = False
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"[INFO] Froze embeddings and first {num_layers} transformer layers")
        print(f"[INFO] Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%} of total)")
        logger.info(f"Froze embeddings and first {num_layers} layers. Trainable: {trainable_params:,} params ({trainable_params/total_params:.1%})")
        
    def run(self):
        """
        Run the fine-tuning process.
        
        Returns:
            Dict with evaluation metrics
        """
        print("\n[INFO] Starting DeBERTa fine-tuning process.")
        logger.info("Starting DeBERTa fine-tuning process.")

        # Load dataset
        print("[INFO] Loading dataset...")
        texts, labels = self.data_module.load_dataset()
        print(f"[INFO] Loaded dataset with {len(texts)} samples.")

        # Create data loaders
        print("[INFO] Creating dataloaders...")
        train_loader, val_loader, test_loader = self.data_module.create_dataloaders(texts, labels)
        print("[INFO] Dataloaders created.")

        # Train model
        print("\n[INFO] Starting training...")
        self.trainer.train(train_loader, val_loader)
        print("[INFO] Training complete.")

        # Evaluate final model performance
        print("[INFO] Evaluating model...")
        metrics = self.evaluate_model(test_loader)
        print(f"[INFO] Model evaluation metrics: {metrics}")

        logger.info("Completed DeBERTa fine-tuning process.")
        print("[INFO] Fine-tuning process completed.\n")
        
        return metrics

    
    def evaluate_model(self, test_loader):
        """
        Evaluate model on test data and generate a comprehensive report.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dict with evaluation metrics
        """
        print("[INFO] Initializing model evaluator...")
        
        model_name = CONFIG["models"]["deberta"]["pretrained_model_name"]
        report_name = "DeBERTa_Classifier"
        evaluator = ModelEvaluator(
            model=self.model, 
            device=self.device,
            model_name=report_name,
            class_names=self.class_names
        )

        # Generate evaluation report
        report_path = f"evaluation_results/{model_name}_evaluation_report.pdf"
        
        print("[INFO] Running evaluation...")
        metrics = evaluator.evaluate_and_report(test_loader, output_path=report_path)

        print(f"[INFO] Evaluation report generated at {report_path}")
        logger.info(f"Evaluation report generated at {report_path}")
        
        return metrics
