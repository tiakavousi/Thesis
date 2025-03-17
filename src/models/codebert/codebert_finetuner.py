import os
import logging
import torch
from pathlib import Path
from typing import Dict, Any
from transformers import (
    RobertaForSequenceClassification, 
    RobertaTokenizerFast
    )

from src.data_pipeline.data_loader import DataModule
from src.training.model_trainer import ModelTrainer
from src.training.model_evaluator import ModelEvaluator
from src.config import CONFIG 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CodeBERTFineTuner:
    """
    Fine-tuner specifically for CodeBERT model.
    """
    def __init__(self):
        """
        Initialize the CodeBERT fine-tuner with configuration.
        """
        self.device = CONFIG["training"]["device"]
        model_name = CONFIG["models"]["codebert"]["pretrained_model_name"]
        num_labels =2
        # num_labels = CONFIG["dataset"]["num_labels"]
        self.class_names = CONFIG["dataset"]["class_names"]

        print(f"[INFO] Initializing CodeBERT fine-tuner with model: {model_name}")
        
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        print("[INFO] Tokenizer loaded successfully.")
        
        # Initialize model
        self.model =RobertaForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        print(f"[INFO] Model {model_name} initialized with {num_labels} labels.")
        
        # Move model to appropriate device
        self.model.to(self.device)
        print(f"[INFO] Model moved to device: {self.device}")
        
        # Initialize data module
        self.data_module = DataModule(self.tokenizer)
        print("[INFO] Data module initialized.")
        
        # Initialize trainer
        self.trainer = ModelTrainer(self.model, model_type="codebert")
        print("[INFO] ModelTrainer initialized.")

        logger.info(f"Initialized CodeBERTFineTuner with model '{model_name}'")
    
    def run(self):
        """
        Run the fine-tuning process.
        
        Returns:
            Dict with evaluation metrics
        """
        print("\n[INFO] Starting CodeBERT fine-tuning process.")
        logger.info("Starting CodeBERT fine-tuning process.")

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

        logger.info("Completed CodeBERT fine-tuning process.")
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
        
        model_name = CONFIG["models"]["codebert"]["pretrained_model_name"]
        report_name = "CodeBERT_Classifier"
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
