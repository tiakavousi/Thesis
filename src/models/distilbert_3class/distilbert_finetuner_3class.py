import logging
from transformers import (
    DistilBertForSequenceClassification, 
    DistilBertTokenizerFast
)
from src.data_pipeline.data_loader import DataModule
from src.training.model_trainer import ModelTrainer
from src.training.model_evaluator import ModelEvaluator
from src.config_three_class import CONFIG_3CLASS

class DistilBertFineTuner3Class:
    """
    Fine-tuner for 3-class DistilBERT model.
    """
    def __init__(self):
        self.device = CONFIG_3CLASS["training"]["device"]
        model_name = CONFIG_3CLASS["models"]["distilbert"]["pretrained_model_name"]
        num_labels = CONFIG_3CLASS["dataset"]["num_labels"]
        self.class_names = CONFIG_3CLASS["dataset"]["class_names"]

        print(f"[INFO] Initializing DistilBERT 3-class fine-tuner with model: {model_name}")
        
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        print("[INFO] Tokenizer loaded successfully.")
        
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        print(f"[INFO] Model {model_name} initialized with {num_labels} labels.")
        
        self.model.to(self.device)
        print(f"[INFO] Model moved to device: {self.device}")
        
        self.data_module = DataModule(self.tokenizer) 
        print("[INFO] Data module initialized.")
        
        # ✅ NEW: Get class weights and pass them to the trainer
        _, _ = self.data_module.load_dataset()
        class_weights = self.data_module.get_class_weights()

        self.trainer = ModelTrainer(self.model, model_type="distilbert", class_weights=class_weights)
        print("[INFO] ModelTrainer initialized.")
   
    def run(self):
        print("\n[INFO] Starting DistilBERT 3-class fine-tuning process.")
        print("[INFO] Loading dataset...")
        texts, labels = self.data_module.load_dataset()
        print(f"[INFO] Loaded dataset with {len(texts)} samples.")

        print("[INFO] Creating dataloaders...")
        train_loader, val_loader, test_loader = self.data_module.create_dataloaders(texts, labels)
        print("[INFO] Dataloaders created.")

        print("\n[INFO] Starting training...")
        self.trainer.train(train_loader, val_loader)
        print("[INFO] Training complete.")

        print("[INFO] Evaluating model...")
        metrics = self.evaluate_model(test_loader)
        print(f"[INFO] Model evaluation metrics: {metrics}")

        print("[INFO] 3-class fine-tuning process completed.\n")
        return metrics

    def evaluate_model(self, test_loader):
        print("[INFO] Initializing model evaluator...")
        model_name = CONFIG_3CLASS["models"]["distilbert"]["pretrained_model_name"]
        report_name = "DistilBERT_3Class_Classifier"
        
        evaluator = ModelEvaluator(
            model=self.model,
            device=self.device,
            model_name=report_name,
            class_names=self.class_names
        )

        report_path = CONFIG_3CLASS["evaluation"]["report_save_path"]

        print("[INFO] Running evaluation...")
        metrics = evaluator.evaluate_and_report(test_loader, output_path=report_path)

        print(f"[INFO] Evaluation report generated at {report_path}")
        return metrics
