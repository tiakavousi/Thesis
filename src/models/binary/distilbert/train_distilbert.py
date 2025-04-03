from src.models.distilbert.distilbert_finetuner import DistilBERTFineTuner

def main():
    """
    Main function to run DistilBERT fine-tuning process.
    Returns:
        DistilBERTFineTuner: The fine-tuner instance with a trained model
    """
    # Create the fine-tuner
    fine_tuner = DistilBERTFineTuner()
    
    # Run the fine-tuning process
    metrics = fine_tuner.run()
    
    # Store metrics as an attribute for easy access
    fine_tuner.metrics = metrics
    
    print(f"Fine-tuning completed with metrics: {metrics}")
    
    return fine_tuner

if __name__ == "__main__":
    main()