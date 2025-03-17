from src.models.codebert.codebert_finetuner import CodeBERTFineTuner

def main():
    """
    Main function to run CodeBERT fine-tuning process.
    Returns:
        CodeBERTFineTuner: The fine-tuner instance with a trained model
    """
    # Create the fine-tuner
    fine_tuner = CodeBERTFineTuner()
    
    # Run the fine-tuning process
    metrics = fine_tuner.run()
    
    # Store metrics as an attribute for easy access
    fine_tuner.metrics = metrics
    
    print(f"Fine-tuning completed with metrics: {metrics}")
    
    return fine_tuner

if __name__ == "__main__":
    main()