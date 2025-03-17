from src.models.deberta.deberta_finetuner import DeBERTaFineTuner

def main():
    """
    Main function to run DeBERTa fine-tuning process.
    Returns:
        DeBERTaFineTuner: The fine-tuner instance with a trained model
    """
    # Create the fine-tuner
    fine_tuner = DeBERTaFineTuner()
    
    # Run the fine-tuning process
    metrics = fine_tuner.run()
    
    # Store metrics as an attribute for easy access
    fine_tuner.metrics = metrics
    
    print(f"Fine-tuning completed with metrics: {metrics}")
    
    return fine_tuner

if __name__ == "__main__":
    main()
