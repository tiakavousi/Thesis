from src.models.codebert_3class.codebert_finetuner_3class import CodeBERTFineTuner3Class

def main():
    """
    Main function to run CodeBERT fine-tuning process for 3-class sentiment classification.
    Returns:
        CodeBERTFineTuner3Class: The fine-tuner instance with a trained model
    """
    # Create the fine-tuner
    fine_tuner = CodeBERTFineTuner3Class()

    # Run the fine-tuning process
    metrics = fine_tuner.run()

    # Store metrics as an attribute for easy access
    fine_tuner.metrics = metrics

    print(f"Fine-tuning completed with metrics: {metrics}")

    return fine_tuner

if __name__ == "__main__":
    main()
