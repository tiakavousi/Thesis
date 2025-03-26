from src.models.deberta_3class.deberta_finetuner_3class import DeBERTaFineTuner3Class

def main():
    """
    Main function to run DeBERTa 3-class fine-tuning process.
    Returns:
        DeBERTaFineTuner3Class: The fine-tuner instance with a trained model
    """
    fine_tuner = DeBERTaFineTuner3Class()
    metrics = fine_tuner.run()
    fine_tuner.metrics = metrics
    print(f"✅ Fine-tuning completed with metrics: {metrics}")
    return fine_tuner

if __name__ == "__main__":
    main()
