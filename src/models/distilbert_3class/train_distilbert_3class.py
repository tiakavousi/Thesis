from src.models.distilbert_3class.distilbert_finetuner_3class import DistilBertFineTuner3Class

def main():
    """
    Main function to run DistilBERT 3-class fine-tuning process.
    Returns:
        DistilBertFineTuner3Class: The fine-tuner instance with a trained model
    """
    fine_tuner = DistilBertFineTuner3Class()
    metrics = fine_tuner.run()
    fine_tuner.metrics = metrics
    print(f"✅ Fine-tuning completed with metrics: {metrics}")
    return fine_tuner

if __name__ == "__main__":
    main()
