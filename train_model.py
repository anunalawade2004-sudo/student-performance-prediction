"""Train and evaluate both models from the command line."""
from data_utils import load_dataset
from model_utils import train_models

def main():
    data = load_dataset()
    reg, clf, metrics, *_ = train_models(data)
    print("\nStudent Performance Prediction - Evaluation")
    print("-" * 48)
    for name, value in metrics.items():
        print(f"{name.upper():12s}: {value:.4f}")
    print("\nModels trained successfully.")
    print("Note: app.py trains the models when the app starts; this script is for standalone evaluation.")

if __name__ == "__main__":
    main()
