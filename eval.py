import os
import cv2
import numpy as np
import torch
import pandas as pd
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Define the CNN Model Architecture used in Training as model is saved as state_dict()
class CaptchaCNN(nn.Module):
    def __init__(self):
        super(CaptchaCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        conv_output_size = 128 * 25 * 6
        self.fc_layers = nn.Sequential(
            nn.Flatten(), nn.Linear(conv_output_size, 1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.4),
        )
        self.digit1 = nn.Linear(512, 10)
        self.digit2 = nn.Linear(512, 10)
        self.digit3 = nn.Linear(512, 10)
        self.digit4 = nn.Linear(512, 10)
        self.digit5 = nn.Linear(512, 10)
        self.digit6 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return [self.digit1(x), self.digit2(x), self.digit3(x), self.digit4(x), self.digit5(x), self.digit6(x)]

def predict_captcha(model, image_path, device):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return "000000"
    
    # Pre-processing to match training data
    blurred = cv2.medianBlur(img, 3)
    _, thresholded = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
    finalimg = cv2.medianBlur(thresholded, 3) / 255.0
    finalimg = np.expand_dims(finalimg, axis=(0, 1))
    img_tensor = torch.FloatTensor(finalimg).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        predicted_digits = [torch.max(output, 1)[1].item() for output in outputs]
    return ''.join(map(str, predicted_digits))

def evaluate(model_path="models/captcha_model_best.pth", labels_csv="data/captcha_data.csv"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = CaptchaCNN()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load validation data
    labels_df = pd.read_csv(labels_csv)
    validation_df = labels_df[labels_df['image_path'].str.startswith('validation-images')]
    if validation_df.empty:
        print("No validation images found")
        return

    true_labels = []
    pred_labels = []
    for _, row in validation_df.iterrows():
        image_path = os.path.join('data', row['image_path'])
        true_label = str(row['solution']).zfill(6)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            continue
        pred_label = predict_captcha(model, image_path, device)
        true_labels.append(true_label)
        pred_labels.append(pred_label)

    if not true_labels:
        print("No valid images processed")
        return

    # CAPTCHA metrics (full sequence)
    captcha_metrics = {
        'Accuracy': accuracy_score(true_labels, pred_labels),
        'Precision': precision_score(true_labels, pred_labels, average='weighted', zero_division=0),
        'Recall': recall_score(true_labels, pred_labels, average='weighted', zero_division=0),
        'F1': f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    }

    # Per-digit accuracy
    true_digits = np.array([list(label) for label in true_labels], dtype=int)
    pred_digits = np.array([list(label) for label in pred_labels], dtype=int)
    per_digit_accuracy = np.mean(true_digits == pred_digits, axis=0)

    # Individual digit accuracy (0-9)
    digit_acc = {d: 0.0 for d in range(10)}  # Initialize all digits
    for digit in range(10):
        mask = true_digits == digit
        if mask.sum() > 0:
            digit_acc[digit] = np.mean(pred_digits[mask] == digit)

    # Print metrics
    print(f"CAPTCHA Metrics: {', '.join([f'{k}: {v:.4f}' for k, v in captcha_metrics.items()])}")
    print(f"Per-Digit Accuracy: {', '.join([f'Digit {i+1}: {acc:.4f}' for i, acc in enumerate(per_digit_accuracy)])}")
    print(f"Individual Digit Accuracy: {', '.join([f'{d}: {acc:.4f}' for d, acc in digit_acc.items()])}")

    # Plot and save
    os.makedirs("metrics", exist_ok=True)

    # Plot CAPTCHA metrics
    plt.bar(captcha_metrics.keys(), captcha_metrics.values())
    plt.ylim(0, 1)
    plt.title("CAPTCHA Metrics")
    plt.savefig("metrics/captcha_metrics.png")
    plt.close()

    # Plot per-digit accuracy
    plt.bar([f"Digit {i+1}" for i in range(6)], per_digit_accuracy)
    plt.ylim(0, 1)
    plt.title("Per-Digit Accuracy")
    plt.savefig("metrics/per_digit_accuracy.png")
    plt.close()

    # Plot individual digit accuracy with all numbers and scores
    bars = plt.bar(range(10), list(digit_acc.values()))
    plt.xticks(range(10))  # Force all numbers 0-9
    plt.ylim(0, 1)
    plt.title("Individual Digit Accuracy (0-9)")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", ha='center', va='bottom')
    plt.savefig("metrics/individual_digit_accuracy.png")
    plt.close()

if __name__ == "__main__":
    evaluate()