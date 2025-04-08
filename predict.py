import os
import cv2
import numpy as np
import torch
import pandas as pd
from torch import nn
import argparse

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
        conv_output_size = 128 * 25 * 6  # 19200
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 1024), nn.ReLU(), nn.Dropout(0.4),
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
        return [self.digit1(x), self.digit2(x), self.digit3(x), 
                self.digit4(x), self.digit5(x), self.digit6(x)]

def predict_captcha(model, image_path, device):
    """Predict captcha for a single image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image at {image_path}")
        return None
    
    # Pre-processing same as training images
    # Threshold value
    thresh_value = 220

    # Step 1: Median blur
    blurred = cv2.medianBlur(img, 3)

    # Step 2: Thresholding
    _, thresholded = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)

    # Step 3: Another median blur to clean edges
    finalimg = cv2.medianBlur(thresholded, 3)
    
    finalimg = finalimg / 255.0
    finalimg = np.expand_dims(finalimg, axis=0)  # Add channel dimension
    finalimg = np.expand_dims(finalimg, axis=0)  # Add batch dimension
    
    img_tensor = torch.FloatTensor(finalimg)
    
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        outputs = model(img_tensor)
        
        predicted_digits = []
        for output in outputs:
            _, predicted = torch.max(output, 1)
            predicted_digits.append(predicted.item())
    
    captcha_text = ''.join(map(str, predicted_digits))
    return captcha_text

def predict_folder(model_name="captcha_model_best.pth", subfolder_name="validation-images", output_csv_name="predictions.csv"):

    # Fixed parent folders
    MODELS_PARENT = "models"
    DATA_PARENT = "data"
    OUTPUT_PARENT = "output"
    
    # Construct full paths
    model_path = os.path.join(MODELS_PARENT, model_name)
    image_folder = os.path.join(DATA_PARENT, subfolder_name)
    output_csv = os.path.join(OUTPUT_PARENT, output_csv_name)
    
    # Ensure parent directories exist
    os.makedirs(MODELS_PARENT, exist_ok=True)
    os.makedirs(DATA_PARENT, exist_ok=True)
    os.makedirs(OUTPUT_PARENT, exist_ok=True)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Check if image folder exists
    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found at {image_folder}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = CaptchaCNN()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get all image files from folder
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = [f for f in os.listdir(image_folder) 
                  if os.path.isfile(os.path.join(image_folder, f)) and 
                  os.path.splitext(f.lower())[1] in valid_extensions]
    
    if not image_files:
        print(f"No image files found in {image_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        try:
            prediction = predict_captcha(model, image_path, device)
            if prediction:
                results.append({
                    'image_path': subfolder_name + "/" + image_file,
                    'predicted_captcha': prediction
                })
                print(f"Image: {image_file}, Prediction: {prediction}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    # Save results to CSV if requested
    if output_csv and results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    
    return results

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Predict captchas from images using a trained model.")
    parser.add_argument("--model-name", type=str, default="captcha_model_best.pth",
                        help="Name of the model file in 'models' folder (default: captcha_model_best.pth)")
    parser.add_argument("--subfolder-name", type=str, default="validation-images",
                        help="Name of the subfolder in 'data' folder (default: validation-images)")
    parser.add_argument("--output-csv-name", type=str, default="predictions.csv",
                        help="Name of the output CSV file in 'output' folder (default: predictions.csv)")
    
    args = parser.parse_args()
    
    # Run prediction with provided or default arguments
    predict_folder(args.model_name, args.subfolder_name, args.output_csv_name)

if __name__ == "__main__":
    main()