
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import from src if running from examples directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.dataset import RegDataset
from src.train import NNTrain
from src.config_loader import Config

def train_conditional_regressor(x, y, snr, config):
    """
    Train a neural network to regress y on x conditioned on SNR.
    Model: y = f(x, snr)
    
    Args:
        x (np.array): Input variable (e.g., predicted shear), shape (N,).
        y (np.array): Target variable (e.g., true shear), shape (N,).
        snr (np.array): Conditioning variable (SNR), shape (N,).
        config (Config): Forklens configuration object.
        
    Returns:
        model (nn.Module): The trained PyTorch model.
        losses (tuple): (train_losses, valid_losses)
    """
    
    # 1. Prepare Data
    # The network expects input features columns. 
    # We want inputs = [x, snr]
    # We want targets = [y]
    
    x = x.reshape(-1)
    y = y.reshape(-1)
    snr = snr.reshape(-1)
    
    # Normalize inputs if necessary. 
    # Neural networks converge faster with normalized data (mean 0, std 1).
    # Here we assume the user might handle normalization, or we pass raw.
    # Given the values are small (-0.1 to 0.1), raw might be fine, 
    # but SNR might be large (10-100). It's GOOD practice to normalize SNR.
    # For this script, we'll feed them raw as the user requested "optimal model" 
    # and MLPs can learn scaling, though normalization is recommended.
    
    inputs = np.column_stack((x, snr)).astype(np.float32)
    targets = y.reshape(-1, 1).astype(np.float32)
    
    print(f"Training on {len(inputs)} samples. Inputs shape: {inputs.shape}")
    
    # 2. Create Dataset
    # Using the RegDataset we verified earlier
    dataset = RegDataset(inputs, targets, config)
    
    # 3. Initialize Trainer
    # Ensure config inputs are set correctly (in_features=2)
    if config.nntrain.in_features != 2:
        print(f"Warning: Config in_features is {config.nntrain.in_features}, but we have 2 inputs (x, snr). Temporarily overwriting.")
        config.nntrain.in_features = 2
        
    trainer = NNTrain(config)
    
    # 4. Run Training
    train_losses, valid_losses = trainer.run(dataset)
    
    return trainer, train_losses, valid_losses

def evaluate_and_plot(trainer, x, y, snr):
    """
    Evaluate the model and visualize results.
    """
    inputs = np.column_stack((x, snr)).astype(np.float32)
    targets = y.reshape(-1, 1).astype(np.float32)
    
    # Create a temporary dataset/loader for prediction
    dummy_config = trainer.config
    ds = RegDataset(inputs, targets, dummy_config)
    dl = DataLoader(ds, batch_size=1000)
    
    # Predict
    res = trainer.predict(dl)
    preds = res['preds']
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original: x vs y (colored by SNR)")
    plt.scatter(x, y, c=snr, s=1, cmap='viridis')
    plt.xlabel('x (Input)')
    plt.ylabel('y (Target)')
    plt.colorbar(label='SNR')
    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'r--', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.title("Model Prediction: pred vs y (colored by SNR)")
    plt.scatter(preds, y, c=snr, s=1, cmap='viridis')
    plt.xlabel('Model Prediction')
    plt.ylabel('y (Target)')
    plt.colorbar(label='SNR')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('regression_results.png')
    print("Results plot saved to regression_results.png")

if __name__ == "__main__":
    # --- DEMO with Synthetic Data mimicking User's plot ---
    
    # Load Config
    config_path = "/home/z/Zekang.Zhang/forklens/examples/cnn/config.yaml" 
    # Assuming relative path from examples/calibration/
    # If standard loading:
    try:
        conf = Config(config_path)
    except:
        # Fallback for testing
        class MockConfig:
            pass
        conf = Config(config_path) # The loader typically needs a valid path

    # Generate synthetic data
    # Scenario: y = x, but with bias dependent on SNR
    # e.g., for low SNR, y is slightly higher than x.
    N = 5000
    x_true = np.linspace(-0.1, 0.1, N)
    snr = np.random.choice([10, 20, 50], N) # 3 distinct SNR levels like the colors in plot
    
    # Define a relationship: y = x + bias(snr) + noise
    # Bias: small snr -> large bias
    bias = 0.02 * (30 / snr) * np.sin(x_true * 10) # Complicated bias
    noise = 0.005 * np.random.randn(N)
    
    y_target = x_true # We want to recover this? 
    # Or usually in calibration: 
    # We observe x_measured and want to predict y_true.
    x_measured = x_true + bias + noise
    
    # So we regress y_target on (x_measured, snr)
    
    print("Training Regressor on Synthetic Data...")
    trainer, t_loss, v_loss = train_conditional_regressor(x_measured, y_target, snr, conf)
    
    evaluate_and_plot(trainer, x_measured, y_target, snr)
