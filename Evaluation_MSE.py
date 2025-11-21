# Standard libraries
import os
import math
import glob
import warnings
from tqdm import tqdm
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import cv2

# PyTorch related libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_skimage  
from PIL import Image

# Custom class imports
import Auto_encoder
import Config
import Dataset

# Ensure GPU visibility before any CUDA operations
os.environ["CUDA_VISIBLE_DEVICES"] = Config.GPUs

# Suppressing warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'       # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'      # Disable oneDNN 
# Fixed CUDA memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# --- Utility Functions ---
def get_device():
    """Determines and returns the appropriate device (GPU if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPU(s).")
        print(f"Available GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    else:
        print("Using CPU.")
    return device

def calculate_psnr(mse):
    """Calculates PSNR from MSE."""
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))

def ssim_loss(y_true, y_pred):
    """Calculate SSIM for each image in the batch"""
    mu_x = torch.mean(y_pred, dim=(2, 3), keepdim=True)
    mu_y = torch.mean(y_true, dim=(2, 3), keepdim=True)
    sigma_x = torch.var(y_pred, dim=(2, 3), keepdim=True)
    sigma_y = torch.var(y_true, dim=(2, 3), keepdim=True)
    covariance = ((y_pred - mu_x) * (y_true - mu_y)).mean(dim=(2, 3), keepdim=True)
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    # Return per-image SSIM as 1D tensor, average over channels
    return ssim.mean(dim=1).squeeze()

def calculate_metrics(original_img_tensor, reconstructed_img_tensor):
    """
    Calculates MSE, PSNR, and SSIM for tensor images.
    Returns per-image metrics as numpy arrays.
    """
    # Calculate per-image MSE loss
    mse_loss = torch.nn.MSELoss(reduction='none')(original_img_tensor, reconstructed_img_tensor)
    mse_loss = mse_loss.mean(dim=[1, 2, 3])  # Average over C,H,W dimensions
    
    # Calculate SSIM
    ssim_val = ssim_loss(original_img_tensor, reconstructed_img_tensor)
    
    # Calculate PSNR
    psnr = -10 * torch.log10(mse_loss)
    
    # Convert to numpy
    mse_np = mse_loss.cpu().numpy()
    psnr_np = psnr.cpu().numpy()
    ssim_np = ssim_val.cpu().numpy()
    
    return mse_np, psnr_np, ssim_np

def extract_scalar(value):
    """Helper function to extract scalar from numpy array or tensor"""
    if isinstance(value, (np.ndarray, np.generic)):
        if value.ndim == 0:  # 0-dimensional array
            return float(value)
        else:  # Multi-dimensional array
            return float(value.flatten()[0])
    elif isinstance(value, torch.Tensor):
        return float(value.item())
    else:
        return float(value)

def save_side_by_side_images(original_np, reconstructed_np, img_path, output_dir, mse, psnr, ssim_val):
    """
    Save original and reconstructed images side by side with metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original_np)
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Reconstructed image
    axes[1].imshow(reconstructed_np)
    axes[1].set_title('Reconstructed', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Add metrics as figure title
    metrics_text = f'MSE: {mse:.4f} | PSNR: {psnr:.2f} dB | SSIM: {ssim_val:.4f}'
    fig.suptitle(metrics_text, fontsize=16, y=0.98, fontweight='bold')
    
    # Add filename as subtitle
    filename_text = os.path.basename(img_path)
    fig.text(0.5, 0.02, filename_text, ha='center', fontsize=10, style='italic')
    
    # Save with original filename
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(output_dir, f'{base_name}_comparison.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# --- Data Loading ---
class CustomEvaluationDataset(TorchDataset):
    """
    A dataset class to load images from nested subfolders and store their paths.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # List of common image file extensions
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif')

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        for entry in os.listdir(root_dir):
            class_path = os.path.join(root_dir, entry)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(image_extensions):
                        self.image_paths.append(os.path.join(class_path, img_name))
            elif entry.lower().endswith(image_extensions):
                # Handle case where images might be directly in root_dir, not in subfolders
                self.image_paths.append(class_path)

        if not self.image_paths:
            print(f"DEBUG: No images found in '{root_dir}' or its direct subfolders with extensions {image_extensions}")
        else:
            print(f"DEBUG: Found {len(self.image_paths)} images in '{root_dir}' and its subfolders.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, Config.tile_size, Config.tile_size), "ERROR:" + img_path

        if self.transform:
            image = self.transform(image)
        return image, img_path

def prepare_evaluation_data(path):
    """Prepares the dataset and DataLoader for evaluation."""
    print("\nPreparing evaluation data....")
    eval_transforms = transforms.Compose([
        transforms.Resize((Config.tile_size, Config.tile_size)),
        transforms.ToTensor(),
    ])
    try:
        eval_dataset = CustomEvaluationDataset(path, transform=eval_transforms)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    if len(eval_dataset) == 0:
        print("No images found in the dataset. Data loader will be empty.")
        return None

    # Use batch_size from config or default to 8
    batch_size = getattr(Config, 'batch_size', 8)
    data_loader_eval = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, 
                                  num_workers=Config.num_workers, drop_last=False)
    print(f"Number of samples in the evaluation data loader: {len(eval_dataset)}")
    return data_loader_eval

# --- Model Loading ---
def load_model(device, model_path):
    """Load model from checkpoint or regular model file"""
    print("\nInitializing the model...")
    loaded_model = Auto_encoder.AutoEncoder()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    
    # Load the checkpoint/model file
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if this is a checkpoint file (with metadata) or regular model file
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # This is a checkpoint file with metadata
        print(f"Loading checkpoint from epoch {checkpoint.get('epoch', 'Unknown')} with training loss {checkpoint.get('loss', 'Unknown'):.4f}")
        state_dict = checkpoint['model_state_dict']
    else:
        # This is a regular model file (state_dict only)
        print("Loading regular model file")
        state_dict = checkpoint
    
    # If the model was saved with DataParallel, remove 'module.' prefix from keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    try:
        loaded_model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"Error loading model state_dict: {e}")
        print("Attempting to load with strict=False (may lead to partially loaded model).")
        loaded_model.load_state_dict(new_state_dict, strict=False)

    loaded_model.to(device)
    
    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1 and not isinstance(loaded_model, nn.DataParallel):
        loaded_model = nn.DataParallel(loaded_model)
        print(f"Model wrapped with DataParallel for {torch.cuda.device_count()} GPUs.")

    loaded_model.eval()
    print("Model loaded successfully!")
    return loaded_model

# --- Main Evaluation Function ---
def evaluate_reconstruction_and_save(model, data_loader, output_excel_path, 
                                    save_images=True, output_image_dir='reconstruction_results'):
    """
    Evaluates the autoencoder's reconstruction quality for all images in the data_loader,
    calculates SSIM, MSE, and PSNR for each, saves results to an Excel sheet,
    saves side-by-side comparison images, and prints overall mean and standard deviation.
    """
    if data_loader is None or len(data_loader) == 0:
        print("No data to evaluate. Skipping evaluation.")
        return

    model.eval()
    device = next(model.parameters()).device

    results = []

    print("\nStarting reconstruction evaluation...")
    with torch.no_grad():
        for batch_idx, (input_images, img_paths) in enumerate(tqdm(data_loader, desc="Processing batches")):
            # Filter out error images
            valid_indices = [i for i, path in enumerate(img_paths) if "ERROR:" not in path]
            if not valid_indices:
                continue
            
            # Get only valid images
            input_images = input_images[valid_indices].to(device)
            valid_paths = [img_paths[i] for i in valid_indices]
            
            # Get reconstructions
            reconstructions = model(input_images)
            
            # Calculate metrics for batch
            mse_batch, psnr_batch, ssim_batch = calculate_metrics(input_images, reconstructions)
            
            # Process each image in batch
            input_np = input_images.cpu().numpy()
            recon_np = reconstructions.cpu().numpy()
            
            for i in range(len(valid_indices)):
                # Convert to HWC format and clip
                original_np = input_np[i].transpose(1, 2, 0)
                reconstructed_np = recon_np[i].transpose(1, 2, 0)
                
                original_np = np.clip(original_np, 0, 1)
                reconstructed_np = np.clip(reconstructed_np, 0, 1)
                
                # Handle NaN/inf
                if np.isnan(reconstructed_np).any() or np.isinf(reconstructed_np).any():
                    print(f"WARNING: Image {valid_paths[i]} contains NaN or inf values!")
                    reconstructed_np = np.nan_to_num(reconstructed_np, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Get metrics for this image (properly extract scalar)
                mse = extract_scalar(mse_batch[i])
                psnr = extract_scalar(psnr_batch[i])
                ssim_val = extract_scalar(ssim_batch[i])
                
                # Save side-by-side comparison image
                if save_images:
                    save_side_by_side_images(original_np, reconstructed_np, valid_paths[i], 
                                            output_image_dir, mse, psnr, ssim_val)

                results.append({
                    "Image Path": valid_paths[i],
                    "MSE": mse,
                    "PSNR": psnr,
                    "SSIM": ssim_val
                })

    if not results:
        print("DEBUG: The 'results' list is empty. No metrics were collected.")
        return

    # Convert results to DataFrame and save to Excel
    df = pd.DataFrame(results)
    df.to_excel(output_excel_path, index=False)
    print(f"\nReconstruction metrics saved to {output_excel_path}")
    if save_images:
        print(f"Side-by-side comparison images saved to {output_image_dir}/")

    # Calculate and print overall statistics
    print("\n--- Overall Reconstruction Statistics ---")
    print(f"Mean MSE: {df['MSE'].mean():.4f} (Std Dev: {df['MSE'].std():.4f})")
    print(f"Mean PSNR: {df['PSNR'].mean():.2f} dB (Std Dev: {df['PSNR'].std():.2f} dB)")
    print(f"Mean SSIM: {df['SSIM'].mean():.4f} (Std Dev: {df['SSIM'].std():.4f})")
    print("---------------------------------------")

# --- Visualization (Optional - for a sample of images) ---
def visualize_reconstruction_with_metrics(model, data_loader, num_images=6):
    """
    Visualize original and reconstructed images side by side with reconstruction metrics (MSE, PSNR, SSIM).
    This function will take the first `num_images` from the data_loader.
    """
    if data_loader is None or len(data_loader) == 0:
        print("No data for visualization. Skipping visualization.")
        return

    model.eval()
    device = next(model.parameters()).device

    input_images_list = []
    reconstructions_list = []
    mse_list = []
    psnr_list = []
    ssim_list = []
    image_paths_for_viz = []

    count = 0
    with torch.no_grad():
        for input_batch, img_paths in data_loader:
            if count >= num_images:
                break
            
            # Filter valid images
            valid_indices = [i for i, path in enumerate(img_paths) if "ERROR:" not in path]
            if not valid_indices:
                continue
            
            input_images = input_batch[valid_indices].to(device)
            valid_paths = [img_paths[i] for i in valid_indices]
            
            reconstructions = model(input_images)
            
            # Calculate metrics
            mse_batch, psnr_batch, ssim_batch = calculate_metrics(input_images, reconstructions)
            
            # Convert to numpy
            input_np = input_images.cpu().numpy()
            recon_np = reconstructions.cpu().numpy()
            
            for i in range(min(len(valid_indices), num_images - count)):
                original_np = input_np[i].transpose(1, 2, 0)
                reconstructed_np = recon_np[i].transpose(1, 2, 0)
                
                original_np = np.clip(original_np, 0, 1)
                reconstructed_np = np.clip(reconstructed_np, 0, 1)
                
                # DEBUG: Check reconstructed image statistics
                print(f"Image {count}: Reconstructed min={reconstructed_np.min():.4f}, "
                      f"max={reconstructed_np.max():.4f}, mean={reconstructed_np.mean():.4f}")
                
                # Check for NaN or inf
                if np.isnan(reconstructed_np).any() or np.isinf(reconstructed_np).any():
                    print(f"WARNING: Image {count} contains NaN or inf values!")
                    reconstructed_np = np.nan_to_num(reconstructed_np, nan=0.0, posinf=1.0, neginf=0.0)

                input_images_list.append(original_np)
                reconstructions_list.append(reconstructed_np)
                mse_list.append(extract_scalar(mse_batch[i]))
                psnr_list.append(extract_scalar(psnr_batch[i]))
                ssim_list.append(extract_scalar(ssim_batch[i]))
                image_paths_for_viz.append(os.path.basename(valid_paths[i]))
                
                count += 1
                if count >= num_images:
                    break

    if not input_images_list:
        print("No images to visualize after filtering.")
        return

    # Create figure with more height to accommodate metrics
    actual_num = len(input_images_list)
    fig = plt.figure(figsize=(2 * actual_num, 6))
    gs = plt.GridSpec(3, actual_num, height_ratios=[4, 4, 1])

    for i in range(actual_num):
        # Original images
        ax1 = fig.add_subplot(gs[0, i])
        ax1.imshow(input_images_list[i])
        ax1.axis('off')
        if i == 0:
            ax1.set_title('Original')
        ax1.text(0.5, -0.1, image_paths_for_viz[i], size=7, ha="center", transform=ax1.transAxes)

        # Reconstructed images
        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(reconstructions_list[i])
        ax2.axis('off')
        if i == 0:
            ax2.set_title('Reconstructed')

        # Add metrics text
        ax3 = fig.add_subplot(gs[2, i])
        ax3.text(0.5, 0.5,
                 f'MSE: {mse_list[i]:.4f}\nPSNR: {psnr_list[i]:.2f}dB\nSSIM: {ssim_list[i]:.4f}',
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=8)
        ax3.axis('off')

    # Add overall metrics
    avg_mse = np.mean(mse_list)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    plt.figtext(0.02, 0.02, f'Average MSE: {avg_mse:.4f}\nAverage PSNR: {avg_psnr:.2f}dB\nAverage SSIM: {avg_ssim:.4f}',
                bbox=dict(facecolor='white', alpha=0.8))

    plt.suptitle('Sample Reconstruction Visualization with Metrics', y=0.98, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    
    # Get device CPU or GPU
    device = get_device()

    # Test samples location
    test_data_base_path = Config.folder_path_test

    # Trained model location
    model_path = Config.model_path_mse

    # Prepare the evaluation data loader
    evaluation_data_loader = prepare_evaluation_data(test_data_base_path)

    # Path to reconstruction metrics
    output_excel_path = Config.reconstruction_metrics_mse
    
    # Output directory for comparison images
    output_image_dir = Config.recon_output_mse

    # Only proceed if data loader was successfully created and has data
    if evaluation_data_loader:
        # Load the trained model
        try:
            model = load_model(device, model_path)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Error loading model: {e}. Exiting evaluation.")
            exit()

        # Perform evaluation, calculate metrics, save to Excel, and save comparison images
        evaluate_reconstruction_and_save(model, evaluation_data_loader, output_excel_path, 
                                        save_images=True, output_image_dir=output_image_dir)

        # Optional: Visualize a few reconstructions with metrics
        print("\nPreparing data for visualization...")
        visual_data_loader = prepare_evaluation_data(test_data_base_path)
        if visual_data_loader:
            visualize_reconstruction_with_metrics(model, visual_data_loader, num_images=6)
        else:
            print("Skipping visualization due to empty or problematic data loader.")
    else:
        print("Skipping overall evaluation and visualization due to empty or problematic data loader.")