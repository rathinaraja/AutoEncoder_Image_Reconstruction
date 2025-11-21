# System related libraries
import os, csv, warnings 
from tqdm import tqdm 
from PIL import Image 
import numpy as np
 
# PyTorch related libraries
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.functional import structural_similarity_index_measure as ssim

# Custom class imports
import Config, Dataset, Auto_encoder

# Ensure GPU visibility before any CUDA operations
os.environ["CUDA_VISIBLE_DEVICES"] = Config.GPUs

# Suppressing warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'       # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'      # Disable oneDNN 
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'  # Memory segments can dynamically grow as needed

# --- Utility Functions --
def get_device():
    # Get the available device for computation (CPU or GPU). 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    if torch.cuda.is_available():
        print(f"\nAvailable GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")    
    else:
        print("\nUsing CPU.")   
    return device

# --- Data augmentation definition ---
def get_data_augmentations():
    # Training dataset augmentation to handle variation and loading as tensor dataset
    train_transforms = transforms.Compose([ 
        # Spatial augmentations
        transforms.Resize((Config.tile_size, Config.tile_size)), 
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), 
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            
        # # Color augmentations
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
            
        # # Noise and blurring
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            
        # Convert to tensor  (It converts the pixel values into 0 to 1)
        transforms.ToTensor(),    
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,0.225]),
        #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    return train_transforms

# --- Loading the autoencoder model ---
def load_model(device):
    # Initialize the model and move it to the available device.
    # If multiple GPUs are available, wrap the model with DataParallel.     
    model = Auto_encoder.AutoEncoder().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model) 
        print(f"\n{torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}") 
    return model

# Define SSIM loss function
def ssim_loss(y_true, y_pred):
    mu_x = torch.mean(y_pred, dim=(2, 3), keepdim=True)
    mu_y = torch.mean(y_true, dim=(2, 3), keepdim=True)
    sigma_x = torch.var(y_pred, dim=(2, 3), keepdim=True)
    sigma_y = torch.var(y_true, dim=(2, 3), keepdim=True)
    covariance = ((y_pred - mu_x) * (y_true - mu_y)).mean(dim=(2, 3), keepdim=True)
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return 1 - ssim.mean()

# --- Saving checkpoints ---
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir): 
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"Model_AE_epoch_{epoch:03d}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# --- Train the model ---
def train_model(model, data_loader, device, num_epochs, criterion, optimizer, checkpoint_dir, save_interval):
    # Train the model with the specified data loader, loss function, and optimizer. 
    train_recon_losses = []
    model.train()  # Set model to training mode 
    
    # Create checkpoint directory 
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        total_loss = 0

        # Create a progress bar for the data loader within the epoch
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for batch in progress_bar:
            batch = batch.to(device)
    
            # Forward pass
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() 
            
            # Update progress bar with current loss
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
        # Calculate and store the average loss for the epoch
        avg_loss = total_loss / len(data_loader)
        train_recon_losses.append(avg_loss)    

        # Print the average loss for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_dir)
        
        print()  # Add blank line for better readability
    
    return train_recon_losses

# --- Saving the training loss ---
def save_training_loss(losses, file_path):
    # Save training loss to a CSV file. 
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])  # Header
        for epoch, value in enumerate(losses, 1):
            writer.writerow([epoch, value])
    print(f"\nTraining loss saved to {file_path}")

# --- Saving the checkpoints ---
def save_model(model, file_path):
    # Save the trained model's state dictionary. 
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), file_path)
    else:
        torch.save(model.state_dict(), file_path)
    print(f"\nFinal model saved to {file_path}")

# --- Loading checkpoints ---
def load_checkpoint(model, optimizer, checkpoint_path, device): 
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        print(f"Resumed from epoch {epoch} with loss {loss:.4f}")
        return epoch, loss
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, float('inf')

# Main function
def main():    
    # Get device CPU or GPU to train the model
    device = get_device()

    # Get data transformations for different augmentations
    train_transforms = get_data_augmentations()

    # training data location
    trainig_set_path = Config.folder_path_train

    # Load dataset
    print("\nLoading training data...")
    tile_dataset_train = Dataset.TileDatasetTrain(trainig_set_path, transform=train_transforms)
    
    print("\nThe number of training samples is: ",len(tile_dataset_train))
    data_loader_train = DataLoader(tile_dataset_train, batch_size=Config.batch_size_train, num_workers=Config.num_workers, shuffle=True)

    # Prepare model
    print("\nInitializing the model...")
    model = load_model(device)

    # Configure optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # L2 regularization with weight_decay
    criterion = ssim_loss   
    print(f"\nUsing SSIM Loss for training")
    
    # Optional: Load from checkpoint to resume training
    # Uncomment the lines below if you want to resume from a specific checkpoint
    # start_epoch, last_loss = load_checkpoint(model, optimizer, "checkpoints/Model_AE_epoch_010.pth", device)
    
    # Train the model with checkpoint saving every 5 epochs
    print(f"\nStarting training with checkpoint saving every 5 epochs...")
    print(f"Checkpoints will be saved in './checkpoints/' directory\n")
    
    checkpoint_dir = Config.model_saved_location_ssim
    save_interval = Config.save_interval
    train_recon_losses = train_model(model=model, data_loader=data_loader_train, device=device, num_epochs=Config.num_epochs, 
                                     criterion=criterion, optimizer=optimizer, checkpoint_dir=checkpoint_dir, save_interval=save_interval)  # Save every 10 epochs

    # Save training loss and final model
    training_loss_file = Config.training_loss_file_ssim
    save_training_loss(train_recon_losses, training_loss_file)

    # Save the final model
    model_saved_location = Config.model_saved_location_ssim    
    save_model(model, f"{checkpoint_dir}/Final_Model_AE_SSIM.pth")

    print("\nTraining pipeline completed!")
    print(f"Final training loss: {train_recon_losses[-1]:.4f}")
    print(f"All checkpoints saved in {checkpoint_dir} directory.\n")

# Run the script
if __name__ == "__main__":
    main()