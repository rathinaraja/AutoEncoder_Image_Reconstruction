#------------------------------------------------------------------
# System parameters
#------------------------------------------------------------------
GPUs = "2"  # Specify GPU IDs to use: "3" or "3,4,5" 
num_workers = 4 
#------------------------------------------------------------------
# Moel training (MSE/SSIM)
#------------------------------------------------------------------ 
# Input folder path for training
folder_path_train = '/data_64T_1/Raja/DEEP_CLUSTER++/DeepCluster++/Test_samples_1/WSI_1/Informative_Part1'  

# Image size
tile_size = 256  

# Numebr of training epochs
num_epochs = 10

# Batch size during training
batch_size_train = 64

model_saved_location_mse = "checkpoints/MSE"              
model_saved_location_ssim = "checkpoints/SSIM"
save_interval = 5

training_loss_file_mse = "training_loss_mse.csv" 
training_loss_file_ssim = "training_loss_ssim.csv" 
#------------------------------------------------------------------
# Moel testing (MSE/SSIM)
#------------------------------------------------------------------ 
# Input folder path for testing
folder_path_test = 'input_images'  

# Batch size during testing
batch_size_test = 32

# Final saved model
model_path_mse = "checkpoints/MSE/model_auto_encoder_reconstruction_mse.pth"
model_path_ssim = "checkpoints/SSIM/model_auto_encoder_reconstruction_ssim.pth" 

# Output folder for the reconstructed image
recon_output_mse = "reconstructed_images/MSE/"
recon_output_ssim = "reconstructed_images/SSIM/"

# Reconstruction metrics for the test set
reconstruction_metrics_mse = "reconstructed_images/validation_reconstruction_metrics_mse.xlsx"
reconstruction_metrics_ssim = "reconstructed_images/validation_reconstruction_metrics_ssim.xlsx" 
#------------------------------------------------------------------ 