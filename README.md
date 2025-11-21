# AutoEncoder for Image Reconstruction

## What is an autoencoder? 

An autoencoder is an unsupervised neural network that compresses an input image into a low-dimensional latent representation and then reconstructs it. By learning to recreate the original image, it captures essential structural and visual patterns, making it useful for feature extraction, denoising, and representation learning, especially in large image datasets.

## How is the autoencoder helpful in DeepCluster++?  

In <a href="https://github.com/rathinaraja/DeepCluster" target="_blank" rel="noopener">DeepCluster++</a>, the autoencoder (AE_CRC) provides domain-specific, morphology-sensitive features that significantly improve cluster quality and diversity during tile selection. By training on 100,000 tiles sampled from tumor and normal WSIs, the autoencoder learns to encode subtle histologic structures into compact latent vectors. These embeddings preserve key tissue characteristics more effectively than generic pretrained models, enabling clearer separation of tissue types and improved detection of edge-case patterns. The SSIM-based reconstruction objective ensures the latent space reflects structural integrity, which leads to more coherent clustering. This strengthened feature space allows DeepCluster++ to identify representative, diverse, and informative tiles for downstream classification and survival modeling.

<div align="center">
  <img src="https://github.com/rathinaraja/AutoEncoder_Image_Reconstruction/blob/main/AutoEncoder_for_WSI.jpg" alt="Example" width="950"/>
  <p><em>Figure: AutoEncoder for image reconstruction </em></p>
</div>

Regardless of the application, the <a href="https://github.com/rathinaraja/AutoEncoder_Image_Reconstruction/blob/main/Auto_encoder.py" target="_blank" rel="noopener">autoencoder</a> architecture can be used to reconstruct images across any domain.

# AutoEncoder Training 

This guide provides step-by-step instructions for training an AutoEncoder model (AE_CRC) on pathology images (256x256 RGB) extracted from whole slide images to achieve better image reconstruction quality. It is then used in the DeepCluster++ framework

---

## Table of Contents
- [Overview](#overview)
- [Step 1: Prepare Input Images](#step-1-prepare-input-images)
- [Step 2: Configure Training Parameters](#step-2-configure-training-parameters)
- [Step 3: Train the Model](#step-3-train-the-model)
- [Step 4: Visualize Reconstruction Results](#step-4-visualize-reconstruction-results)
- [Step 5: Evaluate Model Performance](#step-5-evaluate-model-performance)
- [Output Files and Directories](#output-files-and-directories)

---

This AutoEncoder implementation supports two loss functions:
- **MSE (Mean Squared Error)**: Measures pixel-wise differences between original and reconstructed images
- **SSIM (Structural Similarity Index)**: Evaluates structural and perceptual similarity

**Why SSIM for DeepCluster++?** SSIM is preferred for DeepCluster++ because it better captures perceptual image quality by considering luminance, contrast, and structural information. Unlike MSE which treats all pixel differences equally, SSIM aligns more closely with human visual perception, making it ideal for clustering tasks that rely on meaningful feature representations.

---

## Step 1: Prepare Input Images

Collect your training images and place them in a folder structure. The data loader will automatically read all images regardless of:
- Subfolder organization
- Image file types (supports `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.tif`)

**Example folder structure:**
```
training_data/
├── class_1/
│   ├── image1.png
│   ├── image2.jpg
│   ├── image3.png
│   ├── ...
└── image4.jpg
└── ...
```

---

## Step 2: Configure Training Parameters

Edit the `Config.py` file to set your training and evaluation configuration:
```python
#------------------------------------------------------------------
# System parameters
#------------------------------------------------------------------
GPUs = "2"                # Specify GPU IDs to use: "3" or "3,4,5" 
num_workers = 4           # Number of data loading workers

#------------------------------------------------------------------
# Model training (MSE/SSIM)
#------------------------------------------------------------------ 
# Input folder path for training
folder_path_train = '/data_64T_1/Raja/DEEP_CLUSTER++/DeepCluster++/Test_samples_1/WSI_1/Informative_Part1'  

# Image parameters
tile_size = 256           # Image size for training

# Training parameters
num_epochs = 100           # Number of training epochs
batch_size_train = 64     # Batch size during training

# Model checkpoints
model_saved_location_mse = "checkpoints/MSE"              
model_saved_location_ssim = "checkpoints/SSIM"
save_interval = 5         # Save model every N epochs

# Training logs
training_loss_file_mse = "training_loss_mse.csv" 
training_loss_file_ssim = "training_loss_ssim.csv" 

#------------------------------------------------------------------
# Model testing (MSE/SSIM)
#------------------------------------------------------------------ 
# Input folder path for testing
folder_path_test = 'input_images'  

# Testing parameters
batch_size_test = 32      # Batch size during testing

# Trained model paths
model_path_mse = "sample_models/model_auto_encoder_reconstruction_mse.pth"
model_path_ssim = "sample_models/model_auto_encoder_reconstruction_ssim.pth" 

# Output directories
recon_output_mse = "reconstructed_images/MSE/"
recon_output_ssim = "reconstructed_images/SSIM/"

# Reconstruction metrics
reconstruction_metrics_mse = "reconstructed_images/validation_reconstruction_metrics_mse.xlsx"
reconstruction_metrics_ssim = "reconstructed_images/validation_reconstruction_metrics_ssim.xlsx" 
```

## Step 3: Train the Model

### Training with MSE Loss
```bash
python Training_MSE.py
```

### Training with SSIM Loss
```bash
python Training_SSIM.py
```

### Training Progress

During training, you'll see output similar to:
- [Sample_output_MSE_Training.txt](Sample_output_MSE_Training.txt)
- [Sample_output_SSIM_Training.txt](Sample_output_SSIM_Training.txt)

### Training Loss Logs

Loss values for each epoch are automatically saved:
- `training_loss_mse.csv` - MSE training losses
- `training_loss_ssim.csv` - SSIM training losses

### Saved Models

Trained models are stored in the `checkpoints/` directory:
```
checkpoints/
├── MSE/
│   ├── Model_AE_epoch_005.pth
│   ├── Model_AE_epoch_010.pth
└── SSIM/
    ├── Model_AE_epoch_005.pth
    ├── Model_AE_epoch_010.pth
```

Sample pre-trained models are provided in each folder for reference.

---

## Step 4: Visualize Reconstruction Results

Use the provided Jupyter notebook to visualize reconstruction quality:

1. Open the visualization notebook
2. Specify the model location (MSE or SSIM checkpoint)
3. Use images from the `input_images/` folder
4. View side-by-side comparisons with quality metrics:
   - **MSE Loss**: Mean Squared Error
   - **SSIM Score**: Structural Similarity Index
   - **PSNR**: Peak Signal-to-Noise Ratio

By training the model on a large and diverse set of image samples and allowing it to run until the loss fully converges, the autoencoder learns highly precise representations. As a result, it can generate high-quality reconstructed images, as shown below.

<div align="center">
  <img src="https://github.com/rathinaraja/AutoEncoder_Image_Reconstruction/blob/main/Reconstruction_quality.png" alt="Example" width="950"/>
  <p><em>Figure: AutoEncoder for image reconstruction </em></p>
</div>

---

## Step 5: Evaluate Model Performance

Run comprehensive evaluation on your test set:

### Evaluate MSE Model
```bash
python Evaluation_MSE.py
```

### Evaluate SSIM Model
```bash
python Evaluation_SSIM.py
```

### Evaluation Outputs

**Reconstructed Images:**
- MSE results: `reconstructed_images/MSE/`
- SSIM results: `reconstructed_images/SSIM/`

Each output includes:
- Side-by-side comparison (original vs reconstructed)
- Quality metrics displayed on the image

**Metrics Excel Files:**
- `validation_reconstruction_metrics_mse.xlsx`
- `validation_reconstruction_metrics_ssim.xlsx`

Each Excel file contains:
| Image Path | MSE | PSNR | SSIM |
|------------|-----|------|------|
| path/to/image1.png | 0.0042 | 38.75 | 0.9812 |
| path/to/image2.png | 0.0051 | 36.42 | 0.9756 |

---



---

## Notes

- **GPU Configuration**: Adjust the `GPUs` parameter in `Config.py` to match your available hardware
- **Memory Requirements**: Large batch sizes require more GPU memory. Reduce `batch_size_train` if you encounter OOM errors
- **Training Time**: Training duration depends on dataset size, image resolution, and hardware specifications
- **Model Selection**: SSIM-trained models generally produce more visually pleasing reconstructions and are recommended for DeepCluster++ workflows

---

## Citation

If you use this AutoEncoder implementation in your research, please cite the DeepCluster++ paper.

---

## Support

For issues or questions, please open an issue in the repository or contact the maintainers.
