AutoEncoder_Image_Reconstruction
--------------------------------
### What is an autoencoder? 

An autoencoder is an unsupervised neural network that compresses an input image into a low-dimensional latent representation and then reconstructs it. By learning to recreate the original image, it captures essential structural and visual patterns, making it useful for feature extraction, denoising, and representation learning, especially in large image datasets.

### How is the autoencoder helpful in DeepCluster++?  

In DeepCluster++, the autoencoder (AE_CRC) provides domain-specific, morphology-sensitive features that significantly improve cluster quality and diversity during tile selection. By training on 100,000 tiles sampled from tumor and normal WSIs, the autoencoder learns to encode subtle histologic structures into compact latent vectors. These embeddings preserve key tissue characteristics more effectively than generic pretrained models, enabling clearer separation of tissue types and improved detection of edge-case patterns. The SSIM-based reconstruction objective ensures the latent space reflects structural integrity, which leads to more coherent clustering. This strengthened feature space allows DeepCluster++ to identify representative, diverse, and informative tiles for downstream classification and survival modeling.
