# Mathematical Formulations & Algorithms for Camera-Aware Neural Networks

## Table of Contents
1. [Core Mathematical Foundations](#1-core-mathematical-foundations)
2. [Ray Direction Computation](#2-ray-direction-computation)
3. [Perspective Correction Layer (PCL)](#3-perspective-correction-layer-pcl)
4. [Feature-wise Linear Modulation (FiLM)](#4-feature-wise-linear-modulation-film)
5. [Network Architectures](#5-network-architectures)
6. [Loss Functions](#6-loss-functions)
7. [Training Algorithms](#7-training-algorithms)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Complete Algorithm Pseudocode](#9-complete-algorithm-pseudocode)

---

## 1. Core Mathematical Foundations

### 1.1 Camera Projection Model

The pinhole camera model relates 3D world coordinates to 2D image coordinates:

**Projection Equation:**
$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K [R | t] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

Where:
- $(u, v)$ = 2D image coordinates
- $(X, Y, Z)$ = 3D world coordinates
- $s$ = depth scale factor
- $K$ = camera intrinsic matrix
- $[R | t]$ = camera extrinsic matrix (rotation $R$ and translation $t$)

**Intrinsic Matrix:**
$$
K = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

Where:
- $f_x, f_y$ = focal lengths in pixel units
- $c_x, c_y$ = principal point (image center)

**Extrinsic Matrix:**
$$
[R | t] = \begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z
\end{bmatrix}
$$

Where:
- $R \in SO(3)$ = 3×3 rotation matrix
- $t \in \mathbb{R}^3$ = translation vector

### 1.2 Inverse Projection (Pixel to Ray)

Given a pixel $(u, v)$ and depth $d$, compute 3D point:

**Normalized Image Coordinates:**
$$
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

**3D Camera-Space Point:**
$$
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = d \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

**3D World-Space Point:**
$$
\begin{bmatrix} X_w \\ Y_w \\ Z_w \end{bmatrix} = R^T \left( \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} - t \right)
$$

---

## 2. Ray Direction Computation

### 2.1 Mathematical Formulation

For each pixel $(u, v)$ in an $H \times W$ image, compute normalized ray direction $\mathbf{r}_{uv}$:

**Algorithm:**
$$
\mathbf{r}_{uv} = \text{normalize}\left( K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \right)
$$

Expanded:
$$
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix}
\frac{1}{f_x} & 0 & -\frac{c_x}{f_x} \\
0 & \frac{1}{f_y} & -\frac{c_y}{f_y} \\
0 & 0 & 1
\end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

$$
\mathbf{r}_{uv} = \frac{1}{\sqrt{x^2 + y^2 + 1}} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

### 2.2 C++ Implementation Algorithm

```cpp
Eigen::Tensor<float, 3> computeRayDirections(
    const Eigen::Matrix3f& K, 
    int height, 
    int width
) {
    // Output tensor: (height, width, 3)
    Eigen::Tensor<float, 3> rays(height, width, 3);
    
    // Compute inverse intrinsics
    Eigen::Matrix3f K_inv = K.inverse();
    
    // Extract intrinsic parameters
    float fx = K(0, 0), fy = K(1, 1);
    float cx = K(0, 2), cy = K(1, 2);
    
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            // Pixel coordinates (homogeneous)
            Eigen::Vector3f pixel(u, v, 1.0f);
            
            // Apply inverse intrinsics
            Eigen::Vector3f ray_unnorm = K_inv * pixel;
            
            // Normalize to unit vector
            float norm = ray_unnorm.norm();
            Eigen::Vector3f ray_norm = ray_unnorm / norm;
            
            // Store in tensor
            rays(v, u, 0) = ray_norm(0);
            rays(v, u, 1) = ray_norm(1);
            rays(v, u, 2) = ray_norm(2);
        }
    }
    
    return rays;
}
```

### 2.3 Coordinate System Conventions

**Camera Coordinate System:**
- X-axis: Right
- Y-axis: Down
- Z-axis: Forward (into the scene)

**Ray Direction Properties:**
- Unit length: $||\mathbf{r}_{uv}|| = 1$
- Encodes viewing angle for each pixel
- Independent of scene depth
- Depends only on camera intrinsics

---

## 3. Perspective Correction Layer (PCL)

### 3.1 Motivation

Standard CNNs treat images as regular grids, ignoring perspective distortion. PCL applies geometric correction based on camera parameters.

### 3.2 Mathematical Formulation

**Spatial Transformation:**
$$
\mathbf{F}_{out}(u', v') = \mathcal{G}(\mathbf{F}_{in}, \mathcal{T}(u', v'; K))
$$

Where:
- $\mathbf{F}_{in}$ = input feature map
- $\mathbf{F}_{out}$ = perspective-corrected feature map
- $\mathcal{T}$ = transformation function based on camera $K$
- $\mathcal{G}$ = differentiable grid sampler

**Transformation Function:**
$$
\mathcal{T}(u', v'; K) = K \cdot \text{warp}(K^{-1} \cdot [u', v', 1]^T)
$$

The warp function applies perspective correction:
$$
\text{warp}(\mathbf{p}) = \frac{1}{p_z} \begin{bmatrix} p_x \\ p_y \\ p_z \end{bmatrix}
$$

### 3.3 Differentiable Grid Sampling

Use bilinear interpolation for gradient flow:

$$
\mathbf{F}_{out}(u', v') = \sum_{n,m} \mathbf{F}_{in}(n, m) \cdot \max(0, 1 - |u' - n|) \cdot \max(0, 1 - |v' - m|)
$$

### 3.4 PCL Implementation Algorithm

```cpp
torch::Tensor perspectiveCorrectionLayer(
    const torch::Tensor& features,  // (B, C, H, W)
    const torch::Tensor& K_matrix   // (B, 3, 3)
) {
    int B = features.size(0);  // batch size
    int C = features.size(1);  // channels
    int H = features.size(2);  // height
    int W = features.size(3);  // width
    
    // Generate sampling grid
    auto grid = torch::zeros({B, H, W, 2});
    
    for (int b = 0; b < B; ++b) {
        auto K = K_matrix[b];
        auto K_inv = torch::inverse(K);
        
        for (int v = 0; v < H; ++v) {
            for (int u = 0; u < W; ++u) {
                // Original pixel coordinates
                auto pixel = torch::tensor({u, v, 1.0f});
                
                // Transform to normalized coordinates
                auto norm_coord = torch::matmul(K_inv, pixel);
                
                // Apply perspective correction (identity for now)
                // In practice, this would include depth-dependent warping
                auto corrected = norm_coord / norm_coord[2];
                
                // Back-project to pixel space
                auto new_pixel = torch::matmul(K, corrected);
                
                // Normalize to [-1, 1] for grid_sample
                grid[b][v][u][0] = 2.0f * new_pixel[0].item<float>() / W - 1.0f;
                grid[b][v][u][1] = 2.0f * new_pixel[1].item<float>() / H - 1.0f;
            }
        }
    }
    
    // Apply differentiable sampling
    return torch::nn::functional::grid_sample(
        features, 
        grid,
        torch::nn::functional::GridSampleFuncOptions()
            .mode(torch::kBilinear)
            .padding_mode(torch::kZeros)
            .align_corners(true)
    );
}
```

**Gradient Computation:**
$$
\frac{\partial L}{\partial \mathbf{F}_{in}} = \frac{\partial L}{\partial \mathbf{F}_{out}} \cdot \frac{\partial \mathbf{F}_{out}}{\partial \mathbf{F}_{in}}
$$

Backpropagation through grid_sample is handled automatically by PyTorch.

---

## 4. Feature-wise Linear Modulation (FiLM)

### 4.1 Mathematical Formulation

FiLM conditionally modulates feature maps based on external information (camera parameters):

**FiLM Transformation:**
$$
\text{FiLM}(\mathbf{F}; \gamma, \beta) = \gamma \odot \mathbf{F} + \beta
$$

Where:
- $\mathbf{F} \in \mathbb{R}^{B \times C \times H \times W}$ = feature map
- $\gamma \in \mathbb{R}^{B \times C}$ = learned scale parameters
- $\beta \in \mathbb{R}^{B \times C}$ = learned shift parameters
- $\odot$ = element-wise multiplication (broadcasted)

### 4.2 Camera Embedding

**Embedding Network:**
$$
[\gamma, \beta] = \text{MLP}(\text{flatten}(K, [R|t]))
$$

**MLP Architecture:**
$$
\begin{align}
\mathbf{h}_1 &= \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{x}_{camera} + \mathbf{b}_1) \\
\mathbf{h}_2 &= \text{ReLU}(\mathbf{W}_2 \cdot \mathbf{h}_1 + \mathbf{b}_2) \\
[\gamma, \beta] &= \mathbf{W}_3 \cdot \mathbf{h}_2 + \mathbf{b}_3
\end{align}
$$

Where:
- $\mathbf{x}_{camera} \in \mathbb{R}^{21}$ = concatenated camera parameters (9 from K, 12 from [R|t])
- $\mathbf{h}_1 \in \mathbb{R}^{128}$
- $\mathbf{h}_2 \in \mathbb{R}^{256}$
- $[\gamma, \beta] \in \mathbb{R}^{2C}$

### 4.3 FiLM Implementation Algorithm

```cpp
class FiLMLayer : public torch::nn::Module {
private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc_out{nullptr};
    int feature_channels;
    
public:
    FiLMLayer(int camera_dim, int channels) 
        : feature_channels(channels) {
        // Camera embedding network
        fc1 = register_module("fc1", torch::nn::Linear(camera_dim, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 256));
        fc_out = register_module("fc_out", torch::nn::Linear(256, 2 * channels));
    }
    
    torch::Tensor forward(
        const torch::Tensor& features,      // (B, C, H, W)
        const torch::Tensor& camera_params  // (B, 21)
    ) {
        // Embed camera parameters
        auto h1 = torch::relu(fc1->forward(camera_params));
        auto h2 = torch::relu(fc2->forward(h1));
        auto film_params = fc_out->forward(h2);  // (B, 2*C)
        
        // Split into scale (gamma) and shift (beta)
        auto gamma = film_params.slice(1, 0, feature_channels);  // (B, C)
        auto beta = film_params.slice(1, feature_channels, 2 * feature_channels);
        
        // Reshape for broadcasting: (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1);
        beta = beta.unsqueeze(-1).unsqueeze(-1);
        
        // Apply FiLM transformation
        return gamma * features + beta;
    }
};
```

### 4.4 Gradient Flow

**Forward Pass:**
$$
\mathbf{F}_{out} = \gamma \odot \mathbf{F}_{in} + \beta
$$

**Backward Pass:**
$$
\begin{align}
\frac{\partial L}{\partial \mathbf{F}_{in}} &= \frac{\partial L}{\partial \mathbf{F}_{out}} \odot \gamma \\
\frac{\partial L}{\partial \gamma} &= \sum_{h,w} \frac{\partial L}{\partial \mathbf{F}_{out}} \odot \mathbf{F}_{in} \\
\frac{\partial L}{\partial \beta} &= \sum_{h,w} \frac{\partial L}{\partial \mathbf{F}_{out}}
\end{align}
$$

---

## 5. Network Architectures

### 5.1 Baseline U-Net Architecture

**Encoder:**
$$
\begin{align}
\mathbf{F}_1 &= \text{Conv}_{3 \to 64}(\mathbf{I}) \to \text{ReLU} \to \text{Conv}_{64 \to 64} \to \text{ReLU} \to \text{MaxPool} \\
\mathbf{F}_2 &= \text{Conv}_{64 \to 128}(\mathbf{F}_1) \to \text{ReLU} \to \text{Conv}_{128 \to 128} \to \text{ReLU} \to \text{MaxPool} \\
\mathbf{F}_3 &= \text{Conv}_{128 \to 256}(\mathbf{F}_2) \to \text{ReLU} \to \text{Conv}_{256 \to 256} \to \text{ReLU} \to \text{MaxPool} \\
\mathbf{F}_4 &= \text{Conv}_{256 \to 512}(\mathbf{F}_3) \to \text{ReLU} \to \text{Conv}_{512 \to 512} \to \text{ReLU} \to \text{MaxPool}
\end{align}
$$

**Decoder:**
$$
\begin{align}
\mathbf{D}_1 &= \text{UpConv}_{512 \to 256}(\mathbf{F}_4) \to \text{Concat}(\mathbf{F}_3) \to \text{Conv}_{512 \to 256} \to \text{ReLU} \\
\mathbf{D}_2 &= \text{UpConv}_{256 \to 128}(\mathbf{D}_1) \to \text{Concat}(\mathbf{F}_2) \to \text{Conv}_{256 \to 128} \to \text{ReLU} \\
\mathbf{D}_3 &= \text{UpConv}_{128 \to 64}(\mathbf{D}_2) \to \text{Concat}(\mathbf{F}_1) \to \text{Conv}_{128 \to 64} \to \text{ReLU} \\
\mathbf{D}_{out} &= \text{Conv}_{64 \to 1}(\mathbf{D}_3) \to \text{Sigmoid}
\end{align}
$$

**Output:**
$$
\hat{\mathbf{D}} = d_{max} \cdot \mathbf{D}_{out}
$$

Where $d_{max}$ is maximum depth value (e.g., 10 meters).

### 5.2 Geometry-Aware Network Architecture

**Input Concatenation:**
$$
\mathbf{X}_{input} = \text{Concat}(\mathbf{I}_{RGB}, \mathbf{R}_{rays}) \in \mathbb{R}^{B \times 6 \times H \times W}
$$

**Modified Encoder with Geometric Modules:**
$$
\begin{align}
\mathbf{F}_1 &= \text{Conv}_{6 \to 64}(\mathbf{X}_{input}) \to \text{PCL}(K) \to \text{ReLU} \to \text{Conv}_{64 \to 64} \to \text{ReLU} \\
\mathbf{F}_2 &= \text{FiLM}(\text{Conv}_{64 \to 128}(\mathbf{F}_1), K) \to \text{ReLU} \to \text{MaxPool} \\
\mathbf{F}_3 &= \text{FiLM}(\text{Conv}_{128 \to 256}(\mathbf{F}_2), [R|t]) \to \text{ReLU} \to \text{MaxPool} \\
\mathbf{F}_4 &= \text{Conv}_{256 \to 512}(\mathbf{F}_3) \to \text{SpatialAttention}(\mathbf{R}_{rays}) \to \text{MaxPool}
\end{align}
$$

**Spatial Attention Module:**
$$
\text{Attention}(\mathbf{F}, \mathbf{R}) = \mathbf{F} \odot \sigma(\text{Conv}_{C \to 1}([\mathbf{F}, \mathbf{R}]))
$$

Where $\sigma$ is sigmoid activation.

### 5.3 Network Parameter Counts

**Baseline U-Net:**
$$
\text{Params}_{baseline} \approx 34M
$$

**Geometry-Aware Network:**
$$
\text{Params}_{geo} \approx 36M
$$

Additional parameters from:
- FiLM networks: ~500K
- PCL: No additional parameters (transformation only)
- Attention: ~100K

---

## 6. Loss Functions

### 6.1 Scale-Invariant Depth Loss

**Formulation:**
$$
L_{si}(\hat{\mathbf{D}}, \mathbf{D}) = \frac{1}{N} \sum_{i=1}^{N} (\log \hat{d}_i - \log d_i)^2 - \frac{\lambda}{N^2} \left( \sum_{i=1}^{N} (\log \hat{d}_i - \log d_i) \right)^2
$$

Where:
- $\hat{\mathbf{D}}$ = predicted depth map
- $\mathbf{D}$ = ground truth depth map
- $N$ = number of valid pixels
- $\lambda$ = scale-invariance weight (typically 0.5)

**Physical Interpretation:**
- First term: per-pixel log depth error
- Second term: penalizes global scale shift

**Gradient:**
$$
\frac{\partial L_{si}}{\partial \hat{d}_i} = \frac{2}{N \hat{d}_i} \left[ (\log \hat{d}_i - \log d_i) - \frac{\lambda}{N} \sum_{j=1}^{N} (\log \hat{d}_j - \log d_j) \right]
$$

### 6.2 Gradient Matching Loss

Preserves edge sharpness in depth predictions:

**Formulation:**
$$
L_{grad}(\hat{\mathbf{D}}, \mathbf{D}) = \frac{1}{N} \sum_{i=1}^{N} \left( |\nabla_x \hat{d}_i - \nabla_x d_i| + |\nabla_y \hat{d}_i - \nabla_y d_i| \right)
$$

Where:
- $\nabla_x, \nabla_y$ = image gradients in x and y directions

**Implementation:**
$$
\begin{align}
\nabla_x d_i &= d_{i+1,j} - d_{i,j} \\
\nabla_y d_i &= d_{i,j+1} - d_{i,j}
\end{align}
$$

### 6.3 Combined Loss Function

**Total Loss:**
$$
L_{total} = \alpha \cdot L_{si} + \beta \cdot L_{grad} + \gamma \cdot L_{smooth}
$$

**Smoothness Loss (optional):**
$$
L_{smooth}(\hat{\mathbf{D}}) = \frac{1}{N} \sum_{i=1}^{N} \left( |\nabla_x^2 \hat{d}_i| + |\nabla_y^2 \hat{d}_i| \right) \cdot e^{-|\nabla_x \mathbf{I}_i| - |\nabla_y \mathbf{I}_i|}
$$

This penalizes depth discontinuities except at image edges.

**Typical Hyperparameters:**
- $\alpha = 1.0$ (scale-invariant loss weight)
- $\beta = 0.1$ (gradient matching weight)
- $\gamma = 0.001$ (smoothness weight)

### 6.4 Loss Implementation

```cpp
torch::Tensor scaleInvariantLoss(
    const torch::Tensor& pred_depth,   // (B, 1, H, W)
    const torch::Tensor& gt_depth,     // (B, 1, H, W)
    float lambda = 0.5
) {
    // Create valid mask (ignore zero depth)
    auto valid_mask = (gt_depth > 0);
    
    // Compute log depth
    auto log_pred = torch::log(pred_depth.clamp_min(1e-6));
    auto log_gt = torch::log(gt_depth.clamp_min(1e-6));
    
    // Compute difference
    auto log_diff = log_pred - log_gt;
    auto masked_diff = log_diff.masked_select(valid_mask);
    
    int N = masked_diff.numel();
    
    // First term: mean squared log difference
    auto term1 = (masked_diff * masked_diff).sum() / N;
    
    // Second term: square of mean log difference
    auto term2 = lambda * torch::pow(masked_diff.sum(), 2) / (N * N);
    
    return term1 - term2;
}

torch::Tensor gradientMatchingLoss(
    const torch::Tensor& pred_depth,
    const torch::Tensor& gt_depth
) {
    // Compute gradients using finite differences
    auto pred_grad_x = pred_depth.slice(3, 1, -1) - pred_depth.slice(3, 0, -2);
    auto pred_grad_y = pred_depth.slice(2, 1, -1) - pred_depth.slice(2, 0, -2);
    
    auto gt_grad_x = gt_depth.slice(3, 1, -1) - gt_depth.slice(3, 0, -2);
    auto gt_grad_y = gt_depth.slice(2, 1, -1) - gt_depth.slice(2, 0, -2);
    
    // L1 difference
    auto loss_x = torch::abs(pred_grad_x - gt_grad_x).mean();
    auto loss_y = torch::abs(pred_grad_y - gt_grad_y).mean();
    
    return loss_x + loss_y;
}
```

---

## 7. Training Algorithms

### 7.1 Stochastic Gradient Descent with Adam

**Adam Optimizer Update Rule:**

$$
\begin{align}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla L(\theta_{t-1}) \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) [\nabla L(\theta_{t-1})]^2 \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^t} \\
\hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
\end{align}
$$

**Hyperparameters:**
- Learning rate: $\eta = 10^{-4}$
- First moment decay: $\beta_1 = 0.9$
- Second moment decay: $\beta_2 = 0.999$
- Stability constant: $\epsilon = 10^{-8}$

### 7.2 Learning Rate Schedule

**Cosine Annealing:**
$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t \pi}{T}\right)\right)
$$

Where:
- $\eta_{max} = 10^{-4}$ (initial learning rate)
- $\eta_{min} = 10^{-6}$ (minimum learning rate)
- $T$ = total number of training steps

### 7.3 Training Algorithm Pseudocode

```
Algorithm: Train Camera-Aware Depth Network

Input: 
  - Training dataset D = {(I_i, R_i, K_i, P_i, D_i)} for i=1..N
  - Model M with parameters θ
  - Hyperparameters: epochs E, batch_size B, learning_rate η

Output: Trained model parameters θ*

1. Initialize:
   θ ← random initialization (Xavier/He initialization)
   optimizer ← Adam(θ, lr=η)
   best_val_loss ← ∞
   
2. For epoch = 1 to E:
   
   3. Shuffle training data D
   
   4. For each batch b in D (size B):
      
      5. Load batch data:
         {I_batch, R_batch, K_batch, P_batch, D_batch} ← sample B examples
      
      6. Forward pass:
         X_input ← Concat(I_batch, R_batch)  // Combine RGB + rays
         D_pred ← M(X_input, K_batch, P_batch)
      
      7. Compute loss:
         L_si ← ScaleInvariantLoss(D_pred, D_batch)
         L_grad ← GradientMatchingLoss(D_pred, D_batch)
         L_total ← α·L_si + β·L_grad
      
      8. Backward pass:
         gradients ← ∇_θ L_total
      
      9. Gradient clipping:
         gradients ← clip(gradients, max_norm=1.0)
      
      10. Update parameters:
          θ ← optimizer.step(gradients)
      
      11. Log metrics:
          Log(iteration, L_total, L_si, L_grad)
   
   12. Validation:
       val_loss ← Evaluate(M, D_val)
       
       If val_loss < best_val_loss:
          best_val_loss ← val_loss
          Save_Checkpoint(θ, epoch, "best_model.pt")
       
       If early_stopping(val_loss):
          Break
   
   13. Update learning rate:
       η ← CosineAnnealingSchedule(epoch, E)

14. Return θ*
```

### 7.4 Data Augmentation Strategy

**Geometric Augmentations:**
```
Function: AugmentSample(I, R, K, P, D)
  
  1. Random crop:
     - Crop size: random uniform [0.7, 1.0] × original
     - Update K: adjust c_x, c_y based on crop offset
  
  2. Random horizontal flip (p=0.5):
     - Flip I, D horizontally
     - Flip R[:, :, 0] (x-component of rays)
     - Update P: negate rotation around y-axis
  
  3. Color jitter (applied to I only):
     - Brightness: ±0.2
     - Contrast: ±0.2
     - Saturation: ±0.2
     - Hue: ±0.1
  
  4. Return augmented (I', R', K', P', D')
```

**Important:** Camera parameters (K, P) must be updated consistently with geometric augmentations.

---

## 8. Evaluation Metrics

### 8.1 Depth Accuracy Metrics

**Absolute Relative Error (AbsRel):**
$$
\text{AbsRel} = \frac{1}{