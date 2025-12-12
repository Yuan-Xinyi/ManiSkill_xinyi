import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import joblib
import os

class GMMSampler:
    def __init__(self, n_components=10, random_state=None):
        self.n_components = n_components
        self.model = GaussianMixture(
            n_components=n_components, 
            covariance_type='full', 
            init_params='kmeans',
            random_state=random_state
        )
        self.is_fitted = False

    def fit(self, data):
        """
        Fit the GMM model to the data.
        Args:
            data: (N, 4) numpy array of [Cx, Cy, Theta, Radius]
        """
        if len(data) < self.n_components:
            return 
        
        # Convert to [Cx, Cy, Cos, Sin, Radius] (5 dims) to handle cyclic angle
        cx = data[:, 0]
        cy = data[:, 1]
        theta = data[:, 2]
        radius = data[:, 3]
        
        X = np.column_stack([cx, cy, np.cos(theta), np.sin(theta), radius])
        
        self.model.fit(X)
        self.is_fitted = True

    def save(self, path):
        """Save the fitted GMM model to disk."""
        if self.is_fitted:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
            print(f"GMM saved to {path}")

    def load(self, path):
        """Load a GMM model from disk."""
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_fitted = True
            print(f"GMM loaded from {path}")
        else:
            print(f"GMM file {path} not found.")

    def sample(self, n_samples, radius, alpha=1.0, device="cpu"):
        """
        Sample task vectors conditioned on radius.
        Args:
            n_samples: Number of samples
            radius: float. The target radius for the tasks.
            alpha: Covariance narrowing factor.
            device: torch device.
        Returns:
            task_vectors: (n_samples, 4) tensor of [Sx, Sy, Gx, Gy]
        """
        if not self.is_fitted:
            # Fallback to random if not fitted
            return torch.rand((n_samples, 4), device=device) * 0.2

        # We want to sample P(Y | x) where x is Radius, Y is [Cx, Cy, Cos, Sin]
        # x_val is the condition value
        x_val = float(radius)
        
        means = self.model.means_ # (K, 5)
        covs = self.model.covariances_ # (K, 5, 5)
        weights = self.model.weights_ # (K,)
        
        # Indices
        idx_y = [0, 1, 2, 3] # Cx, Cy, Cos, Sin
        idx_x = [4]          # Radius
        
        cond_means = []
        cond_covs = []
        cond_weights = []
        
        # Calculate conditional distribution for each component
        for k in range(self.n_components):
            mu = means[k]
            sigma = covs[k]
            
            mu_y = mu[idx_y]
            mu_x = mu[idx_x] # scalar-ish
            
            sigma_yy = sigma[np.ix_(idx_y, idx_y)]
            sigma_yx = sigma[np.ix_(idx_y, idx_x)]
            sigma_xy = sigma[np.ix_(idx_x, idx_y)]
            sigma_xx = sigma[np.ix_(idx_x, idx_x)]
            
            # Inverse of sigma_xx (1x1 matrix)
            try:
                sigma_xx_inv = 1.0 / (sigma_xx[0, 0] + 1e-8)
            except:
                sigma_xx_inv = 1.0
            
            # Conditional Mean: mu_y + sigma_yx * inv(sigma_xx) * (x - mu_x)
            diff = x_val - mu_x[0]
            mu_y_cond = mu_y + sigma_yx.flatten() * sigma_xx_inv * diff
            
            # Conditional Cov: sigma_yy - sigma_yx * inv(sigma_xx) * sigma_xy
            sigma_y_cond = sigma_yy - sigma_yx @ (sigma_xy * sigma_xx_inv)
            
            # Apply alpha narrowing to the spatial distribution
            sigma_y_cond *= alpha
            
            cond_means.append(mu_y_cond)
            cond_covs.append(sigma_y_cond)
            
            # Update component weight: w_k * P(x | k)
            # P(x | k) is PDF of N(mu_x, sigma_xx)
            var_x = sigma_xx[0, 0]
            denom = np.sqrt(2 * np.pi * var_x)
            num = np.exp(-0.5 * (diff**2) / (var_x + 1e-8))
            prob_x = num / (denom + 1e-8)
            
            cond_weights.append(weights[k] * prob_x)
            
        # Normalize weights
        cond_weights = np.array(cond_weights)
        if cond_weights.sum() == 0:
            cond_weights = np.ones_like(cond_weights) / len(cond_weights)
        else:
            cond_weights /= cond_weights.sum()
            
        # 1. Sample components based on conditional weights
        component_indices = np.random.choice(
            self.n_components, size=n_samples, p=cond_weights
        )
        
        samples_y = np.zeros((n_samples, 4))
        
        # 2. Sample from conditional Gaussians
        for k in range(self.n_components):
            mask = (component_indices == k)
            count = np.sum(mask)
            if count == 0:
                continue
            
            # Add jitter for numerical stability
            cov = cond_covs[k] + np.eye(4) * 1e-6
            pts = np.random.multivariate_normal(cond_means[k], cov, size=count)
            samples_y[mask] = pts
            
        # 3. Convert [Cx, Cy, Cos, Sin] + Radius -> [Sx, Sy, Gx, Gy]
        t_samples = torch.from_numpy(samples_y).float().to(device)
        cx = t_samples[:, 0:1]
        cy = t_samples[:, 1:2]
        cos_t = t_samples[:, 2:3]
        sin_t = t_samples[:, 3:4]
        
        # Normalize direction vector
        norm = torch.sqrt(cos_t**2 + sin_t**2)
        cos_t = cos_t / (norm + 1e-8)
        sin_t = sin_t / (norm + 1e-8)
        
        # Calculate Start/Goal
        r = x_val
        off_x = r * cos_t
        off_y = r * sin_t
        
        sx = cx + off_x
        sy = cy + off_y
        gx = cx - off_x
        gy = cy - off_y
        
        return torch.cat([sx, sy, gx, gy], dim=1)
