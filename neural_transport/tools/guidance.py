import torch
import torch.nn as nn


class XCO2Guidance(nn.Module):
    """Guidance function for column-averaged XCO2 observations.
    
    Computes gradients that steer the 3D CO2 profile toward matching
    sparse column-averaged observations from OCO-2.
    """
    
    def __init__(
        self,
        obs_mask,           # [B, 1, Nlat, Nlon] - where we have observations
        obs_values,         # [B, 1, Nlat, Nlon] - XCO2 observations (normalized)
        averaging_kernel,   # [B, C, Nlat, Nlon] - vertical weighting
        guidance_scale=1.0, # Strength of guidance
        loss_type='mse',    # 'mse' or 'l1'
    ):
        super().__init__()
        self.obs_mask = obs_mask
        self.obs_values = obs_values.detach()
        self.ak = averaging_kernel
        self.guidance_scale = guidance_scale
        self.loss_type = loss_type
    
    def compute_xco2(self, x):
        """Compute column-averaged XCO2 from 3D profile.
        
        Args:
            x: [B, C, Nlat, Nlon] - 3D CO2 field
            
        Returns:
            xco2: [B, 1, Nlat, Nlon] - column average
        """
        # Sum over vertical levels weighted by averaging kernel
        xco2 = (self.ak * x).sum(dim=1, keepdim=True)  # [B, 1, Nlat, Nlon]
        return xco2
    
    def compute_loss(self, x_pred, x_denoised):
        """Compute observation loss.

        Args:
            x_pred: [B, C, Nlat, Nlon] - current noisy prediction
            x_denoised: [B, C, Nlat, Nlon] - denoised prediction from model
            
        Returns:
            loss: scalar tensor
        """
        # Use denoised prediction for column average
        xco2_pred = self.compute_xco2(x_denoised)

        # Compute loss only where we have observations
        if self.loss_type == 'mse':
            loss = ((xco2_pred - self.obs_values) ** 2 * self.obs_mask).sum()
        elif self.loss_type == 'l1':
            loss = (torch.abs(xco2_pred - self.obs_values) * self.obs_mask).sum()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Normalize by number of observations
        n_obs = self.obs_mask.sum().clamp(min=1)
        loss = loss / n_obs

        return loss

    def get_gradient(self, x_current, x_denoised, retain_graph=False):
        """Compute guidance gradient.
        
        Args:
            x_current: [B, C, Nlat, Nlon] - current noisy state (requires_grad=True)
            x_denoised: [B, C, Nlat, Nlon] - denoised prediction
            retain_graph: whether to retain computation graph
            
        Returns:
            grad: [B, C, Nlat, Nlon] - gradient for guidance
        """
        loss = self.compute_loss(x_current, x_denoised)
        
        # Compute gradient w.r.t. x_current
        grad = torch.autograd.grad(
            outputs=loss,
            inputs=x_current,
            retain_graph=retain_graph,
            create_graph=False
        )[0]
        
        return self.guidance_scale * grad