'''
Example implementation of a U-Net model with ReFrame support.
This example demonstrates how to integrate ReFrame into a U-Net architecture.
The U-Net is not trained. Randomized weights and inputs are used to simulate the forward pass and demonstrate the caching functionality of ReFrame.
This code is for demonstration purposes only.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import time, copy, random

class DoubleConv(nn.Module):
    """(Conv => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, enable_reframe=True, reframe_cache_policy="delta"):
        super().__init__()

        # Parameters for ReFrame
        self.reframe = enable_reframe # Enable or disable ReFrame
        self.reframe_refresh_cache = True # Whether to refresh the cache; first run will always refresh
        self.reframe_cache_policy = reframe_cache_policy # Cache policy for ReFrame
        self.reframe_feature_cache = None # Cache for ReFrame features
        self.reframe_cached_input = None # Cached input for ReFrame to compute MAPE on inputs
        self.reframe_current_iteration = 0 # Current iteration for ReFrame
        self.reframe_saved_frames = 0 # Track how many frames have used ReFrame

        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    '''
    Initialize weights for the U-Net model.
    '''
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)


    '''
    Forward pass for the U-Net model for both baseline and ReFrame implementations.
    '''
    def forward(self, x):
        # Forward pass for baseline U-Net without ReFrame
        if not self.reframe:
            # Downsampling
            d1 = self.down1(x)
            d2 = self.down2(self.pool1(d1))
            d3 = self.down3(self.pool2(d2))
            bn = self.bottleneck(self.pool3(d3))

            # Upsampling
            u3 = self.up3(bn)
            u3 = torch.cat([u3, d3], dim=1) # Level 3 skip connection
            u3 = self.up_conv3(u3)

            u2 = self.up2(u3)
            u2 = torch.cat([u2, d2], dim=1) # Level 2 skip connection
            u2 = self.up_conv2(u2)

            u1 = self.up1(u2)
            u1 = torch.cat([u1, d1], dim=1) # Level 1 skip connection
            u1 = self.up_conv1(u1)

        # Forward pass with ReFrame enabled
        else:
            # Determine if we need to refresh the cache

            # Frame Delta policy
            if "delta" in self.reframe_cache_policy:
                current_input = x
                cached_input = self.reframe_cached_input

                SMAPE_THRESHOLD = 0.1  # Example threshold for MAPE
                if cached_input is None:
                    print(f"[Iter. {self.reframe_current_iteration}]: ReFrame cache REFRESH")
                    self.reframe_refresh_cache = True
                else:
                    # Calculate SMAPE
                    diff = torch.abs(current_input - cached_input).mean().item()
                    smape = 2 * diff / (torch.abs(current_input) + torch.abs(cached_input)).mean().item()
                    if smape > SMAPE_THRESHOLD:
                        print(f"[Iter. {self.reframe_current_iteration}]: ReFrame cache REFRESH")
                        self.reframe_refresh_cache = True
                    else:
                        print(f"[Iter. {self.reframe_current_iteration}]: ReFrame cache USE")
                        self.reframe_refresh_cache = False

            # Every-N policy
            elif "N" in self.reframe_cache_policy:
                every_n_iterations = int(self.reframe_cache_policy.split("-")[1])
                if self.reframe_current_iteration % every_n_iterations == 0:
                    print(f"[Iter. {self.reframe_current_iteration}]: ReFrame cache REFRESH")
                    self.reframe_refresh_cache = True
                else:
                    print(f"[Iter. {self.reframe_current_iteration}]: ReFrame cache USE")
                    self.reframe_refresh_cache = False

            # Unknown policy, always refresh
            else:
                self.reframe_refresh_cache = True


            # On a cache refresh, run network as normal but store to cache
            if self.reframe_refresh_cache:
                # Store the input for MAPE calculation
                print(f"[Iter. {self.reframe_current_iteration}]: Updating cached input...")
                self.reframe_cached_input = x.clone()

                # Downsampling
                d1 = self.down1(x)
                d2 = self.down2(self.pool1(d1))
                d3 = self.down3(self.pool2(d2))
                bn = self.bottleneck(self.pool3(d3))

                # Upsampling
                u3 = self.up3(bn)
                u3 = torch.cat([u3, d3], dim=1) # Level 3 skip connection
                u3 = self.up_conv3(u3)

                u2 = self.up2(u3)
                u2 = torch.cat([u2, d2], dim=1) # Level 2 skip connection
                u2 = self.up_conv2(u2)

                u1 = self.up1(u2)

                # Store feature to cache for ReFrame before concatenation
                print(f"[Iter. {self.reframe_current_iteration}]: Storing ReFrame features...")
                self.reframe_feature_cache = u1.clone()

                u1 = torch.cat([u1, d1], dim=1) # Level 1 skip connection
                u1 = self.up_conv1(u1)

            # Not a cache refresh, use cached features
            else:
                self.reframe_saved_frames += 1
                d1 = self.down1(x)
                # Use cached features instead of recomputing
                print(f"[Iter. {self.reframe_current_iteration}]: Using cached ReFrame features...")
                u1 = self.reframe_feature_cache
                u1 = torch.cat([u1, d1], dim=1)
                u1 = self.up_conv1(u1)

        self.reframe_current_iteration += 1
        return self.out_conv(u1)

# Example usage:
if __name__ == "__main__":
    EXAMPLE_ITERATIONS = 10

    # Set up models
    baseline_model = UNet(in_channels=1, out_channels=1, enable_reframe=False)
    baseline_model._init_weights()
    baseline_model.eval()
    
    reframe_cache_policy = "delta" # options: "delta", "N-2", "N-5", etc.
    reframe_model = UNet(in_channels=1, out_channels=1, enable_reframe=True, reframe_cache_policy=reframe_cache_policy)
    reframe_model.load_state_dict(copy.deepcopy(baseline_model.state_dict()))

    # Create a sequence of random inputs to simulate a forward pass
    input_x = [torch.randn(1, 1, 128, 128)]
    for i in range(EXAMPLE_ITERATIONS-1):
        # Modify the input slightly to simulate different frames
        new_input_x = input_x[-1] + torch.randn(1, 1, 128, 128) * 0.1 * random.uniform(0.5, 1.5)
        input_x.append(new_input_x)

    baseline_outputs = []
    start_time = time.time()
    for i in range(EXAMPLE_ITERATIONS):
        x = input_x[i]
        out = baseline_model(x)
        baseline_outputs.append(out)
    end_time = time.time()
    baseline_time = end_time - start_time

    print("Running ReFrame U-Net...")
    reframe_outputs = []
    start_time = time.time()
    for i in range(EXAMPLE_ITERATIONS):
        x = input_x[i]
        out = reframe_model(x)
        reframe_outputs.append(out)
    end_time = time.time()
    reframe_time = end_time - start_time

    print("=====================================")
    print(f"Baseline U-Net time: {baseline_time:.3f} seconds")
    print(f"ReFrame U-Net time: {reframe_time:.3f} seconds")
    print(f"ReFrame speedup: {baseline_time / reframe_time:.2f}x")
    print(f"ReFrame skipped frames: {reframe_model.reframe_saved_frames}")

    # Compute the difference between outputs
    print("\nComparing outputs:")
    largest_diff = 0
    for i in range(EXAMPLE_ITERATIONS):
        diff = torch.abs(baseline_outputs[i] - reframe_outputs[i])
        if diff.mean().item() > largest_diff:
            largest_diff = diff.mean().item()
        print(f"Output difference for frame {i}: {diff.mean().item():.2f} ({baseline_outputs[i].mean().item():.4f} vs. {reframe_outputs[i].mean().item():.4f})")
    print(f"Largest output difference: {largest_diff:.2f}")

    print("=====================================")
    print("Example complete.")