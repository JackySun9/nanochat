#!/usr/bin/env python3
"""
Test script to verify MPS support in nanochat
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mps_support():
    """Test MPS device detection and initialization"""
    print("Testing MPS support for nanochat...")
    
    # Test PyTorch MPS availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.backends.mps.is_available():
        print("✓ MPS is available on this system")
        
        # Test device creation
        device = torch.device("mps")
        print(f"✓ Successfully created MPS device: {device}")
        
        # Test tensor operations on MPS
        try:
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.mm(x, y)
            print(f"✓ Successfully performed matrix multiplication on MPS: {z.shape}")
        except Exception as e:
            print(f"✗ Error performing operations on MPS: {e}")
            return False
            
    else:
        print("✗ MPS is not available on this system")
        return False
    
    # Test nanochat common module
    try:
        from nanochat.common import compute_init, get_memory_usage, get_device_type
        print("✓ Successfully imported nanochat.common")
        
        # Test device initialization
        ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
        print(f"✓ Device initialization successful: {device}")
        print(f"✓ Device type: {get_device_type()}")
        print(f"✓ Memory usage: {get_memory_usage():.2f}MiB")
        
    except Exception as e:
        print(f"✗ Error testing nanochat.common: {e}")
        return False
    
    print("\n🎉 All MPS tests passed! nanochat should work on your M1 Mac Pro.")
    return True

if __name__ == "__main__":
    success = test_mps_support()
    sys.exit(0 if success else 1)
