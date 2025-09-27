# check_cuda.py
import torch
import subprocess
import os
import glob

def check_cuda_windows():
    print("=== CUDA Detection on Windows ===\n")
    
    # 1. Check PyTorch CUDA availability
    print("1. PyTorch CUDA Info:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA devices: {torch.cuda.device_count()}")
    else:
        print("   ❌ CUDA not available in PyTorch")
    
    # 2. Check common nvcc locations
    print("\n2. Searching for nvcc...")
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\CUDA",
        os.environ.get('CUDA_PATH', ''),
        os.environ.get('CUDA_PATH_V11_8', ''),
        os.environ.get('CUDA_PATH_V12_0', ''),
        os.environ.get('CUDA_PATH_V12_3', ''),
    ]
    
    for path in cuda_paths:
        if path and os.path.exists(path):
            nvcc_path = os.path.join(path, "bin", "nvcc.exe")
            if os.path.exists(nvcc_path):
                print(f"   ✅ Found nvcc at: {nvcc_path}")
                try:
                    result = subprocess.run([nvcc_path, "--version"], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        lines = result.stdout.split('\n')
                        for line in lines[:3]:  # Show first 3 lines
                            if line.strip():
                                print(f"      {line}")
                        break
                except Exception as e:
                    print(f"   ❌ Error running nvcc: {e}")
            else:
                print(f"   📁 CUDA path exists: {path} (but nvcc not found)")
    
    # 3. Check environment variables
    print("\n3. Environment Variables:")
    cuda_vars = {k: v for k, v in os.environ.items() if 'CUDA' in k.upper()}
    for var, value in cuda_vars.items():
        print(f"   {var}: {value}")
    
    # 4. Check if CUDA DLLs are available
    print("\n4. CUDA DLL Check:")
    try:
        cudart_path = glob.glob(r"C:\Windows\System32\cudart*.dll")
        if cudart_path:
            print(f"   ✅ CUDA runtime DLLs found: {len(cudart_path)}")
        else:
            print("   ❌ No CUDA runtime DLLs found in System32")
    except:
        pass

if __name__ == "__main__":
    check_cuda_windows()