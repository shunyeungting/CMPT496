import sys
import os
import platform
import pkg_resources
import shutil
import torch

def check_python_version():
    required_python_version = (3, 7)
    if sys.version_info < required_python_version:
        print(f"Error: Python {required_python_version[0]}.{required_python_version[1]} or higher is required.")
        return False
    print(f"Python version is sufficient: {sys.version}")
    return True

def check_installed_packages():
    required_packages = {
        "numpy": "1.21.0",
        "pandas": "1.3.0",
        # Add other dependencies required by manualDatagenerator
    }

    all_packages_ok = True

    for package, required_version in required_packages.items():
        try:
            pkg_resources.require(f"{package}>={required_version}")
            print(f"{package} is installed and meets the version requirement: {required_version} or higher")
        except pkg_resources.DistributionNotFound:
            print(f"Error: {package} is not installed.")
            all_packages_ok = False
        except pkg_resources.VersionConflict as e:
            print(f"Error: {package} version conflict: {e}")
            all_packages_ok = False
    
    return all_packages_ok

def check_environment_variables():
    # Example: if your program relies on specific environment variables, add checks here
    required_env_vars = []
    all_vars_ok = True

    for var in required_env_vars:
        if var not in os.environ:
            print(f"Error: Environment variable {var} is not set.")
            all_vars_ok = False
        else:
            print(f"Environment variable {var} is set.")
    
    return all_vars_ok

def check_os():
    os_info = platform.system()
    os_details = platform.platform()
    print(f"Operating System: {os_info}")
    print(f"OS Details: {os_details}")
    return os_info

def check_virtual_environment():
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Virtual environment detected.")
        return True
    else:
        print("Warning: Not using a virtual environment. It is recommended to use one.")
        return False

def check_disk_space(required_space_gb=5):
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (2**30)
    print(f"Free disk space: {free_gb} GB")
    if free_gb < required_space_gb:
        print(f"Error: Not enough disk space. At least {required_space_gb} GB is required.")
        return False
    return True

def check_network_connectivity():
    try:
        # Example of a simple check: try to reach a reliable site like Google
        import urllib.request
        urllib.request.urlopen("http://www.google.com", timeout=5)
        print("Network connectivity is available.")
        return True
    except:
        print("Warning: No network connectivity detected.")
        return False

def check_gpu_availability():
    os_info = platform.system()
    if os_info == "Darwin":  # Check if the OS is macOS
        if platform.machine() == "arm64":  # Check if it's an Apple Silicon chip (M1, M2, M3, etc.)
            print("Apple Silicon (M1/M2/M3) detected.")
            print("CUDA is not available on Apple Silicon. PyTorch will use CPU or Metal for acceleration.")
            # Here you could add a check for Metal API support if PyTorch supports it
            return True
        else:
            print("macOS detected but not running on Apple Silicon. No GPU acceleration available.")
            return False
    elif torch.cuda.is_available():
        print("GPU is available and can be used for acceleration.")
        return True
    else:
        print("GPU is not available. Training may be slower.")
        return False
    

def main():
    print("Starting environment check...\n")
    
    os_info = check_os()
    python_ok = check_python_version()
    packages_ok = check_installed_packages()
    env_vars_ok = check_environment_variables()
    venv_ok = check_virtual_environment()
    disk_ok = check_disk_space()
    network_ok = check_network_connectivity()
    gpu_ok = check_gpu_availability()
    
    if os_info == "Windows":
        print("Detected Windows OS.")
        # Possibly add other Windows-specific checks
    elif os_info == "Darwin":
        print("Detected macOS.")
        # Possibly add other macOS-specific checks
    else:
        print("Warning: Operating system not fully supported by this script.")
    
    print("\nEnvironment check complete.")
    
    if python_ok and packages_ok and env_vars_ok and venv_ok and disk_ok and network_ok and gpu_ok:
        print("Your environment is ready for running manualDatagenerator.")
    else:
        print("There are issues with your environment. Please address the errors above.")

if __name__ == "__main__":
    main()