import os
import platform
import subprocess
import tensorflow as tf

def setup_compute_environment():
    """
    Automatically detect available compute devices and set appropriate environment variables.
    
    Priority:
    1. NVIDIA GPUs (set CUDA_VISIBLE_DEVICES)
    2. Apple Silicon GPUs (configure for Metal)
    3. CPU fallback
    """
    
    # Check for NVIDIA GPUs first
    nvidia_gpus = detect_nvidia_gpus()
    if nvidia_gpus:
        gpu_ids = ",".join(map(str, range(len(nvidia_gpus))))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"Detected {len(nvidia_gpus)} NVIDIA GPU(s), setting CUDA_VISIBLE_DEVICES={gpu_ids}")
        for i, gpu in enumerate(nvidia_gpus):
            print(f"   GPU {i}: {gpu}")
        return "nvidia"
    
    # Check for Apple Silicon
    if is_apple_silicon():
        # For Apple Silicon, we don't need CUDA_VISIBLE_DEVICES
        # TensorFlow Metal will automatically use the GPU
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # Remove if set
        print("Detected Apple Silicon GPU, using Metal Performance Shaders")
        return "apple_metal"
    
    # CPU fallback
    os.environ['CUDA_VISIBLE_DEVICES'] = ""  # Force CPU-only
    print("No GPU detected, using CPU")
    return "cpu"

def detect_nvidia_gpus():
    """Detect NVIDIA GPUs using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        gpus = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

def is_apple_silicon():
    """Check if running on Apple Silicon (M1/M2/M3/etc)."""
    if platform.system() != 'Darwin':
        return False
    
    try:
        # Check if it's Apple Silicon
        result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
        return result.stdout.strip() == 'arm64'
    except subprocess.CalledProcessError:
        return False

def get_tensorflow_device_info():
    """Get detailed TensorFlow device information (call after TF import)."""
    try:        
        print("\nTensorFlow Device Information:")
        print(f"TensorFlow version: {tf.__version__}")
        
        # List all devices
        devices = tf.config.list_physical_devices()
        for device in devices:
            print(f"   {device}")
        
        # GPU-specific info
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\nTensorFlow detected {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    if details:
                        print(f"\tDetails: {details}")
                except:
                    pass
        else:
            print("\tNo GPUs detected by TensorFlow")
            
    except ImportError:
        print("TensorFlow not available for device detection")

# Example usage function
def configure_and_test():
    """Configure compute environment and test TensorFlow setup."""
    print("🔍 Detecting compute devices...")
    device_type = setup_compute_environment()
    
    print("\nTesting TensorFlow configuration...")
    try:        
        # Test basic TensorFlow functionality
        with tf.device('/GPU:0' if device_type in ['nvidia', 'apple_metal'] else '/CPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"TensorFlow test successful: {c.numpy()}")
        
        get_tensorflow_device_info()
        
    except Exception as e:
        print(f"TensorFlow test failed: {e}")
    
    return device_type

if __name__ == "__main__":
    configure_and_test()