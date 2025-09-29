import GPUtil
import torch

print("=" * 60)
print("GPU CONFIGURATION")
print("=" * 60)

# PyTorch CUDA info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"\nGPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")

    props = torch.cuda.get_device_properties(0)
    print(f"Total VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"CUDA Compute Capability: {props.major}.{props.minor}")

    # Current memory usage
    print(
        f"\nCurrent VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
    )
    print(f"Current VRAM reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    # GPUtil info
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"\nGPU {gpu.id}: {gpu.name}")
        print(f"  Memory Free: {gpu.memoryFree}MB / {gpu.memoryTotal}MB")
        print(f"  Memory Used: {gpu.memoryUsed}MB ({gpu.memoryUtil*100:.1f}%)")
        print(f"  GPU Load: {gpu.load*100:.1f}%")
        print(f"  Temperature: {gpu.temperature}°C")
else:
    print("\n⚠️ CUDA not available - PyTorch will use CPU only!")

print("=" * 60)
