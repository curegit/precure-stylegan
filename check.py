from chainer import cuda
from chainer import print_runtime_info

# Show chainer environment
print_runtime_info()
print(f"CUDA: {'Available' if cuda.available else 'Not Available'}")
print(f"cuDNN: {'Available' if cuda.cudnn_enabled else 'Not Available'}")
