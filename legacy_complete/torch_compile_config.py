
# Fix for torch.compile graph breaks
import torch._dynamo as dynamo
dynamo.config.capture_scalar_outputs = True

# Disable C++ compilation cache to avoid zuf0 errors
import torch._inductor.config as inductor_config
inductor_config.cpp.enable_kernel_profile = False
inductor_config.triton.unique_kernel_names = True

# Set environment variables for debugging
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
