## Note
1. must set MPS for CUDA, otherwise GPU usage may explode when running multiple processes:
    ```sh
    # Check if you have permissions (may need sudo)
    nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

    # Start MPS daemon
    export CUDA_VISIBLE_DEVICES=0
    sudo nvidia-cuda-mps-control -d

    # Verify it's running
    ps aux | grep nvidia-cuda-mps
    ```