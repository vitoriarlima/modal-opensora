# Code to run on Modal.

# NOTE: This code is not working.

# Sources I am referencing:
# https://github.com/hpcaitech/Open-Sora
# https://github.com/hpcaitech/Open-Sora/blob/main/docs/installation.md
# https://github.com/hpcaitech/Open-Sora?tab=readme-ov-file#installation
# https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2
# https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3





import string
import time
from pathlib import Path
import modal

# Initialize the Modal app
app = modal.App("trying-sora-environment-build-v6")

cuda_version = "12.1.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.9",
                              force_build = True)
    .apt_install("git", 
                #  "sudo"
                 )  
    .run_commands(
        "pip install -U pip setuptools wheel",
    )
    .pip_install_from_requirements("requirements-cu121.txt",)
    .run_commands([

        "pip install tensornvme",
        "pip install git+https://github.com/vitoriarlima/Open-Sora.git",

        # "pip install torch==2.2.2 torchvision==0.17.2 xformers --index-url https://download.pytorch.org/whl/cu121",
        # "apt-get install libcudnn8", 
        # "sudo apt-get install libcudnn8 libcudnn8-dev",
                    ],
                   )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/models",
    })
)


# Define volumes for model weights and outputs
VOLUME_NAME = "opensora-outputs"
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
OUTPUTS_PATH = Path("/outputs")  # Remote path for saving video outputs

MODEL_VOLUME_NAME = "opensora-model"
model = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
MODEL_PATH = Path("/models")  # Remote path for saving model weights

# Time constants
MINUTES = 60
HOURS = 60 * MINUTES


@app.function(
    image=image,
    volumes={
        MODEL_PATH: model,
        OUTPUTS_PATH: outputs
    },
    gpu="any",  # Specify GPU requirements
    timeout=20 * MINUTES,
    # ignore_lockerfile=True,
    
)
def download_models():
    """
    Download both the VAE and STDiT models from HuggingFace.
    """

    from opensora.models.vae.vae import VideoAutoencoderPipeline

    # Initialize VAE model
    vae = VideoAutoencoderPipeline.from_pretrained(
        "hpcai-tech/OpenSora-VAE-v1.2",
        torch_dtype=torch.float16
    )


    from opensora.models.stdit.stdit3 import STDiT3
    
    # Initialize STDiT model
    stdit = STDiT3.from_pretrained(
        "hpcai-tech/OpenSora-STDiT-v3",
        torch_dtype=torch.float16
    )
    
    return "Models downloaded successfully"
