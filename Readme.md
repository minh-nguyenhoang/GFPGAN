This is the GFPGAN module taken from https://github.com/TencentARC/GFPGAN. Please refer to the original Github for license and usage.

# Installation

!git clone https://github.com/minh-nguyenhoang/gfpgan.git \\
!pip install basicsr\\
!pip install facexlib\\
!pip install -r gfpgan/requirements.txt\\

# Usage

Please refer to the original Github for more infomation, but here is an example on how to use the model

from gfpgan import GFPGANer

restorer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None)

cropped_faces, restored_faces, _ = restorer.enhance('your image')
