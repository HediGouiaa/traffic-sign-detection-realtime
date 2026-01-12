import os
import glob
import ultralytics
import torch
if torch.cuda.is_available():
    print(" GPU is available")
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print(" GPU is NOT available")
ultralytics.checks()
home=os.getcwd()
print("Current working directory:", home)
