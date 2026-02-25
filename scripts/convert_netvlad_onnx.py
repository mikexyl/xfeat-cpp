import torch
from model import NetVLAD # assuming lyakaap structure

model = torch.load('checkpoints/pitts30k_vgg16_netvlad.pth.tar')
model.eval()

dummy_input = torch.randn(1, 3, 480, 640) 

torch.onnx.export(model, dummy_input, "netvlad.onnx", 
                  export_params=True, 
                  opset_version=12, 
                  do_constant_folding=True,
                  input_names=['input'], 
                  output_names=['output'])