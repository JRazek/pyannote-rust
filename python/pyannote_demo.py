from pyannote.audio import Model
import numpy as np
import torch
from torch.nn.functional import normalize

import print_result

import os

token = os.getenv('TOKEN')

model = Model.from_pretrained("pyannote/segmentation", 
                              use_auth_token=token,)

with open('pcm_16le_speech.raw', 'rb') as f:
    raw_data = np.frombuffer(f.read(), dtype=np.int16)

assert raw_data.shape[0] == 80000

tensor_data = torch.tensor(raw_data).float().unsqueeze(0).unsqueeze(0)
tensor_data = normalize(tensor_data, dim=2)

#print(tensor_data.shape)  # Should print torch.Size([1, 1, 80000])

model.eval()

output = model.forward(tensor_data)

if isinstance(output, torch.Tensor):
    tensor_output = output
else:
    tensor_output, = output

print(tensor_output.shape)
#
print(tensor_output)
#
#print(model)
#
#numpy_output = tensor_output.detach().numpy()
##print(numpy_output.shape)
#
#print_result.plot_tensor(numpy_output)
#print_result.test()

import torch.onnx;

torch.onnx.export(model, tensor_data, "/home/user/rust_projects/pyannote_testing/model.onnx", opset_version=11)
