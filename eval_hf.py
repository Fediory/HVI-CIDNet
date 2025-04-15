from net.CIDNet import CIDNet
import os
import json
import safetensors.torch as sf
from huggingface_hub import hf_hub_download
import argparse
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import platform
from PIL import Image

eval_parser = argparse.ArgumentParser(description='EvalHF')
eval_parser.add_argument('--path', type=str, default="Fediory/HVI-CIDNet-LOLv1-wperc", help='You can change this path to our method weights mentioned here: https://huggingface.co/papers/2502.20272.')
eval_parser.add_argument('--input_img', type=str, default="../datasets/DICM/01.jpg", help='The path of your image.')
eval_parser.add_argument('--alpha_s', type=float, default=1.0)
eval_parser.add_argument('--alpha_i', type=float, default=1.0)
eval_parser.add_argument('--gamma', type=float, default=1.0)
el = eval_parser.parse_args()

def from_pretrained(cls, pretrained_model_name_or_path: str):
    model_id = str(pretrained_model_name_or_path)

    config_file = hf_hub_download(repo_id=model_id, filename="config.json", repo_type="model")
    config = None
    if config_file is not None:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)


    model_file = hf_hub_download(repo_id=model_id, filename="model.safetensors", repo_type="model")
    # instance = sf.load_model(cls, model_file, strict=False)
    state_dict  = sf.load_file(model_file)
    cls.load_state_dict(state_dict, strict=False) 
    return cls



model = CIDNet().cuda()
model = from_pretrained(cls=model,pretrained_model_name_or_path=el.path)
model.eval()

pil2tensor = transforms.Compose([transforms.ToTensor()])
img = Image.open(el.input_img).convert('RGB')
input = pil2tensor(img)
factor = 8
h, w = input.shape[1], input.shape[2]
H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
padh = H - h if h % factor != 0 else 0
padw = W - w if w % factor != 0 else 0
input = F.pad(input.unsqueeze(0), (0,padw,0,padh), 'reflect')
with torch.no_grad():
    model.trans.alpha_s = el.alpha_s
    model.trans.alpha = el.alpha_i
    model.trans.gated = True
    model.trans.gated2 = True
    output = model(input.cuda()**el.gamma)
        

output = torch.clamp(output.cuda(),0,1).cuda()
output = output[:, :, :h, :w]
enhanced_img = transforms.ToPILImage()(output.squeeze(0))
output_folder = './output_hf'
if not os.path.exists(output_folder):          
    os.mkdir(output_folder)  
item = el.input_img
name = item.split('/')[-1]
enhanced_img.save(output_folder + "/" + name)