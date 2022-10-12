import transdict
import mindspore as ms
import torch
from mindspore import load_checkpoint

torch_model = torch.load('darknet53-a628ea1b.pth')
ms_model = load_checkpoint('backbone_darknet53.ckpt')
torch_model2 =torch_model['state_dict']

def trans_para(torch_model2):
    ms_model = []
    for k, v in torch_model2.items():
        ms_k = transdict.annalysename(k)
        if ms_k:
            ms_model.append({"name":ms_k, "data":ms.Tensor(v.numpy())})
    return ms_model

ms_model=trans_para(torch_model2)
ms.save_checkpoint(ms_model, 'myms_darknet.ckpt')