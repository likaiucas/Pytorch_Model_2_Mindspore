import torch
import mindspore 
from mindspore import load_checkpoint
import os

def traversal_params(pth_file_path, ckpt_file_path):
    # load pth file as a dictionary
    torch_params_dict = torch.load(pth_file_path)
    # traversal a params dictionary
    for k, v in torch_params_dict.items():
        print("param_key: ", k)
        print("param_value: ", v)

    # load mindspore ckpt file as a dictionary
    mind_params_dict = load_checkpoint(ckpt_file_path)
    for k, v in mind_params_dict.items():
        print("param_key: ", k)
        print("param_value: ", v)

def list_txt(path, l=None):
    '''

    :param path: 储存list的位置
    :param list: list数据
    :return: None/relist 当仅有path参数输入时为读取模式将txt读取为list
             当path参数和list都有输入时为保存模式将list保存为txt
    '''
    if l != None:
        file = open(path, 'w')
        for n in l:
            file.write(n+'\n')
        file.close()
        return None


def traversal_params2list(pth_file_path, ckpt_file_path):
    # load pth file as a dictionary
    torch_params_dict = torch.load(pth_file_path)
    torch_params_dict = torch_params_dict['state_dict']
    # traversal a params dictionary
    torch_para = [k for k, v in torch_params_dict.items()]

    # load mindspore ckpt file as a dictionary
    mind_params_dict = load_checkpoint(ckpt_file_path)
    ms_para = [k for k, v in mind_params_dict.items()]
    list_txt(pth_file_path.replace('.pth', '.txt'), torch_para)
    list_txt(ckpt_file_path.replace('.ckpt', '.txt'), ms_para)
    return torch_para, ms_para
    
        
traversal_params2list('darknet53-a628ea1b.pth', 'myms_darknet.ckpt')