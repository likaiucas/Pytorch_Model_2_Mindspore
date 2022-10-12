# Pytorch_Model_2_Mindspore
## First use comapre.py to generate 2 .txt about the dictionary of 2 state_dict. 
## Then, with the help of feature from https://github.com/WinMerge/winmerge, we manually create an mapping that represents the relationship of two "theorotically" same models in physical. And create transdict.py which supports the mapping. 
## Finally, use convert_darkent.py to translate .pth to .ckpt

# some common relation:
     mindspore.ckpt <-> torch.pth
convolution: weight <-> weight
batchnorm:   beta   <-> weight
             gamma  <-> bias
             moving_mean <-> running_mean
             moving_variance <-> running_var
             None <-> num_batches_tracked
