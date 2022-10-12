torch2ms_dict={
    'conv1.conv.weight':"feature_map.backbone.conv0.0.weight",
    'conv1.bn.weight':'feature_map.backbone.conv0.1.beta',
    'conv1.bn.bias':'feature_map.backbone.conv0.1.gamma',
    'conv1.bn.running_mean':'feature_map.backbone.conv0.1.moving_mean',
    'conv1.bn.running_var':'feature_map.backbone.conv0.1.moving_variance',
    'conv1.bn.num_batches_tracked':None,
    "conv_res_block{}.conv.conv.weight":'feature_map.backbone.conv{}.0.weight',
    "conv_res_block{}.conv.bn.weight":'feature_map.backbone.conv{}.1.beta',
    'conv_res_block{}.conv.bn.bias':'feature_map.backbone.conv{}.1.gamma',
    'conv_res_block{}.conv.bn.running_mean':'feature_map.backbone.conv{}.1.moving_mean',
    "conv_res_block{}.conv.bn.running_var":'feature_map.backbone.conv{}.1.moving_variance',
    "conv_res_block{}.conv.bn.num_batches_tracked":None,
    "conv_res_block{}.res{}.conv{}.conv.weight":'feature_map.backbone.layer{}.{}.conv{}.0.weight',
    'conv_res_block{}.res{}.conv{}.bn.weight':"feature_map.backbone.layer{}.{}.conv{}.1.beta",
    'conv_res_block{}.res{}.conv{}.bn.bias':'feature_map.backbone.layer{}.{}.conv{}.1.gamma',
    'conv_res_block{}.res{}.conv{}.bn.running_mean':"feature_map.backbone.layer{}.{}.conv{}.1.moving_mean",
    'conv_res_block{}.res{}.conv{}.bn.running_var':'feature_map.backbone.layer{}.{}.conv{}.1.moving_variance',
    'conv_res_block{}.res{}.conv{}.bn.num_batches_tracked':None
}

ms2torch_dict={v:k for k, v in torch2ms_dict.items() if v}

def ms2torch(namecut):
    def _5(key):
        if key[2][-1]=='0':
            return ms2torch_dict['.'.join(key)]
        else:
            m = key[2][-1]
            key[2]=key[2].replace(key[0][-1],'{}')
            return ms2torch_dict['.'.join(key)].format(m)
    def _7(key):
        m = key[2][-1]
        n = key[3][-1]
        p = key[4][-1]
        key[2]=key[2].replace(key[2][-1],'{}')
        key[3]=key[3].replace(key[3][-1],'{}')
        key[4]=key[4].replace(key[4][-1],'{}')
        return ms2torch_dict['.'.join(key)].format(m,n,p)
    fun = {
        5:_5,
        7:_7,
    }
    length=len(namecut)
    return fun[length](namecut)


def torch2ms(namecut):
    def _3(key):
        return torch2ms_dict['.'.join(key)]
    def _4(key):
        m = key[0][-1]
        key[0]=key[0].replace(key[0][-1],'{}')
        if torch2ms_dict['.'.join(key)]:
            return torch2ms_dict['.'.join(key)].format(m)
        else:
            return None
    def _5(key):
        m = key[0][-1]
        n = key[1][-1]
        p = key[2][-1]
        key[0]=key[0].replace(key[0][-1],'{}')
        key[1]=key[1].replace(key[1][-1],'{}')
        key[2]=key[2].replace(key[2][-1],'{}')
        if torch2ms_dict['.'.join(key)]:
            return torch2ms_dict['.'.join(key)].format(m, n, p)
        else:
            return None
    fun = {
        3:_3,
        4:_4,
        5:_5,
    }
    length=len(namecut)
    return fun[length](namecut)

def annalysename(key, mode = torch2ms):
    namecut = key.split('.')
    return mode(namecut)

print(annalysename('conv_res_block1.conv.conv.weight'))
print(annalysename('feature_map.backbone.layer5.5.conv5.1.moving_variance', mode = ms2torch))
