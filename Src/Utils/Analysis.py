import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np
import torchvision



# def getsize(In):

#     if isinstance(In,(list,tuple)):
#         pass
#     if isinstance(In,(OrderedDict)):
#         pass
#     if isinstance(In)
#         type(list):
#         type(tuple):
#         type(OrderedDict):pass
#         type(torchvision.models.detection.image_list.ImageList):pass
#         type(torch.Tensor):
    
width=76

def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    model.to(device)
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
 
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

           
            
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size



            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()


    print(
        "\033[5;35m# ---------------------------------------------------------------------------- #\n"+
        "#                      summary module for backbone network                     #\n"+
        "# ---------------------------------------------------------------------------- #\033[0m"
        )
    

    summary_str += "\033[36m# "+"="*width+" #" + "\033[0m\n"
    line_new = "\033[36m{:>20}  \033[35m{:>25} \033[34m{:>15}".format(
        "Layer (type)", "Output Shape", "Param")
    summary_str += line_new + "\033[0m\n"
    summary_str += "\033[36m# "+"="*width+" #" + "\033[0m\n"
    total_params = 0
    total_output = 0
    trainable_params = 0



    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "\033[36m#\033[1;36m{:>30}  \033[32m{:>25} \033[32m{:>15}\033[36m".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "     #\n"






    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size
 
    summary_str += "\033[0m\033[36m# "+"="*width+"" + " #\033[0m\n"
    summary_str += "\033[36m# Total params: \033[1;33m{0:,}".format(total_params) + " \033[0m\n"
    summary_str += "\033[36m# Trainable params: \033[1;33m{0:,}".format(trainable_params) + " \033[0m\n"
    summary_str += "\033[36m# Non-trainable params: \033[1;33m{0:,}".format(total_params -
                                                        trainable_params) + " \033[0m\n"

    summary_str += "\033[36m# "+"="*width+"" + " #\033[0m\n"

    summary_str += "\033[36m# Input size (Pixel): \033[1;33m%s" %str(input_size) + " \033[0m\n"
    summary_str += "\033[36m# Input size (MB): \033[1;33m%0.2f" % total_input_size + " \033[0m\n"
    summary_str += "\033[36m# Forward/backward pass size (MB): \033[1;33m%0.2f" % total_output_size + " \033[0m\n"
    summary_str += "\033[36m# Params size (MB):\033[1;33m %0.2f" % total_params_size + " \033[0m\n"
    summary_str += "\033[36m# Estimated Total Size (MB): \033[1;33m%0.2f" % total_size + " \033[0m\n"
    summary_str += "\033[36m# "+"="*width+"" + " #\033[0m"
   
    # ------------------------------- memorymodule ------------------------------- #

    return summary_str, (total_params, trainable_params)
    


def graminfo():
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    str_="\033[1;36m# ===== Used GRAM:\033[1;32m{used} \033[36mGB - \033[1;33m{total} \n\033[1;36m# ===== Free GRAM:\033[1;32m{free} \033[36mGB - \033[1;33m{total}".format(
        used=float(meminfo.used)/1024**3,
        total=float(meminfo.total)/1024**3,
        free=float(meminfo.free)/1024**3
    )
    str_+="\n\033[0m\033[36m# "+"="*width+"" + " #\033[0m\n"
    print(str_)






def main():
    import torchvision.models as models
    model= models.resnet101(pretrained=False)
    model.eval()
    _,total_size=summary(model,(3,512,512),batch_size=4,device="cuda:0")
    graminfo()

            
   

if __name__ == '__main__':
    main()
    