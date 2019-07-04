import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np


def summary(model, input1_size, input2_size, batch_size=-1, device="cuda", display_func=print):

    def register_hook(module):

        def hook(module_, tensor, output):
            class_name = str(module_.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(tensor[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module_, "weight") and hasattr(module_.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module_.weight.size())))
                summary[m_key]["trainable"] = module_.weight.requires_grad
            if hasattr(module_, "bias") and hasattr(module_.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module_.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input1_size, tuple):
        input1_size = [input1_size]

    if isinstance(input2_size, tuple):
        input2_size = [input2_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(batch_size, *in_size).to(device=device, dtype=torch.float) for in_size in input1_size]
    x2 = [torch.rand(batch_size, *in_size).to(device=device, dtype=torch.float) for in_size in input2_size]
    # disp_func(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # disp_func(x.shape)
    model(*x, *x2)

    # Delete unnecessary tensors. Adding this just in case.
    x.clear()
    x2.clear()

    # remove these hooks
    for h in hooks:
        h.remove()
    display_func('Model : ' + str(model.__class__).split("'")[-2])
    display_func("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    display_func(line_new)
    display_func("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        display_func(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input1_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    display_func("================================================================")
    display_func("Total params: {0:,}".format(total_params))
    display_func("Trainable params: {0:,}".format(trainable_params))
    display_func("Non-trainable params: {0:,}".format(total_params - trainable_params))
    display_func("----------------------------------------------------------------")
    display_func("Input size (MB): %0.2f" % total_input_size)
    display_func("Forward/backward pass size (MB): %0.2f" % total_output_size)
    display_func("Params size (MB): %0.2f" % total_params_size)
    display_func("Estimated Total Size (MB): %0.2f" % total_size)
    display_func("----------------------------------------------------------------")
    return summary
