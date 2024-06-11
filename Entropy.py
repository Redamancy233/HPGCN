import numpy as np
import torch
device = torch.device("cuda:0")
def EntropySampling(model, uldata, classNum):
    n = classNum * 2
    model.eval()
    out = torch.tensor([]).cuda()
    with torch.no_grad():
        for inputs, _ in uldata:
            inputs = inputs.cuda()
            output, _ = model(inputs)
            out = torch.cat((out, output), 0)
    out = torch.clamp(out, 1e-7)
    log_probs = torch.log(out)
    U = (out * log_probs).sum(1)
    selected = U.sort()[1][:n]
    selectidx = selected.detach().cpu().numpy()
    return selectidx