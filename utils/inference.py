import torch
import pandas as pd
import torch.nn.functional as F

def inference(test_loader, model, device):
    model.eval()
    with torch.no_grad():
        preds, gts, ids = [], [], []
        for i, data in enumerate(test_loader):
            inputs, masks, labels, image_ids = data
            masks = masks.type(torch.BoolTensor).to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            proba = F.softmax(outputs, dim=1)
            pred = torch.argmax(proba, dim=1)
            pred = pred.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            preds.extend(list(pred))
            gts.extend(list(labels))
            ids.extend(list(image_ids))
    res = pd.DataFrame({'image_id': ids, 'prediction': preds, 'label': gts})
    return res
