import torch

@torch.no_grad()
def predict(model,visual,tactile,bank,keypoints=None):
    model.eval()
    result=model(visual,tactile,bank.embeddings.to(visual.device),keypoints,return_details=True)
    result['predicted_indices']=result['scores'].argmax(-1)
    result['predicted_categories']=[bank.category_names[i] for i in result['predicted_indices'].tolist()]
    return result
