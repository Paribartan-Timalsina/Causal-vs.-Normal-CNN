import torch
import torch.nn.functional as F

def predict(model, img_tensor, device):
    """
    Make prediction on an image tensor.
    
    Args:
        model: The neural network model
        img_tensor: Image tensor (can be (3,28,28) or (1,3,28,28))
        device: torch device
    
    Returns:
        predicted: Predicted digit
        confidence: Confidence percentage
        probs: Probability distribution over all digits
    """
    with torch.no_grad():
        # Ensure tensor has batch dimension
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        prob = F.softmax(output, dim=1)
        confidence, predicted = torch.max(prob, 1)

    return predicted.item(), confidence.item() * 100, prob.cpu().numpy()[0]
