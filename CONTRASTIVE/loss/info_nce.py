import torch

def info_nce_loss(z1, z2, temperature=0.5):
    
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)
    
    batch_size = z1.shape[0]
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    similarity_matrix = similarity_matrix / temperature
    exp_sim_matrix = torch.exp(similarity_matrix)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(labels.device)
    exp_sim_matrix = exp_sim_matrix.masked_fill(mask, 0)

    loss = -torch.log(exp_sim_matrix / exp_sim_matrix.sum(dim=1, keepdim=True) + 1e-9)
    loss = loss * labels
    loss = loss.sum() / (2 * batch_size)
    return loss

def entropy_loss(model):
    entropy_loss = 0

    for param in model.parameters():
        
        if param.requires_grad:
            if param.numel() == 1:
                entropy_loss += torch.log(param)
            else:
                param_flat = param.view(-1)
                # prob = torch.softmax(param_flat, dim=0)
                # entropy_loss += (-1)*torch.sum(prob * torch.log(prob + 1e-8))
                dist =  torch.distributions.Categorical(logits=param_flat)

                entropy_loss += dist.entropy()

    return (-1)*entropy_loss