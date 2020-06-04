import torch


def compute_accuracy(probs, target):
    r"""
    :param probs: Matrix (NxKxE), E number of nets in the ensemble
    :param target: Vector (N,)
    :return: (Float)
    """
    avg_probs_per_sample = probs.mean(-1)  # Average over the last dim which is of size E=number of models in the ensemble
    avg_pred_per_sample = avg_probs_per_sample.argmax(1)  # Accuracy computation
    overall_acc = (avg_pred_per_sample == target).float().sum().item() / len(avg_pred_per_sample)
    return overall_acc


def compute_entropy(probs):
    avg_prob_per_sample = probs.mean(-1)  # Average over the last dim which is of size E=number of models in the ensemble
    avg_ent_per_sample = torch.distributions.Categorical(probs=avg_prob_per_sample).entropy().detach().cpu().numpy()
    return avg_ent_per_sample


def compute_cross_entropy(probs, target):
    r"""
    Sends back a distribution
    """
    avg_probs_per_sample = probs.mean(
        -1)
    xe = torch.nn.CrossEntropyLoss(reduction='none')
    return xe(avg_probs_per_sample, target).detach().cpu().numpy()
