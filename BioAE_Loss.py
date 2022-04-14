"""
This Script, provides useful custom losses to train BioAE network.
"""
import torch
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F


LAMBDA = 1000


class AESSLoss(Module):
    """
    Auto Encoder loss + lambda * Steady State (Sv) Loss
    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    """

    def __init__(self, reduction: str = 'mean') -> None:
        super(AESSLoss, self).__init__()
        self.reduction = reduction

    def forward(self,
                output: Tensor,
                ae_target: Tensor) -> Tensor:
        """
        Calculation of the loss
        :param output: The output of the BioAE network. Neurons 0 to num_genes-1 belong to the Auto Encoder,
                       and neurons num_genes to num_genes + num_metabolites - 1 belong to the Sv network.
                       Length = num_genes + num_metabolites
        :param ae_target: The input of the Auto Encoder (gene expressions), which also should be its output.
                          Length = num_genes
        :return: Auto Encoder loss + lambda * Steady State (Sv) Loss
        """
        # ########## Extracting Parameters ##########
        # num_metabolites = 6124
        # num_genes = 1713
        # batch_size = 4
        batch_size = output.shape[0]  # Should be equal to ae_target.shape[0]
        num_output_neurons = output.shape[1]
        num_genes = ae_target.shape[1]
        num_metabolites = num_output_neurons - num_genes
        # ########## Splitting the Output ##########
        ae_output = output[:, :num_genes]
        ss_output = output[:, num_genes:]
        # ######### Calculating AE and SS Losses ###########
        ae_loss = F.l1_loss(ae_output, ae_target, reduction=self.reduction)
        sv_zero_vec = torch.zeros((batch_size, num_metabolites))
        ss_loss = F.l1_loss(ss_output, sv_zero_vec, reduction=self.reduction)  # or torch.norm(ss_output, 1)
        # loss_weights = torch.cat((torch.ones(num_genes), LAMBDA * torch.ones(NUM_METs)), dim=0)
        # result = (loss_weights * torch.abs(output - target)).mean()
        return ae_loss + LAMBDA * ss_loss
