"""
This Script, provides useful custom losses to train BioAE network.
"""
import torch
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F


LAMBDA1 = 10
LAMBDA2 = 0.5


class AESSLoss(Module):
    """
    Auto Encoder loss + lambda1 * Steady State (Sv) Loss + lambda2 * Parsimony Loss
    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    Note: Based on the specific type of output parsing in forward,
          this loss should be used only with ** BioAE ** network.
    """

    def __init__(self, reduction: str = 'mean') -> None:
        super(AESSLoss, self).__init__()
        self.reduction = reduction

    def forward(self,
                network_output: Tensor,
                ae_target: Tensor) -> Tensor:
        """
        Calculation of the loss
        :param network_output: The output of the BioAE network. Neurons 0 to num_genes-1 belong to the Auto Encoder,
                       and neurons num_genes to num_genes + num_metabolites - 1 belong to the Sv network.
                       Length = num_genes + num_metabolites
        :param ae_target: The input of the Auto Encoder (gene expressions), which also should be its output.
                          Length = num_genes
        :return: Auto Encoder loss + lambda * Steady State (Sv) Loss + lambda2 * Parsimony Loss
        """
        # ########## Extracting Parameters ##########
        # batch_size = network_output.shape[0]  # Should be equal to ae_target.shape[0]
        # num_output_neurons = network_output.shape[1]
        # num_genes = ae_target.shape[1]
        # num_metabolites = num_output_neurons - num_genes
        # ########## Splitting the Output ##########
        # ae_output = network_output[:, :num_genes]
        # ss_output = network_output[:, num_genes:]
        ae_output = network_output['Decoded']
        ss_output = network_output['Metabolites']
        rxn_output = network_output['Full_Reactions']
        # ######### Calculating AE, SS, and Parsimony Losses ###########
        ae_loss = F.l1_loss(ae_output, ae_target, reduction=self.reduction)
        # ###########
        # sv_zero_vec = torch.zeros((batch_size, num_metabolites)))
        ss_loss = F.mse_loss(ss_output, torch.zeros(ss_output.shape), reduction=self.reduction)
        # ss_loss = torch.norm(ss_output, 1) / (ss_output.shape[0] * ss_output.shape[1])
        # ############
        # reactions_zero_vec = torch.zeros((batch_size, num_reactions))
        parsimony_loss = F.l1_loss(rxn_output, torch.zeros(rxn_output.shape), reduction=self.reduction)
        # parsimony_loss = torch.norm(rxn_output, 1)/ (rxn_output.shape[0] * rxn_output.shape[1])
        # ############
        # loss_weights = torch.cat((torch.ones(num_genes), LAMBDA * torch.ones(NUM_METs)), dim=0)
        # total_loss = (loss_weights * torch.abs(output - target)).mean()
        total_loss = ae_loss + LAMBDA1 * ss_loss + LAMBDA2 * parsimony_loss
        return total_loss


class ImplicitAELoss(Module):
    """
    Auto Encoder loss + lambda1 * Bounds Loss + lambda2 * Parsimony Loss
    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    Note: Based on the specific type of output parsing in forward,
          this loss should be used only with ** ImplicitBioAE ** network.
    """
    def __init__(self, reduction: str = 'mean') -> None:
        super(ImplicitAELoss, self).__init__()
        self.reduction = reduction

    def forward(self,
                network_output: Tensor,
                ae_target: Tensor) -> Tensor:
        """
        Calculation of the loss
        :param network_output: The output of the ImplicitBioAE network. It is a dict in the format of
                               {'Decoded': decoded, 'Projected_Reactions': projected_reactions}
        :param ae_target: The input of the Auto Encoder (gene expressions), which also should be its output.
                          Length = num_genes
        :return: Auto Encoder loss + lambda1 * Bounds Loss + lambda2 * Parsimony Loss
        """
        ae_output = network_output['Decoded']
        rxn_output = network_output['Projected_Reactions']
        bounds_exceed_output = network_output['DeadZone_Bounds']
        # ######### Calculating AE, Bounds, and Parsimony Losses ###########
        ae_loss = F.l1_loss(ae_output, ae_target, reduction=self.reduction)
        # ###########
        bounds_loss = F.l1_loss(bounds_exceed_output, torch.zeros(bounds_exceed_output.shape), reduction=self.reduction)
        # ############
        parsimony_loss = F.l1_loss(rxn_output, torch.zeros(rxn_output.shape), reduction=self.reduction)
        # ############
        total_loss = ae_loss + LAMBDA1 * bounds_loss + LAMBDA2 * parsimony_loss
        return total_loss
