"""
This script, provides custom Layers and Network Models required for our
    Transcriptomics to Fluxomics translation task.
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter
import math
import numpy as np
from GPR_ConnectionsBuilder import GPRMapParser


# ######################### Custom Layers #############################
class GPRLayer(nn.Module):
    """
    This module, provides a masked linear layer, i.e., output = (weight * mask) @ input + bias.
    The mask defines the connection pattern between input and output nodes,
        and is used for inducing GPR rules on the meaningful nodes of genes, complexes, and reactions.
    """
    def __init__(self,
                 connections_matrix: np.ndarray,
                 device=None,
                 dtype=None):
        """
        :param connections_matrix: The 0/1 matrix (np array) defining the connection pattern between input and output
        :param device: cpu, gpu
        :param dtype: e.g.: torch.float
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GPRLayer, self).__init__()
        self.input_dim = connections_matrix.shape[1]
        self.output_dim = connections_matrix.shape[0]
        self._connections_mask = torch.tensor(data=connections_matrix,
                                              device=device,
                                              dtype=torch.float,
                                              requires_grad=False)
        self._weights = Parameter(torch.empty((self.output_dim, self.input_dim), **factory_kwargs))
        self.bias = Parameter(torch.empty(self.output_dim, **factory_kwargs))
        self.reset_parameters()
        # self.masked_weights = torch.mul(self._weights, self._connections_mask)  # Elementwise Multiplication

    def reset_parameters(self) -> None:
        """
        Initializing self._weights and self.bias.
        Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        https://github.com/pytorch/pytorch/issues/57109
        """
        init.kaiming_uniform_(self._weights, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.input_dim)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, _input: Tensor) -> Tensor:
        return F.linear(input=_input, weight=self._weights * self._connections_mask, bias=self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_dim, self.output_dim, self.bias is not None
        )


class FixedLayer(nn.Module):
    """
    This module, provides a linear layer with pre-defined weights and biases.
    This layer is primarily used for the Sv=At+b layer.
    """
    def __init__(self,
                 weights: np.ndarray,
                 bias: np.ndarray,
                 device=None):
        """
        :param weights: Fixes linear weights
        :param bias: Fixed linear biases
        :param device: cpu, gpu
        """
        super(FixedLayer, self).__init__()
        self.input_dim = weights.shape[1]
        self.output_dim = weights.shape[0]
        self.weight = torch.tensor(data=weights, dtype=torch.float, device=device, requires_grad=False)
        self.bias = torch.tensor(data=bias, dtype=torch.float, device=device, requires_grad=False)

    def forward(self, _input: Tensor) -> Tensor:
        return F.linear(input=_input, weight=self.weight, bias=self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_dim, self.output_dim, self.bias is not None
        )
# #####################################################################


# #################################  Custom Model ####################################
class BioAE(nn.Module):
    """
    This module is the complete Auto Encoder used for translating Transcriptomics data into Fluxomics data,
        which satisfies the steady-state conditions (Sv=0).
    Note: Because of the specific output format of forward (as a dict),
          only use AESSLoss as this network's loss.
    """
    def __init__(self,
                 gpr_info: GPRMapParser,
                 stoic_weights: np.ndarray,
                 stoic_bias: np.ndarray):
        """
        :param gpr_info: A GPRMapParser object to make desired GPRLayers
        :param stoic_weights: Fixed weight matrix (A=S.diag(u-l)) for the Steady State layer; in At+b=0
        :param stoic_bias: Fixed bias vector (b=Sl) for the Steady State layer; in At+b=0
        """
        super(BioAE, self).__init__()
        genes_to_complexes_connection_matrix = gpr_info.make_genes_to_complexes_connection_matrix()
        complexes_to_reactions_connection_matrix = gpr_info.make_complexes_to_reactions_connection_matrix()
        # ###################### Encoder ########################
        self.encoder = nn.Sequential(
            GPRLayer(connections_matrix=genes_to_complexes_connection_matrix),  # Genes to Cmps
            nn.ReLU(),
            GPRLayer(connections_matrix=complexes_to_reactions_connection_matrix),  # Cmps to gene_Reactions
            nn.ReLU()
        )
        # ###################### Decoder ########################
        self.decoder = nn.Sequential(
            GPRLayer(connections_matrix=complexes_to_reactions_connection_matrix.transpose()),  # gene_Reactions to Cmps
            nn.ReLU(),
            GPRLayer(connections_matrix=genes_to_complexes_connection_matrix.transpose()),  # Cmps to Genes
            nn.Sigmoid()
        )
        # ############## Gene-Associated Reactions -- to --> All Reactions ##############
        num_g_reactions = gpr_info.get_num_g_reactions()
        num_all_reactions = stoic_weights.shape[1]
        if num_all_reactions != gpr_info.get_num_all_reactions():
            raise ValueError("Number of Reactions are not compatible in your GPR and Stoichiometry files.")
        self.fill_reactions = nn.Sequential(
            nn.Linear(in_features=num_g_reactions, out_features=num_all_reactions),  # gene_Reactions to Reactions
            nn.Sigmoid()  # Based on our implementation, It is crucial for us to have numbers in [0,1] here
        )
        # ################### Reactions -- to --> Metabolites ##################
        self.steady_state_net = FixedLayer(weights=stoic_weights, bias=stoic_bias)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        filled_reactions = self.fill_reactions(encoded)
        metabolites = self.steady_state_net(filled_reactions)
        # output = torch.cat((decoded, metabolites), dim=1)
        output = {'Decoded': decoded, 'Metabolites': metabolites, 'Full_Reactions': filled_reactions}
        return output
# ##############################################################################


# ###############################  Implicit Constraint Model ##################################
class ImplicitBioAE(nn.Module):
    """
    This module is the Auto Encoder used for translating Transcriptomics data into Fluxomics data,
        with implicit steady-state conditions (Sv=0).
    Note: Because of the specific output format of forward (as a dict),
          only use ImplicitAELoss as this network's loss.
    """
    def __init__(self,
                 gpr_info: GPRMapParser,
                 stoic_kernel_projector: np.ndarray):
        """

        :param gpr_info: A GPRMapParser object to make desired GPRLayers
        :param stoic_kernel_projector: Fixed weight matrix (P = N.NT) for the Projection layer
        """
        super(ImplicitBioAE, self).__init__()
        genes_to_complexes_connection_matrix = gpr_info.make_genes_to_complexes_connection_matrix()
        complexes_to_reactions_connection_matrix = gpr_info.make_complexes_to_reactions_connection_matrix()
        # ###############################################
        num_g_reactions = gpr_info.get_num_g_reactions()
        num_all_reactions = stoic_kernel_projector.shape[0]  # = stoic_kernel_projector.shape[1]
        if num_all_reactions != gpr_info.get_num_all_reactions():
            raise ValueError("Number of Reactions are not compatible in your GPR and Stoichiometry files.")
        # ###################### Encoder ########################
        self.encoder = nn.Sequential(
            GPRLayer(connections_matrix=genes_to_complexes_connection_matrix),  # Genes to Cmps
            nn.ReLU(),
            GPRLayer(connections_matrix=complexes_to_reactions_connection_matrix),  # Complexes to Enzymes
            nn.ReLU(),
            nn.Linear(in_features=num_g_reactions, out_features=num_all_reactions)  # Enzymes to Full Reactions
            # If using nn.Sigmoid(), use l+(u-l)x after it and if using nn.ReLU(), use x+l after it.
        )
        # ###################### Projection ########################
        self.projector = FixedLayer(weights=stoic_kernel_projector, bias=np.zeros(num_all_reactions))
        # ###################### Decoder ########################
        self.decoder = nn.Sequential(
            nn.Linear(in_features=num_all_reactions, out_features=num_g_reactions),  # Full Reactions to Enzymes
            nn.ReLU(),
            GPRLayer(connections_matrix=complexes_to_reactions_connection_matrix.transpose()),  # Enzymes to Complexes
            nn.ReLU(),
            GPRLayer(connections_matrix=genes_to_complexes_connection_matrix.transpose()),  # Complexes to Genes
            nn.ReLU()
        )
        # ###############################################

    def forward(self, x):
        encoded = self.encoder(x)
        projected_reactions = self.projector(encoded)
        decoded = self.decoder(projected_reactions)
        # output = torch.cat((decoded, projected), dim=1)
        output = {'Decoded': decoded, 'Projected_Reactions': projected_reactions}
        return output
# ##############################################################################
