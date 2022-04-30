"""
This mother script, trains an Auto Encoder on Transcriptomics data to translate it into Fluxomixs data
    satisfying steady-state constraints
"""
import os.path
import pandas as pd
from BioNNDatasets import CustomTranscriptomicsDataset
from GPR_ConnectionsBuilder import GPRMapParser, Stoichiometry
from CustomModels import BioAE
from BioAE_Loss import AESSLoss
from NN_Learning import train, test
from torch.utils.data import DataLoader, ConcatDataset
import torch


# NUM_METs = 6124
# NUM_RXNs = 8593
# NUM_G_RXNs = 4942
# NUM_CMPs = 9136
# NUM_GENs = 1713
# Note: the reactions without any gene inside the human genome are not excluded here
# LAMBDA = 1000
LEARNING_RATE = 1e-1  # -3
WEIGHT_DECAY = 1e-8
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {DEVICE} device')


# ##################################  Loading a Dataset #####################################
def prepare_dataloaders(dataset_dir: str,
                        train_annotations_filepath: str,
                        test_annotations_filepath: str):
    """
    This method, reads BioDatasets from their pathways and returns two DataLoaders for train and test
    :param dataset_dir: The filepath for the GenesRegData.pkl
    :param train_annotations_filepath: The filepath for the train_annotations.csv
    :param test_annotations_filepath: The filepath for the test_annotations.csv
    :return: train_dataloader, test_dataloader
    """
    train_data = CustomTranscriptomicsDataset(annotations_file=train_annotations_filepath, dataset_dir=dataset_dir)
    test_data = CustomTranscriptomicsDataset(annotations_file=test_annotations_filepath, dataset_dir=dataset_dir)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader, test_dataloader
# ############################################################################################


# ##################################  Building and Training Neural Net #####################################
def make_and_save_model(train_dataloader: DataLoader,
                        test_dataloader: DataLoader,
                        stoichiometric_data: Stoichiometry,
                        gpr_info: GPRMapParser,
                        model_saving_filepath: str) -> None:
    """
    This method, handle the training of the model and saving it
    :param train_dataloader: A DataLoader containing train dataset
    :param test_dataloader: A DataLoader containing test dataset
    :param stoichiometric_data: A Stoichiometry object denoting the A and b data
    :param gpr_info: A GPRMapParser object containing GPR information
    :param model_saving_filepath: The pathway to save the model.pth
    :return: -
    """
    stoic_weights = stoichiometric_data.get_a_matrix()
    stoic_bias = stoichiometric_data.get_b_vector()
    model = BioAE(gpr_info=gpr_info,
                  stoic_weights=stoic_weights,
                  stoic_bias=stoic_bias).to(DEVICE)
    print(model)

    loss_fn = AESSLoss()  # nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, device=DEVICE)
        test(dataloader=test_dataloader, model=model, loss_fn=loss_fn, device=DEVICE)
    print("Done!")

    torch.save(model.state_dict(), model_saving_filepath)
    print("Saved PyTorch Model State to " + model_saving_filepath)


def load_and_test_model(test_dataloader: DataLoader,
                        stoichiometric_data: Stoichiometry,
                        gpr_info: GPRMapParser,
                        saved_model_filepath: str,
                        folder_to_save_predictions: str,
                        filepath_to_save_annotation: str) -> None:
    """
    This method, loads a trained model and evaluates it on test date
    :param test_dataloader: A DataLoader containing test dataset
    :param stoichiometric_data: A Stoichiometry object denoting the A and b data
    :param gpr_info: A GPRMapParser object containing GPR information
    :param saved_model_filepath: The pathway where the model.pth is saved
    :param folder_to_save_predictions: The folder path to save t-fluxes predictions
    :param filepath_to_save_annotation: The filepath to save predictions annotation .csv file
    :return: -
    """
    stoic_weights = stoichiometric_data.get_a_matrix()
    stoic_bias = stoichiometric_data.get_b_vector()
    model = BioAE(gpr_info=gpr_info,
                  stoic_weights=stoic_weights,
                  stoic_bias=stoic_bias)
    model.load_state_dict(torch.load(saved_model_filepath))
    model.eval()
    reactions_identifiers = None
    samples_stat = {}
    if not os.path.exists(folder_to_save_predictions):
        os.makedirs(folder_to_save_predictions)
    all_filenames = []
    all_labels = []
    with torch.no_grad():
        for x, cancer_types in test_dataloader:
            x = x.to(DEVICE)
            x = x.float()
            network_output = model(x)
            # print(f'Predicted: "{network_output}"')
            t_reactions_predictions = network_output['Full_Reactions'].numpy()  # len = BATCH_SIZE
            if not reactions_identifiers:
                num_reactions = len(t_reactions_predictions[0])
                reactions_identifiers = ['R' + str(i) for i in range(num_reactions)]
            # ########## Saving predictions one by one #########
            for batch in range(BATCH_SIZE):
                t_reactions_prediction = t_reactions_predictions[batch]
                # prediction_dict = dict(zip(reactions_identifiers, reactions_prediction))
                cancer_type = cancer_types[batch]
                if cancer_type in samples_stat.keys():
                    samples_stat[cancer_type] += 1
                else:
                    samples_stat[cancer_type] = 0
                sample_t_fluxome_df = pd.DataFrame({'ReactionID': reactions_identifiers,
                                                    't': t_reactions_prediction})
                """
                Note: To have access to samples names (e.g. TCGA-06-0675-11A-32R-A36H-07),
                      we should define their labels in the Annotation.csv (in the TissuesDataCreator.py file)
                      just these name, and read cancer types later during post processing.
                      For our purpose, it hasn't been necessary.
                """
                sample_filename = cancer_type + '_' + str(samples_stat[cancer_type]) + '.csv'
                sample_filepath = os.path.join(folder_to_save_predictions, sample_filename)
                sample_t_fluxome_df.to_csv(sample_filepath)
                all_filenames.append(sample_filename)
                all_labels.append(cancer_type)
    annotation_df = pd.DataFrame({'FileName': all_filenames, 'Label': all_labels})
    annotation_df.to_csv(filepath_to_save_annotation)


def main():
    # train_dataloader, test_dataloader = prepare_dataloaders(
    #     dataset_dir="./Human_Tissues_Dataset/Tissues_data/GenesRegData.pkl",
    #     train_annotations_filepath="./Human_Tissues_Dataset/train_annotations.csv",
    #     test_annotations_filepath="./Human_Tissues_Dataset/test_annotations.csv")
    tumor_dataset = CustomTranscriptomicsDataset(
        annotations_file="./Human Tumors Dataset/9264 Tumor Samples Annotation.csv",
        dataset_dir="./Human Tumors Dataset/9264 Tumor Samples")
    normal_dataset = CustomTranscriptomicsDataset(
        annotations_file="./Human Tumors Dataset/Normal Samples Annotation.csv",
        dataset_dir="./Human Tumors Dataset/Normal Samples")
    merged_dataset = ConcatDataset([tumor_dataset, normal_dataset])

    tumor_dataloader = DataLoader(tumor_dataset, batch_size=BATCH_SIZE, shuffle=True)
    normal_dataloader = DataLoader(normal_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_dataloader = DataLoader(merged_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(merged_dataset, batch_size=BATCH_SIZE, shuffle=True)

    stoichiometric_data = Stoichiometry(a_matrix_filepath="Data/A.txt",
                                        b_vector_filepath="Data/b.txt")
    gpr_info = GPRMapParser(gpr_data_filepath="./Data/Cmp_Map.txt")
    make_and_save_model(train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        stoichiometric_data=stoichiometric_data,
                        gpr_info=gpr_info,
                        model_saving_filepath="model.pth")
    load_and_test_model(test_dataloader=tumor_dataloader,
                        stoichiometric_data=stoichiometric_data,
                        gpr_info=gpr_info,
                        saved_model_filepath="model.pth",
                        folder_to_save_predictions="./Human Tumors Dataset/9264 Tumor Predicted tFluxes",
                        filepath_to_save_annotation="./Human Tumors Dataset/9264 Tumor Predictions Annotation.csv")
    load_and_test_model(test_dataloader=normal_dataloader,
                        stoichiometric_data=stoichiometric_data,
                        gpr_info=gpr_info,
                        saved_model_filepath="model.pth",
                        folder_to_save_predictions="./Human Tumors Dataset/Normal Predicted tFluxes",
                        filepath_to_save_annotation="./Human Tumors Dataset/Normal Predictions Annotation.csv")
# ############################################################################################


if __name__ == '__main__':
    main()
