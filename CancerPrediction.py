import os
import pandas as pd
from sklearn.svm import SVC


def t_to_flux(t: dict, lb: dict, ub: dict) -> list:
    """
    This method, translates t predictions into fluxomics
    :param t: t predictions dict, in the format of {'Rxxx': value}
    :param lb: lb dict, in the format of {'Rxxx': value}
    :param ub: ub dict, in the format of {'Rxxx': value}
    :return: flux_vec: predicted fluxes list
    """
    flux_dict = {}  # predicted fluxes, in the format of {'Rxxx': value}
    for rxn_key in t.keys():
        flux_dict[rxn_key] = lb[rxn_key] + (ub[rxn_key] - lb[rxn_key]) * t[rxn_key]
    flux_vec = list(flux_dict.values())  # Note: we have assumed that keys are sorted from 'R0' to 'R8592'
    return flux_vec


def read_samples(annotation: pd.DataFrame, samples_folder: str) -> list:
    """
    This method, reads t predictions from samples_folder based on annotation
    :param annotation: Filtered annotation file, with columns = ['FileName', 'Label']
    :param samples_folder: The folder in which t predictions reside
    :return: all_samples: list of t_predictions dict
    """
    all_samples = []
    for _, sample_info in annotation.iterrows():
        sample_filepath = os.path.join(samples_folder, sample_info['FileName'])
        sample_df = pd.read_csv(sample_filepath)
        # sample_t_predictions = sample_df['t']
        ids_col_key = sample_df.columns[-2]  # 'GeneID' or 'ReactionID'
        values_col_keu = sample_df.columns[-1]  # sample_id, 'Flux', or 't'
        sample_t_predictions = dict(
            zip(sample_df[ids_col_key],
                sample_df[values_col_keu])
        )
        all_samples.append(sample_t_predictions)
    return all_samples


def load_data(tumor_annotation_filepath: str,
              normal_annotation_filepath: str,
              tumor_samples_folder: str,
              normal_samples_folder: str,
              cancer_type: str):
    """
    This method, loads positive and negative cancer-specific samples from annotation_files
    :param tumor_annotation_filepath: Annotation file of the tumor samples (or predictions)
    :param normal_annotation_filepath: Annotation file of the normals samples (or predictions)
    :param tumor_samples_folder: The folder in which tumor t predictions exist
    :param normal_samples_folder: The folder in which normal t predictions exist
    :param cancer_type: The cancer type which is desired to fit a model in it
    :return: cancer_t_samples, cancer_labels
    """
    # ##################### Loading Annotation files ########################
    tumor_annotation = pd.read_csv(tumor_annotation_filepath)
    normal_annotation = pd.read_csv(normal_annotation_filepath)
    # ##################### Filtering Annotation files ########################
    cancer_positive_annotation = tumor_annotation[tumor_annotation['Label'] == cancer_type]
    cancer_negative_annotation = normal_annotation[normal_annotation['Label'] == cancer_type]
    # ####################### Loading samples ########################
    cancer_positive_samples = read_samples(annotation=cancer_positive_annotation,
                                           samples_folder=tumor_samples_folder)
    cancer_negative_samples = read_samples(annotation=cancer_negative_annotation,
                                           samples_folder=normal_samples_folder)
    # ###################### Merging Sample #########################
    cancer_all_samples = cancer_positive_samples + cancer_negative_samples
    cancer_labels = [1] * len(cancer_positive_samples) + [0] * len(cancer_negative_samples)
    return cancer_all_samples, cancer_labels


def translate_t_predictions_to_flux(t_samples: list, lb_filepath: str, ub_filepath: str) -> list:
    """
    This method, translates elements of t_samples into flux predictions
    :param t_samples: list of t predictions dicts
    :param lb_filepath: Filepath to lb(_fva).txt
    :param ub_filepath:  Filepath to ub(_fva).txt
    :return: flux_samples
    """
    # ######## Reading lb ########
    lb = {}
    with open(lb_filepath, 'r') as file:
        lb_data = file.readlines()
        for _row in lb_data:
            _row_items = _row[:-1].split(':\t')
            lb[_row_items[0]] = float(_row_items[1])
    # ######## Reading ub ########
    ub = {}
    with open(ub_filepath, 'r') as file:
        ub_data = file.readlines()
        for _row in ub_data:
            _row_items = _row[:-1].split(':\t')
            ub[_row_items[0]] = float(_row_items[1])
    # ######## Translating t to flux ########
    flux_samples = []
    for t_sample in t_samples:
        flux_samples.append(t_to_flux(t=t_sample, lb=lb, ub=ub))
    return flux_samples


def classify(features_data: list, labels: list):
    """
    This method, fits a SVM model on (features_data, labels)
    :param features_data: X
    :param labels: y (0 or 1)
    :return: -
    """
    classifier = SVC()
    classifier.fit(X=features_data, y=labels)
    print("Prediction Accuracy: ", classifier.score(X=features_data, y=labels))


def classify_transcriptome(cancer_type: str):
    cancer_all_samples, cancer_labels = load_data(
        tumor_annotation_filepath="./Human Tumors Dataset/9264 Tumor Samples Annotation.csv",
        normal_annotation_filepath="./Human Tumors Dataset/Normal Samples Annotation.csv",
        tumor_samples_folder="./Human Tumors Dataset/9264 Tumor Samples",
        normal_samples_folder="./Human Tumors Dataset/Normal Samples",
        cancer_type=cancer_type)
    cancer_samples_list = [list(cancer_sample.values()) for cancer_sample in cancer_all_samples]
    classify(features_data=cancer_samples_list, labels=cancer_labels)


def classify_fluxome(cancer_type: str):
    cancer_all_samples, cancer_labels = load_data(
        tumor_annotation_filepath="./Human Tumors Dataset/9264 Tumor Predictions Annotation.csv",
        normal_annotation_filepath="./Human Tumors Dataset/Normal Predictions Annotation.csv",
        tumor_samples_folder="./Human Tumors Dataset/9264 Tumor Predicted Fluxes",
        normal_samples_folder="./Human Tumors Dataset/Normal Predicted Fluxes",
        cancer_type=cancer_type)
    # cancer_all_samples = translate_t_predictions_to_flux(t_samples=cancer_t_samples,
    #                                                      lb_filepath="./Data/lb_fva.txt",
    #                                                      ub_filepath="./Data/ub_fva.txt")
    cancer_samples_list = [list(cancer_sample.values()) for cancer_sample in cancer_all_samples]
    classify(features_data=cancer_samples_list, labels=cancer_labels)


def main():
    cancer_type = 'BRCA'
    classify_transcriptome(cancer_type=cancer_type)
    classify_fluxome(cancer_type=cancer_type)


if __name__ == "__main__":
    main()
