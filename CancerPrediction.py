import os
import pandas as pd
from sklearn.svm import SVC


def t_to_flux(t, lb, ub):
    """
    This method, translates t predictions into fluxomics
    :param t: t predictions
    :param lb: lb
    :param ub: ub
    :return: prediction fluxes
    """
    # ToDo
    return lb + (ub - lb) * t


def read_samples(annotation: pd.DataFrame, samples_folder: str) -> list:
    """
    This method, reads t predictions from samples_folder based on annotation
    :param annotation: Filtered annotation file, with columns = ['FileName', 'Label']
    :param samples_folder: The folder in which t predictions reside
    :return: all_samples
    """
    all_samples = []
    for _, sample_info in annotation.iterrows():
        sample_filepath = os.path.join(samples_folder, sample_info['FileName'])
        sample_df = pd.read_csv(sample_filepath)
        sample_t_predictions = sample_df['t']
        all_samples.append(sample_t_predictions)
    return all_samples


def load_data(tumor_prediction_annotation_filepath: str,
              normal_prediction_annotation: str,
              tumor_predictions_folder: str,
              normal_predictions_folder: str,
              cancer_type: str):
    """
    This method, loads positive and negative cancer-specific samples from annotation_files
    :param tumor_prediction_annotation_filepath: Annotation file of the tumor t predictions
    :param normal_prediction_annotation: Annotation file of the normals t predictions
    :param tumor_predictions_folder: The folder in which tumor t predictions exist
    :param normal_predictions_folder: The folder in which normal t predictions exist
    :param cancer_type: The cancer type which is desired to fit a model in it
    :return: cancer_t_samples, cancer_labels
    """
    # ##################### Loading Annotation files ########################
    tumor_prediction_annotation = pd.read_csv(tumor_prediction_annotation_filepath)
    normal_prediction_annotation = pd.read_csv(normal_prediction_annotation)
    # ##################### Filtering Annotation files ########################
    cancer_positive_annotation = tumor_prediction_annotation[tumor_prediction_annotation['Label'] == cancer_type]
    cancer_negative_annotation = normal_prediction_annotation[normal_prediction_annotation['Label'] == cancer_type]
    # ####################### Loading samples ########################
    cancer_positive_samples = read_samples(annotation=cancer_positive_annotation,
                                           samples_folder=tumor_predictions_folder)
    cancer_negative_samples = read_samples(annotation=cancer_negative_annotation,
                                           samples_folder=normal_predictions_folder)
    # ###################### Merging Sample #########################
    cancer_t_samples = cancer_positive_samples + cancer_negative_samples
    cancer_labels = [1] * len(cancer_positive_samples) + [0] * len(cancer_negative_samples)
    return cancer_t_samples, cancer_labels


def translate_t_predictions_to_flux(t_samples: list, lb_filepath: str, ub_filepath: str) -> list:
    """
    This method, translates elements of t_samples into flux predictions
    :param t_samples: list of t predictions
    :param lb_filepath: Filepath to lb(_fva).txt
    :param ub_filepath:  Filepath to ub(_fva).txt
    :return: flux_samples
    """
    # ToDo: parse lb_data, ub_data; make t_samples include 'Rxxx' keys
    with open(lb_filepath, 'r') as file:
        lb_data = file.readlines()
    with open(ub_filepath, 'r') as file:
        ub_data = file.readlines()
    flux_samples = []
    for t_sample in t_samples:
        flux_samples.append(t_to_flux(t=t_sample, lb=lb_data, ub=ub_data))
    return flux_samples


def classify(features_data: list, labels: list):
    """
    This method, fits a SVM model on (features_data, labels)
    :param features_data: X
    :param labels: y
    :return: -
    """
    classifier = SVC()
    classifier.fit(X=features_data, y=labels)
    # ToDo: return something! (prediction_error or accuracy on a test sample)


def main():
    cancer_t_samples, cancer_labels = load_data(
        tumor_prediction_annotation_filepath="./Human Tumors Dataset/9264 Tumor Samples Annotation.csv",
        normal_prediction_annotation="./Human Tumors Dataset/Normal Samples Annotation.csv",
        tumor_predictions_folder="./Human Tumors Dataset/9264 Tumor Predicted tFluxes",
        normal_predictions_folder="./Human Tumors Dataset/Normal Predicted tFluxes",
        cancer_type='BRCA')
    cancer_all_samples = translate_t_predictions_to_flux(t_samples=cancer_t_samples,
                                                         lb_filepath="./Data/lb_fva.txt",
                                                         ub_filepath="./Data/ub_fva.txt")
    classify(features_data=cancer_all_samples, labels=cancer_labels)


if __name__ == "__main__":
    main()
