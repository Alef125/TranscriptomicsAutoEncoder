"""
This script, organizes the tissues_data into distinct files (to be further read by torch.Dataset),
    and translates the gene_ids into the format defined in the MetabolicModelParser.py
"""

import cobra
# from cobra.flux_analysis import flux_variability_analysis
# import math
# from gtfparse import read_gtf
import json
import pandas as pd
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense

# load the dataset
recon_model = cobra.io.read_sbml_model("Data/recon_2.2.xml")
#
all_reactions = recon_model.reactions
# all_metabolites = recon_model.metabolites
# all_met_ids = [_met.id for _met in all_metabolites]

"""
df = read_gtf("./gencode.v38.annotation.gtf")

print(df.columns)
> Index(['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand',
       'frame', 'gene_id', 'gene_type', 'gene_name', 'level', 'hgnc_id',
       'havana_gene', 'transcript_id', 'transcript_type', 'transcript_name',
       'transcript_support_level', 'tag', 'havana_transcript', 'exon_number',
       'exon_id', 'ont', 'protein_id', 'ccdsid'],
      dtype='object')
      
print(df.iloc[5])
> seqname                                                   chr1
source                                                  HAVANA
feature                                             transcript
start                                                    12010
end                                                      13670
score                                                      NaN
strand                                                       +
frame                                                        0
gene_id                                      ENSG00000223972.5
gene_type                   transcribed_unprocessed_pseudogene
gene_name                                              DDX11L1
level                                                        2
hgnc_id                                             HGNC:37102
havana_gene                               OTTHUMG00000000961.2
transcript_id                                ENST00000450305.2
transcript_type             transcribed_unprocessed_pseudogene
transcript_name                                    DDX11L1-201
transcript_support_level                                    NA
tag                                    basic,Ensembl_canonical
havana_transcript                         OTTHUMT00000002844.2
exon_number                                                   
exon_id                                                       
ont                                    PGO:0000005,PGO:0000019
protein_id                                                    
ccdsid                                                        
Name: 5, dtype: object
"""

def prepare_regulations_data():
    with open("Data/Genes_Translation.json", 'r') as f:
        map1 = json.load(f)
    with open("Data/Genes_index_Map.json", 'r') as f:
        map2 = json.load(f)
    t_data = pd.read_csv("Data/Tissue_Data.csv")
    cols = ['genes', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14',
            'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T24', 'T25', 'T26', 'T27', 'T28',
            'T29', 'T30', 'T31', 'T32']
    t_cols = ['adipose tissue fpkm', 'adrenal gland fpkm', 'appendix fpkm', 'bone marrow fpkm', 'brain fpkm',
              'colon fpkm', 'duodenum fpkm', 'endometrium fpkm', 'esophagus fpkm', 'fallopian tube fpkm',
              'gallbladder fpkm', 'heart muscle fpkm', 'kidney fpkm', 'liver fpkm', 'lung fpkm', 'lymph node fpkm',
              'ovary fpkm', 'pancreas fpkm', 'placenta fpkm', 'prostate fpkm', 'rectum fpkm', 'salivary gland fpkm',
              'skeletal muscle fpkm', 'skin fpkm', 'small intestine fpkm', 'smooth muscle fpkm', 'spleen fpkm',
              'stomach fpkm', 'testis fpkm', 'thyroid gland fpkm', 'tonsil fpkm', 'urinary bladder fpkm']
    my_data = {}
    for _col in cols:
        my_data[_col] = []
    for index, row in t_data.iterrows():
        en_id = row['ensg_id']
        if en_id in map1.keys():
            key1 = map1[en_id]
            if key1 in map2.keys():
                g_name = map2[key1]
                tissues_info = row[t_cols]
                my_data['genes'].append(g_name)
                for cnt in range(1, len(cols)):
                    my_data[cols[cnt]].append(tissues_info[cnt-1])
            else:
                print(key1)
        else:
            print(en_id)
    my_df = pd.DataFrame(my_data)
    my_df.to_pickle("./GenesRegData.pkl")


labels_num = list(range(29, 33))
labels = ['T' + str(_label) for _label in labels_num]
data = {'file_name': [_label + '.pkl' for _label in labels], 'label': labels}
# Create DataFrame
df = pd.DataFrame(data)
df.to_csv("./Human_Tissues_Dataset/test_annotations.csv", header=False, index=False)


# df = read_gtf("gencode.v38.annotation.gtf")
# with open("./hgnc_complete_set.json", 'r') as f:
#     df = json.load(f)
# my_map = {}
# for _gene in df['response']['docs']:
#     if 'hgnc_id' in _gene.keys() and 'ensembl_gene_id' in _gene.keys():
#         xx = _gene['ensembl_gene_id']
#         yy = _gene['hgnc_id']
#         my_map[xx] = yy
#
# with open("./Genes_Translation.json", 'w') as f:
#     json.dump(my_map, f)

# filter DataFrame to gene entries on chrY
# df_genes = df[df["feature"] == "gene"]
# print(df)

# dataset = openpyxl.load_workbook("./1260419__excel_tabless1-s18.xlsx")
# Sheets: "S1. All genes", "S2. Tissue enriched", "S3. Group enriched", "S4. Enhanced", "S5. Missing genes",
#         "S6. GO analysis", "S7. SP-TM", "S8. Transcription factors", "S9. Druggable proteins",
#         "S10. Cancer genes", "S11. FPKM Cell-lines", "S12. Genes with different splic",
#         "S13.", "S14.", "S15.", "S16", "S17.", "S18. Full FPKM dataset, tissues"

# dataset_sheet = dataset.active
# for row in dataset_sheet.iter_rows(max_row=6):
#     for cell in row:
#         print(cell.value, end=" ")
#     print()
