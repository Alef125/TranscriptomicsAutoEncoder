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


def make_dict_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        txt_lines = f.readlines()
    map_dict = {}
    for a_line in txt_lines:
        line_sides = a_line.split("\t:")
        line_key = line_sides[0]
        map_dict[line_key] = line_sides[1][:-1]
    return map_dict


def check_free_reactions():
    rxns_map = make_dict_from_txt("Data/Reactions_Map.txt")
    with open("Data/Free_Reactions.txt", 'r') as f:
        free_rxns = f.readlines()
    for a_rxn in free_rxns:
        rxn_id = rxns_map[a_rxn[:-1]]
        rxn_object = all_reactions.get_by_id(rxn_id)
        rxn_lb = rxn_object.lower_bound
        rxn_ub = rxn_object.upper_bound
        if rxn_lb != -1000.0 or rxn_ub != 1000.0:
            print(len(rxn_object.metabolites))


def run_fva():
    solution = recon_model.optimize()
    v_max = solution.objective_value
    f = 0.9
    biomass_reaction = recon_model.reactions.get_by_id("biomass_reaction")
    fva_cons = recon_model.problem.Constraint(biomass_reaction.flux_expression, lb=f * v_max)
    recon_model.add_cons_vars(fva_cons)
    fva_lbs = []
    fva_ubs = []
    for _reaction in all_reactions:
        recon_model.objective = _reaction
        # Max
        recon_model.objective_direction = "max"
        try:
            max_sol = recon_model.optimize()
            if max_sol.status != "optimal":
                print(_reaction.id)
                raise Exception
            fva_ubs.append(max_sol.objective_value)
        except cobra.exceptions.OptimizationError:
            fva_ubs.append(1000.0)
        # Min
        recon_model.objective_direction = "min"
        try:
            min_sol = recon_model.optimize()
            if min_sol.status != "optimal":
                print(_reaction.id)
                raise Exception
            fva_lbs.append(min_sol.objective_value)
        except cobra.exceptions.OptimizationError:
            fva_lbs.append(-1000.0)
    return fva_lbs, fva_ubs


def save_bound(_bounds, save_path):
    with open(save_path, 'w') as f:
        for i in range(len(_bounds)):
            _bound = _bounds[i]
            _str = 'R' + str(i) + ':\t' + str(_bound) + '\n'
            f.writelines([_str])


def read_bound(bound_path):
    with open(bound_path, 'r') as f:
        bound_info = f.readlines()
        _bound = []
        for _rxn_txt in bound_info:
            _parts = _rxn_txt.split(':\t')
            rxn_val = _parts[1]
            _bound.append(round(float(rxn_val[:-1]), 3))
    return _bound


def read_s(s_path):
    with open(s_path, 'r') as f:
        s_info = f.readlines()
        s_matrix = np.zeros((6124, 8593))
        for element_info in s_info:
            s_parts = element_info.split(':\t')
            ind_parts = s_parts[0].split(',')
            i_ind = int(ind_parts[0][1:])
            j_ind = int(ind_parts[1][1:])
            s_val = float(s_parts[1][:-1])
            s_matrix[i_ind, j_ind] = s_val
    return s_matrix


def make_b(_lb, s_matrix):
    return s_matrix.dot(np.array(_lb))


def save_b(_b, save_path):
    with open(save_path, 'w') as f:
        for i in range(len(_b)):
            element = _b[i]
            _str = 'M' + str(i) + ':\t' + str(element) + '\n'
            f.writelines([_str])


def save_a(s_path, _lb, _ub, save_path):
    diff_bound = [_ub[i] - _lb[i] for i in range(len(_lb))]
    with open(s_path, 'r') as f:
        s_info = f.readlines()
        with open(save_path, 'w') as save_f:
            for element_info in s_info:
                s_parts = element_info.split(':\t')
                ind_parts = s_parts[0].split(',')
                # i_ind = int(ind_parts[0][1:])
                j_ind = int(ind_parts[1][1:])
                s_val = float(s_parts[1][:-1])
                a_val = s_val * diff_bound[j_ind]
                if a_val != 0.0:
                    save_str = s_parts[0] + ':\t' + str(a_val) + '\n'
                    save_f.writelines([save_str])


# lbs, ubs = run_fva()
# save_bound(lbs, "./lb_fva.txt")
# save_bound(ubs, "./ub_fva.txt")

# check_free_reactions()
lb = read_bound("Data/lb_fva.txt")
ub = read_bound("Data/ub_fva.txt")
S = read_s("Data/S.txt")
# b = make_b(lb, S)
# save_b(b, "./b_fva.txt")
# save_a("./S.txt", lb, ub, "./A_fva.txt")

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


# met_info = {}
# for i in range(len(all_metabolites)):
#     _key = 'M' + str(i)
#     met_info[_key] = 0
# for j in range(len(all_reactions)):
#     _rxn = all_reactions[j]
#     _lb = _rxn.lower_bound
#     if _lb == -math.inf:
#         _lb = -1000.0
#     for _met, val in _rxn.metabolites.items():
#         _key = 'M' + str(all_met_ids.index(_met.id))
#         met_info[_key] += val * _lb
#         the_str = 'M' + str(all_met_ids.index(_met.id)) + ',' + 'R' + str(j) + ':\t' + str(val * _lb)
#
# with open("./b.txt", 'w') as f:
#     for _key, ele in met_info.items():
#         _str = _key + ':\t' + str(ele) + '\n'
#         f.writelines([_str])

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
