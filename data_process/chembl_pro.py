import pandas as pd
import numpy as np
act_df = pd.read_csv('activity_metrix_raw.csv', index_col=0)


ATOM_SYMBOLS = ['H', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Descriptors import MolWt
from molvs import Standardizer


def smile_standard(smiles:str):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return 0
    s = Standardizer()
    if MolWt(mol) > 2000:
        return 0
    BadAtom = 0
    count = 0
    for atom in mol.GetAtoms():
        count+=1
        if (atom.GetSymbol() in ATOM_SYMBOLS) is not True:
            BadAtom = 1
            break
    if BadAtom==1 or count==1:
           return 0
    mol = s.standardize(mol)
    mol = s.fragment_parent(mol)
    try:
        remover = CHem.SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)
    finally:
        SMILES = Chem.MolToSmiles(mol)
        return SMILES


import time
all_smiles = list(act_df.index)
clean_smiles = []
start_now = time.time()
for j, i in enumerate(all_smiles):
    clean_smiles.append(smile_standard(i))
    if j%1000 == 1:
        print(f'{(time.time() - start_now) / (j+1) * (780000-j)} second left', end='\r')
act_df.index = clean_smiles
act_df.drop(index=0)
smiles_file = open('clean_chembl_smiles.txt', 'w')
smiles_file.write(str(clean_smiles))
smiles_file.close()
act_df.to_csv('activity_metrix_clean.csv')
