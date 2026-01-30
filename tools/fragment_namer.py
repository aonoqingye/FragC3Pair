from typing import Dict, List, Tuple, Optional, Iterable
from rdkit import Chem
from rdkit.Chem import AllChem


# 一小撮常见基团/骨架 SMARTS，可自行扩充
_SMARTS_DB: List[Tuple[str, str]] = [
    ("Phenyl", "c1ccccc1"),
    ("Pyridine", "n1ccccc1"),
    ("Pyrimidine", "n1ccncc1"),
    ("Indole", "c1cc2cc[nH]c2cc1"),
    ("Aniline (Ar-NH2)", "[NX3H2]c"),
    ("Primary amine", "[NX3;H2][CX4]"),
    ("Secondary amine", "[NX3;H1]([CX4])[CX4]"),
    ("Tertiary amine", "[NX3]([CX4])([CX4])[CX4]"),
    ("Amide", "C(=O)N"),
    ("Carboxylic acid", "C(=O)[OH]"),
    ("Ester", "C(=O)O[CX4]"),
    ("Alcohol (aliph.)", "[CX4][OH]"),
    ("Phenol", "c[OH]"),
    ("Carbonyl (ketone)", "C(=O)[CX4]"),
    ("Aldehyde", "C(=O)[H]"),
    ("Nitrile", "C#N"),
    ("Halide-F", "[F]"),
    ("Halide-Cl", "[Cl]"),
    ("Halide-Br", "[Br]"),
    ("Halide-I", "[I]"),
    ("Sulfonamide", "S(=O)(=O)N"),
    ("Sulfonyl", "S(=O)(=O)"),
    ("Thiol", "[SH]"),
    ("Thioether", "S[CX4]"),
    ("Quaternary ammonium", "[NX4+]"),
]

def _submol_from_atom_indices(parent: Chem.Mol, atom_indices: Iterable[int]) -> Optional[Chem.Mol]:
    aset = list({int(a) for a in atom_indices})
    if not aset:
        return None
    rw = Chem.RWMol()
    old2new = {}
    for aidx in aset:
        a = parent.GetAtomWithIdx(int(aidx))
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        na.SetIsAromatic(a.GetIsAromatic())
        old2new[int(aidx)] = rw.AddAtom(na)
    for bond in parent.GetBonds():
        u = bond.GetBeginAtomIdx(); v = bond.GetEndAtomIdx()
        if u in old2new and v in old2new:
            rw.AddBond(old2new[u], old2new[v], bond.GetBondType())
    m = rw.GetMol()
    try:
        m.UpdatePropertyCache(strict=False); Chem.SanitizeMol(m)
    except Exception:
        pass
    try:
        AllChem.Compute2DCoords(m)
    except Exception:
        pass
    return m

def smiles_for_fragment(parent: Chem.Mol, atom_indices: Iterable[int]) -> Optional[str]:
    m = _submol_from_atom_indices(parent, atom_indices)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None

def guess_name_for_fragment(parent: Chem.Mol, atom_indices: Iterable[int]) -> Optional[str]:
    sub = _submol_from_atom_indices(parent, atom_indices)
    if sub is None:
        return None
    for name, smarts in _SMARTS_DB:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        if sub.HasSubstructMatch(patt):
            return name
    return None

def guess_name_with_rulebank(submol: Chem.Mol, rulebank: Optional[List[Tuple[str, Chem.Mol]]]) -> Optional[str]:
    if not rulebank:
        return None
    # 规则已按“更具体优先”排好序
    for name, patt in rulebank:
        if submol.HasSubstructMatch(patt):
            return name
    return None

def describe_fragment(
    parent: Chem.Mol, atom_indices: Iterable[int]
) -> Dict[str, str]:
    m = _submol_from_atom_indices(parent, atom_indices)
    smi = Chem.MolToSmiles(m, canonical=True) if m is not None else ""
    return {"smiles": smi}