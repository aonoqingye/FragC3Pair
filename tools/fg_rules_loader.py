import os, re
from typing import List, Tuple, Dict, Iterable, Optional
from rdkit import Chem
from rdkit.Chem import RDConfig

def _read_tsv_lines(path: str) -> List[List[str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("//"):  # 跳注释
                continue
            parts = ln.split("\t")
            # 容错：去掉多余空格
            parts = [p.strip() for p in parts if p.strip()!=""]
            if parts:
                rows.append(parts)
    return rows

def load_functional_group_hierarchy(root: Optional[str]=None) -> List[Tuple[str,str]]:
    """
    解析 Functional_Group_Hierarchy.txt:
    形如:
      Name \t SMARTS \t Label [\t RemovalReaction]
    返回: [(hierName, smarts)]
    """
    if root is None:
        root = RDConfig.RDDataDir  # anaconda下通常指到 share/RDKit/Data
    path = os.path.join(root, "Functional_Group_Hierarchy.txt")
    out = []
    rows = _read_tsv_lines(path)
    for cols in rows:
        if len(cols) >= 2:
            name, smarts = cols[0], cols[1]
            # 用层次名作“可读名称”，如 "Amine.Primary.Aromatic"
            out.append((name, smarts))
    return out  # e.g. ("Amine.Primary", "…SMARTS…")
    # 参考文件内容：含 AcidChloride/CarboxylicAcid/Amine 等主类与 Aromatic/Aliphatic 子类。:contentReference[oaicite:2]{index=2}

def load_functional_groups_flat(root: Optional[str]=None) -> List[Tuple[str,str]]:
    """
    解析 FunctionalGroups.txt:
    形如:
      Label \t SMARTS \t Notes
    返回: [(label, smarts)]
    """
    if root is None:
        root = RDConfig.RDDataDir
    path = os.path.join(root, "FunctionalGroups.txt")
    out = []
    rows = _read_tsv_lines(path)
    for cols in rows:
        if len(cols) >= 2:
            label, smarts = cols[0], cols[1]
            out.append((label, smarts))
    return out
    # 参考文件内容：-NO2/-CF3/-tBu/-C(=O)O 等片段标签与SMARTS，非常适合直接命名。:contentReference[oaicite:3]{index=3}

def build_fg_smart_db(root: Optional[str]=None) -> List[Tuple[str, Chem.Mol]]:
    """
    将两份规则合并，并按“更具体优先”的顺序编译SMARTS为Mol。
    - 层次库(大类.子类) 放前面
    - 扁平库(短片段) 放后面
    - 按 SMARTS 长度降序，尽量保证“更大/更具体”的规则先匹配
    """
    h = load_functional_group_hierarchy(root)
    f = load_functional_groups_flat(root)
    merged = h + f

    # 去重：按(name, smarts)去重
    seen = set()
    uniq: List[Tuple[str,str]] = []
    for name, smarts in merged:
        key = (name, smarts)
        if key not in seen:
            seen.add(key); uniq.append(key)

    # 预编译，并按 SMARTS 文本长度降序
    compiled: List[Tuple[str, Chem.Mol]] = []
    for name, smarts in uniq:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None:
            compiled.append((name, patt))
    compiled.sort(key=lambda x: -len(Chem.MolToSmarts(x[1]) or ""))  # 更长先试
    return compiled
