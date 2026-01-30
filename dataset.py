import os
import sys
import csv
import torch
import numpy as np
from tqdm import tqdm

from itertools import islice
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset


class PairDataset(InMemoryDataset):
    def __init__(
        self,
        root="/tmp",
        dataset="",
        xd1=None,
        xd2=None,
        xt=None,
        y=None,
        xt_feature1=None,
        transform=None,
        pre_transform=None,
        smile_graph=None,
    ):
        """
        Initialization function: try to load existing cached data, otherwise execute process to create graph data.

        Parameter description:
        - xd: drug name
        - xt: cell line ID list
        - y: label list
        - xt_feature1: cell line expression feature
        - smile_graph: "SMILES → molecular graph" corresponding mapping (dict)
        """

        super(PairDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
            print("Use existing data files")
        else:
            self.process(xd1, xd2, xt, xt_feature1, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
            print("Create a new data file")

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + ".pt"]

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature1(self, cellId, cell_features):
        """Find the corresponding feature vector in xt_feature1 according to cellId"""
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return False

    def get_data(self, slice):
        d = [self.data[i] for i in slice]
        return PairDataset(d)

    """
    Customize the process method to fit the task of drug-target affinity prediction
    Inputs:
    XD - list of DRUG_NAME, XT: list of encoded target (categorical or one-hot),
    Y: list of labels (i.e. affinity)
    Return: PyTorch-Geometric format processed data 
    """

    def process(self, xd1, xd2, xt, xt_feature1, y, graph):
        assert len(xd1) == len(xd2) and len(xd2) == len(xt) and len(xt) == len(y)
        data_list = []
        slices = [0]

        # ✅ 将 expr_dict 构建移出循环，避免重复计算
        expr_dict = {row[0]: np.asarray(row[1:], dtype=float) for row in xt_feature1}

        # 进度条：总数=样本数
        pbar = tqdm(range(len(xd1)), desc="Building dataset", unit="sample")

        for i in pbar:
            drug1 = xd1[i]  # drug name
            drug2 = xd2[i]  # drug name
            target = xt[i]  # cell line ID
            labels = y[i]  # label

            data = DATA.Data()
            data.drug1_id = drug1
            data.drug2_id = drug2
            data.cell_id = target

            # Get cell line expression characteristics 1
            cell1 = expr_dict.get(target, None)
            if cell1 is None:
                pbar.close()
                print("Cell feature1 not found for target:", target)
                sys.exit()

            # Processing cell features
            data.cell1 = torch.from_numpy(cell1[None, :]).float()

            graph1 = graph[drug1]
            graph2 = graph[drug2]
            # 保持兼容：仍把 BRICS 图放到 data.graph
            data.graph1 = graph1.get('brics', None)
            data.graph1_fg = graph1.get('fg', None)
            data.graph1_murcko = graph1.get('murcko', None)
            data.graph1_ringpaths = graph1.get('ringpaths', None)
            data.graph2 = graph2.get('brics', None)
            data.graph2_fg = graph2.get('fg', None)
            data.graph2_murcko = graph2.get('murcko', None)
            data.graph2_ringpaths = graph2.get('ringpaths', None)
            data.y = torch.Tensor([labels])

            data_list.append(data)

            # 可选：每隔固定步数更新显示一些轻量信息，避免频繁 set_postfix 降速
            # if i % 200 == 0:
            #     pbar.set_postfix_str(f"last_target={str(target)[:12]}")

        pbar.close()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
