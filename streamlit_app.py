import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from tqdm import tqdm
import pickle
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool
import random

# Thiết lập seed để đảm bảo tính tái lặp
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

# Định nghĩa lớp MyConv
class MyConv(nn.Module):
    def __init__(self, node_dim, edge_dim, dropout_p, arch='GIN', mlp_layers=1):
        super().__init__()
        if arch == 'GIN':
            h = nn.Sequential()
            for _ in range(mlp_layers - 1):
                h.append(nn.Linear(node_dim, node_dim))
                h.append(nn.ReLU())
            h.append(nn.Linear(node_dim, node_dim))
            self.gine_conv = GINEConv(h, edge_dim=edge_dim)
            self.batch_norm = nn.BatchNorm1d(node_dim)
            self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, edge_index, edge_attr):
        x = self.gine_conv(x, edge_index, edge_attr)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

# Định nghĩa lớp MyGNN
class MyGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, dropout_p, arch='GIN', num_layers=3, mlp_layers=1):
        super().__init__()
        self.convs = nn.ModuleList(
            [MyConv(node_dim, edge_dim, dropout_p=dropout_p, arch=arch, mlp_layers=mlp_layers)
             for _ in range(num_layers)]
        )

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        return x

# Định nghĩa lớp MyFinalNetwork
class MyFinalNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, arch, num_layers, dropout_mlp, dropout_gin, embedding_dim, mlp_layers, pooling_method):
        super().__init__()
        node_dim = (node_dim - 1) + 118 + 1
        edge_dim = (edge_dim - 1) + 21 + 1

        self.gnn = MyGNN(node_dim, edge_dim, dropout_p=dropout_gin, arch=arch, num_layers=num_layers, mlp_layers=mlp_layers)

        if pooling_method == 'add':
            self.pooling_fn = global_add_pool
        elif pooling_method == 'mean':
            self.pooling_fn = global_mean_pool
        elif pooling_method == 'max':
            self.pooling_fn = global_max_pool
        else:
            raise ValueError("Phương pháp pooling không hợp lệ")

        self.head = nn.Sequential(
            nn.BatchNorm1d(node_dim),
            nn.Dropout(p=dropout_mlp),
            nn.Linear(node_dim, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(p=dropout_mlp),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x0 = F.one_hot(x[:, 0].to(torch.int64), num_classes=118+1).float()
        edge_attr0 = F.one_hot(edge_attr[:, 0].to(torch.int64), num_classes=21+1).float()
        x = torch.cat([x0, x[:, 1:]], dim=1)
        edge_attr = torch.cat([edge_attr0, edge_attr[:, 1:]], dim=1)

        node_out = self.gnn(x, edge_index, edge_attr)
        graph_out = self.pooling_fn(node_out, batch)
        return self.head(graph_out)

# Định nghĩa device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tham số mô hình
node_dim = 72
edge_dim = 14
best_params = {
    'embedding_dim': 256,
    'num_layer': 7,
    'dropout_mlp': 0.28210247642451436,
    'dropout_gin': 0.12555795277599677,
    'mlp_layers': 2,
    'pooling_method': 'mean'
}

# Tải mô hình GIN
@st.cache_resource
def load_model():
    model = MyFinalNetwork(
        node_dim=node_dim,
        edge_dim=edge_dim,
        arch='GIN',
        num_layers=best_params['num_layer'],
        dropout_mlp=best_params['dropout_mlp'],
        dropout_gin=best_params['dropout_gin'],
        embedding_dim=best_params['embedding_dim'],
        mlp_layers=best_params['mlp_layers'],
        pooling_method=best_params['pooling_method']
    ).to(device)
    with open('GIN_597_562_cpu.pkl', 'rb') as f:
        state_dict = torch.load(f, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Hàm chuẩn hóa SMILES
def standardize_smiles(batch):
    uc = rdMolStandardize.Uncharger()
    md = rdMolStandardize.MetalDisconnector()
    te = rdMolStandardize.TautomerEnumerator()

    standardized_list = []
    for smi in tqdm(batch, desc='Processing . . .'):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                Chem.SanitizeMol(mol, sanitizeOps=(Chem.SANITIZE_ALL ^ Chem.SANITIZE_CLEANUP ^ Chem.SANITIZE_PROPERTIES))
                cleanup = rdMolStandardize.Cleanup(mol)
                normalized = rdMolStandardize.Normalize(cleanup)
                uncharged = uc.uncharge(normalized)
                fragment = uc.uncharge(rdMolStandardize.FragmentParent(uncharged))
                ionized = rdMolStandardize.Reionize(fragment)
                disconnected = md.Disconnect(ionized)
                tautomer = te.Canonicalize(disconnected)
                smiles = Chem.MolToSmiles(tautomer, isomericSmiles=False, canonical=True)
                standardized_list.append(smiles)
            else:
                standardized_list.append(None)
        except Exception:
            standardized_list.append(None)
    return standardized_list

# Định nghĩa featurizer
featurizer = MultiHotAtomFeaturizer.v2()
featurizer_bond = MultiHotBondFeaturizer()

# Hàm chuyển SMILES thành dữ liệu PyTorch Geometric
def smi_to_pyg(smi, y=None):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    id_pairs = ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    atom_pairs = [z for (i, j) in id_pairs for z in ((i, j), (j, i))]

    bonds = (mol.GetBondBetweenAtoms(i, j) for (i, j) in atom_pairs)
    atom_features = [featurizer(a) for a in mol.GetAtoms()]
    bond_features = [featurizer_bond(b) for b in bonds]

    data = Data(
        edge_index=torch.LongTensor(list(zip(*atom_pairs))),
        x=torch.FloatTensor(atom_features),
        edge_attr=torch.FloatTensor(bond_features),
        mol=mol,
        smiles=smi
    )

    if y is not None:
        data.y = torch.FloatTensor([[y]])
    return data

# Định nghĩa lớp Dataset
class MyDataset(Dataset):
    def __init__(self, standardized):
        mols = [smi_to_pyg(smi, y=None) for smi in tqdm(standardized, total=len(standardized))]
        self.X = [m for m in mols if m]

    def __getitem__(self, idx):
        return self.X[idx]

    def __len__(self):
        return len(self.X)

# Giao diện Streamlit
st.title("Ứng dụng dự đoán GIN Model")

st.write("""
Ứng dụng này sử dụng mô hình Graph Isomorphism Network (GIN) để dự đoán dựa trên đầu vào SMILES.
Bạn có thể nhập SMILES thủ công hoặc tải lên tệp CSV chứa SMILES.
SMILES sẽ được chuẩn hóa, chuyển thành dữ liệu đồ thị, sau đó dự đoán bằng mô hình GIN.
""")

# Phần nhập liệu
input_type = st.radio("Chọn kiểu nhập liệu:", ("Nhập thủ công", "Tải lên CSV"))

if input_type == "Nhập thủ công":
    smiles_input = st.text_area("Nhập SMILES (mỗi dòng một SMILES):")
    smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
else:
    column_name = st.text_input("Nhập tên cột chứa SMILES trong CSV:", "smiles")
    uploaded_file = st.file_uploader("Tải lên tệp CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if column_name in df.columns:
            smiles_list = df[column_name].tolist()
        else:
            st.error(f"Không tìm thấy cột '{column_name}' trong tệp CSV.")
            st.stop()

# Nút dự đoán
if st.button("Dự đoán"):
    if not smiles_list:
        st.write("Vui lòng cung cấp đầu vào SMILES.")
    else:
        with st.spinner("Đang chuẩn hóa SMILES..."):
            standardized_smiles = standardize_smiles(smiles_list)

        # Phân loại SMILES hợp lệ và không hợp lệ
        valid_smiles = [smi for smi in standardized_smiles if smi is not None]
        valid_indices = [i for i, smi in enumerate(standardized_smiles) if smi is not None]
        invalid_smiles = [smiles_list[i] for i, smi in enumerate(standardized_smiles) if smi is None]

        if invalid_smiles:
            st.write("Các SMILES sau không hợp lệ và không thể xử lý:")
            for smi in invalid_smiles:
                st.write(smi)

        if not valid_smiles:
            st.write("Tất cả SMILES đầu vào không hợp lệ. Không thể thực hiện dự đoán.")
        else:
            with st.spinner("Đang chuyển đổi thành dữ liệu đồ thị..."):
                dataset = MyDataset(valid_smiles)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

            # Dự đoán
            with st.spinner("Đang thực hiện dự đoán..."):
                predictions = [np.nan] * len(smiles_list)
                valid_predictions = []
                for batch in dataloader:
                    batch = batch.to(device)
                    with torch.no_grad():
                        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    valid_predictions.extend(pred.cpu().numpy().flatten())

                # Ánh xạ dự đoán về danh sách ban đầu
                for idx, pred in zip(valid_indices, valid_predictions):
                    predictions[idx] = pred

            # Tạo DataFrame kết quả
            results_df = pd.DataFrame({
                'SMILES gốc': smiles_list,
                'SMILES chuẩn hóa': [smi if smi is not None else 'Không hợp lệ' for smi in standardized_smiles],
                'Dự đoán': predictions
            })

            # Hiển thị kết quả
            st.write("Kết quả dự đoán:")
            st.dataframe(results_df)

            # Nút tải xuống kết quả
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Tải xuống kết quả dưới dạng CSV",
                data=csv,
                file_name="du_doan.csv",
                mime="text/csv"
            )
