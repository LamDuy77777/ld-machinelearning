import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from tqdm import tqdm
import pickle
import xgboost as xgb
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool
import random
from rdkit import DataStructs

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

# Tắt thông báo lỗi từ RDKit
RDLogger.DisableLog('rdApp.*')

# Định nghĩa device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                st.write(f"Invalid SMILES: {smi}")
        except Exception as e:
            st.write(f"An error occurred with SMILES {smi}: {str(e)}")
            standardized_list.append(None)
    return standardized_list

# Hàm chuyển đổi SMILES thành đặc trưng Morgan fingerprint cho XGBoost
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fp)
    else:
        return np.zeros(2048)

# Hàm chuyển đổi SMILES thành ECFP4 cho AD
def smiles_to_ecfp4(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return fp

# Hàm tính Tanimoto Distance
def tanimoto_distance(fp1, fp2):
    return 1 - DataStructs.TanimotoSimilarity(fp1, fp2)

# Lớp AD để tính toán điểm SDC
class AD:
    def __init__(self, train_data, nBits=2048, radius=2):
        self.train_data = train_data
        self.nBits = nBits
        self.radius = radius
        self.train_fps = None

    def fit(self):
        self.train_fps = []
        for smiles in tqdm(self.train_data, desc="Processing training SMILES"):
            fp = smiles_to_ecfp4(smiles, self.radius, self.nBits)
            if fp is not None:
                self.train_fps.append(fp)
            else:
                st.write(f"Invalid SMILES in training data: {smiles}")

    def get_score(self, smiles):
        test_fp = smiles_to_ecfp4(smiles, self.radius, self.nBits)
        if test_fp is None:
            return np.nan
        sdc = 0.0
        for train_fp in self.train_fps:
            td = tanimoto_distance(test_fp, train_fp)
            if td >= 1:
                continue
            exponent = -3 * td / (1 - td)
            sdc += np.exp(exponent)
        return sdc if sdc > 0 else np.nan

# Định nghĩa lớp MyConv cho GIN
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
        edge_dim = (node_dim - 1) + 21 + 1
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

# Tải mô hình XGBoost
@st.cache_resource
def load_xgb_model():
    with open('xgboost_binary_10nM.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Tải mô hình GIN
@st.cache_resource
def load_gin_model():
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
        state_dict = torch.load(f, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Tải và khởi tạo AD
@st.cache_resource
def load_ad_model():
    train_df = pd.read_csv('data_AD_classification_streamlit.csv')
    train_smiles = train_df['standardized'].drop_duplicates().tolist()
    ad = AD(train_data=train_smiles)
    ad.fit()
    return ad

# Hàm chuyển SMILES thành dữ liệu PyTorch Geometric cho GIN
featurizer = MultiHotAtomFeaturizer.v2()
featurizer_bond = MultiHotBondFeaturizer()

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

# Định nghĩa lớp Dataset cho GIN
class MyDataset(Dataset):
    def __init__(self, standardized):
        mols = [smi_to_pyg(smi, y=None) for smi in tqdm(standardized, total=len(standardized))]
        self.X = [m for m in mols if m]
    def __getitem__(self, idx):
        return self.X[idx]
    def __len__(self):
        return len(self.X)

# Giao diện Streamlit
st.title("Dự đoán với Mô hình XGBoost và GIN (bao gồm Miền Ứng dụng)")

st.write("""
Ứng dụng này sử dụng mô hình XGBoost để phân loại SMILES (0 hoặc 1) và mô hình GIN để dự đoán giá trị pEC50 cho tất cả SMILES hợp lệ.
Kết quả bao gồm miền ứng dụng (AD) với ngưỡng SDC = 7.019561595570336e-06, phân loại dự đoán là "Reliable" hoặc "Unreliable".
Bạn có thể nhập SMILES thủ công hoặc tải lên tệp CSV chứa SMILES.
""")

# Phần nhập liệu
input_type = st.radio("Chọn kiểu nhập liệu:", ("Nhập thủ công", "Tải lên CSV"))

if input_type == "Nhập thủ công":
    smiles_input = st.text_area("Nhập SMILES (mỗi dòng một SMILES):")
    smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
else:
    column_name = st.text_input("Nhập tên cột chứa SMILES trong CSV:", "SMILES")
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
        valid_pairs = [(orig, std) for orig, std in zip(smiles_list, standardized_smiles) if std is not None]
        invalid_smiles = [smiles_list[i] for i, smi in enumerate(standardized_smiles) if smi is None]

        if invalid_smiles:
            st.write("Các SMILES sau không hợp lệ và không thể xử lý:")
            for smi in invalid_smiles:
                st.write(smi)

        if not valid_pairs:
            st.write("Tất cả SMILES đầu vào không hợp lệ. Không thể thực hiện dự đoán.")
        else:
            valid_orig, valid_std = zip(*valid_pairs)

            # Dự đoán phân loại với XGBoost
            with st.spinner("Đang thực hiện dự đoán phân loại với XGBoost..."):
                xgb_features = [smiles_to_features(smi) for smi in valid_std]
                xgb_model = load_xgb_model()
                xgb_predictions = xgb_model.predict(np.array(xgb_features))

            # Tính toán SDC cho AD
            with st.spinner("Đang tính toán miền ứng dụng (SDC)..."):
                ad_model = load_ad_model()
                sdc_scores = [ad_model.get_score(smi) for smi in valid_std]
                ad_labels = ["Reliable" if score >= 7.019561595570336e-06 else "Unreliable" for score in sdc_scores]

            # Dự đoán pEC50 với GIN cho tất cả SMILES hợp lệ
            with st.spinner("Đang chuyển đổi thành dữ liệu đồ thị cho GIN..."):
                dataset = MyDataset(valid_std)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

            with st.spinner("Đang thực hiện dự đoán pEC50 với GIN..."):
                gin_model = load_gin_model()
                gin_predictions = []
                for batch in dataloader:
                    batch = batch.to(device)
                    with torch.no_grad():
                        pred = gin_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    gin_predictions.extend(pred.cpu().numpy().flatten())

            # Tạo DataFrame kết quả
            result_df = pd.DataFrame({
                'SMILES gốc': valid_orig,
                'SMILES chuẩn hóa': valid_std,
                'Dự đoán XGBoost': xgb_predictions,
                'Dự đoán pEC50 (GIN)': gin_predictions,
                'Applicability Domain': ad_labels
            })

            # Hiển thị kết quả
            st.write("Kết quả dự đoán cho tất cả SMILES hợp lệ:")
            st.dataframe(result_df)

            # Nút tải xuống kết quả
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Tải xuống kết quả dưới dạng CSV",
                data=csv,
                file_name="du_doan.csv",
                mime="text/csv"
            )
