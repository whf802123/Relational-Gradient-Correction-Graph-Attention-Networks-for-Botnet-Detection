import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

CSV_PATH = 'C:\\Users\\whf80\\Desktop\\DW-GAT\\ICASSP\\CTU13.csv'
WINDOW_SIZE = 1000
BATCH_SIZE = 100
EVOLVE_EPOCHS_FIRST = 10
EVOLVE_EPOCHS_INC = 2
EVOLVE_HIDDEN_CHANNELS = 32
HIDDEN_CHANNELS = 8
CORR_THRESHOLD = 0.1
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
LR = 0.005
WEIGHT_DECAY = 0.0
SEED = 42
ENABLE_WINDOW_ANALYSIS = True
WINDOW_METRIC_AVERAGE = 'weighted'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
if not 0 < BATCH_SIZE <= WINDOW_SIZE:
    raise ValueError('Require 0 < BATCH_SIZE <= WINDOW_SIZE.')
if WINDOW_SIZE % BATCH_SIZE != 0:
    raise ValueError('WINDOW_SIZE must be divisible by BATCH_SIZE to avoid skipped initial samples.')
if not np.isclose(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO, 1.0):
    raise ValueError('Train/validation/test ratios must sum to 1.')
df = pd.read_csv(CSV_PATH)
if len(df) < WINDOW_SIZE:
    raise ValueError('Dataset is smaller than WINDOW_SIZE; reduce WINDOW_SIZE and BATCH_SIZE.')
if 'Label' not in df.columns:
    raise KeyError("Column 'Label' was not found. Please confirm that the CTU13 CSV label column is named Label.")
feature_cols = [c for c in df.columns if c not in ('num', 'Label')]
if len(feature_cols) == 0:
    raise RuntimeError('No usable feature columns were found.')
labels = df['Label'].astype(int).to_numpy(dtype=np.int64)
unique_labels = np.unique(labels)
if len(unique_labels) != 2 or set(unique_labels.tolist()) != {0, 1}:
    raise RuntimeError(f'This code expects binary CTU13 labels 0/1, but found {unique_labels.tolist()}.')
label_to_id = {'Normal': 0, 'Botnet': 1}
id_to_label = {0: 'Normal', 1: 'Botnet'}
feat_df = df[feature_cols].apply(pd.to_numeric, errors='coerce')
feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
N_total = len(df)
all_idx = np.arange(N_total)
(train_idx, holdout_idx) = train_test_split(all_idx, test_size=VALIDATION_RATIO + TEST_RATIO, stratify=labels, random_state=SEED, shuffle=True)
(val_idx, test_idx) = train_test_split(holdout_idx, test_size=TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO), stratify=labels[holdout_idx], random_state=SEED, shuffle=True)
medians = feat_df.iloc[train_idx].median(numeric_only=True)
feat_df = feat_df.fillna(medians).fillna(0.0)
scaler = StandardScaler(with_mean=True, with_std=True)
features = np.empty_like(feat_df.values, dtype=np.float64)
features[train_idx] = scaler.fit_transform(feat_df.iloc[train_idx].values.astype(float))
features[test_idx] = scaler.transform(feat_df.iloc[test_idx].values.astype(float))
features[val_idx] = scaler.transform(feat_df.iloc[val_idx].values.astype(float))
N = len(features)
is_train = np.zeros(N, dtype=bool)
is_train[train_idx] = True
is_test = np.zeros(N, dtype=bool)
is_test[test_idx] = True
is_val = np.zeros(N, dtype=bool)
is_val[val_idx] = True
print(f'Data split: train={len(train_idx)} ({TRAIN_RATIO:.0%}), validation={len(val_idx)} ({VALIDATION_RATIO:.0%}), test={len(test_idx)} ({TEST_RATIO:.0%})')
train_unique_ids = np.unique(labels[train_idx])
NUM_CLASSES = len(train_unique_ids)
if NUM_CLASSES != 2:
    raise RuntimeError('The training set does not contain both Normal and Botnet classes.')
print('CTU13 label mapping:', id_to_label)
print(f'Samples: total={N_total}, train={len(train_idx)}, test={len(test_idx)}')
print(f'Features: {len(feature_cols)}')

def _safe_row_corrcoef(x_np: np.ndarray) -> np.ndarray:
    if x_np.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    if x_np.shape[0] == 1:
        return np.ones((1, 1), dtype=np.float64)
    corr = np.corrcoef(x_np)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr

def build_train_graph(x_train_np: np.ndarray):
    n = x_train_np.shape[0]
    if n == 0:
        return torch.empty((2, 0), dtype=torch.long, device=DEVICE)
    corr = _safe_row_corrcoef(x_train_np)
    adj = np.abs(corr) >= CORR_THRESHOLD
    np.fill_diagonal(adj, False)
    (src, dst) = np.where(adj)
    self_nodes = np.arange(n, dtype=np.int64)
    src = np.concatenate([src.astype(np.int64), self_nodes])
    dst = np.concatenate([dst.astype(np.int64), self_nodes])
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long, device=DEVICE)
    return edge_index

def build_inductive_eval_graph(x_train_np: np.ndarray, x_test_np: np.ndarray):
    n_train = x_train_np.shape[0]
    n_test = x_test_np.shape[0]
    n_total = n_train + n_test
    if n_test == 0:
        return (None, None)
    x_eval_np = np.concatenate([x_train_np, x_test_np], axis=0)
    corr = _safe_row_corrcoef(x_eval_np)
    src_list = []
    dst_list = []
    if n_train > 0:
        corr_tt = corr[:n_train, :n_train]
        adj_tt = np.abs(corr_tt) >= CORR_THRESHOLD
        np.fill_diagonal(adj_tt, False)
        (src_tt, dst_tt) = np.where(adj_tt)
        src_list.append(src_tt.astype(np.int64))
        dst_list.append(dst_tt.astype(np.int64))
        corr_train_test = corr[:n_train, n_train:]
        (train_src, test_col) = np.where(np.abs(corr_train_test) >= CORR_THRESHOLD)
        if train_src.size > 0:
            src_list.append(train_src.astype(np.int64))
            dst_list.append((n_train + test_col).astype(np.int64))
    self_nodes = np.arange(n_total, dtype=np.int64)
    src_list.append(self_nodes)
    dst_list.append(self_nodes)
    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long, device=DEVICE)
    x_eval_tensor = torch.tensor(x_eval_np, dtype=torch.float, device=DEVICE)
    return (x_eval_tensor, edge_index)

class EvolveGCNHLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.summary_score = nn.Parameter(torch.empty(in_channels))
        self.input_kernels = nn.Parameter(torch.empty(3, in_channels, in_channels))
        self.state_kernels = nn.Parameter(torch.empty(3, in_channels, in_channels))
        self.gate_bias = nn.Parameter(torch.zeros(3, in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.initial_weight)
        nn.init.uniform_(self.summary_score, -self.in_channels ** (-0.5), self.in_channels ** (-0.5))
        for gate in range(3):
            nn.init.xavier_uniform_(self.input_kernels[gate])
            nn.init.xavier_uniform_(self.state_kernels[gate])
        nn.init.zeros_(self.gate_bias)

    def summarize(self, x):
        if x.ndim != 2 or x.size(1) != self.in_channels or x.size(0) == 0:
            raise ValueError('EvolveGCN-H requires a nonempty [nodes, features] tensor.')
        score_direction = self.summary_score / self.summary_score.norm().clamp_min(1e-08)
        scores = x @ score_direction
        k = min(self.out_channels, x.size(0))
        selected = torch.topk(scores, k=k, sorted=True).indices
        if k < self.out_channels:
            selected = torch.cat([selected, selected[-1:].expand(self.out_channels - k)])
        selected_x = x.index_select(0, selected)
        gates = scores.index_select(0, selected).tanh().unsqueeze(1)
        return (selected_x * gates).transpose(0, 1)

    def evolve(self, x, previous_weight=None):
        old_weight = self.initial_weight if previous_weight is None else previous_weight
        if old_weight.shape != self.initial_weight.shape:
            raise ValueError('The temporal weight state has an incompatible shape.')
        summary = self.summarize(x)
        input_terms = torch.matmul(self.input_kernels, summary) + self.gate_bias
        recurrent_terms = torch.matmul(self.state_kernels[:2], old_weight)
        (update, reset) = torch.sigmoid(input_terms[:2] + recurrent_terms).unbind(0)
        candidate = torch.tanh(input_terms[2] + self.state_kernels[2] @ (reset * old_weight))
        return old_weight + update * (candidate - old_weight)

def normalized_gcn_adjacency(edge_index, num_nodes, dtype, device):
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
    (src, dst) = edge_index
    adjacency[dst, src] = 1.0
    inverse_sqrt_degree = adjacency.sum(dim=1).clamp_min(1.0).rsqrt()
    return inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]

class EvolveGCNClassifier(nn.Module):

    def __init__(self, in_channels, first_hidden_channels, hidden_channels, num_classes=2):
        super().__init__()
        self.egcn1 = EvolveGCNHLayer(in_channels, first_hidden_channels)
        self.egcn2 = EvolveGCNHLayer(first_hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, previous_state=None, *, evolve=True, return_state=False):
        if previous_state is None:
            if not evolve:
                raise ValueError('Evaluation requires committed EvolveGCN weights.')
            previous_state = (None, None)
        if len(previous_state) != 2:
            raise ValueError('Expected one temporal weight matrix per GCN layer.')
        adjacency = normalized_gcn_adjacency(edge_index, x.size(0), x.dtype, x.device)
        weight1 = self.egcn1.evolve(x, previous_state[0]) if evolve else previous_state[0]
        h1 = F.elu(adjacency @ (x @ weight1))
        weight2 = self.egcn2.evolve(h1, previous_state[1]) if evolve else previous_state[1]
        hidden = F.elu(adjacency @ (h1 @ weight2))
        logits = self.classifier(hidden)
        if return_state:
            return (logits, hidden, (weight1, weight2))
        return (logits, hidden)

def detach_temporal_state(state):
    return tuple((weight.detach().clone() for weight in state))
features_window = deque(maxlen=WINDOW_SIZE)
labels_window = deque(maxlen=WINDOW_SIZE)
index_window = deque(maxlen=WINDOW_SIZE)
model = None
optimizer = None
criterion = None
temporal_state = None
if NUM_CLASSES == 2:
    train_labels_only = labels[train_idx]
    pos_ratio = (train_labels_only == 1).mean() + 1e-08
    w_neg = 1.0 / max(1e-08, 1.0 - pos_ratio)
    w_pos = 1.0 / max(1e-08, pos_ratio)
else:
    w_neg = w_pos = 1.0
y_true_test = []
y_pred_test = []
y_prob_test_all = []
hidden_test = []
window_metrics = []
global_idx = 0
stream_window_id = 0

def select_training_indices(n_nodes, new_train_mask_local, first_window=False):
    if n_nodes == 0:
        return None
    if first_window:
        return torch.arange(n_nodes, device=DEVICE)
    new_idx = torch.where(new_train_mask_local)[0]
    return new_idx if new_idx.numel() > 0 else None

def _loss_on_used_nodes(logits, y_train_local, used_idx):
    if used_idx is None or used_idx.numel() == 0:
        return None
    return criterion(logits[used_idx], y_train_local[used_idx])
for start in tqdm(range(0, N, BATCH_SIZE), desc='EvolveGCN-H batches'):
    batch_feats = features[start:start + BATCH_SIZE]
    batch_labels = labels[start:start + BATCH_SIZE]
    bsz = len(batch_feats)
    for i in range(bsz):
        features_window.append(batch_feats[i])
        labels_window.append(int(batch_labels[i]))
        index_window.append(global_idx)
        global_idx += 1
    if len(features_window) < WINDOW_SIZE:
        continue
    first_window = model is None
    x_win_np = np.asarray(features_window, dtype=np.float64)
    y_win_np = np.asarray(labels_window, dtype=np.int64)
    idx_win_np = np.asarray(index_window, dtype=np.int64)
    new_mask_full = np.zeros(WINDOW_SIZE, dtype=bool)
    if first_window:
        new_mask_full[:] = True
    else:
        new_mask_full[-bsz:] = True
    train_mask_full = is_train[idx_win_np]
    test_mask_full = is_test[idx_win_np]
    x_train_np = x_win_np[train_mask_full]
    y_train_np = y_win_np[train_mask_full]
    new_train_mask_local_np = new_mask_full[train_mask_full]
    if x_train_np.shape[0] == 0:
        continue
    x_train_tensor = torch.tensor(x_train_np, dtype=torch.float, device=DEVICE)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.long, device=DEVICE)
    new_train_mask_local = torch.tensor(new_train_mask_local_np, dtype=torch.bool, device=DEVICE)
    train_edge_index = build_train_graph(x_train_np)
    if model is None:
        model = EvolveGCNClassifier(in_channels=x_train_tensor.shape[1], hidden_channels=HIDDEN_CHANNELS, first_hidden_channels=EVOLVE_HIDDEN_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        if NUM_CLASSES == 2:
            class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float, device=DEVICE)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        epochs_now = EVOLVE_EPOCHS_FIRST
    else:
        epochs_now = EVOLVE_EPOCHS_INC
    previous_state = temporal_state
    for _ in range(epochs_now):
        model.train()
        used_idx = select_training_indices(x_train_tensor.size(0), new_train_mask_local, first_window=first_window)
        if used_idx is None:
            continue
        (logits_train, _hidden_train) = model(x_train_tensor, train_edge_index, previous_state=previous_state)
        ce_loss = _loss_on_used_nodes(logits_train, y_train_tensor, used_idx)
        if ce_loss is None:
            continue
        loss = ce_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        (_, _, current_state) = model(x_train_tensor, train_edge_index, previous_state=previous_state, return_state=True)
        temporal_state = detach_temporal_state(current_state)
    if first_window:
        eval_test_mask_full = test_mask_full
    else:
        eval_test_mask_full = new_mask_full & test_mask_full
    if eval_test_mask_full.any():
        x_new_test_np = x_win_np[eval_test_mask_full]
        y_new_test_np = y_win_np[eval_test_mask_full]
        (x_eval_tensor, eval_edge_index) = build_inductive_eval_graph(x_train_np, x_new_test_np)
        model.eval()
        with torch.no_grad():
            (logits_eval, hidden_eval) = model(x_eval_tensor, eval_edge_index, previous_state=temporal_state, evolve=False)
            probs_eval = torch.softmax(logits_eval, dim=1)
            n_train_ctx = x_train_np.shape[0]
            test_slice = slice(n_train_ctx, n_train_ctx + len(x_new_test_np))
            logits_test = logits_eval[test_slice]
            probs_test = probs_eval[test_slice]
            hidden_test_batch = hidden_eval[test_slice]
            window_pred = torch.argmax(logits_test, dim=1).cpu().numpy()
            y_true_test.extend(y_new_test_np.tolist())
            y_pred_test.extend(window_pred.tolist())
            y_prob_test_all.append(probs_test.cpu().numpy())
            hidden_test.extend(hidden_test_batch.cpu().numpy().tolist())
    if ENABLE_WINDOW_ANALYSIS and test_mask_full.any():
        x_window_test_np = x_win_np[test_mask_full]
        y_window_test_np = y_win_np[test_mask_full]
        (x_window_eval_tensor, window_eval_edge_index) = build_inductive_eval_graph(x_train_np, x_window_test_np)
        model.eval()
        with torch.no_grad():
            (window_logits_eval, _) = model(x_window_eval_tensor, window_eval_edge_index, previous_state=temporal_state, evolve=False)
            n_train_ctx = x_train_np.shape[0]
            window_logits_test = window_logits_eval[n_train_ctx:n_train_ctx + len(x_window_test_np)]
            window_pred_full = torch.argmax(window_logits_test, dim=1).cpu().numpy()
        window_metrics.append({'window': stream_window_id, 'stream_start': int(idx_win_np[0]), 'stream_end': int(idx_win_np[-1]), 'n_test': int(len(y_window_test_np)), 'accuracy': float(accuracy_score(y_window_test_np, window_pred_full)), 'precision': float(precision_score(y_window_test_np, window_pred_full, average=WINDOW_METRIC_AVERAGE, zero_division=0)), 'recall': float(recall_score(y_window_test_np, window_pred_full, average=WINDOW_METRIC_AVERAGE, zero_division=0)), 'f1': float(f1_score(y_window_test_np, window_pred_full, average=WINDOW_METRIC_AVERAGE, zero_division=0))})
    stream_window_id += 1
y_true_test = np.asarray(y_true_test, dtype=np.int64)
y_pred_test = np.asarray(y_pred_test, dtype=np.int64)
if len(y_prob_test_all) > 0:
    y_prob_test_all = np.vstack(y_prob_test_all)
else:
    y_prob_test_all = np.empty((0, NUM_CLASSES), dtype=np.float64)
print('\n=== Evaluation on Random Test Split (15%) ===')
print(f'Expected hold-out samples: {len(test_idx)}')
print(f'Actually evaluated samples: {len(y_true_test)}')
if len(y_true_test) == 0:
    print('The test set is empty. Check the data split or window settings.')
else:
    unique_ids = np.arange(NUM_CLASSES)
    target_names = [id_to_label[i] for i in unique_ids]
    report = classification_report(y_true_test, y_pred_test, output_dict=True, digits=4, labels=unique_ids, target_names=target_names, zero_division=0)
    print('Classification Report (Hold-out Test):')
    for lab in target_names:
        m = report[lab]
        print(f"{lab:<28} precision: {m['precision'] * 100:.2f}%  recall: {m['recall'] * 100:.2f}%  f1-score: {m['f1-score'] * 100:.2f}%")
    print(f"{'accuracy':<28}: {accuracy_score(y_true_test, y_pred_test) * 100:.2f}%")
    for k in ['macro avg', 'weighted avg']:
        m = report[k]
        print(f"{k:<28} precision: {m['precision'] * 100:.2f}%  recall: {m['recall'] * 100:.2f}%  f1-score: {m['f1-score'] * 100:.2f}%")
    try:
        labels_order = unique_ids.tolist()
        display_names = [id_to_label[i] for i in labels_order]
        cm = confusion_matrix(y_true_test, y_pred_test, labels=labels_order)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
        (fig_cm, ax_cm) = plt.subplots(figsize=(5.6, 5.0))
        disp.plot(values_format='d', cmap='Blues', colorbar=False, ax=ax_cm)
        ax_cm.set_xlabel('Predicted label')
        ax_cm.set_ylabel('True label')
        plt.setp(ax_cm.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        fig_cm.tight_layout()
        plt.show()
        plt.close(fig_cm)
    except Exception as e:
        print('Error while plotting the confusion matrix:', e)
    try:
        if NUM_CLASSES > 2:
            if y_prob_test_all.shape[0] == 0:
                print('[ROC] No test probabilities were collected. Skipping ROC plotting.')
            else:
                classes_sorted = unique_ids.tolist()
                y_true_bin = label_binarize(y_true_test, classes=classes_sorted)
                plt.figure(figsize=(7.2, 5.4))
                plotted = False
                for cls_id in classes_sorted:
                    col = classes_sorted.index(cls_id)
                    y_true_c = y_true_bin[:, col]
                    y_score_c = y_prob_test_all[:, cls_id]
                    if y_true_c.sum() == 0 or y_true_c.sum() == len(y_true_c):
                        continue
                    (fpr, tpr, _) = roc_curve(y_true_c, y_score_c)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{id_to_label[cls_id]} (AUC={roc_auc:.3f})')
                    plotted = True
                if plotted:
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.legend(fontsize=8, loc='lower right')
                    plt.tight_layout()
                    plt.show()
                else:
                    print('[ROC] No class has both positive and negative test samples. Multiclass ROC cannot be plotted.')
        elif np.unique(y_true_test).size == 2:
            (fpr, tpr, _) = roc_curve(y_true_test, y_prob_test_all[:, 1])
            roc_auc = auc(fpr, tpr)
            (fig_roc, ax_roc) = plt.subplots(figsize=(7.2, 5.4))
            ax_roc.plot(fpr, tpr, label=f'Botnet (AUC={roc_auc:.3f})')
            ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('EvolveGCN-H ROC')
            ax_roc.legend(loc='lower right')
            fig_roc.tight_layout()
            plt.show()
            plt.close(fig_roc)
        else:
            print('[ROC] Both classes are required for binary ROC.')
    except Exception as e:
        print('Error while computing or plotting ROC:', e)
if ENABLE_WINDOW_ANALYSIS:
    if len(window_metrics) > 0:
        window_df = pd.DataFrame(window_metrics)
        window_df = window_df.sort_values('window').reset_index(drop=True)
        print('\n=== Window-level Performance ===')
        print(f'Evaluated windows: {len(window_df)}')
        print(f"Mean test samples/window: {window_df['n_test'].mean():.2f}")
        print(f"Mean window Accuracy: {window_df['accuracy'].mean() * 100:.2f}%")
        print(f"Mean window F1 ({WINDOW_METRIC_AVERAGE}): {window_df['f1'].mean() * 100:.2f}%")
        (fig_wp, ax_wp) = plt.subplots(figsize=(7.2, 4.6))
        ax_wp.plot(window_df['window'], window_df['accuracy'] * 100.0, linewidth=1.5, label='Accuracy')
        ax_wp.plot(window_df['window'], window_df['f1'] * 100.0, linewidth=1.5, label='F1-score')
        ax_wp.set_xlabel('Window Index')
        ax_wp.set_ylabel('Performance (%)')
        ax_wp.set_title('Window-level Performance during Incremental Learning')
        ax_wp.set_ylim(0.0, 101.0)
        ax_wp.grid(True, alpha=0.3)
        ax_wp.legend()
        fig_wp.tight_layout()
        plt.show()
        plt.close(fig_wp)
    else:
        print('\n[Window Analysis] No window-level test samples were available.')
TSNE_MAX_PER_CLASS = 1000
try:
    if len(hidden_test) > 10:
        hidden_test_np = np.asarray(hidden_test, dtype=np.float64)
        labels_tsne = np.asarray(y_true_test[:len(hidden_test_np)], dtype=np.int64)
        print('\n=== t-SNE Diagnostics ===')
        print(f'Raw embedding shape: {hidden_test_np.shape}')
        finite_mask = np.isfinite(hidden_test_np).all(axis=1)
        if not finite_mask.all():
            bad = int((~finite_mask).sum())
            print(f'[t-SNE] Dropped {bad} samples containing NaN/Inf.')
            hidden_test_np = hidden_test_np[finite_mask]
            labels_tsne = labels_tsne[finite_mask]
        if len(hidden_test_np) <= 10:
            print('t-SNE: Too few valid test embeddings. Skipping visualization.')
        else:
            dim_std = hidden_test_np.std(axis=0)
            useful_dims = dim_std > 1e-10
            if useful_dims.sum() < 2:
                print('t-SNE: Fewer than two informative embedding dimensions remain. Reliable visualization is not possible.')
            else:
                if useful_dims.sum() != hidden_test_np.shape[1]:
                    removed = hidden_test_np.shape[1] - int(useful_dims.sum())
                    print(f'[t-SNE] Removed {removed} near-constant dimensions.')
                x_tsne = hidden_test_np[:, useful_dims]
                x_tsne = StandardScaler().fit_transform(x_tsne)
                x_tsne = np.nan_to_num(x_tsne, nan=0.0, posinf=10.0, neginf=-10.0)
                x_tsne = np.clip(x_tsne, -10.0, 10.0)
                rng = np.random.default_rng(SEED)
                selected = []
                for cls_id in np.unique(labels_tsne):
                    cls_idx = np.where(labels_tsne == cls_id)[0]
                    if len(cls_idx) > TSNE_MAX_PER_CLASS:
                        cls_idx = rng.choice(cls_idx, size=TSNE_MAX_PER_CLASS, replace=False)
                    selected.append(np.asarray(cls_idx, dtype=np.int64))
                selected = np.concatenate(selected)
                selected.sort()
                x_tsne = x_tsne[selected]
                labels_plot = labels_tsne[selected]
                n_tsne = len(x_tsne)
                perplexity = min(30.0, max(5.0, (n_tsne - 1) / 3.0))
                perplexity = min(perplexity, float(n_tsne - 1))
                print(f'Samples plotted: {n_tsne}')
                print(f'Input dimensions after filtering: {x_tsne.shape[1]}')
                print(f'Perplexity: {perplexity:.2f}')
                emb_2d = TSNE(n_components=2, random_state=SEED, perplexity=perplexity, init='pca', learning_rate='auto').fit_transform(x_tsne)
                print(f'2D range: x=[{emb_2d[:, 0].min():.3f}, {emb_2d[:, 0].max():.3f}], y=[{emb_2d[:, 1].min():.3f}, {emb_2d[:, 1].max():.3f}]')
                (fig_tsne, ax_tsne) = plt.subplots(figsize=(7.0, 6.0))
                for cls_id in np.unique(labels_plot):
                    mask = labels_plot == cls_id
                    ax_tsne.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=12, alpha=0.75, label=id_to_label.get(int(cls_id), str(cls_id)))
                ax_tsne.set_xlabel('t-SNE Dim 1')
                ax_tsne.set_ylabel('t-SNE Dim 2')
                ax_tsne.set_title('t-SNE of EvolveGCN-H Test Embeddings')
                ax_tsne.legend(fontsize=7, markerscale=1.5, loc='best')
                fig_tsne.tight_layout()
                plt.show()
                plt.close(fig_tsne)
    else:
        print('t-SNE: Too few test embeddings. Skipping visualization.')
except Exception as e:
    print('t-SNE visualization failed:', e)
