import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression
import cv2
from model.clip import build_model
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, GraphNorm, SAGEConv
from torch_geometric.data import Batch, Data
from .layers import FPN, Projector, TransformerDecoder, MultiTaskProjector

class ComplexGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(ComplexGCN, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.norm1 = GraphNorm(hidden_dim * num_heads)
        self.dim_match1 = nn.Linear(input_dim, hidden_dim * num_heads)

        self.sage = SAGEConv(hidden_dim * num_heads, hidden_dim)
        self.norm2 = GraphNorm(hidden_dim)
        self.dim_match2 = nn.Linear(hidden_dim * num_heads, hidden_dim)

        self.gcn = GCNConv(hidden_dim, hidden_dim)
        self.norm3 = GraphNorm(hidden_dim)

        # self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # GAT + 残差连接
        x_residual = self.dim_match1(x)
        x = self.gat1(x, edge_index)
        x = self.norm1(x)
        x = F.leaky_relu(x) + x_residual

        # GraphSAGE + 残差连接
        x_residual = self.dim_match2(x)
        x = self.sage(x, edge_index)
        x = self.norm2(x)
        x = F.leaky_relu(x) + x_residual

        # GCN + 残差连接
        x_residual = x  # 假设维度匹配
        x = self.gcn(x, edge_index)
        x = self.norm3(x)
        x = F.leaky_relu(x) + x_residual

        # 输出层
        # x = self.fc(x)

        return x
class CROG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Flags for ablation study
        self.use_contrastive = cfg.use_contrastive
        self.use_pretrained_clip = cfg.use_pretrained_clip
        self.use_grasp_masks = cfg.use_grasp_masks
        
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        print(f"Load pretrained CLIP: {self.use_pretrained_clip}")
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len, self.use_pretrained_clip).float()
        # Multi-Modal FPN
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.yolo_model = DetectMultiBackend(cfg.yolo_weights, device=self.device, dnn=False,
                                             data='yolov5/data/coco128.yaml', fp16=False)
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        self.yolo_model.to(self.device)
        # 加载预训练的 GCN 模型
        self.gcn_model = ComplexGCN(input_dim=512, hidden_dim=512, output_dim=512).to(self.device)
        self.gcn_model.load_state_dict(torch.load(cfg.gcn_weights, map_location=self.device))
        for param in self.gcn_model.parameters():
            param.requires_grad = False
        self.gcn_model.to(self.device)

        self.feature_reducer = nn.Linear(1024, 512)
        self.fc_reduce = nn.Linear(692224, 512).to(self.device)
        self.num_nodes = 31

        # Decoder
        if self.use_contrastive:
            print("Use contrastive learning module")
            # Decoder
            self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                            d_model=cfg.vis_dim,
                                            nhead=cfg.num_head,
                                            dim_ffn=cfg.dim_ffn,
                                            dropout=cfg.dropout,
                                            return_intermediate=cfg.intermediate)
        else:
            print("Disable contrastive learning module")
        if self.use_grasp_masks:
            # Projector
            print("Use grasp masks")
            self.proj = MultiTaskProjector(cfg.word_dim, cfg.vis_dim // 2, 3)
        else:
            print("Disable grasp masks")
            self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)
    def apply_attention_to_vis(self, vis, gcn_output, bboxes):
        """
        将 GCN 输出作为注意力权重应用到 fq 特征图中。
        """
        for i, bbox in enumerate(bboxes):
            if i >= gcn_output.size(0):
                break
            x1, y1, x2, y2 = bbox
            # 计算缩放因子
            scale_x = 52 / 640
            scale_y = 52 / 480

            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)

            # 确保宽高有效
            if x2 <= x1 or y2 <= y1:
                continue  # 跳过无效的 bbox

            # 将节点特征转换为注意力权重
            attention_map = gcn_output[i].view(1, -1, 1, 1)
            attention_map = F.interpolate(attention_map, size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False)
            attention_map = torch.sigmoid(attention_map)

            updated_region = vis[0][:, :, y1:y2, x1:x2] * (1 + attention_map)
            # 将 vis 转换为 list 以便进行修改
            vis = list(vis)
            # 进行修改
            vis[0] = vis[0].clone()  # 克隆以避免 in-place 修改
            vis[0][:, :, y1:y2, x1:x2] = updated_region
            # 如果后续代码需要 vis 是 tuple 类型，则转换回 tuple
            vis = tuple(vis)

        return vis
    def get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def pad_node_features(self, node_features, max_nodes=31, feature_dim=512):
        """
        Pads node features to ensure there are exactly `max_nodes`.
        """
        num_nodes = node_features.size(0)
        if num_nodes < max_nodes:
            padding_size = max_nodes - num_nodes
            padding = torch.zeros((padding_size, feature_dim), device=node_features.device)
            padded_features = torch.cat([node_features, padding], dim=0)
        else:
            padded_features = node_features[:max_nodes]
        return padded_features

    def pad_edge_index(self, edge_index, max_nodes=31, max_edges=30):
        if edge_index.numel() == 0:
            # 当 edge_index 为空时，返回填充的张量
            return torch.zeros((2, max_edges), dtype=torch.long, device=edge_index.device)

        num_edges = edge_index.size(1)
        if num_edges < max_edges:
            padding_size = max_edges - num_edges
            padding = torch.zeros((2, padding_size), dtype=torch.long, device=edge_index.device)
            padded_edge_index = torch.cat([edge_index, padding], dim=1)
        else:
            padded_edge_index = edge_index[:, :max_edges]
        return padded_edge_index

    def build_edges(self, bboxes):
        """
                基于物体的相对位置（左右或上下，保留差距较大的边）构建边，并为每条边附加方向标签
                - bboxes: 检测框列表，格式为 [[x1, y1, x2, y2], ...]
                返回 edge_index 和 edge_attr，分别表示边索引和边的方向标签。
                """
        centers = [self.get_center(bbox) for bbox in bboxes]  # 提取每个检测框的中心点
        centers = np.array(centers)  # 转换为 numpy 数组
        num_nodes = len(centers)

        edge_list = []
        edge_attr = []  # 用于存储边的方向信息，0: 左, 1: 右, 2: 上, 3: 下

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue  # 排除自身连接

                # 物体 A 和物体 B 的中心点
                center_A = centers[i]
                center_B = centers[j]

                # 计算水平距离和垂直距离
                x_distance = abs(center_A[0] - center_B[0])  # 水平距离
                y_distance = abs(center_A[1] - center_B[1])  # 垂直距离

                # 增加一定的容差，以确保边的构建
                tolerance = 0  # 容差值（可以根据需要调整）

                # 保留差距较大的距离作为边，并赋予方向标签
                if x_distance > y_distance + tolerance:  # 水平距离明显大于垂直距离
                    if center_A[0] < center_B[0]:  # A 在 B 的左侧
                        edge_list.append([i, j])  # A -> B 的边
                        edge_attr.append(1)  # 右侧
                    elif center_A[0] > center_B[0]:  # A 在 B 的右侧
                        edge_list.append([i, j])  # A -> B 的边
                        edge_attr.append(0)  # 左侧
                elif y_distance > x_distance + tolerance:  # 垂直距离明显大于水平距离
                    if center_A[1] < center_B[1]:  # A 在 B 的上方
                        edge_list.append([i, j])  # A -> B 的边
                        edge_attr.append(3)  # 下侧
                    elif center_A[1] > center_B[1]:  # A 在 B 的下方
                        edge_list.append([i, j])  # A -> B 的边
                        edge_attr.append(2)  # 上侧

        # 将 edge_list 转换为 torch.tensor，并转置为 (2, num_edges)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().cuda()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long).cuda()  # 边的方向标签

        return edge_index, edge_attr
    def extract_features(self, imgdet):
        # Preprocess image

        imgdet = imgdet.cpu().numpy()
        img_resized = letterbox(imgdet, 640, stride=32, auto=True)[0]  # Resize image
        img_resized = img_resized.transpose(2, 0, 1)[::-1]  # HWC to CHW
        img_resized = np.ascontiguousarray(img_resized)

        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).to(self.device)
        img_tensor = img_tensor.float() / 255.0  # Scale to 0-1
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # 推理

        pred = self.yolo_model(img_tensor, augment=False, visualize=False)

        # 应用 NMS
        pred = non_max_suppression(pred, 0.55, 0.45, None, False, max_det=1000)
        node_features, bboxes = torch.zeros((31, 512), device=self.device), []


        for i, det in enumerate(pred[0]):
            if i >= self.num_nodes:
                break
            x1, y1, x2, y2, conf, cls = map(int, det[:6])
            x1 = max(0, min(x1, 640))
            x2 = max(0, min(x2, 640))
            y1 = max(0, min(y1, 480))
            y2 = max(0, min(y2, 480))
            bboxes.append([x1, y1, x2, y2])
            crop = imgdet[y1:y2, x1:x2]

            cropped_img_resized = cv2.resize(crop, (416, 416))
            crop_tensor = torch.from_numpy(cropped_img_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(
                self.device)
            feature_vector = self.backbone.encode_image(crop_tensor)[1]


            node_features[i] = self.fc_reduce(feature_vector.flatten())

        edge_index, edge_attr = self.build_edges(bboxes) if bboxes else (
        torch.empty((2, 0), dtype=torch.long), torch.empty(0))

        node_features = self.pad_node_features(node_features, max_nodes=self.num_nodes)
        edge_index = self.pad_edge_index(edge_index, max_nodes=self.num_nodes, max_edges=30)

        return node_features, edge_index, edge_attr, bboxes



    def forward(self, img, imgdet, word, mask=None, grasp_qua_mask=None, grasp_sin_mask=None, grasp_cos_mask=None, grasp_wid_mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(img)
        print(f"vis1 stats: min={vis[0].min()}, max={vis[0].max()}, has_nan={torch.isnan(vis[0]).any()}")

        word, state = self.backbone.encode_text(word)
        # 确保同步
        for i in range(imgdet.size(0)):
            node_features, edge_index, edge_attr, bboxes = self.extract_features(imgdet[i])
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=node_features)
            if edge_index.numel() == 0:  # 检查 edge_index 是否为空
                print("Edge index is empty, skipping GCN processing.")
                gcn_output = torch.zeros(node_features.size(0), 512,
                                         device=node_features.device)  # 使用默认填充值
            else:
                gcn_output = self.gcn_model(data.x, data.edge_index)

            # 将 gcn_output 作为注意力权重应用到 fq 特征图中
            vis = self.apply_attention_to_vis(vis, gcn_output, bboxes)

        # 打印中间结果的统计信息
        print(f"vis stats: min={vis[0].min()}, max={vis[0].max()}, has_nan={torch.isnan(vis[0]).any()}")
        print(f"state stats: min={state.min()}, max={state.max()}, has_nan={torch.isnan(state).any()}")
        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state)
        print(f"fq stats: min={fq.min()}, max={fq.max()}, has_nan={torch.isnan(fq).any()}")

        b, c, h, w = fq.size()
        
        if self.use_contrastive:
            fq = self.decoder(fq, word, pad_mask)
            fq = fq.reshape(b, c, h, w)

        if self.use_grasp_masks:
            
            # b, 1, 104, 104
            pred, grasp_qua_pred, grasp_sin_pred, grasp_cos_pred, grasp_wid_pred = self.proj(fq, state)

            if self.training:
                # resize mask
                if pred.shape[-2:] != mask.shape[-2:]:
                    mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()
                    grasp_qua_mask = F.interpolate(grasp_qua_mask, grasp_qua_pred.shape[-2:], mode='nearest').detach()
                    grasp_sin_mask = F.interpolate(grasp_sin_mask, grasp_sin_pred.shape[-2:], mode='nearest').detach()
                    grasp_cos_mask = F.interpolate(grasp_cos_mask, grasp_cos_pred.shape[-2:], mode='nearest').detach()
                    grasp_wid_mask = F.interpolate(grasp_wid_mask, grasp_wid_pred.shape[-2:], mode='nearest').detach()

                # Ratio Augmentation
                total_area = mask.shape[2] * mask.shape[3]
                coef = 1 - (mask.sum(dim=(2,3)) / total_area)

                # Generate weight
                weight = mask * 0.5 + 1

                loss = F.binary_cross_entropy_with_logits(pred, mask, weight=weight)
                grasp_qua_loss = F.smooth_l1_loss(grasp_qua_pred, grasp_qua_mask)
                grasp_sin_loss = F.smooth_l1_loss(grasp_sin_pred, grasp_sin_mask)
                grasp_cos_loss = F.smooth_l1_loss(grasp_cos_pred, grasp_cos_mask)
                grasp_wid_loss = F.smooth_l1_loss(grasp_wid_pred, grasp_wid_mask)

                # @TODO adjust coef of different loss items
                total_loss = loss + grasp_qua_loss + grasp_sin_loss + grasp_cos_loss + grasp_wid_loss

                loss_dict = {}
                loss_dict["m_ins"] = loss.item()
                loss_dict["m_qua"] = grasp_qua_loss.item()
                loss_dict["m_sin"] = grasp_sin_loss.item()
                loss_dict["m_cos"] = grasp_cos_loss.item()
                loss_dict["m_wid"] = grasp_wid_loss.item()

                # loss = F.binary_cross_entropy_with_logits(pred, mask, reduction="none").sum(dim=(2,3))
                # loss = torch.dot(coef.squeeze(), loss.squeeze()) / (mask.shape[0] * mask.shape[2] * mask.shape[3])

                return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask), total_loss, loss_dict
            else:
                return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)

        else:
            # b, 1, 104, 104
            pred = self.proj(fq, state)

            if self.training:
                # resize mask
                if pred.shape[-2:] != mask.shape[-2:]:
                    mask = F.interpolate(mask, pred.shape[-2:],
                                        mode='nearest').detach()
                loss = F.binary_cross_entropy_with_logits(pred, mask)
                loss_dict = {}
                loss_dict["m_ins"] = loss.item()
                loss_dict["m_qua"] = 0
                loss_dict["m_sin"] = 0
                loss_dict["m_cos"] = 0
                loss_dict["m_wid"] = 0
                return (pred.detach(), None, None, None, None), (mask, None, None, None, None), loss, loss_dict
            else:
                return pred.detach(), mask