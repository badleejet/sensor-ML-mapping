#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np

class ConvLSTMCell(nn.Module):
    """
    ConvLSTM セル実装
    畳み込み層とLSTMを組み合わせた時系列処理モジュール
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # 入力テンソルの形状を確認
        b, c, h, w = input_tensor.size()
        
        # 現在の隠れ状態の形状が入力と一致するか確認
        if h_cur.size()[2:] != (h, w):
            # 必要に応じて隠れ状態のサイズを調整
            h_cur = nn.functional.interpolate(h_cur, size=(h, w), mode='nearest')
            c_cur = nn.functional.interpolate(c_cur, size=(h, w), mode='nearest')
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class SimplifiedMultiPersonPredictor(nn.Module):
    """
    簡易版多人数予測モデル
    CNN特徴抽出 + LSTM時系列処理 + 座標・信頼度予測
    """
    def __init__(self, max_people=2, input_channels=2):
        super().__init__()
        
        self.max_people = max_people
        self.input_channels = input_channels
        
        # CNN特徴抽出
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 特徴マップサイズの計算
        feature_height = 34 // 4  # 高さ
        feature_width = 1         # 幅
        self.feature_size = 128 * feature_height * feature_width
        
        # LSTM時系列処理
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
        
        # 座標予測
        self.coordinate_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, max_people * 2),
            nn.Sigmoid()  # 0-1の範囲に制限
        )
        
        # 信頼度予測
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, max_people),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        time_steps = x.size(1)
        
        # 時系列データのフレームごとに特徴抽出
        features = []
        for t in range(time_steps):
            # 現在のフレームを取得
            frame = x[:, t]  # [batch_size, height, channels]
            
            # 畳み込み処理のためにチャンネルを最初の次元に移動
            frame = frame.permute(0, 2, 1).unsqueeze(3)
            
            # 特徴抽出
            frame_features = self.feature_extractor(frame)
            
            # フラット化
            frame_features = frame_features.reshape(batch_size, -1)
            
            features.append(frame_features)
        
        # 時系列特徴をスタック
        sequence = torch.stack(features, dim=1)
        
        # LSTM処理
        lstm_out, _ = self.lstm(sequence)
        
        # 最後の時間ステップを使用
        final_features = lstm_out[:, -1]
        
        # 座標と信頼度の予測
        coordinates = self.coordinate_head(final_features)
        coordinates = coordinates.view(batch_size, self.max_people, 2)
        
        confidences = self.confidence_head(final_features)
        
        # 座標を実際のエリアサイズにスケーリング (2×2.5m)
        scale_factors = torch.tensor([2.0, 2.5], device=coordinates.device)
        scaled_coordinates = coordinates * scale_factors.view(1, 1, 2)
        
        return scaled_coordinates, confidences


def custom_loss_for_mchannel(pred_coords, pred_conf, true_coords, valid_counts, 
                            current_epoch=0, max_epochs=100):
    """
    マルチチャンネルヒートマップデータ用のカスタム損失関数
    
    Args:
        pred_coords: 予測座標 (batch_size, max_people, 2) - スケーリング済み (0-2, 0-2.5)
        pred_conf: 予測信頼度 (batch_size, max_people)
        true_coords: 正解座標 (batch_size, max_people, 2) - スケーリング済み (0-2, 0-2.5)
        valid_counts: 各バッチの有効人数 (batch_size,)
        current_epoch: 現在のエポック数
        max_epochs: 最大エポック数
    """
    # 段階的な閾値スケジュール設定
    progress_percentage = current_epoch / max_epochs * 100
    
    # 最初の50%のエポックでは閾値なし、50%-80%では0.5m、80%以降は0.25mの閾値
    # 部屋のサイズが小さくなったので閾値も調整
    if progress_percentage < 50:
        # 最初の50%は閾値なし
        threshold_active = False
        distance_threshold = float('inf')
    elif progress_percentage < 80:
        # 50-80%は中程度の閾値
        threshold_active = False
        # threshold_active = True
        distance_threshold = 0.5  # 2×2.5mの部屋なので閾値も小さく
    else:
        # 80%以降は厳しい閾値
        threshold_active = False
        # threshold_active = True
        distance_threshold = 0.25  # より厳しい閾値
    
    batch_size = pred_coords.shape[0]
    device = pred_coords.device
    
    # バッチ全体の損失を初期化（変更: 0.0からテンソルに変更）
    total_loss = torch.tensor(0.0, device=device)
    
    for i in range(batch_size):
        valid_count = valid_counts[i].item()
        
        if valid_count == 0:
            # すべての信頼度スコアを0に近づける
            conf_loss = pred_conf[i].mean()
            total_loss = total_loss + conf_loss  # 変更: += から明示的な加算に変更
            continue
        
        # 有効な座標のみを使用
        valid_true_coords = true_coords[i, :valid_count]
        
        # 予測座標と実際の座標間の距離行列を計算 
        pred = pred_coords[i].unsqueeze(1)  # (max_people, 1, 2)
        true = valid_true_coords.unsqueeze(0)  # (1, valid_count, 2)
        dist_matrix = torch.sqrt(((pred - true) ** 2).sum(dim=2))  # (max_people, valid_count)
        
        # 改良されたマッチング処理
        matched_true = []  # すでにマッチングされた正解インデックスを記録
        min_dists = torch.zeros_like(pred_conf[i])
        matched_pred_indices = []  # マッチングされた予測インデックスを記録
        
        for pred_idx in range(len(pred_coords[i])):
            distances = dist_matrix[pred_idx].clone()  # 距離のコピーを作成
            
            # すでにマッチングされた正解を除外して最小距離を探す
            while True:
                min_dist, min_idx = torch.min(distances), torch.argmin(distances)
                if min_idx.item() not in matched_true or len(matched_true) >= valid_count:
                    break
                # 変更: in-place操作を避ける
                tmp_distances = distances.clone()
                tmp_distances[min_idx] = float('inf')
                distances = tmp_distances
            
            # 閾値チェックを適用（閾値が有効な場合のみ）
            if not threshold_active or min_dist <= distance_threshold:
                # 変更: in-place操作を避ける
                tmp_min_dists = min_dists.clone()
                tmp_min_dists[pred_idx] = min_dist
                min_dists = tmp_min_dists
                
                if min_idx.item() not in matched_true and len(matched_true) < valid_count:
                    matched_true.append(min_idx.item())
                    matched_pred_indices.append(pred_idx)
            else:
                # 閾値を超える場合は大きなペナルティ値を設定
                # 変更: in-place操作を避ける
                tmp_min_dists = min_dists.clone()
                tmp_min_dists[pred_idx] = distance_threshold * 2
                min_dists = tmp_min_dists
        
        # 座標の損失
        # マッチングされた予測の損失
        if matched_pred_indices:
            # マッチングされた予測のみを考慮（信頼度で重み付け）
            idx_tensor = torch.tensor(matched_pred_indices, device=device)
            matched_dists = torch.index_select(min_dists, 0, idx_tensor)
            matched_conf = torch.index_select(pred_conf[i], 0, idx_tensor)
            
            if len(matched_dists) > 0:
                coord_loss_matched = torch.sum(matched_dists * matched_conf) / len(matched_dists)
            else:
                coord_loss_matched = torch.tensor(0.0, device=device)
        else:
            coord_loss_matched = torch.tensor(0.0, device=device)
        
        # マッチングされなかった真値に対するペナルティ
        unmatched_count = valid_count - len(matched_true)
        
        # 閾値が有効な場合のみペナルティを適用
        if threshold_active and unmatched_count > 0:
            # 学習段階に応じたペナルティの強度
            if progress_percentage < 80:
                penalty_weight = 0.5  # 中程度のペナルティ
            else:
                penalty_weight = 1.0  # 強いペナルティ
                
            unmatched_penalty = torch.tensor(unmatched_count * distance_threshold * penalty_weight, device=device)
        else:
            unmatched_penalty = torch.tensor(0.0, device=device)
        
        # 座標の損失（マッチングされたものとペナルティの合計）
        if valid_count > 0:
            coord_loss = (coord_loss_matched + unmatched_penalty) / valid_count
        else:
            coord_loss = torch.tensor(0.0, device=device)
        
        # 信頼度スコアの損失
        conf_target = torch.zeros_like(pred_conf[i])
        
        # 閾値が有効でない場合は元の処理
        if not threshold_active:
            # 変更: in-place操作を避ける
            tmp_conf_target = conf_target.clone()
            tmp_conf_target[:valid_count] = 1
            conf_target = tmp_conf_target
        else:
            # 閾値が有効な場合はマッチングされた予測のみ高信頼度に
            tmp_conf_target = conf_target.clone()
            for idx in matched_pred_indices:
                tmp_conf_target[idx] = 1.0
            conf_target = tmp_conf_target
                
        conf_loss = nn.BCELoss()(pred_conf[i], conf_target)
        
        # 合計損失
        sample_loss = coord_loss + conf_loss
        total_loss = total_loss + sample_loss  # 変更: += から明示的な加算に変更
    
    # バッチサイズで割って平均損失を返す
    return total_loss / batch_size


def model_summary(model, input_size=(1, 50, 34, 2)):
    """
    モデルのサマリーを表示する関数
    
    Args:
        model: 要約するモデル
        input_size: 入力サイズ (batch_size, time_steps, height, channels)
    """
    print("=" * 50)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 50)
    
    # モデルのパラメータ数を計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # モデルの構造を表示
    print("\nModel Architecture:")
    print(model)
    
    # 入力サイズでテスト実行
    try:
        with torch.no_grad():
            dummy_input = torch.randn(*input_size)
            coords, conf = model(dummy_input)
            print(f"\nInput shape: {dummy_input.shape}")
            print(f"Output coordinates shape: {coords.shape}")
            print(f"Output confidence shape: {conf.shape}")
    except Exception as e:
        print(f"\nError during forward pass: {e}")
    
    print("=" * 50)