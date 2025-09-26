#!/usr/bin/env python
# coding: utf-8

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class MultichannelDataset(Dataset):
    def __init__(self, json_files_dir, max_people=2):
        """
        マルチチャンネルヒートマップデータセット
        Args:
            json_files_dir: JSONファイルが格納されているディレクトリパス
            max_people: 最大人数（デフォルト: 2）
        """
        self.samples = []
        self.max_people = max_people
        
        # ディレクトリ内のすべてのJSONファイルを取得
        json_files = glob.glob(os.path.join(json_files_dir, "*.json"))
        print(f"Found {len(json_files)} JSON files in {json_files_dir}")
        
        skipped_windows = 0
        total_windows = 0
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            scenario_id = data.get('scenario_id', os.path.basename(json_file))
            
            for window in data['windows']:
                total_windows += 1
                
                # ground truthの確認: 両方が[0,0]の場合はスキップ
                ground_truth = window['ground_truth']
                
                # すべてのground truthが[0,0]かどうかをチェック
                all_zeros = True
                for person in ground_truth:
                    if person['position'] != [0, 0]:
                        all_zeros = False
                        break
                
                # 両方とも[0,0]の場合はこのデータをスキップ
                if all_zeros:
                    skipped_windows += 1
                    continue
                
                # センサーデータの抽出
                sensor_data = []
                for sensor in window['sensors']:
                    heatmap = np.array(sensor['heatmap_data'])
                    sensor_data.append(heatmap)
                
                # マルチチャンネルヒートマップ作成 [50, 34, 2]
                multi_channel_heatmap = np.stack(sensor_data, axis=-1)
                
                # 有効なground truthの抽出と正規化
                positions = []
                confidences = []
                valid_count = 0
                
                for person in ground_truth:
                    if person['position'] != [0, 0]:  # 実際に人がいる場合
                        # 2×2.5mエリア内での座標を0～1の範囲に正規化
                        normalized_x = person['position'][0] / 2.0
                        normalized_y = person['position'][1] / 2.5
                        positions.append([normalized_x, normalized_y])
                        confidences.append(1.0)  # 実際の人は信頼度1.0
                        valid_count += 1
                
                # 最大人数に合わせてパディング
                while len(positions) < self.max_people:
                    positions.append([0, 0])  # ダミー座標
                    confidences.append(0.0)   # 信頼度0
                
                # データセットに追加
                self.samples.append({
                    'heatmap': torch.tensor(multi_channel_heatmap, dtype=torch.float32),
                    'positions': torch.tensor(positions, dtype=torch.float32),
                    'confidences': torch.tensor(confidences, dtype=torch.float32),
                    'valid_count': valid_count,
                    'scenario_id': scenario_id,
                    'window_id': window['window_id']
                })
        
        # サンプルをシナリオIDと窓IDでソート（時系列性を保持）
        self.samples.sort(key=lambda x: (x['scenario_id'], x['window_id']))
        
        print(f"Total windows processed: {total_windows}")
        print(f"Skipped windows (all zeros): {skipped_windows}")
        print(f"Windows added to dataset: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class HeatmapDataset(Dataset):
    """推論用のデータセット - JSONファイルからセンサーデータを読み込む"""
    
    def __init__(self, json_file_path):
        # JSONファイルを読み込む
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)
        
        self.windows = self.data.get('windows', [])
        print(f"Loaded {len(self.windows)} windows from JSON file")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        
        # センサーからヒートマップデータを取得
        sensors = window.get('sensors', [])
        
        # センサーデータをチェック
        if len(sensors) > 0:
            sensor_heatmaps = []
            
            for sensor in sensors:
                if 'heatmap_data' in sensor:
                    sensor_heatmap = sensor['heatmap_data']
                    
                    if len(sensor_heatmap) > 0:
                        # ヒートマップデータを配列に変換
                        try:
                            sensor_heatmap_array = np.array(sensor_heatmap)
                            sensor_heatmaps.append(sensor_heatmap_array)
                        except Exception as e:
                            print(f"  Error converting sensor heatmap to array: {e}")
            
            # すべてのセンサーのヒートマップを結合
            if sensor_heatmaps:
                try:
                    heatmap_data = np.stack(sensor_heatmaps, axis=-1)
                except Exception as e:
                    print(f"  Error stacking sensor heatmaps: {e}")
                    heatmap_data = sensor_heatmaps[0]
            else:
                heatmap_data = np.array([])
        else:
            heatmap_data = np.array([])
        
        # Ground truthの取得
        ground_truth = window.get('ground_truth', [])
        
        # 有効な位置情報（非ゼロ位置）を抽出
        valid_positions = []
        valid_ids = []
        
        for i, item in enumerate(ground_truth):
            position = item.get('position', [0, 0])
            obj_id = item.get('object_id', i+1)
            
            # 非ゼロ位置のみを有効とする
            if position[0] != 0 or position[1] != 0:
                valid_positions.append(position)
                valid_ids.append(obj_id)
        
        # メタデータ
        metadata = {
            'window_id': window.get('window_id', idx),
            'timestamp_start': window.get('timestamp_start', ''),
            'timestamp_end': window.get('timestamp_end', ''),
            'valid_positions': valid_positions,
            'valid_ids': valid_ids
        }
        
        # ヒートマップデータをテンソルに変換
        if heatmap_data.size > 0:
            heatmap_tensor = torch.tensor(heatmap_data, dtype=torch.float32)
        else:
            heatmap_tensor = torch.tensor([], dtype=torch.float32)
        
        return {
            'heatmap': heatmap_tensor,
            'metadata': metadata
        }


def create_data_loaders(train_dir, val_dir, test_dir=None, batch_size=16, max_people=2, preserve_order=True):
    """
    トレーニング、検証、テスト用のデータローダーを作成
    Args:
        train_dir: トレーニングデータのJSONファイルが格納されているディレクトリ
        val_dir: 検証データのJSONファイルが格納されているディレクトリ
        test_dir: テストデータのJSONファイルが格納されているディレクトリ（オプション）
        batch_size: バッチサイズ
        max_people: 最大人数
        preserve_order: 時系列順序を保持するかどうか
    Returns:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    """
    # GPU利用可能性をチェックしてpin_memoryを設定
    pin_memory = torch.cuda.is_available()
    
    print("Creating training dataset...")
    train_dataset = MultichannelDataset(train_dir, max_people=max_people)
    
    # データセットが空でないかチェック
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty. Check if data exists in {train_dir}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not preserve_order,  # preserve_orderがTrueならシャッフルしない
        num_workers=0,
        pin_memory=pin_memory
    )
    
    print("\nCreating validation dataset...")
    val_dataset = MultichannelDataset(val_dir, max_people=max_people)
    
    # データセットが空でないかチェック
    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty. Check if data exists in {val_dir}")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 検証セットは常にシャッフルしない
        num_workers=0,
        pin_memory=pin_memory
    )
    
    test_loader = None
    test_dataset = None
    if test_dir:
        print("\nCreating test dataset...")
        test_dataset = MultichannelDataset(test_dir, max_people=max_people)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # テストセットは常にシャッフルしない
            num_workers=0,
            pin_memory=pin_memory
        )
        print(f"\nTest data: {len(test_dataset)} samples")
    
    print(f"\nSummary:")
    print(f"Training data: {len(train_dataset)} samples")
    print(f"Validation data: {len(val_dataset)} samples")
    if test_loader:
        print(f"Test data: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def visualize_ground_truth_distribution(train_dataset, val_dataset, test_dataset):
    """各データセットのground truthの分布を散布図で表示"""
    plt.figure(figsize=(15, 5))
    
    # トレーニングデータのプロット
    plt.subplot(1, 3, 1)
    plot_positions(train_dataset, "Training Data")
    
    # 検証データのプロット
    plt.subplot(1, 3, 2)
    plot_positions(val_dataset, "Validation Data")
    
    # テストデータのプロット
    if test_dataset is not None:
        plt.subplot(1, 3, 3)
        plot_positions(test_dataset, "Test Data")
    
    plt.tight_layout()
    plt.show()


def plot_positions(dataset, title):
    """1つのデータセットの位置座標を散布図でプロット"""
    # 位置データの抽出
    all_positions = []
    for sample in dataset.samples:
        positions = sample['positions'].numpy()
        confidences = sample['confidences'].numpy()
        
        # 信頼度が0より大きい位置のみ抽出（有効な位置）
        for pos, conf in zip(positions, confidences):
            if conf > 0:
                # 元のスケールに戻す（0-1から0-2, 0-2.5へ）
                real_x = pos[0] * 2.0
                real_y = pos[1] * 2.5
                all_positions.append([real_x, real_y])
    
    # NumPy配列に変換
    if all_positions:
        all_positions = np.array(all_positions)
        
        # 散布図プロット
        plt.scatter(all_positions[:, 0], all_positions[:, 1], alpha=0.5)
        plt.xlim(0, 2)
        plt.ylim(0, 2.5)
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(f'{title} - {len(all_positions)} Points')
        plt.grid(True)
    else:
        plt.text(1, 1.25, "No valid positions", ha='center', va='center')
        plt.xlim(0, 2)
        plt.ylim(0, 2.5)
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(title)
        plt.grid(True)


def debug_data_loading():
    """
    データの読み込みとデータローダーの作成をデバッグするための関数
    """
    # データディレクトリのパス設定
    train_dir = './dataset2/train'
    val_dir = './dataset2/val'
    test_dir = './dataset2/test'  # テストデータがある場合
    
    # データローダーの作成（時系列順序を保持）
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_data_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=4,  # デバッグ用に小さいバッチサイズ
        max_people=2,
        preserve_order=True  # 時系列順序を保持
    )
    
    # ground truthの分布を可視化
    print("\nVisualizing ground truth distribution...")
    visualize_ground_truth_distribution(train_dataset, val_dataset, test_dataset)
    
    # データサンプルの確認
    print("\nChecking training data sample:")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx > 0:
            break
            
        # バッチのサイズと形状を確認
        heatmap = batch['heatmap']
        positions = batch['positions']
        confidences = batch['confidences']
        valid_counts = batch['valid_count']
        
        print(f"Batch shape - Heatmap: {heatmap.shape}")
        print(f"Batch shape - Positions: {positions.shape}")
        print(f"Batch shape - Confidences: {confidences.shape}")
        print(f"Valid counts: {valid_counts}")
        
        # 最初のサンプルの詳細確認
        print("\nFirst sample details:")
        print(f"Heatmap min: {heatmap[0].min()}, max: {heatmap[0].max()}")
        print(f"Positions: {positions[0]}")
        print(f"Confidences: {confidences[0]}")
        print(f"Valid count: {valid_counts[0]}")
        
        # ヒートマップの統計情報
        for channel in range(heatmap.shape[-1]):
            channel_data = heatmap[0, :, :, channel]
            print(f"Channel {channel} stats - Min: {channel_data.min()}, Max: {channel_data.max()}, Mean: {channel_data.mean()}")
    
    # 各データローダーからバッチを取得できることを確認
    print("\nVerifying data loaders...")
    
    loader_names = ["Training", "Validation", "Test"]
    loaders = [train_loader, val_loader, test_loader]
    
    for name, loader in zip(loader_names, loaders):
        if loader is None:
            print(f"{name} loader: Not available")
            continue
            
        try:
            batch = next(iter(loader))
            print(f"{name} loader: Success - Got batch with {len(batch['heatmap'])} samples")
        except Exception as e:
            print(f"{name} loader: Error - {str(e)}")