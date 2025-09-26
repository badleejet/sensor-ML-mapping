#!/usr/bin/env python
# coding: utf-8

import json
import csv
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from data_loader import HeatmapDataset
from models import SimplifiedMultiPersonPredictor


def run_inference(model_path, test_json_path, device=None):
    """
    モデルを使用してテストデータに対して推論を実行
    
    Args:
        model_path: 学習済みモデルのパス
        test_json_path: テストデータのJSONファイルパスまたはディレクトリパス
        device: 計算デバイス
    
    Returns:
        results: 推論結果のリスト
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # テストデータのパスがディレクトリかファイルかを判定
    test_json_files = []
    if os.path.isdir(test_json_path):
        # ディレクトリの場合、すべてのJSONファイルを取得
        test_json_files = glob.glob(os.path.join(test_json_path, "*.json"))
        print(f"Found {len(test_json_files)} JSON files in directory: {test_json_path}")
    elif os.path.isfile(test_json_path):
        # ファイルの場合、そのファイルのみ
        test_json_files = [test_json_path]
        print(f"Using single test file: {test_json_path}")
    else:
        raise ValueError(f"Test path does not exist: {test_json_path}")
    
    if not test_json_files:
        raise ValueError(f"No JSON files found in: {test_json_path}")
    
    # モデルの読み込み
    model = SimplifiedMultiPersonPredictor(max_people=2, input_channels=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 全ファイルの推論結果を格納
    all_results = []
    
    for json_file in test_json_files:
        print(f"\nProcessing file: {os.path.basename(json_file)}")
        
        # データセットの読み込み
        try:
            test_dataset = HeatmapDataset(json_file)
        except Exception as e:
            print(f"  Error loading dataset from {json_file}: {e}")
            continue
        
        # 各ウィンドウに対して推論を実行
        file_results = []
        
        for idx in range(len(test_dataset)):
            try:
                sample = test_dataset[idx]
                heatmap = sample['heatmap']
                metadata = sample['metadata']
                
                # 空のテンソルをチェック
                if heatmap.numel() == 0:
                    print(f"  Warning: Empty heatmap tensor in window {metadata['window_id']}")
                    continue
                
                # バッチ次元を追加
                if len(heatmap.shape) == 3:  # [time, distance, channels]
                    heatmap = heatmap.unsqueeze(0)
                
                # デバイスに転送
                heatmap = heatmap.to(device)
                
                # 推論を実行
                with torch.no_grad():
                    try:
                        pred_coords, pred_conf = model(heatmap)
                        
                        # 結果を出力
                        pred_coords_np = pred_coords.cpu().numpy()[0]
                        pred_conf_np = pred_conf.cpu().numpy()[0]
                        
                        # 結果を保存
                        result = {
                            'file_name': os.path.basename(json_file),
                            'window_id': metadata['window_id'],
                            'timestamp_start': metadata['timestamp_start'],
                            'timestamp_end': metadata['timestamp_end'],
                            'ground_truth': {
                                'positions': metadata['valid_positions'],
                                'ids': metadata['valid_ids']
                            },
                            'predictions': [{
                                'position': coord.tolist(),
                                'confidence': float(conf)
                            } for coord, conf in zip(pred_coords_np, pred_conf_np)]
                        }
                        
                        file_results.append(result)
                        
                    except Exception as e:
                        print(f"  Error during model forward pass for window {metadata['window_id']}: {e}")
                
            except Exception as e:
                print(f"  Error processing window {idx} in {json_file}: {e}")
        
        print(f"  Processed {len(file_results)} windows successfully")
        all_results.extend(file_results)
    
    print(f"\nTotal processed results: {len(all_results)}")
    return all_results


def visualize_ground_truth_and_predictions(results, title="Ground Truth and Predictions", 
                                         figsize=(12, 8), conf_threshold=0.3, 
                                         add_trajectory=True, save_path=None, group_by_file=True):
    """
    Ground Truthと予測結果を一つのグラフに可視化
    
    Args:
        results: 推論結果のリスト
        title: グラフタイトル
        figsize: 図のサイズ
        conf_threshold: 信頼度閾値
        add_trajectory: 軌跡を表示するかどうか
        save_path: 保存先パス（Noneなら保存しない）
        group_by_file: ファイル別に色分けするかどうか
    """
    if not results:
        print("No results to visualize")
        return
    
    # 図を作成
    plt.figure(figsize=figsize)
    
    # プロットの範囲とラベルを設定
    plt.xlim(0, 2)
    plt.ylim(0, 2.5)
    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # ファイル別またはウィンドウ別に色を定義
    if group_by_file:
        # ファイル名でグループ化
        file_names = list(set(result.get('file_name', 'unknown') for result in results))
        num_groups = len(file_names)
        group_key = lambda x: x.get('file_name', 'unknown')
        group_names = file_names
    else:
        # ウィンドウ別（従来通り）
        num_groups = len(results)
        group_key = lambda x: results.index(x)
        group_names = [f"Window {result['window_id']}" for result in results]
    
    cmap = plt.cm.tab10 if num_groups <= 10 else plt.cm.viridis
    colors = [cmap(i/max(1, num_groups-1)) for i in range(num_groups)]
    
    # 軌跡を追跡するための辞書
    gt_trajectory_points = {}
    pred_trajectory_points = {}
    
    # 各結果を可視化
    legend_entries = []
    processed_groups = set()
    
    for result in results:
        if group_by_file:
            file_name = result.get('file_name', 'unknown')
            group_idx = file_names.index(file_name)
            group_label = f"File: {file_name}"
        else:
            group_idx = results.index(result)
            timestamp = result["timestamp_start"].split("_")[1][:6] if "_" in result["timestamp_start"] else result["timestamp_start"]
            formatted_timestamp = f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:6]}"
            group_label = f'Window {result["window_id"]} ({formatted_timestamp})'
        
        color = colors[group_idx % len(colors)]
        
        # 凡例エントリを追加（重複を避ける）
        group_identifier = group_label if group_by_file else result['window_id']
        if group_identifier not in processed_groups:
            legend_entries.append(
                mpatches.Patch(color=color, alpha=0.7, label=group_label)
            )
            processed_groups.add(group_identifier)
        
        # Ground Truthの可視化
        for gt_idx, pos in enumerate(result['ground_truth']['positions']):
            # Ground Truthポイントをプロット
            plt.scatter(pos[0], pos[1], marker='o', s=100, 
                       color=color, alpha=0.8, edgecolors='black', linewidth=1.5)
            
            # ラベルを追加
            label_text = f'GT{gt_idx+1}'
            if group_by_file:
                label_text += f'\n{result["window_id"]}'
            
            plt.annotate(label_text, 
                        (pos[0] + 0.03, pos[1] + 0.03), 
                        color='black', 
                        fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor=color))
            
            # 軌跡を追跡
            trajectory_key = f"{file_name}_{result['window_id']}_{gt_idx}" if group_by_file else f"{result['window_id']}_{gt_idx}"
            if trajectory_key not in gt_trajectory_points:
                gt_trajectory_points[trajectory_key] = []
            gt_trajectory_points[trajectory_key].append((group_idx, pos))
        
        # 予測結果の可視化
        predictions = result['predictions']
        
        for pred_idx, pred in enumerate(predictions):
            pos = pred['position']
            conf = pred['confidence']
            
            # 信頼度が閾値を超える場合のみ表示
            if conf >= conf_threshold:
                # 予測ポイントをプロット (×マーカー)
                plt.scatter(pos[0], pos[1], marker='x', s=100, 
                          color=color, alpha=0.8, linewidth=2)
                
                # ラベル
                label_text = f'P{pred_idx}\n({conf:.2f})'
                if group_by_file:
                    label_text += f'\n{result["window_id"]}'
                
                plt.annotate(label_text, 
                          (pos[0] - 0.15, pos[1] - 0.15), 
                          color='black', 
                          fontsize=7,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor=color))
                
                # Ground Truthとの距離誤差を計算して表示
                if result['ground_truth']['positions']:
                    # 最も近いGround Truthを見つける
                    min_dist = float('inf')
                    closest_gt_idx = -1
                    
                    for gt_idx, gt_pos in enumerate(result['ground_truth']['positions']):
                        dist = np.sqrt((pos[0] - gt_pos[0])**2 + (pos[1] - gt_pos[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_gt_idx = gt_idx
                    
                    # 距離誤差を表示
                    if closest_gt_idx >= 0:
                        plt.annotate(f'Err: {min_dist:.2f}m', 
                                   (pos[0] - 0.15, pos[1] - 0.30), 
                                   color='red', 
                                   fontsize=6,
                                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, edgecolor='red'))
                
                # 軌跡を追跡
                trajectory_key = f"{file_name}_{result['window_id']}_{pred_idx}" if group_by_file else f"{result['window_id']}_{pred_idx}"
                if trajectory_key not in pred_trajectory_points:
                    pred_trajectory_points[trajectory_key] = []
                pred_trajectory_points[trajectory_key].append((group_idx, pos))
    
    # 軌跡を描画（オプション）
    if add_trajectory:
        # Ground Truthの軌跡を描画
        for traj_key, points in gt_trajectory_points.items():
            if len(points) > 1:  # 2点以上ある場合のみ軌跡を描画
                # ウィンドウインデックスでソート
                points.sort(key=lambda x: x[0])
                
                # 位置を抽出
                x_coords = [p[1][0] for p in points]
                y_coords = [p[1][1] for p in points]
                
                # 軌跡線をプロット
                plt.plot(x_coords, y_coords, '--', color='blue', alpha=0.4, linewidth=1)
        
        # 予測結果の軌跡を描画
        for traj_key, points in pred_trajectory_points.items():
            if len(points) > 1:  # 2点以上ある場合のみ軌跡を描画
                # ウィンドウインデックスでソート
                points.sort(key=lambda x: x[0])
                
                # 位置を抽出
                x_coords = [p[1][0] for p in points]
                y_coords = [p[1][1] for p in points]
                
                # 軌跡線をプロット
                plt.plot(x_coords, y_coords, '-.', color='red', alpha=0.4, linewidth=1)
    
    # Ground Truthと予測の凡例
    type_legend_entries = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Ground Truth'),
        Line2D([0], [0], marker='x', color='black', markersize=8, label='Prediction')
    ]
    
    # 軌跡の凡例
    if add_trajectory:
        trajectory_legend_entries = [
            Line2D([0], [0], linestyle='--', color='blue', alpha=0.6, linewidth=1.5, label='GT Trajectory'),
            Line2D([0], [0], linestyle='-.', color='red', alpha=0.6, linewidth=1.5, label='Pred Trajectory')
        ]
        type_legend_entries.extend(trajectory_legend_entries)
    
    # 凡例を結合
    all_legends = legend_entries + type_legend_entries
    plt.legend(handles=all_legends, loc='upper right', 
              framealpha=0.9, frameon=True, fontsize=8)
    
    plt.tight_layout()
    
    # 図を保存（パスが提供された場合）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def calculate_metrics(results, conf_threshold=0.3, distance_threshold=0.5):
    """
    推論結果の評価指標を計算
    
    Args:
        results: 推論結果のリスト
        conf_threshold: 信頼度閾値
        distance_threshold: 距離閾値
    
    Returns:
        metrics: 評価指標の辞書
    """
    if not results:
        return None
    
    total_gt = 0
    total_pred = 0
    total_matches = 0
    total_distance = 0.0
    
    for result in results:
        # Ground Truth情報
        gt_positions = result['ground_truth']['positions']
        valid_count = len(gt_positions)
        total_gt += valid_count
        
        # 予測結果の情報
        predictions = result['predictions']
        valid_preds = [pred for pred in predictions if pred['confidence'] >= conf_threshold]
        total_pred += len(valid_preds)
        
        # Ground Truthと予測のマッチング
        if valid_count > 0 and valid_preds:
            # 距離行列の計算
            distances = np.zeros((len(valid_preds), valid_count))
            
            for i, pred in enumerate(valid_preds):
                pred_pos = pred['position']
                for j, gt_pos in enumerate(gt_positions):
                    dist = np.sqrt((pred_pos[0] - gt_pos[0])**2 + (pred_pos[1] - gt_pos[1])**2)
                    distances[i, j] = dist
            
            # 各予測に対して最も近いGround Truthを見つける
            for i, pred in enumerate(valid_preds):
                min_dist = np.min(distances[i])
                if min_dist <= distance_threshold:
                    total_matches += 1
                    total_distance += min_dist
    
    # 評価指標の計算
    precision = total_matches / total_pred if total_pred > 0 else 0
    recall = total_matches / total_gt if total_gt > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_distance = total_distance / total_matches if total_matches > 0 else float('inf')
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_distance': mean_distance,
        'total_gt': total_gt,
        'total_pred': total_pred,
        'total_matches': total_matches
    }
    
    return metrics


def print_metrics(metrics, conf_threshold=0.3, distance_threshold=0.5):
    """
    評価指標を表示
    """
    if metrics is None:
        print("No metrics to display")
        return
    
    print("\n===== Evaluation Metrics =====")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Distance threshold: {distance_threshold} m")
    print(f"Total ground truth objects: {metrics['total_gt']}")
    print(f"Total predictions (conf > {conf_threshold}): {metrics['total_pred']}")
    print(f"Total matches (dist < {distance_threshold}m): {metrics['total_matches']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Mean distance error: {metrics['mean_distance']:.4f} m")
    print("==============================")


def export_results_to_csv(results, output_csv_path, conf_threshold=0.3):
    """
    推論結果をCSVファイルにエクスポート
    
    Args:
        results: 推論結果
        output_csv_path: 出力CSVファイルのパス
        conf_threshold: 信頼度の閾値
    """
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        # CSVヘッダーを定義（ファイル名を追加）
        fieldnames = [
            'file_name', 'window_id', 
            'ground_truth_pos1_x', 'ground_truth_pos1_y', 
            'ground_truth_pos2_x', 'ground_truth_pos2_y',
            'pred_pos1_x', 'pred_pos1_y', 'pred_conf1', 
            'pred_pos2_x', 'pred_pos2_y', 'pred_conf2',
            'distance_error1', 'distance_error2'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # ファイル名を取得
            file_name = result.get('file_name', 'unknown')
            
            # Ground truthの座標を取得
            gt_positions = result['ground_truth']['positions']
            gt_pos1 = gt_positions[0] if len(gt_positions) > 0 else [0, 0]
            gt_pos2 = gt_positions[1] if len(gt_positions) > 1 else [0, 0]
            
            # 予測結果を取得（信頼度でフィルタリング）
            predictions = [pred for pred in result['predictions'] if pred['confidence'] >= conf_threshold]
            
            # 予測座標と信頼度を初期化
            pred_pos1 = [0, 0]
            pred_pos2 = [0, 0]
            pred_conf1 = 0.0
            pred_conf2 = 0.0
            
            # 予測結果を設定
            if len(predictions) > 0:
                pred_pos1 = predictions[0]['position']
                pred_conf1 = predictions[0]['confidence']
            if len(predictions) > 1:
                pred_pos2 = predictions[1]['position']
                pred_conf2 = predictions[1]['confidence']
            
            # 距離誤差を計算
            dist_error1 = np.sqrt((pred_pos1[0] - gt_pos1[0])**2 + (pred_pos1[1] - gt_pos1[1])**2) if len(gt_positions) > 0 else float('inf')
            dist_error2 = np.sqrt((pred_pos2[0] - gt_pos2[0])**2 + (pred_pos2[1] - gt_pos2[1])**2) if len(gt_positions) > 1 else float('inf')
            
            # CSVに書き出し
            writer.writerow({
                'file_name': file_name,
                'window_id': result['window_id'],
                'ground_truth_pos1_x': gt_pos1[0], 'ground_truth_pos1_y': gt_pos1[1],
                'ground_truth_pos2_x': gt_pos2[0], 'ground_truth_pos2_y': gt_pos2[1],
                'pred_pos1_x': pred_pos1[0], 'pred_pos1_y': pred_pos1[1], 'pred_conf1': pred_conf1,
                'pred_pos2_x': pred_pos2[0], 'pred_pos2_y': pred_pos2[1], 'pred_conf2': pred_conf2,
                'distance_error1': dist_error1,
                'distance_error2': dist_error2
            })
    
    print(f"Results exported to {output_csv_path}")
    print(f"Total rows: {len(results)}")
    
    # ファイル別の統計も表示
    file_stats = {}
    for result in results:
        file_name = result.get('file_name', 'unknown')
        if file_name not in file_stats:
            file_stats[file_name] = 0
        file_stats[file_name] += 1
    
    print("Results by file:")
    for file_name, count in file_stats.items():
        print(f"  - {file_name}: {count} windows")


def run_inference_and_visualize(model_path, test_json_path, conf_threshold=0.3, 
                              distance_threshold=0.5, add_trajectory=True,
                              save_path=None, export_csv_path=None):
    """
    モデルの推論、評価、可視化、CSVエクスポートを一連の流れで実行
    
    Args:
        model_path: 学習済みモデルのパス
        test_json_path: テストデータのJSONファイルパス
        conf_threshold: 信頼度閾値
        distance_threshold: 距離閾値
        add_trajectory: 軌跡を表示するかどうか
        save_path: 可視化結果の保存パス
        export_csv_path: CSV出力パス
    
    Returns:
        results: 推論結果
        metrics: 評価指標
    """
    print(f"Loading data from {test_json_path}")
    print(f"Using model weights from {model_path}")
    
    # 推論を実行
    results = run_inference(model_path, test_json_path)
    
    # 評価指標を計算
    metrics = calculate_metrics(results, conf_threshold, distance_threshold)
    print_metrics(metrics, conf_threshold, distance_threshold)
    
    # 結果を可視化
    title = f"Ground Truth and Predictions (Conf>{conf_threshold}, Dist<{distance_threshold}m)"
    visualize_ground_truth_and_predictions(
        results, 
        title=title,
        conf_threshold=conf_threshold,
        add_trajectory=add_trajectory,
        save_path=save_path
    )
    
    # CSVにエクスポート（オプション）
    if export_csv_path:
        export_results_to_csv(results, export_csv_path, conf_threshold)
    
    return results, metrics


def main_inference():
    """
    メインの推論関数
    """
    # パスを設定
    model_path = "./outputs/best_model.pth"  # 学習済みモデル
    
    # テストディレクトリまたはファイルを自動検出
    test_paths = [
        "./dataset3/test",      # ディレクトリ全体
        "./dataset2/test",      # ディレクトリ全体
        "./dataset1/test",      # ディレクトリ全体
        "./dataset/test",       # ディレクトリ全体
        "./dataset3/test/slide_two_sekkin_1_rev_2.json",  # 個別ファイル（従来の指定）
        "./dataset2/test/slide_two_sekkin_1_rev_2.json",  # 個別ファイル（従来の指定）
    ]
    
    test_json_path = None
    for path in test_paths:
        if os.path.exists(path):
            if os.path.isdir(path) and len(glob.glob(os.path.join(path, "*.json"))) > 0:
                test_json_path = path
                print(f"Using test directory: {test_json_path}")
                break
            elif os.path.isfile(path):
                test_json_path = path
                print(f"Using test file: {test_json_path}")
                break
    
    if test_json_path is None:
        print("Error: No test data found in any of the expected locations:")
        for path in test_paths:
            print(f"  - {path}")
        return None, None
    
    # 信頼度と距離の閾値を設定
    conf_threshold = 0.3  # 信頼度閾値
    distance_threshold = 0.5  # 距離閾値（メートル単位）
    
    # 推論と可視化を実行
    results, metrics = run_inference_and_visualize(
        model_path=model_path,
        test_json_path=test_json_path,
        conf_threshold=conf_threshold,
        distance_threshold=distance_threshold,
        add_trajectory=True,
        save_path="./outputs/ground_truth_and_predictions.png",
        export_csv_path="./outputs/inference_results.csv"
    )
    
    return results, metrics


if __name__ == "__main__":
    # 推論の実行
    results, metrics = main_inference()