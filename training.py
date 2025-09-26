#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data_loader import create_data_loaders
from models import SimplifiedMultiPersonPredictor, custom_loss_for_mchannel


def train_model(model, train_loader, val_loader, test_loader=None, epochs=100, 
                learning_rate=0.001, device=None, scheduler_type='cosine', save_dir='./outputs'):
    """
    モデルをトレーニングする関数
    
    Args:
        model: トレーニングするモデル
        train_loader: トレーニングデータローダー
        val_loader: 検証データローダー
        test_loader: テストデータローダー（オプション）
        epochs: トレーニングエポック数
        learning_rate: 初期学習率
        device: 計算デバイス（CPUまたはGPU）
        scheduler_type: 学習率スケジューラのタイプ ('cosine', 'step', 'plateau', 'onecycle', 'none')
        save_dir: モデルと結果の保存ディレクトリ
    
    Returns:
        train_losses, val_losses: トレーニングと検証の損失履歴
        best_model_path: 最良モデルの保存パス
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    model = model.to(device)
    
    # 保存ディレクトリの作成
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学習率スケジューラの設定
    scheduler = None
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        print(f"Using Cosine Annealing LR scheduler")
    elif scheduler_type == 'step':
        step_size = epochs // 3
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
        print(f"Using Step LR scheduler (step_size={step_size}, gamma=0.1)")
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        print(f"Using ReduceLROnPlateau scheduler (factor=0.5, patience=5)")
    elif scheduler_type == 'onecycle':
        total_steps = epochs * len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate * 10, 
                                                 total_steps=total_steps, div_factor=10,
                                                 pct_start=0.3, final_div_factor=100)
        print(f"Using OneCycle LR scheduler")
    else:
        print(f"No LR scheduler used")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = None
    
    # データローダーの有効性をチェック
    if len(train_loader) == 0:
        raise ValueError("Training data loader is empty. Please check your training data.")
    if len(val_loader) == 0:
        raise ValueError("Validation data loader is empty. Please check your validation data.")
    
    # トレーニングループ
    for epoch in range(epochs):
        # 学習モード
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # データをデバイスに転送
            heatmap = batch['heatmap'].to(device)
            positions = batch['positions'].to(device)
            valid_counts = batch['valid_count'].to(device)
            
            # 勾配をゼロにリセット
            optimizer.zero_grad()
            
            # 順伝播
            pred_coords, pred_conf = model(heatmap)
            
            # 損失計算
            loss = custom_loss_for_mchannel(
                pred_coords, 
                pred_conf, 
                positions,
                valid_counts,
                current_epoch=epoch,
                max_epochs=epochs
            )
            
            # 逆伝播と最適化
            loss.backward()
            
            # 勾配クリッピング（大きな勾配更新を防ぐ）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # OneCycleの場合はバッチごとに学習率を更新
            if scheduler_type == 'onecycle':
                scheduler.step()
            
            train_loss += loss.item()
        
        # エポックごとの平均損失（ゼロ除算エラーを防ぐ）
        if len(train_loader) > 0:
            train_loss /= len(train_loader)
        else:
            train_loss = 0.0
        train_losses.append(train_loss)
        
        # 検証
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # データをデバイスに転送
                heatmap = batch['heatmap'].to(device)
                positions = batch['positions'].to(device)
                valid_counts = batch['valid_count'].to(device)
                
                # 順伝播
                pred_coords, pred_conf = model(heatmap)
                
                # 損失計算
                loss = custom_loss_for_mchannel(
                    pred_coords, 
                    pred_conf, 
                    positions,
                    valid_counts,
                    current_epoch=epoch,
                    max_epochs=epochs
                )
                
                val_loss += loss.item()
        
        # エポックごとの平均検証損失（ゼロ除算エラーを防ぐ）
        if len(val_loader) > 0:
            val_loss /= len(val_loader)
        else:
            val_loss = 0.0
        val_losses.append(val_loss)
        
        # 学習率の更新
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            elif scheduler_type != 'onecycle':
                scheduler.step()
        
        # 現在の学習率を表示
        current_lr = optimizer.param_groups[0]['lr']
        
        # 最良モデルの保存（エラーハンドリング付き）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                # 複数の保存先を試行
                model_save_paths = [
                    os.path.join(save_dir, 'best_model.pth'),
                    './best_model.pth',
                    f'./model_epoch_{epoch+1}_best.pth'
                ]
                
                saved = False
                for save_path in model_save_paths:
                    try:
                        # 保存先ディレクトリを確認・作成
                        save_path_dir = os.path.dirname(os.path.abspath(save_path))
                        if save_path_dir and not os.path.exists(save_path_dir):
                            os.makedirs(save_path_dir, exist_ok=True)
                        
                        torch.save(model.state_dict(), save_path)
                        print(f'Epoch {epoch+1}: New best model saved to {save_path} with validation loss: {val_loss:.6f}')
                        best_model_path = save_path
                        saved = True
                        break
                    except Exception as e:
                        print(f'Warning: Could not save model to {save_path}. Error: {e}')
                        continue
                
                if not saved:
                    print(f'Warning: Could not save model at epoch {epoch+1}')
                    
            except Exception as e:
                print(f'Warning: Model saving failed at epoch {epoch+1}. Error: {e}')
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, LR: {current_lr:.8f}')
    
    # 学習曲線の表示と保存
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 画像の保存
    try:
        training_curve_path = os.path.join(save_dir, 'training_curve.png')
        plt.savefig(training_curve_path, dpi=300, bbox_inches='tight')
        print(f"Training curve saved to {training_curve_path}")
    except Exception as e:
        print(f"Could not save training curve: {e}")
    
    plt.show()
    
    # テストデータがある場合は評価
    if test_loader is not None:
        test_loss = evaluate_model(model, test_loader, device, epochs-1, epochs)
        print(f'\nTest Loss: {test_loss:.6f}')
    
    return train_losses, val_losses, best_model_path


def evaluate_model(model, data_loader, device, current_epoch=0, max_epochs=100):
    """
    モデルを評価する関数
    
    Args:
        model: 評価するモデル
        data_loader: 評価用データローダー
        device: 計算デバイス
        current_epoch: 現在のエポック
        max_epochs: 最大エポック数
    
    Returns:
        average_loss: 平均損失
    """
    if len(data_loader) == 0:
        print("Warning: Data loader is empty, returning 0.0 loss")
        return 0.0
        
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # データをデバイスに転送
            heatmap = batch['heatmap'].to(device)
            positions = batch['positions'].to(device)
            valid_counts = batch['valid_count'].to(device)
            
            # 順伝播
            pred_coords, pred_conf = model(heatmap)
            
            # 損失計算
            loss = custom_loss_for_mchannel(
                pred_coords, 
                pred_conf, 
                positions,
                valid_counts,
                current_epoch=current_epoch,
                max_epochs=max_epochs
            )
            
            total_loss += loss.item()
    
    average_loss = total_loss / len(data_loader)
    return average_loss


def calculate_metrics(predictions, ground_truths, conf_threshold=0.3, distance_threshold=0.5):
    """
    予測精度の評価指標を計算
    
    Args:
        predictions: 予測結果のリスト
        ground_truths: 正解データのリスト
        conf_threshold: 信頼度閾値
        distance_threshold: 距離閾値
    
    Returns:
        metrics: 評価指標の辞書
    """
    total_gt = 0
    total_pred = 0
    total_matches = 0
    total_distance = 0.0
    
    for pred_coords, pred_conf, gt_positions in zip(predictions[0], predictions[1], ground_truths):
        # Ground Truth情報
        valid_count = len(gt_positions)
        total_gt += valid_count
        
        # 予測結果の情報（信頼度でフィルタリング）
        valid_preds = []
        for i, conf in enumerate(pred_conf):
            if conf >= conf_threshold:
                valid_preds.append(pred_coords[i])
        total_pred += len(valid_preds)
        
        # Ground Truthと予測のマッチング
        if valid_count > 0 and valid_preds:
            # 距離行列の計算
            distances = np.zeros((len(valid_preds), valid_count))
            
            for i, pred_pos in enumerate(valid_preds):
                for j, gt_pos in enumerate(gt_positions):
                    dist = np.sqrt((pred_pos[0] - gt_pos[0])**2 + (pred_pos[1] - gt_pos[1])**2)
                    distances[i, j] = dist
            
            # 各予測に対して最も近いGround Truthを見つける
            for i, pred_pos in enumerate(valid_preds):
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
    評価指標を表示する関数
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


def main_training():
    """
    メインのトレーニング関数
    """
    # データディレクトリの設定
    train_dir = './dataset3/train'
    val_dir = './dataset3/val'
    test_dir = './dataset3/test'
    
    # データローダーの作成
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_data_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=8,
        max_people=2,
        preserve_order=True
    )
    
    # モデルの初期化
    model = SimplifiedMultiPersonPredictor(max_people=2, input_channels=2)
    
    # モデルのトレーニング
    print("\n==== Starting Model Training ====")
    train_losses, val_losses, best_model_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=100,
        learning_rate=0.001,
        scheduler_type='cosine',
        save_dir='./outputs'
    )
    
    print(f"\nTraining completed!")
    if best_model_path:
        print(f"Best model saved at: {best_model_path}")
    
    return model, train_losses, val_losses, best_model_path


if __name__ == "__main__":
    # トレーニングの実行
    model, train_losses, val_losses, best_model_path = main_training()