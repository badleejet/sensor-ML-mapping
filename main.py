#!/usr/bin/env python
# coding: utf-8

"""
メイン実行スクリプト
データローダー、モデル定義、トレーニング、可視化の各コンポーネントを統合
"""

import os
import sys
import argparse
from data_loader import create_data_loaders, debug_data_loading
from models import SimplifiedMultiPersonPredictor, model_summary
from training import train_model, evaluate_model, main_training
from visualization import run_inference_and_visualize, main_inference


def setup_directories():
    """必要なディレクトリを作成"""
    directories = ['./outputs', './models', './results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created/verified: {directory}")


def run_training_pipeline(config):
    """
    トレーニングパイプラインを実行
    
    Args:
        config: 設定辞書
    """
    print("=" * 60)
    print("STARTING TRAINING PIPELINE")
    print("=" * 60)
    
    # データローダーの作成
    print("\n1. Creating data loaders...")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_data_loaders(
        train_dir=config['train_dir'],
        val_dir=config['val_dir'],
        test_dir=config['test_dir'],
        batch_size=config['batch_size'],
        max_people=config['max_people'],
        preserve_order=config['preserve_order']
    )
    
    # モデルの初期化
    print("\n2. Initializing model...")
    model = SimplifiedMultiPersonPredictor(
        max_people=config['max_people'], 
        input_channels=config['input_channels']
    )
    
    # モデルの概要表示
    if config['show_model_summary']:
        print("\n3. Model Summary:")
        model_summary(model)
    
    # モデルのトレーニング
    print("\n4. Starting training...")
    train_losses, val_losses, best_model_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        scheduler_type=config['scheduler_type'],
        save_dir=config['save_dir']
    )
    
    print(f"\nTraining completed successfully!")
    if best_model_path:
        print(f"Best model saved at: {best_model_path}")
    
    return best_model_path, train_losses, val_losses


def run_inference_pipeline(config, model_path=None):
    """
    推論パイプラインを実行
    
    Args:
        config: 設定辞書
        model_path: 学習済みモデルのパス
    """
    print("=" * 60)
    print("STARTING INFERENCE PIPELINE")
    print("=" * 60)
    
    if model_path is None:
        model_path = config['model_path']
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, None
    
    # 推論と可視化を実行
    results, metrics = run_inference_and_visualize(
        model_path=model_path,
        test_json_path=config['test_json_path'],
        conf_threshold=config['conf_threshold'],
        distance_threshold=config['distance_threshold'],
        add_trajectory=config['add_trajectory'],
        save_path=config['save_visualization_path'],
        export_csv_path=config['export_csv_path']
    )
    
    print("\nInference completed successfully!")
    return results, metrics


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Multi-person tracking model training and inference')
    parser.add_argument('--mode', choices=['train', 'inference', 'both', 'debug'], default='both',
                        help='Execution mode: train, inference, both, or debug')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (not implemented)')
    parser.add_argument('--data_dir', type=str, default='./dataset3',
                        help='Base directory for dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--model_path', type=str, default='./outputs/best_model.pth',
                        help='Path to save/load model')
    parser.add_argument('--test_json', type=str, default='./dataset3/test/slide3_two_sekkin_4_rev.json',
                        help='Path to test JSON file')
    
    args = parser.parse_args()
    
    # 必要なディレクトリを作成
    setup_directories()
    
    # 設定を作成
    config = {
        # データ設定
        'train_dir': os.path.join(args.data_dir, 'train'),
        'val_dir': os.path.join(args.data_dir, 'val'),
        'test_dir': os.path.join(args.data_dir, 'test'),
        
        # モデル設定
        'max_people': 2,
        'input_channels': 2,
        'show_model_summary': True,
        
        # トレーニング設定
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'scheduler_type': 'cosine',
        'preserve_order': True,
        'save_dir': './outputs',
        
        # 推論設定
        'model_path': args.model_path,
        'test_json_path': args.test_json,
        'conf_threshold': 0.3,
        'distance_threshold': 0.5,
        'add_trajectory': True,
        'save_visualization_path': './outputs/inference_visualization.png',
        'export_csv_path': './outputs/inference_results.csv'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 実行モードに応じて処理を分岐
    if args.mode == 'debug':
        print("\n" + "=" * 60)
        print("RUNNING DEBUG MODE")
        print("=" * 60)
        debug_data_loading()
        
    elif args.mode == 'train':
        model_path, train_losses, val_losses = run_training_pipeline(config)
        
    elif args.mode == 'inference':
        results, metrics = run_inference_pipeline(config)
        
    elif args.mode == 'both':
        # トレーニングを実行
        model_path, train_losses, val_losses = run_training_pipeline(config)
        
        # トレーニング完了後に推論を実行
        if model_path:
            print("\n" + "=" * 60)
            print("TRANSITIONING TO INFERENCE")
            print("=" * 60)
            results, metrics = run_inference_pipeline(config, model_path)
        else:
            print("Training failed, skipping inference.")
    
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
