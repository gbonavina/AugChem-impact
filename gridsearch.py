from augchem.core import Augmentator
import itertools
import os 
from pathlib import Path
import json
from datetime import datetime
import tqdm
import time
import pandas as pd
from preprocessing import PreProcessing
from lstm_regressor import LSTM, LSTMTrainer
import torch
import torch.nn as nn 
import numpy as np

class GridSearch:
    def __init__(self, dataset_path='QM9.csv', output_dir='grid_search_datasets', model_config=None):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # params I got that were the best w/ optuna (baysean optimization) for an LSTM regressor
        self.model_config = model_config or {
            'epochs': 100,
            'batch_size': 32, 
            'lr': 0.00038419788129315696,
            'dropout': 0.3520475751955372,
            'weight_decay': 1.0528482325186746e-06,
            'num_layers': 2,
            'hidden_dim': 256,
            'embedding_dim': 64
        }

        smiles_methods = ['enumeration', 'mask', 'delete', 'swap', 'fusion']
        all_combinations = []
        
        # fazer as combinações dos 5 métodos (120 combinações!)
        for r in range(1, len(smiles_methods) + 1):
            for combo in itertools.combinations(smiles_methods, r):
                all_combinations.append(list(combo))

        self.param_grid = {
            'mask_ratio': [0.05, 0.1, 0.2, 0.3, 0.5],
            'delete_ratio': [0.05, 0.1, 0.2, 0.3, 0.5],
            'augment_percentage': [0.1, 0.3, 0.5, 0.7, 1.0],
            'augmentation_methods': all_combinations
        }

    def generate_parameter_combinations(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = []
        for methods in self.param_grid['augmentation_methods']:
            for aug_perc in self.param_grid['augment_percentage']:
                needs_mask_ratio = 'mask' in methods
                needs_delete_ratio = 'delete' in methods

                if needs_mask_ratio and needs_delete_ratio:
                    for mask_r in self.param_grid['mask_ratio']:
                        for delete_r in self.param_grid['delete_ratio']:
                            combinations.append({
                                'augmentation_methods': methods,
                                'augment_percentage': aug_perc,
                                'mask_ratio': mask_r,
                                'delete_ratio': delete_r
                            })

                elif needs_mask_ratio:
                    for mask_r in self.param_grid['mask_ratio']:
                        combinations.append({
                                'augmentation_methods': methods,
                                'augment_percentage': aug_perc,
                                'mask_ratio': mask_r,
                                'delete_ratio': 0.0
                            })
                        
                elif needs_delete_ratio:
                    for delete_r in self.param_grid['delete_ratio']:
                        combinations.append({
                                'augmentation_methods': methods,
                                'augment_percentage': aug_perc,
                                'mask_ratio': 0.0,
                                'delete_ratio': delete_r
                            })
                        
                else:
                    combinations.append({
                                'augmentation_methods': methods,
                                'augment_percentage': aug_perc,
                                'mask_ratio': 0.0,
                                'delete_ratio': 0.0
                    })


        return combinations

    def apply_augmentation(self, params):
        aug = Augmentator(seed=42)
    
        augment_params = {
            'dataset': self.dataset_path,
            'augmentation_methods': params['augmentation_methods'],
            'augment_percentage': params['augment_percentage'],
            'col_to_augment': 'SMILES_1',
            'property_col': 'Property_0',
            'isUnsupervised': False
        }
    
        if 'mask' in params['augmentation_methods']:
            augment_params['mask_ratio'] = params['mask_ratio']
            
        if 'delete' in params['augmentation_methods']:
            augment_params['delete_ratio'] = params['delete_ratio']
    
        augmented_data = aug.SMILES.augment_data(**augment_params)
    
        return augmented_data
    
    def _make_json_serializable(self, obj):       
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif hasattr(obj, 'item'):  # Para tensors PyTorch
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def train_model_on_dataset(self, augmented_data):
        try: 
            prep = PreProcessing()
            data_info = prep.prepare_data_from_df(augmented_data)

            model = LSTM(
                vocab_size=data_info['vocab_size'],
                embedding_dim=self.model_config['embedding_dim'],
                hidden_dim=self.model_config['hidden_dim'],
                num_layers=self.model_config['num_layers'],
                padding_idx=data_info['padding_idx'],
                dropout=self.model_config['dropout']
            )

            trainer = LSTMTrainer(model=model)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.model_config['lr'],
                weight_decay=self.model_config['weight_decay']
            )

            train_losses, val_losses = trainer.train_model(
                train_loader=data_info['train_loader'],
                val_loader=data_info['val_loader'],
                num_epochs=self.model_config['epochs'],
                criterion=criterion,
                optimizer=optimizer
            )

            eval_results = trainer.evaluate_model(
                test_dataloader=data_info['test_loader'],
                criterion=criterion,
                scaler=data_info['scaler']
            )

            return {
                'status': 'success',
                'metrics': {
                    'losses': {
                        'train_loss': float(train_losses[-1]),  
                        'val_loss': float(val_losses[-1]),      
                        'test_loss': float(eval_results['loss']) 
                    },
                    'regression_metrics': {
                        'test_r2': float(eval_results['r2']),     
                        'test_mae': float(eval_results['mae']),   
                        'test_mse': float(eval_results['mse']),  
                        'test_rmse': float(eval_results['rmse'])
                    }
                },
                'data_stats': self._make_json_serializable(data_info['data_stats'])  
            }
        
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def run_grid_search(self):
        """Pipeline otimizado: gera → treina → salva resultados → descarta"""
        combinations = self.generate_parameter_combinations()
        print(f"Starting optimized grid search with {len(combinations)} experiments")
        print(f"Memory optimization: Datasets generated and discarded immediately")
        print(f"Results will be saved in: {self.output_dir}")

        experiments_log = []
        successful_count = 0

        for i, params in enumerate(tqdm.tqdm(combinations, desc='Grid Search Progress')):
            experiment_start_time = datetime.now()
            
            methods_str = ', '.join(params['augmentation_methods'])
            print(f"\n[{i+1}/{len(combinations)}] Methods: {methods_str}")
            print(f"   Augment %: {params['augment_percentage']}")
            if 'mask' in params['augmentation_methods']:
                print(f"   Mask ratio: {params['mask_ratio']}")
            if 'delete' in params['augmentation_methods']:
                print(f"   Delete ratio: {params['delete_ratio']}")
                
            try: 
                # ETAPA 1: Gerar dataset aumentado em memória
                print("   Generating augmented dataset...")
                augmented_data = self.apply_augmentation(params)
                print(f"   Dataset generated: {len(augmented_data)} samples")
                
                # ETAPA 2: Treinar modelo imediatamente
                print("   Training neural network...")
                training_results = self.train_model_on_dataset(augmented_data)
                
                experiment_duration = (datetime.now() - experiment_start_time).total_seconds()
                
                experiment_result = {
                    'experiment_id': i,
                    'status': training_results['status'],
                    'parameters': params,
                    'metrics': training_results.get('metrics', {}),
                    'data_stats': training_results.get('data_stats', {}),
                    'duration_seconds': experiment_duration,
                    'dataset_size': len(augmented_data),
                    'timestamp': datetime.now().isoformat()
                }

                experiments_log.append(experiment_result)
                
                if training_results['status'] == 'success':
                    successful_count += 1
                    r2_score = training_results['metrics']['regression_metrics']['test_r2']
                    test_loss = training_results['metrics']['losses']['test_loss']
                    print(f"   Success! Test R²: {r2_score:.3f}, Loss: {test_loss:.4f}")
                else:
                    print(f"   ❌ Training failed: {training_results.get('error', 'Unknown error')}")
                
                del augmented_data
                
                if (i + 1) % 10 == 0:
                    self.save_intermediate_progress(experiments_log, i + 1)
                    print(f"   Progress saved: {successful_count}/{i+1} successful")

            except Exception as e:
                print(f"   ❌ Pipeline failed: {str(e)}")
                experiments_log.append({
                    'experiment_id': i,
                    'status': 'error',
                    'parameters': params,
                    'error': str(e),
                    'duration_seconds': (datetime.now() - experiment_start_time).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Análise final e salvamento
        print(f"\n Grid Search completed!")
        print(f"✅ Successful experiments: {successful_count}/{len(combinations)}")
        print(f"📊 Success rate: {successful_count/len(combinations)*100:.1f}%")
        
        self.generate_analysis_files(experiments_log)
        
        return experiments_log
    
    def save_intermediate_progress(self, experiments_log, completed_count):
        """Salva progresso intermediário para não perder dados"""
        temp_results = {
            'progress_info': {
                'completed_experiments': completed_count,
                'timestamp': datetime.now().isoformat(),
                'model_config': self.model_config
            },
            'experiments': experiments_log
        }
        
        temp_file = self.output_dir / f'progress_checkpoint_{completed_count}.json'
        with open(temp_file, 'w') as f:
            json.dump(temp_results, f, indent=2)
    
    def generate_analysis_files(self, experiments_log):
        """Gera arquivos CSV otimizados para análise do artigo"""
        successful = [exp for exp in experiments_log if exp['status'] == 'success']
        
        if not successful:
            print("❌ No successful experiments for analysis")
            return
        
        # CSV principal para análise estatística
        analysis_rows = []
        for exp in successful:
            params = exp['parameters']
            metrics = exp.get('metrics', {})
            
            if 'losses' in metrics and 'regression_metrics' in metrics:
                row = {
                    'experiment_id': exp['experiment_id'],
                    'method_combination': '|'.join(params['augmentation_methods']),
                    'num_methods': len(params['augmentation_methods']),
                    'augment_percentage': params['augment_percentage'],
                    'mask_ratio': params.get('mask_ratio', 'N/A'),
                    'delete_ratio': params.get('delete_ratio', 'N/A'),
                    'test_loss': metrics['losses']['test_loss'],
                    'test_r2': metrics['regression_metrics']['test_r2'],
                    'test_mae': metrics['regression_metrics']['test_mae'],
                    'test_rmse': metrics['regression_metrics']['test_rmse'],
                    'dataset_size': exp.get('dataset_size', 'N/A'),
                    'duration_seconds': exp.get('duration_seconds', 'N/A'),
                    'timestamp': exp['timestamp']
                }
                analysis_rows.append(row)
        
        if analysis_rows:
            # Criar DataFrame e salvar
            df_analysis = pd.DataFrame(analysis_rows)
            df_analysis.to_csv(self.output_dir / 'paper_analysis_results.csv', index=False)
            
            # Top 10 melhores experimentos
            top_10 = df_analysis.nsmallest(10, 'test_loss')
            top_10.to_csv(self.output_dir / 'top_10_experiments.csv', index=False)
            
            print(f"📈 Analysis files created:")
            print(f"   Main results: {self.output_dir / 'paper_analysis_results.csv'}")
            print(f"   Top 10: {self.output_dir / 'top_10_experiments.csv'}")
            
            # Mostrar melhor resultado
            best_exp = df_analysis.loc[df_analysis['test_loss'].idxmin()]
            print(f"\n🏆 BEST EXPERIMENT:")
            print(f"   ID: {best_exp['experiment_id']}")
            print(f"   Methods: {best_exp['method_combination']}")
            print(f"   Test Loss: {best_exp['test_loss']:.4f}")
            print(f"   Test R²: {best_exp['test_r2']:.3f}")
            print(f"   Augment %: {best_exp['augment_percentage']}")
        else:
            print("❌ No valid metrics found for analysis")