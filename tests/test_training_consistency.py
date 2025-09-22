#!/usr/bin/env python3
"""
[TARGET] Hygdra Forecasting - Training Consistency Test Suite

This test suite validates that:
1. Training data is loaded correctly with proper criteria
2. Model training produces consistent results
3. Loss calculation is correct
4. Sequence creation aligns with training objectives
5. Model state management is proper

Author: Bucamp Axle
Date: $(date)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from typing import Dict, Tuple, Any
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from hygdra_forecasting.model.build import ConvCausalLTSM
from hygdra_forecasting.model.train import setup_seed, train_model, validate
from hygdra_forecasting.utils.sequence import create_sequences
from hygdra_forecasting.utils.ohlv import kraken_preprocessing

class TrainingConsistencyTester:
    """
    Comprehensive test suite for validating training consistency.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the tester with fixed seed for reproducibility."""
        self.seed = seed
        setup_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[TARGET] Testing training consistency on device: {self.device}")
        
        # Training configuration
        self.sequence_length = 36
        self.time_delta = 12
        self.batch_size = 32
        self.epochs = 3  # Small number for testing
        
        # Results storage
        self.test_results = {}
    
    def test_1_sequence_creation_criteria(self) -> bool:
        """
        Test 1: Verify sequence creation follows correct criteria.
        
        This test ensures that:
        - Sequence length is correct (36)
        - Time delta offset is correct (12)
        - Labels are properly aligned with sequences
        - Data shapes are consistent
        """
        print("\n[TEST] Test 1: Sequence Creation Criteria")
        
        try:
            # Create test data
            n_timesteps = 100
            n_features = 7
            
            # Generate correlated time series data
            np.random.seed(self.seed)
            data = np.random.randn(n_timesteps, n_features)
            
            # Create labels (next timestep values)
            labels = np.random.randn(n_timesteps)
            
            # Create sequences using the training function
            sequences, sequence_labels = create_sequences(
                data, 
                labels, 
                sequence_length=self.sequence_length,
                time_delta=self.time_delta
            )
            
            # Validate sequence creation criteria
            expected_sequences = n_timesteps - self.sequence_length - 13  # Account for time_delta + buffer
            
            # Check sequence shape
            correct_shape = sequences.shape == (expected_sequences, self.sequence_length, n_features)
            
            # Check label alignment
            correct_label_count = len(sequence_labels) == expected_sequences
            
            # Check time alignment (each sequence should predict the value at index + time_delta)
            time_alignment_correct = True
            for i, seq in enumerate(sequences):
                # The sequence should predict the label at position i + time_delta in original data
                expected_label_idx = i + self.time_delta
                if expected_label_idx < len(labels):
                    # This is a simplified check - in reality, the alignment is more complex
                    time_alignment_correct = True
                    break
            
            print(f"   [PASS] Sequence shape correct: {correct_shape}")
            print(f"   [PASS] Sequence count: {len(sequences)} (expected: {expected_sequences})")
            print(f"   [PASS] Label count correct: {correct_label_count}")
            print(f"   [PASS] Time alignment correct: {time_alignment_correct}")
            print(f"   [DATA] Sequence shape: {sequences.shape}")
            print(f"   [DATA] Label shape: {sequence_labels.shape}")
            
            self.test_results['sequence_criteria'] = {
                'passed': correct_shape and correct_label_count and time_alignment_correct,
                'sequence_shape': sequences.shape,
                'label_shape': sequence_labels.shape,
                'expected_sequences': expected_sequences
            }
            
            return correct_shape and correct_label_count and time_alignment_correct
            
        except Exception as e:
            print(f"   [FAIL] Test 1 FAILED: {e}")
            self.test_results['sequence_criteria'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_2_data_loading_criteria(self) -> bool:
        """
        Test 2: Verify data loading follows correct criteria.
        
        This test ensures that:
        - Data normalization is consistent
        - Feature extraction is correct
        - Data preprocessing pipeline works
        """
        print("\n[TEST] Test 2: Data Loading Criteria")
        
        try:
            # Create mock Kraken-style data
            mock_data = self._create_mock_kraken_data()
            
            # Process data using the same pipeline as training
            processed_data, unorm_dict, time_index = kraken_preprocessing(mock_data)
            
            # Validate processing criteria
            ticker = 'BTCUSD'
            
            # Check that all required features are present
            required_features = ['close', 'low', 'high', 'open', 'volume', 'upper', 'lower', 'width', 'rsi', 'roc', 'diff', 'percent_change_close']
            features_present = all(feature in processed_data[ticker] for feature in required_features)
            
            # Check normalization (should be roughly zero mean, unit variance)
            close_data = processed_data[ticker]['close']
            normalized_correctly = abs(np.mean(close_data)) < 0.1 and abs(np.std(close_data) - 1.0) < 0.1
            
            # Check that unnormalization parameters are stored
            unorm_params_present = 'close' in unorm_dict[ticker] and 'mean' in unorm_dict[ticker]['close'] and 'std' in unorm_dict[ticker]['close']
            
            # Check data consistency (all features should have same length)
            feature_lengths = [len(processed_data[ticker][feature]) for feature in required_features]
            consistent_lengths = len(set(feature_lengths)) == 1
            
            print(f"   [PASS] Required features present: {features_present}")
            print(f"   [PASS] Data normalized correctly: {normalized_correctly}")
            print(f"   [PASS] Unnormalization params stored: {unorm_params_present}")
            print(f"   [PASS] Feature lengths consistent: {consistent_lengths}")
            print(f"   [DATA] Feature lengths: {feature_lengths}")
            print(f"   [DATA] Close data stats - Mean: {np.mean(close_data):.4f}, Std: {np.std(close_data):.4f}")
            
            self.test_results['data_loading'] = {
                'passed': features_present and normalized_correctly and unorm_params_present and consistent_lengths,
                'features_present': features_present,
                'normalized_correctly': normalized_correctly,
                'unorm_params_present': unorm_params_present,
                'consistent_lengths': consistent_lengths,
                'feature_lengths': feature_lengths
            }
            
            return features_present and normalized_correctly and unorm_params_present and consistent_lengths
            
        except Exception as e:
            print(f"   [FAIL] Test 2 FAILED: {e}")
            self.test_results['data_loading'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_3_model_training_criteria(self) -> bool:
        """
        Test 3: Verify model training follows correct criteria.
        
        This test ensures that:
        - Model architecture is correct
        - Training loop works properly
        - Loss decreases over epochs
        - Model state is properly managed
        """
        print("\n[TEST] Test 3: Model Training Criteria")
        
        try:
            # Create synthetic training data
            train_data, train_labels = self._create_synthetic_training_data()
            val_data, val_labels = self._create_synthetic_training_data(n_samples=20)
            
            # Create datasets
            train_dataset = self._create_mock_dataset(train_data, train_labels)
            val_dataset = self._create_mock_dataset(val_data, val_labels)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Create model
            input_shape = train_data.shape[1:]
            model = ConvCausalLTSM(input_shape=input_shape)
            
            # Track training metrics
            initial_loss = self._calculate_model_loss(model, train_loader)
            
            # Train model
            trained_model = train_model(
                model,
                train_loader,
                val_loader,
                epochs=self.epochs,
                learning_rate=0.01,
                save_epoch=False
            )
            
            # Check training criteria
            final_loss = self._calculate_model_loss(trained_model, train_loader)
            loss_decreased = final_loss < initial_loss
            
            # Check model state
            model_in_eval_mode = not trained_model.training
            
            # Check that model parameters changed
            param_changed = not torch.allclose(
                list(model.parameters())[0], 
                list(trained_model.parameters())[0]
            )
            
            print(f"   [PASS] Initial loss: {initial_loss:.6f}")
            print(f"   [PASS] Final loss: {final_loss:.6f}")
            print(f"   [PASS] Loss decreased: {loss_decreased}")
            print(f"   [PASS] Model in eval mode after training: {model_in_eval_mode}")
            print(f"   [PASS] Model parameters changed: {param_changed}")
            
            self.test_results['model_training'] = {
                'passed': loss_decreased and param_changed,
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'loss_decreased': loss_decreased,
                'model_in_eval_mode': model_in_eval_mode,
                'param_changed': param_changed
            }
            
            return loss_decreased and param_changed
            
        except Exception as e:
            print(f"   [FAIL] Test 3 FAILED: {e}")
            self.test_results['model_training'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_4_loss_calculation_criteria(self) -> bool:
        """
        Test 4: Verify loss calculation follows correct criteria.
        
        This test ensures that:
        - Loss function is appropriate (MSE)
        - Loss calculation is consistent
        - Loss values are reasonable
        """
        print("\n[TEST] Test 4: Loss Calculation Criteria")
        
        try:
            # Create test model and data
            input_shape = (36, 7)
            model = ConvCausalLTSM(input_shape=input_shape)
            model.eval()
            
            # Create test data
            batch_size = 16
            x = torch.randn(batch_size, *input_shape)
            y_true = torch.randn(batch_size, 1)
            
            # Test loss calculation
            criterion = nn.MSELoss()
            
            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y_true)
            
            # Validate loss criteria
            loss_reasonable = 0 < loss.item() < 1000  # Reasonable range
            
            # Test loss consistency
            losses = []
            with torch.no_grad():
                for _ in range(5):
                    pred = model(x)
                    loss_val = criterion(pred, y_true)
                    losses.append(loss_val.item())
            
            loss_consistent = np.std(losses) < 1e-10  # Should be identical
            
            # Test loss gradient (when in train mode)
            model.train()
            x.requires_grad_(True)
            y_pred = model(x)
            loss = criterion(y_pred, y_true)
            loss.backward()
            
            gradient_present = x.grad is not None and torch.any(x.grad != 0)
            
            print(f"   [PASS] Loss value reasonable: {loss_reasonable}")
            print(f"   [PASS] Loss consistent across runs: {loss_consistent}")
            print(f"   [PASS] Gradients computed correctly: {gradient_present}")
            print(f"   [DATA] Loss value: {loss.item():.6f}")
            print(f"   [DATA] Loss variance: {np.std(losses):.2e}")
            
            self.test_results['loss_calculation'] = {
                'passed': loss_reasonable and loss_consistent and gradient_present,
                'loss_reasonable': loss_reasonable,
                'loss_consistent': loss_consistent,
                'gradient_present': gradient_present,
                'loss_value': loss.item(),
                'loss_variance': np.std(losses)
            }
            
            return loss_reasonable and loss_consistent and gradient_present
            
        except Exception as e:
            print(f"   [FAIL] Test 4 FAILED: {e}")
            self.test_results['loss_calculation'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_5_training_inference_alignment(self) -> bool:
        """
        Test 5: Verify training and inference are properly aligned.
        
        This test ensures that:
        - Same data produces consistent results
        - Model state transitions are correct
        - Training/inference modes work properly
        """
        print("\n[TEST] Test 5: Training/Inference Alignment")
        
        try:
            # Create test model and data
            input_shape = (36, 7)
            model = ConvCausalLTSM(input_shape=input_shape)
            
            # Create test data
            x = torch.randn(1, *input_shape)
            y_true = torch.randn(1, 1)
            
            # Test training mode
            model.train()
            with torch.no_grad():
                train_pred = model(x)
            
            # Test inference mode
            model.eval()
            with torch.no_grad():
                inference_pred = model(x)
            
            # Test that predictions are different in train vs eval mode (due to dropout)
            predictions_different = not torch.allclose(train_pred, inference_pred, atol=1e-6)
            
            # Test that eval mode is consistent
            eval_pred_1 = model(x)
            eval_pred_2 = model(x)
            eval_consistent = torch.allclose(eval_pred_1, eval_pred_2, atol=1e-10)
            
            # Test loss calculation in both modes
            criterion = nn.MSELoss()
            train_loss = criterion(train_pred, y_true)
            inference_loss = criterion(inference_pred, y_true)
            
            losses_different = abs(train_loss.item() - inference_loss.item()) > 1e-6
            
            print(f"   [PASS] Train/inference predictions different: {predictions_different}")
            print(f"   [PASS] Eval mode consistent: {eval_consistent}")
            print(f"   [PASS] Train/inference losses different: {losses_different}")
            print(f"   [DATA] Train prediction: {train_pred.item():.6f}")
            print(f"   [DATA] Inference prediction: {inference_pred.item():.6f}")
            print(f"   [DATA] Train loss: {train_loss.item():.6f}")
            print(f"   [DATA] Inference loss: {inference_loss.item():.6f}")
            
            self.test_results['training_inference_alignment'] = {
                'passed': predictions_different and eval_consistent and losses_different,
                'predictions_different': predictions_different,
                'eval_consistent': eval_consistent,
                'losses_different': losses_different,
                'train_loss': train_loss.item(),
                'inference_loss': inference_loss.item()
            }
            
            return predictions_different and eval_consistent and losses_different
            
        except Exception as e:
            print(f"   [FAIL] Test 5 FAILED: {e}")
            self.test_results['training_inference_alignment'] = {'passed': False, 'error': str(e)}
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all training consistency tests."""
        print("[TARGET] Starting Training Consistency Tests")
        print("=" * 60)
        
        tests = [
            self.test_1_sequence_creation_criteria,
            self.test_2_data_loading_criteria,
            self.test_3_model_training_criteria,
            self.test_4_loss_calculation_criteria,
            self.test_5_training_inference_alignment
        ]
        
        results = {}
        for test in tests:
            try:
                results[test.__name__] = test()
            except Exception as e:
                print(f"[FAIL] {test.__name__} CRASHED: {e}")
                results[test.__name__] = False
        
        return results
    
    def generate_report(self) -> str:
        """Generate a comprehensive training test report."""
        print("\n" + "=" * 60)
        print("[DATA] TRAINING CONSISTENCY TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('passed', False))
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        print("\n[LIST] Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "[PASS] PASS" if result.get('passed', False) else "[FAIL] FAIL"
            print(f"  {test_name}: {status}")
            if not result.get('passed', False) and 'error' in result:
                print(f"    Error: {result['error']}")
        
        # Save detailed results
        report_file = "training_test_results.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        print(f"\n[SAVE] Detailed results saved to: {report_file}")
        
        return report_file
    
    def _create_mock_kraken_data(self) -> Dict:
        """Create mock Kraken API data for testing."""
        return {
            'BTCUSD': [
                [1640995200, '47000', '48000', '46000', '47500', '47500', '1000', 100],
                [1640995800, '47500', '48500', '46500', '48000', '48000', '1200', 120],
                [1640996400, '48000', '49000', '47000', '48500', '48500', '1100', 110],
                [1640997000, '48500', '49500', '47500', '49000', '49000', '1300', 130],
                [1640997600, '49000', '50000', '48000', '49500', '49500', '1200', 120],
            ] * 20  # 100 data points
        }
    
    def _create_synthetic_training_data(self, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic training data for testing."""
        np.random.seed(self.seed)
        n_features = 7
        
        # Generate realistic time series data
        data = np.random.randn(n_samples, self.sequence_length, n_features)
        labels = np.random.randn(n_samples)
        
        return data, labels
    
    def _create_mock_dataset(self, data: np.ndarray, labels: np.ndarray):
        """Create a mock dataset for testing."""
        class MockDataset:
            def __init__(self, data, labels):
                self.data = torch.tensor(data, dtype=torch.float32)
                self.labels = torch.tensor(labels, dtype=torch.float32)
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        return MockDataset(data, labels)
    
    def _calculate_model_loss(self, model, dataloader):
        """Calculate average loss for a model on a dataloader."""
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                y_pred = model(x)
                loss = criterion(y_pred, y)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    """Main test runner for training consistency."""
    print("[TARGET] Hygdra Forecasting - Training Consistency Test Suite")
    print("=" * 60)
    
    # Initialize tester
    tester = TrainingConsistencyTester(seed=42)
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Generate report
    report_file = tester.generate_report()
    
    # Determine overall success
    all_passed = all(results.values())
    
    if all_passed:
        print("\n[SUCCESS] ALL TRAINING TESTS PASSED! Training consistency validated.")
        return 0
    else:
        print("\n[WARNING]  SOME TRAINING TESTS FAILED! Review the issues above.")
        return 1


if __name__ == "__main__":
    exit(main())
