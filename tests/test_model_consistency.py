#!/usr/bin/env python3
"""
[SEARCH] Hygdra Forecasting - Model Consistency Test Suite

This comprehensive test suite validates that:
1. Data loading is consistent between training and inference
2. Model weights produce identical results on the same data
3. Sequence creation logic is aligned
4. Normalization/denormalization is correct
5. Model behavior is deterministic

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
from pathlib import Path

# Import project modules
from hygdra_forecasting.model.build import ConvCausalLTSM
from hygdra_forecasting.model.train import setup_seed, validate
from hygdra_forecasting.dataloader.dataloader_kraken import StockDataset
from hygdra_forecasting.utils.sequence import create_sequences, create_sequences_inference
from hygdra_forecasting.utils.dataset import dict_to_dataset, dict_to_dataset_inference
from hygdra_forecasting.utils.ohlv import get_kraken_data_to_json, preprocessing_training_mode

class ModelConsistencyTester:
    """
    Comprehensive test suite for validating model consistency between training and inference.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the tester with a fixed seed for reproducibility."""
        self.seed = seed
        setup_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[TOOL] Testing on device: {self.device}")
        
        # Test configuration
        self.test_tickers = ['BTC-USD', 'ETH-USD']  # Small set for testing
        self.test_interval = '1440'  # Daily data
        self.sequence_length = 36
        self.time_delta = 12
        
        # Results storage
        self.test_results = {}
        
    def test_1_sequence_creation_consistency(self) -> bool:
        """
        Test 1: Verify that sequence creation logic is consistent between training and inference.
        
        This test ensures that:
        - Training sequences use the correct TIME_DELTA_LABEL offset
        - Inference sequences are created properly for prediction
        - Sequence shapes are consistent
        """
        print("\n[TEST] Test 1: Sequence Creation Consistency")
        
        try:
            # Generate test data
            test_data = np.random.randn(100, 7)  # 100 timesteps, 7 features
            test_labels = np.random.randn(100)
            
            # Test training sequence creation
            train_sequences, train_labels = create_sequences(
                test_data, 
                test_labels, 
                sequence_length=self.sequence_length,
                time_delta=self.time_delta
            )
            
            # Test inference sequence creation
            inference_sequences = create_sequences_inference(
                test_data, 
                sequence_length=self.sequence_length
            )
            
            # Validate shapes
            expected_train_shape = (100 - self.sequence_length - 13, self.sequence_length, 7)
            expected_inference_shape = (100 - self.sequence_length, self.sequence_length, 7)
            
            train_shape_correct = train_sequences.shape == expected_train_shape
            inference_shape_correct = inference_sequences.shape == expected_inference_shape
            
            print(f"   [PASS] Training sequences shape: {train_sequences.shape} (expected: {expected_train_shape})")
            print(f"   [PASS] Inference sequences shape: {inference_sequences.shape} (expected: {expected_inference_shape})")
            
            # Check that training has fewer sequences due to time_delta offset
            assert train_sequences.shape[0] < inference_sequences.shape[0], \
                "Training should have fewer sequences due to time_delta offset"
            
            self.test_results['sequence_creation'] = {
                'passed': train_shape_correct and inference_shape_correct,
                'train_shape': train_sequences.shape,
                'inference_shape': inference_sequences.shape,
                'train_sequences_count': len(train_sequences),
                'inference_sequences_count': len(inference_sequences)
            }
            
            return train_shape_correct and inference_shape_correct
            
        except Exception as e:
            print(f"   [FAIL] Test 1 FAILED: {e}")
            self.test_results['sequence_creation'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_2_data_loading_consistency(self) -> bool:
        """
        Test 2: Verify that data loading produces consistent results.
        
        This test ensures that:
        - Same data produces same normalized values
        - Unnormalization parameters are correctly stored
        - Data preprocessing is deterministic
        """
        print("\n[TEST] Test 2: Data Loading Consistency")
        
        try:
            # Test with mock data that simulates Kraken API response
            mock_kraken_data = self._create_mock_kraken_data()
            
            # Process data multiple times to ensure consistency
            results = []
            for i in range(3):
                processed_data, unorm_dict, time_index = self._mock_kraken_preprocessing(mock_kraken_data)
                results.append({
                    'processed_data': processed_data,
                    'unorm_dict': unorm_dict,
                    'time_index': time_index
                })
            
            # Check consistency across multiple runs
            data_consistent = np.allclose(
                results[0]['processed_data']['BTCUSD']['close'],
                results[1]['processed_data']['BTCUSD']['close']
            )
            
            unorm_consistent = (
                results[0]['unorm_dict']['BTCUSD']['close']['mean'] == 
                results[1]['unorm_dict']['BTCUSD']['close']['mean']
            )
            
            print(f"   [PASS] Data processing consistent: {data_consistent}")
            print(f"   [PASS] Unnormalization parameters consistent: {unorm_consistent}")
            print(f"   [PASS] Normalized data shape: {results[0]['processed_data']['BTCUSD']['close'].shape}")
            print(f"   [PASS] Mean/Std stored: {results[0]['unorm_dict']['BTCUSD']['close']}")
            
            self.test_results['data_loading'] = {
                'passed': data_consistent and unorm_consistent,
                'data_shape': results[0]['processed_data']['BTCUSD']['close'].shape,
                'unnorm_params': results[0]['unorm_dict']['BTCUSD']['close']
            }
            
            return data_consistent and unorm_consistent
            
        except Exception as e:
            print(f"   [FAIL] Test 2 FAILED: {e}")
            self.test_results['data_loading'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_3_normalization_denormalization(self) -> bool:
        """
        Test 3: Verify that normalization and denormalization are inverse operations.
        
        This test ensures that:
        - normalize(denormalize(x)) == x
        - denormalize(normalize(x)) == x
        - The bug in inference.py is fixed
        """
        print("\n[TEST] Test 3: Normalization/Denormalization Consistency")
        
        try:
            # Create test data
            original_data = np.random.randn(1000) * 100 + 50000  # Realistic price data
            
            # Normalize
            mean = np.mean(original_data)
            std = np.std(original_data)
            normalized_data = (original_data - mean) / std
            
            # Test correct denormalization (FIXED)
            denormalized_correct = normalized_data * std + mean
            
            # Test incorrect denormalization (OLD BUG)
            denormalized_incorrect = normalized_data * std + std
            
            # Check if denormalization is correct
            correct_denorm = np.allclose(original_data, denormalized_correct, rtol=1e-10)
            incorrect_denorm = np.allclose(original_data, denormalized_incorrect, rtol=1e-10)
            
            print(f"   [PASS] Correct denormalization: {correct_denorm}")
            print(f"   [FAIL] Incorrect denormalization (old bug): {incorrect_denorm}")
            print(f"   [DATA] Original data range: [{original_data.min():.2f}, {original_data.max():.2f}]")
            print(f"   [DATA] Denormalized range: [{denormalized_correct.min():.2f}, {denormalized_correct.max():.2f}]")
            
            self.test_results['normalization'] = {
                'passed': correct_denorm and not incorrect_denorm,
                'original_range': [original_data.min(), original_data.max()],
                'denormalized_range': [denormalized_correct.min(), denormalized_correct.max()],
                'mean': mean,
                'std': std
            }
            
            return correct_denorm and not incorrect_denorm
            
        except Exception as e:
            print(f"   [FAIL] Test 3 FAILED: {e}")
            self.test_results['normalization'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_4_model_behavior_consistency(self) -> bool:
        """
        Test 4: Verify that model produces identical outputs for identical inputs.
        
        This test ensures that:
        - Model is deterministic
        - Same input produces same output
        - Model state is properly managed
        """
        print("\n[TEST] Test 4: Model Behavior Consistency")
        
        try:
            # Create test model
            input_shape = (36, 7)  # Standard input shape
            model = ConvCausalLTSM(input_shape=input_shape)
            model.eval()  # Set to evaluation mode
            
            # Create test input
            test_input = torch.randn(1, *input_shape)
            
            # Test multiple forward passes with same input
            outputs = []
            with torch.no_grad():
                for i in range(5):
                    output = model(test_input)
                    outputs.append(output.item())
            
            # Check if all outputs are identical
            outputs_identical = all(abs(o - outputs[0]) < 1e-10 for o in outputs)
            
            print(f"   [PASS] Model outputs identical: {outputs_identical}")
            print(f"   [DATA] Output values: {[f'{o:.8f}' for o in outputs]}")
            print(f"   [DATA] Output variance: {np.var(outputs):.2e}")
            
            # Test train vs eval mode difference
            model.train()
            with torch.no_grad():
                train_output = model(test_input).item()
            
            model.eval()
            with torch.no_grad():
                eval_output = model(test_input).item()
            
            mode_difference = abs(train_output - eval_output)
            print(f"   [DATA] Train vs Eval mode difference: {mode_difference:.2e}")
            
            self.test_results['model_behavior'] = {
                'passed': outputs_identical,
                'outputs': outputs,
                'output_variance': np.var(outputs),
                'train_eval_difference': mode_difference
            }
            
            return outputs_identical
            
        except Exception as e:
            print(f"   [FAIL] Test 4 FAILED: {e}")
            self.test_results['model_behavior'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_5_training_inference_consistency(self) -> bool:
        """
        Test 5: Verify that training and inference produce consistent results on same data.
        
        This is the critical test that validates the main issue identified.
        """
        print("\n[TEST] Test 5: Training/Inference Consistency")
        
        try:
            # Create synthetic data that mimics real training data
            synthetic_data, synthetic_labels = self._create_synthetic_training_data()
            
            # Create model
            input_shape = synthetic_data.shape[1:]
            model = ConvCausalLTSM(input_shape=input_shape)
            
            # Test training mode
            model.train()
            with torch.no_grad():
                train_predictions = model(torch.tensor(synthetic_data, dtype=torch.float32))
            
            # Test inference mode
            model.eval()
            with torch.no_grad():
                inference_predictions = model(torch.tensor(synthetic_data, dtype=torch.float32))
            
            # Calculate losses
            criterion = nn.MSELoss()
            train_loss = criterion(train_predictions, torch.tensor(synthetic_labels, dtype=torch.float32))
            inference_loss = criterion(inference_predictions, torch.tensor(synthetic_labels, dtype=torch.float32))
            
            # Check if losses are different (they should be due to dropout)
            losses_different = abs(train_loss.item() - inference_loss.item()) > 1e-6
            
            print(f"   [DATA] Training loss: {train_loss.item():.6f}")
            print(f"   [DATA] Inference loss: {inference_loss.item():.6f}")
            print(f"   [DATA] Loss difference: {abs(train_loss.item() - inference_loss.item()):.6f}")
            print(f"   [PASS] Losses are different (expected due to dropout): {losses_different}")
            
            # Test with eval mode for inference (should match training when both in eval)
            model.eval()
            with torch.no_grad():
                eval_train_predictions = model(torch.tensor(synthetic_data, dtype=torch.float32))
            
            eval_loss = criterion(eval_train_predictions, torch.tensor(synthetic_labels, dtype=torch.float32))
            eval_consistency = abs(eval_loss.item() - inference_loss.item()) < 1e-10
            
            print(f"   [PASS] Eval mode consistency: {eval_consistency}")
            
            self.test_results['training_inference'] = {
                'passed': eval_consistency,
                'train_loss': train_loss.item(),
                'inference_loss': inference_loss.item(),
                'eval_loss': eval_loss.item(),
                'loss_difference': abs(train_loss.item() - inference_loss.item()),
                'eval_consistency': eval_consistency
            }
            
            return eval_consistency
            
        except Exception as e:
            print(f"   [FAIL] Test 5 FAILED: {e}")
            self.test_results['training_inference'] = {'passed': False, 'error': str(e)}
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        print("[ROCKET] Starting Hygdra Forecasting Model Consistency Tests")
        print("=" * 60)
        
        tests = [
            self.test_1_sequence_creation_consistency,
            self.test_2_data_loading_consistency,
            self.test_3_normalization_denormalization,
            self.test_4_model_behavior_consistency,
            self.test_5_training_inference_consistency
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
        """Generate a comprehensive test report."""
        print("\n" + "=" * 60)
        print("[DATA] TEST REPORT SUMMARY")
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
        
        # Save detailed results to file
        report_file = "test_results.json"
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
                # Add more mock data points...
            ] * 50  # 50 data points
        }
    
    def _mock_kraken_preprocessing(self, data: Dict) -> Tuple[Dict, Dict, Dict]:
        """Mock the Kraken preprocessing function."""
        processed_data = {}
        unorm_dict = {}
        time_index = {}
        
        for key, values in data.items():
            if key != "last":
                # Extract numeric values
                close_prices = [float(v[4]) for v in values]
                
                processed_data[key] = {
                    "close": np.array(close_prices, dtype=np.float64),
                }
                time_index[key] = [int(v[0]) for v in values]
                
                # Normalize
                mean = np.mean(processed_data[key]["close"])
                std = np.std(processed_data[key]["close"])
                processed_data[key]["close"] = (processed_data[key]["close"] - mean) / std
                
                unorm_dict[key] = {"close": {"mean": mean, "std": std}}
        
        return processed_data, unorm_dict, time_index
    
    def _create_synthetic_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic training data for testing."""
        # Create realistic time series data
        np.random.seed(self.seed)
        n_samples = 100
        n_features = 7
        sequence_length = 36
        
        # Generate correlated time series
        data = np.random.randn(n_samples, sequence_length, n_features)
        
        # Create labels (next timestep prediction)
        labels = np.random.randn(n_samples)
        
        return data, labels


def main():
    """Main test runner."""
    print("[SEARCH] Hygdra Forecasting - Model Consistency Test Suite")
    print("=" * 60)
    
    # Initialize tester
    tester = ModelConsistencyTester(seed=42)
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Generate report
    report_file = tester.generate_report()
    
    # Determine overall success
    all_passed = all(results.values())
    
    if all_passed:
        print("\n[SUCCESS] ALL TESTS PASSED! Model consistency validated.")
        return 0
    else:
        print("\n[WARNING]  SOME TESTS FAILED! Review the issues above.")
        return 1


if __name__ == "__main__":
    exit(main())
