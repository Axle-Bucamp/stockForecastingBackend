#!/usr/bin/env python3
"""
[ROCKET] Hygdra Forecasting - Master Test Runner

This script runs all test suites to validate the complete system:
1. Model consistency tests
2. Inference fix validation tests  
3. Training consistency tests

Author: Bucamp Axle
Date: $(date)
"""

import sys
import os
import subprocess
import json
from datetime import datetime
from typing import Dict, List, Any

class MasterTestRunner:
    """
    Master test runner that orchestrates all test suites.
    """
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        
    def run_test_suite(self, test_file: str, test_name: str) -> Dict[str, Any]:
        """Run a specific test suite and return results."""
        print(f"\n{'='*60}")
        print(f"[TEST] Running {test_name}")
        print(f"{'='*60}")
        
        try:
            # Run the test file
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Parse results
            test_result = {
                'file': test_file,
                'name': test_name,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Try to load JSON results if available
            json_files = [
                'test_results.json',
                'training_test_results.json'
            ]
            
            for json_file in json_files:
                if os.path.exists(json_file):
                    try:
                        with open(json_file, 'r') as f:
                            json_data = json.load(f)
                            test_result['detailed_results'] = json_data
                    except:
                        pass
            
            return test_result
            
        except subprocess.TimeoutExpired:
            return {
                'file': test_file,
                'name': test_name,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Test timed out after 5 minutes',
                'success': False,
                'timestamp': datetime.now().isoformat(),
                'error': 'Timeout'
            }
        except Exception as e:
            return {
                'file': test_file,
                'name': test_name,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites."""
        print("[ROCKET] Hygdra Forecasting - Master Test Runner")
        print("=" * 60)
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Define test suites
        test_suites = [
            {
                'file': 'test_inference_fix.py',
                'name': 'Inference Fix Validation Tests',
                'description': 'Validates the normalization bug fix in inference.py'
            },
            {
                'file': 'test_model_consistency.py', 
                'name': 'Model Consistency Tests',
                'description': 'Validates overall model consistency and behavior'
            },
            {
                'file': 'test_training_consistency.py',
                'name': 'Training Consistency Tests', 
                'description': 'Validates training pipeline and data loading'
            }
        ]
        
        # Run each test suite
        for suite in test_suites:
            print(f"\n[LIST] {suite['name']}")
            print(f"   Description: {suite['description']}")
            
            if not os.path.exists(suite['file']):
                print(f"   [FAIL] Test file not found: {suite['file']}")
                self.test_results[suite['name']] = {
                    'success': False,
                    'error': f'Test file not found: {suite["file"]}',
                    'timestamp': datetime.now().isoformat()
                }
                continue
            
            result = self.run_test_suite(suite['file'], suite['name'])
            self.test_results[suite['name']] = result
            
            # Print summary
            if result['success']:
                print(f"   [PASS] {suite['name']} PASSED")
            else:
                print(f"   [FAIL] {suite['name']} FAILED")
                if result.get('stderr'):
                    print(f"   Error: {result['stderr']}")
        
        return self.test_results
    
    def generate_master_report(self) -> str:
        """Generate a comprehensive master test report."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print("\n" + "=" * 80)
        print("[DATA] MASTER TEST REPORT")
        print("=" * 80)
        
        total_suites = len(self.test_results)
        successful_suites = sum(1 for result in self.test_results.values() if result.get('success', False))
        
        print(f"Test Run Duration: {duration}")
        print(f"Total Test Suites: {total_suites}")
        print(f"Successful Suites: {successful_suites}")
        print(f"Failed Suites: {total_suites - successful_suites}")
        print(f"Overall Success Rate: {successful_suites/total_suites*100:.1f}%")
        
        print("\n[LIST] Test Suite Results:")
        for suite_name, result in self.test_results.items():
            status = "[PASS] PASSED" if result.get('success', False) else "[FAIL] FAILED"
            print(f"  {suite_name}: {status}")
            
            if not result.get('success', False):
                if result.get('error'):
                    print(f"    Error: {result['error']}")
                elif result.get('stderr'):
                    print(f"    Error: {result['stderr']}")
        
        # Generate recommendations
        print("\n[TARGET] Recommendations:")
        if successful_suites == total_suites:
            print("  [SUCCESS] All test suites passed! The system is working correctly.")
            print("  [PASS] The normalization bug fix has been validated.")
            print("  [PASS] Model consistency between training and inference is confirmed.")
            print("  [PASS] Training pipeline is functioning properly.")
        else:
            print("  [WARNING]  Some test suites failed. Please review the issues above.")
            failed_suites = [name for name, result in self.test_results.items() if not result.get('success', False)]
            print(f"  [TOOL] Focus on fixing: {', '.join(failed_suites)}")
        
        # Save master report
        master_report = {
            'test_run_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'total_suites': total_suites,
                'successful_suites': successful_suites,
                'success_rate': successful_suites/total_suites*100
            },
            'test_results': self.test_results,
            'recommendations': self._generate_recommendations()
        }
        
        report_file = f"master_test_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(master_report, f, indent=2, default=str)
        
        print(f"\n[SAVE] Master report saved to: {report_file}")
        return report_file
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if all(result.get('success', False) for result in self.test_results.values()):
            recommendations.extend([
                "All test suites passed successfully!",
                "The normalization bug fix in inference.py is working correctly.",
                "Model consistency between training and inference is validated.",
                "The training pipeline is functioning properly.",
                "The system is ready for production use."
            ])
        else:
            recommendations.extend([
                "Some test suites failed - review the detailed results above.",
                "Focus on fixing the failed test suites before proceeding.",
                "Ensure all data loading and preprocessing is consistent.",
                "Verify that model training and inference use the same criteria."
            ])
            
            # Specific recommendations based on failed tests
            for suite_name, result in self.test_results.items():
                if not result.get('success', False):
                    if 'inference' in suite_name.lower():
                        recommendations.append("Pay special attention to the inference pipeline normalization fix.")
                    elif 'training' in suite_name.lower():
                        recommendations.append("Review the training data loading and sequence creation logic.")
                    elif 'consistency' in suite_name.lower():
                        recommendations.append("Ensure model behavior is consistent across different modes.")
        
        return recommendations


def main():
    """Main test runner."""
    print("[ROCKET] Hygdra Forecasting - Master Test Runner")
    print("=" * 60)
    
    # Initialize master test runner
    runner = MasterTestRunner()
    
    # Run all tests
    results = runner.run_all_tests()
    
    # Generate master report
    report_file = runner.generate_master_report()
    
    # Determine overall success
    all_passed = all(result.get('success', False) for result in results.values())
    
    if all_passed:
        print("\n[SUCCESS] ALL TEST SUITES PASSED!")
        print("[PASS] The system is fully validated and ready for use.")
        return 0
    else:
        print("\n[WARNING]  SOME TEST SUITES FAILED!")
        print("[FAIL] Please review and fix the issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())
