#!/usr/bin/env python3
"""
Analyze Vanta Ledger HRM Dataset
"""

import numpy as np

def analyze_dataset():
    """Analyze the dataset contents"""
    
    # Load dataset
    data = np.load('dataset/data/vanta-ledger-hrm/vanta_ledger_hrm_dataset.npz')
    
    print("📊 Vanta Ledger HRM Dataset Analysis")
    print("=" * 40)
    
    # Basic info
    print(f"📁 Total samples: {len(data['inputs']):,}")
    print(f"🔤 Input sequence length: {data['inputs'].shape[1]} tokens")
    print(f"🎯 Target sequence length: {data['targets'].shape[1]} tokens")
    print(f"🏷️  Task types: {list(np.unique(data['task_types']))}")
    print()
    
    # Task breakdown
    print("📈 Sample breakdown by task:")
    for task in np.unique(data['task_types']):
        count = np.sum(data['task_types'] == task)
        percentage = count / len(data) * 100
        print(f"  • {task}: {count:,} samples ({percentage:.1f}%)")
    print()
    
    # Data format
    print("🔍 Data format:")
    print(f"  • Inputs: {data['inputs'].dtype} (tokenized text)")
    print(f"  • Targets: {data['targets'].dtype} (tokenized responses)")
    print(f"  • Task types: {data['task_types'].dtype} (string labels)")
    print()
    
    # Sample data
    print("📝 Sample data (first 3 samples):")
    for i in range(min(3, len(data['inputs']))):
        print(f"  Sample {i+1}:")
        print(f"    Input: {data['inputs'][i][:10]}... (length: {data['inputs'][i].shape[0]})")
        print(f"    Target: {data['targets'][i][:10]}... (length: {data['targets'][i].shape[0]})")
        print(f"    Task: {data['task_types'][i]}")
        print()

if __name__ == "__main__":
    analyze_dataset()
