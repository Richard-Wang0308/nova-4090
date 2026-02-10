"""Simple script to split training data into train/val."""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Split training data into train/val")
    parser.add_argument('--input', type=str, default='data/train.csv', help='Input CSV file')
    parser.add_argument('--train_output', type=str, default='data/train_split.csv', help='Output train CSV')
    parser.add_argument('--val_output', type=str, default='data/val.csv', help='Output val CSV')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio (default: 0.1)')
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} samples from {args.input}")
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split
    split_idx = int(len(df) * (1 - args.val_ratio))
    df_train = df[:split_idx]
    df_val = df[split_idx:]
    
    # Save
    df_train.to_csv(args.train_output, index=False)
    df_val.to_csv(args.val_output, index=False)
    
    print(f"Split complete:")
    print(f"  Train: {len(df_train)} samples -> {args.train_output}")
    print(f"  Val: {len(df_val)} samples -> {args.val_output}")

if __name__ == '__main__':
    main()
