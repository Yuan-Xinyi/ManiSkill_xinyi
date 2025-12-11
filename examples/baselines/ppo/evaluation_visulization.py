import json
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from collections import defaultdict

def visualize(jsonl_path, output_dir):
    data = []
    print(f"Reading data from {jsonl_path}...")
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if not data:
        print("No data found in the file.")
        return

    # Sort by iteration
    data.sort(key=lambda x: x['iteration'])
    
    iterations = [x['iteration'] for x in data]
    
    # Extract radii keys from the first entry (assuming all have same radii)
    first_metrics = data[0]['metrics']
    # Filter keys that start with "radius_"
    radius_keys = [k for k in first_metrics.keys() if k.startswith("radius_")]
    # Sort radii numerically
    radius_keys.sort(key=lambda x: float(x.split('_')[1]))
    
    print(f"Found {len(iterations)} checkpoints and {len(radius_keys)} radii.")

    # Prepare data for plotting
    radius_data = defaultdict(list)
    
    for entry in data:
        metrics = entry['metrics']
        for r_key in radius_keys:
            if r_key in metrics:
                # Use success_once as the primary success metric, fallback to eval_success_once_mean
                val = metrics[r_key].get('success_once', metrics[r_key].get('eval_success_once_mean', 0.0))
                radius_data[r_key].append(val)
            else:
                radius_data[r_key].append(0.0) # Handle missing data

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Success Rate vs Iteration for each Radius (Line Chart)
    plt.figure(figsize=(14, 8))
    
    # Use default matplotlib color cycle for better distinctness
    for i, r_key in enumerate(radius_keys):
        radius_val = float(r_key.split('_')[1])
        plt.plot(iterations, radius_data[r_key], marker='o', markersize=4, linewidth=1.5, label=f'R={radius_val:.2f}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Success Rate (Success Once)')
    plt.title('Success Rate vs Iteration for Different Radii')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small', ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'success_rate_vs_iteration.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved line plot to {output_path}")
    plt.close()

    # Plot 2: Heatmap (Iteration vs Radius)
    # X-axis: Iteration, Y-axis: Radius
    
    Z = []
    for r_key in radius_keys:
        Z.append(radius_data[r_key])
    
    Z = np.array(Z) # Shape: (num_radii, num_iterations)
    
    plt.figure(figsize=(12, 6))
    
    # Create meshgrid for pcolormesh
    # We need edges for pcolormesh, so we extend the range slightly
    # Or just use imshow with aspect auto
    
    plt.imshow(Z, aspect='auto', cmap='RdYlGn', origin='lower', vmin=0, vmax=1,
               extent=[min(iterations), max(iterations), 0, len(radius_keys)])
    
    # Set Y-ticks to radius values
    # Show every Nth label if too many
    step = max(1, len(radius_keys) // 10)
    plt.yticks(np.arange(len(radius_keys)) + 0.5, [f"{float(k.split('_')[1]):.2f}" for k in radius_keys], fontsize=8)
    # Thin out y-ticks if needed
    if len(radius_keys) > 20:
        plt.yticks(np.arange(0, len(radius_keys), step) + 0.5, 
                   [f"{float(radius_keys[i].split('_')[1]):.2f}" for i in range(0, len(radius_keys), step)])

    plt.colorbar(label='Success Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Radius')
    plt.title('Success Rate Heatmap (Green=High, Red=Low)')
    plt.tight_layout()
    
    output_path_heatmap = os.path.join(output_dir, 'success_rate_heatmap.png')
    plt.savefig(output_path_heatmap, dpi=300)
    print(f"Saved heatmap to {output_path_heatmap}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default="evaluation_results.jsonl", help="Path to the jsonl file")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save plots")
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl):
        print(f"Error: File {args.jsonl} not found.")
    else:
        # If output dir is '.', try to save in the same dir as jsonl
        if args.output_dir == ".":
            args.output_dir = os.path.dirname(os.path.abspath(args.jsonl))
        
        visualize(args.jsonl, args.output_dir)
