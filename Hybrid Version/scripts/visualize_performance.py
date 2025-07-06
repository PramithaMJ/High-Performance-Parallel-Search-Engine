#!/usr/bin/env python3
# Script to visualize performance data from hybrid MPI+OpenMP search engine

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    # Find the CSV file
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    csv_path = os.path.join(data_dir, "hybrid_metrics.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find data file at {csv_path}")
        print("Run the performance_benchmark.sh script first to generate performance data.")
        sys.exit(1)
    
    # Read the data
    df = pd.read_csv(csv_path)
    
    # Add a total threads column
    df['Total_Cores'] = df['MPI_Processes'] * df['OMP_Threads']
    
    # Calculate average times by configuration
    avg_times = df.groupby(['MPI_Processes', 'OMP_Threads'])['Processing_Time_ms'].mean().reset_index()
    
    # Create a plot to show the impact of different MPI/OpenMP configurations
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Performance by configuration (3D bar chart)
    ax1 = plt.subplot(2, 2, 1, projection='3d')
    
    # Get unique MPI process counts and OpenMP thread counts
    mpi_procs = sorted(avg_times['MPI_Processes'].unique())
    omp_threads = sorted(avg_times['OMP_Threads'].unique())
    
    # Create mesh grid for 3D bar chart
    xpos, ypos = np.meshgrid(range(len(mpi_procs)), range(len(omp_threads)))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    
    # Create a dictionary to easily look up times for a given configuration
    time_dict = {}
    for _, row in avg_times.iterrows():
        time_dict[(row['MPI_Processes'], row['OMP_Threads'])] = row['Processing_Time_ms']
    
    # Get heights for each configuration
    heights = []
    for i, x in enumerate(xpos):
        y = ypos[i]
        mpi_proc = mpi_procs[x]
        omp_thread = omp_threads[y]
        if (mpi_proc, omp_thread) in time_dict:
            heights.append(time_dict[(mpi_proc, omp_thread)])
        else:
            heights.append(0)
    
    # Convert to numpy array
    heights = np.array(heights)
    
    # Draw 3D bar chart
    width = depth = 0.7
    ax1.bar3d(xpos, ypos, zpos, width, depth, heights, shade=True, color='skyblue', edgecolor='darkblue')
    
    # Set labels
    ax1.set_title('Query Processing Time by MPI and OpenMP Configuration')
    ax1.set_xlabel('MPI Processes')
    ax1.set_ylabel('OpenMP Threads')
    ax1.set_zlabel('Avg. Processing Time (ms)')
    
    # Set ticks
    ax1.set_xticks(range(len(mpi_procs)))
    ax1.set_yticks(range(len(omp_threads)))
    ax1.set_xticklabels(mpi_procs)
    ax1.set_yticklabels(omp_threads)
    
    # Plot 2: Processing time by total cores
    ax2 = plt.subplot(2, 2, 2)
    
    # Group by total cores
    by_total = avg_times.copy()
    by_total['Total_Cores'] = by_total['MPI_Processes'] * by_total['OMP_Threads']
    total_cores_avg = by_total.groupby('Total_Cores')['Processing_Time_ms'].mean().reset_index()
    
    # Sort by total cores
    total_cores_avg = total_cores_avg.sort_values('Total_Cores')
    
    # Plot line and points
    ax2.plot(total_cores_avg['Total_Cores'], total_cores_avg['Processing_Time_ms'], 
             marker='o', linestyle='-', color='green', linewidth=2, markersize=8)
    
    # Add annotations
    for x, y in zip(total_cores_avg['Total_Cores'], total_cores_avg['Processing_Time_ms']):
        ax2.annotate(f"{y:.2f}ms", 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    ax2.set_title('Processing Time by Total Core Count')
    ax2.set_xlabel('Total Cores (MPI × OpenMP)')
    ax2.set_ylabel('Avg. Processing Time (ms)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Heatmap of processing time
    ax3 = plt.subplot(2, 1, 2)
    
    # Create pivot table for heatmap
    pivot = avg_times.pivot(index="MPI_Processes", columns="OMP_Threads", values="Processing_Time_ms")
    
    # Plot heatmap
    im = ax3.imshow(pivot, cmap='coolwarm_r')  # Use reverse colormap so lower (better) times are in blue
    
    # Add colorbar
    plt.colorbar(im, ax=ax3, label='Processing Time (ms)')
    
    # Set labels
    ax3.set_title('Processing Time Heatmap by MPI Processes and OpenMP Threads')
    ax3.set_xlabel('OpenMP Threads')
    ax3.set_ylabel('MPI Processes')
    
    # Set ticks
    ax3.set_xticks(range(len(pivot.columns)))
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_xticklabels(pivot.columns)
    ax3.set_yticklabels(pivot.index)
    
    # Add annotations to heatmap cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            if not np.isnan(pivot.iloc[i, j]):
                ax3.text(j, i, f"{pivot.iloc[i, j]:.1f}ms", 
                        ha="center", va="center", color="black", fontweight='bold')
    
    # Add overall title
    plt.suptitle("Hybrid MPI+OpenMP Search Engine Performance Analysis", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    output_path = os.path.join(data_dir, "hybrid_performance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Visualization saved to: {output_path}")
    
    # Show plot if not in a script
    plt.show()

if __name__ == "__main__":
    main()
