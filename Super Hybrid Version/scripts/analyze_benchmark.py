#!/usr/bin/env python3
"""
Super Hybrid Search Engine Benchmark Analysis and Visualization
Analyzes CSV results from the comprehensive benchmark suite and generates insights
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from datetime import datetime
import argparse

def load_and_clean_data(csv_file):
    """Load and clean the benchmark data"""
    try:
        df = pd.read_csv(csv_file)
        
        # Clean numeric columns
        numeric_columns = ['MPI_Processes', 'OpenMP_Threads', 'CUDA_Devices', 
                          'Total_Parallel_Units', 'Duration_Seconds', 
                          'Documents_Processed', 'Throughput_Docs_Per_Sec', 'Memory_Usage_MB']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter out failed runs for performance analysis
        df_success = df[df['Notes'] == 'Success'].copy()
        
        return df, df_success
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def analyze_technology_performance(df):
    """Analyze performance by technology type"""
    print("\n" + "="*80)
    print("TECHNOLOGY PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Group by technology type (extracted from configuration name)
    df['Technology'] = df['Configuration'].apply(lambda x: 
        'Super_Hybrid' if 'Super_Hybrid' in x else
        'CUDA_OpenMP' if 'CUDA_OpenMP' in x else
        'OpenMP_MPI' if 'OpenMP_MPI' in x else
        'CUDA_MPI' if 'CUDA_MPI' in x else
        'CUDA_Only' if 'CUDA_Only' in x else
        'OpenMP_Only' if 'OpenMP_Only' in x else
        'MPI_Only' if 'MPI_Only' in x else
        'Serial'
    )
    
    # Performance statistics by technology
    perf_stats = df.groupby('Technology').agg({
        'Duration_Seconds': ['mean', 'std', 'min'],
        'Throughput_Docs_Per_Sec': ['mean', 'std', 'max'],
        'Documents_Processed': ['mean', 'sum'],
        'Memory_Usage_MB': ['mean', 'max']
    }).round(3)
    
    print("\nPerformance Statistics by Technology:")
    print("-" * 60)
    print(perf_stats.to_string())
    
    # Calculate speedup relative to serial baseline
    if 'Serial' in df['Technology'].values:
        serial_time = df[df['Technology'] == 'Serial']['Duration_Seconds'].mean()
        speedup_data = []
        
        for tech in df['Technology'].unique():
            if tech != 'Serial':
                tech_time = df[df['Technology'] == tech]['Duration_Seconds'].mean()
                if tech_time > 0:
                    speedup = serial_time / tech_time
                    speedup_data.append({'Technology': tech, 'Speedup': speedup})
        
        if speedup_data:
            speedup_df = pd.DataFrame(speedup_data)
            print(f"\nSpeedup vs Serial Baseline (Serial time: {serial_time:.3f}s):")
            print("-" * 60)
            for _, row in speedup_df.iterrows():
                print(f"{row['Technology']:20} {row['Speedup']:8.2f}x")
    
    return df

def analyze_scalability(df):
    """Analyze scalability characteristics"""
    print("\n" + "="*80)
    print("SCALABILITY ANALYSIS")
    print("="*80)
    
    # Analyze scaling with parallel units
    if 'Total_Parallel_Units' in df.columns:
        scaling_data = df.groupby('Total_Parallel_Units').agg({
            'Duration_Seconds': 'mean',
            'Throughput_Docs_Per_Sec': 'mean'
        }).reset_index()
        
        print("\nScaling with Total Parallel Units:")
        print("-" * 40)
        for _, row in scaling_data.iterrows():
            print(f"Units: {row['Total_Parallel_Units']:6.0f} | "
                  f"Duration: {row['Duration_Seconds']:6.3f}s | "
                  f"Throughput: {row['Throughput_Docs_Per_Sec']:6.2f} docs/s")
    
    # Analyze MPI scaling
    mpi_scaling = df[df['MPI_Processes'] > 1].groupby('MPI_Processes').agg({
        'Duration_Seconds': 'mean',
        'Throughput_Docs_Per_Sec': 'mean'
    }).reset_index()
    
    if not mpi_scaling.empty:
        print(f"\nMPI Process Scaling:")
        print("-" * 40)
        for _, row in mpi_scaling.iterrows():
            print(f"Processes: {row['MPI_Processes']:2.0f} | "
                  f"Duration: {row['Duration_Seconds']:6.3f}s | "
                  f"Throughput: {row['Throughput_Docs_Per_Sec']:6.2f} docs/s")
    
    # Analyze OpenMP scaling
    omp_scaling = df[df['OpenMP_Threads'] > 1].groupby('OpenMP_Threads').agg({
        'Duration_Seconds': 'mean',
        'Throughput_Docs_Per_Sec': 'mean'
    }).reset_index()
    
    if not omp_scaling.empty:
        print(f"\nOpenMP Thread Scaling:")
        print("-" * 40)
        for _, row in omp_scaling.iterrows():
            print(f"Threads: {row['OpenMP_Threads']:2.0f} | "
                  f"Duration: {row['Duration_Seconds']:6.3f}s | "
                  f"Throughput: {row['Throughput_Docs_Per_Sec']:6.2f} docs/s")

def create_visualizations(df, output_dir):
    """Create comprehensive visualizations"""
    print(f"\n Creating visualizations in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Technology Performance Comparison
    plt.figure(figsize=(14, 10))
    
    # Filter data for main comparison
    df_main = df[df['Phase'].isin(['Baseline', 'Dual_Tech', 'Super_Hybrid'])].copy()
    
    if not df_main.empty:
        plt.subplot(2, 2, 1)
        tech_perf = df_main.groupby('Technology')['Throughput_Docs_Per_Sec'].mean().sort_values(ascending=True)
        bars = plt.barh(range(len(tech_perf)), tech_perf.values)
        plt.yticks(range(len(tech_perf)), tech_perf.index)
        plt.xlabel('Average Throughput (docs/sec)')
        plt.title('Technology Performance Comparison')
        plt.grid(axis='x', alpha=0.3)
        
        # Color bars by performance
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    # 2. Execution Time vs Configuration
    plt.subplot(2, 2, 2)
    if 'Total_Parallel_Units' in df.columns:
        scatter_data = df[df['Total_Parallel_Units'] <= 10000]  # Filter extreme values
        if not scatter_data.empty:
            plt.scatter(scatter_data['Total_Parallel_Units'], 
                       scatter_data['Duration_Seconds'],
                       c=scatter_data['CUDA_Devices'], 
                       cmap='viridis', alpha=0.7)
            plt.colorbar(label='CUDA Devices')
            plt.xlabel('Total Parallel Units')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Execution Time vs Parallelization')
            plt.yscale('log')
    
    # 3. MPI vs OpenMP Scaling
    plt.subplot(2, 2, 3)
    pivot_data = df.pivot_table(values='Throughput_Docs_Per_Sec', 
                               index='MPI_Processes', 
                               columns='OpenMP_Threads', 
                               aggfunc='mean')
    if not pivot_data.empty:
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title('Throughput Heatmap: MPI vs OpenMP')
        plt.ylabel('MPI Processes')
        plt.xlabel('OpenMP Threads')
    
    # 4. Phase Performance Comparison
    plt.subplot(2, 2, 4)
    phase_perf = df.groupby('Phase')['Duration_Seconds'].mean().sort_values()
    if not phase_perf.empty:
        bars = plt.bar(range(len(phase_perf)), phase_perf.values)
        plt.xticks(range(len(phase_perf)), phase_perf.index, rotation=45)
        plt.ylabel('Average Duration (seconds)')
        plt.title('Performance by Test Phase')
        plt.yscale('log')
        
        # Color bars by performance
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/super_hybrid_performance_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Detailed Scaling Analysis
    plt.figure(figsize=(16, 12))
    
    # Technology scaling
    plt.subplot(2, 3, 1)
    for tech in df['Technology'].unique():
        tech_data = df[df['Technology'] == tech]
        if len(tech_data) > 1:
            plt.plot(tech_data['Total_Parallel_Units'], 
                    tech_data['Throughput_Docs_Per_Sec'], 
                    'o-', label=tech, alpha=0.7)
    plt.xlabel('Total Parallel Units')
    plt.ylabel('Throughput (docs/sec)')
    plt.title('Scaling by Technology')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # GPU Ratio Analysis
    gpu_ratio_data = df[df['Configuration'].str.contains('GPU_Ratio', na=False)]
    if not gpu_ratio_data.empty:
        plt.subplot(2, 3, 2)
        gpu_ratio_data['GPU_Ratio'] = gpu_ratio_data['Configuration'].str.extract(r'GPU_Ratio_([0-9.]+)').astype(float)
        gpu_perf = gpu_ratio_data.groupby('GPU_Ratio')['Throughput_Docs_Per_Sec'].mean()
        plt.plot(gpu_perf.index, gpu_perf.values, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('GPU/CPU Ratio')
        plt.ylabel('Throughput (docs/sec)')
        plt.title('GPU/CPU Ratio Optimization')
        plt.grid(alpha=0.3)
    
    # Memory Usage Analysis
    plt.subplot(2, 3, 3)
    if 'Memory_Usage_MB' in df.columns:
        memory_by_tech = df.groupby('Technology')['Memory_Usage_MB'].mean().sort_values()
        bars = plt.bar(range(len(memory_by_tech)), memory_by_tech.values)
        plt.xticks(range(len(memory_by_tech)), memory_by_tech.index, rotation=45)
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage by Technology')
        plt.yscale('log')
        
        # Color bars by usage
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    # Success Rate Analysis
    plt.subplot(2, 3, 4)
    df_all = df  # Include failed runs
    success_rate = df_all.groupby('Technology').apply(
        lambda x: (x['Notes'] == 'Success').sum() / len(x) * 100
    ).sort_values(ascending=False)
    
    bars = plt.bar(range(len(success_rate)), success_rate.values)
    plt.xticks(range(len(success_rate)), success_rate.index, rotation=45)
    plt.ylabel('Success Rate (%)')
    plt.title('Reliability by Technology')
    plt.ylim(0, 100)
    
    # Color bars by success rate
    colors = plt.cm.RdYlGn(success_rate.values / 100)
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Efficiency Analysis (Throughput per Parallel Unit)
    plt.subplot(2, 3, 5)
    df['Efficiency'] = df['Throughput_Docs_Per_Sec'] / df['Total_Parallel_Units']
    efficiency_by_tech = df.groupby('Technology')['Efficiency'].mean().sort_values(ascending=False)
    
    bars = plt.bar(range(len(efficiency_by_tech)), efficiency_by_tech.values)
    plt.xticks(range(len(efficiency_by_tech)), efficiency_by_tech.index, rotation=45)
    plt.ylabel('Efficiency (docs/sec/unit)')
    plt.title('Parallel Efficiency')
    plt.yscale('log')
    
    # Query vs Index Performance
    plt.subplot(2, 3, 6)
    query_data = df[df['Phase'] == 'Query_Processing']
    index_data = df[df['Phase'] == 'Index_Building']
    
    if not query_data.empty and not index_data.empty:
        phases = ['Query Processing', 'Index Building']
        avg_times = [query_data['Duration_Seconds'].mean(), 
                    index_data['Duration_Seconds'].mean()]
        
        bars = plt.bar(phases, avg_times)
        plt.ylabel('Average Duration (seconds)')
        plt.title('Query vs Index Performance')
        plt.yscale('log')
        
        colors = ['lightblue', 'lightcoral']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/super_hybrid_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Visualizations saved to {output_dir}/")

def generate_recommendations(df):
    """Generate performance optimization recommendations"""
    print("\n" + "="*80)
    print("PERFORMANCE OPTIMIZATION RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    # Find best performing configuration
    best_config = df.loc[df['Throughput_Docs_Per_Sec'].idxmax()]
    recommendations.append(f"🏆 Best Overall Configuration: {best_config['Configuration']}")
    recommendations.append(f"   - Throughput: {best_config['Throughput_Docs_Per_Sec']:.2f} docs/sec")
    recommendations.append(f"   - Configuration: {best_config['MPI_Processes']} MPI × {best_config['OpenMP_Threads']} OpenMP × {best_config['CUDA_Devices']} CUDA")
    
    # Technology recommendations
    tech_performance = df.groupby('Technology')['Throughput_Docs_Per_Sec'].mean().sort_values(ascending=False)
    
    if len(tech_performance) > 1:
        best_tech = tech_performance.index[0]
        recommendations.append(f"\n Best Technology Combination: {best_tech}")
        recommendations.append(f"   - Average throughput: {tech_performance.iloc[0]:.2f} docs/sec")
        
        if 'Super_Hybrid' in tech_performance.index:
            super_hybrid_rank = list(tech_performance.index).index('Super_Hybrid') + 1
            recommendations.append(f"   - Super Hybrid ranking: #{super_hybrid_rank} out of {len(tech_performance)}")
    
    # Scaling recommendations
    if 'Total_Parallel_Units' in df.columns:
        efficiency = df['Throughput_Docs_Per_Sec'] / df['Total_Parallel_Units']
        optimal_units = df.loc[efficiency.idxmax(), 'Total_Parallel_Units']
        recommendations.append(f"\n⚡ Optimal Parallel Units: {optimal_units:.0f}")
        recommendations.append(f"   - Best efficiency: {efficiency.max():.4f} docs/sec/unit")
    
    # GPU ratio recommendations
    gpu_ratio_data = df[df['Configuration'].str.contains('GPU_Ratio', na=False)]
    if not gpu_ratio_data.empty:
        gpu_ratio_data['GPU_Ratio'] = gpu_ratio_data['Configuration'].str.extract(r'GPU_Ratio_([0-9.]+)').astype(float)
        best_ratio_idx = gpu_ratio_data['Throughput_Docs_Per_Sec'].idxmax()
        best_ratio = gpu_ratio_data.loc[best_ratio_idx, 'GPU_Ratio']
        recommendations.append(f"\n Optimal GPU/CPU Ratio: {best_ratio:.1f}")
        recommendations.append(f"   - Throughput at optimal ratio: {gpu_ratio_data.loc[best_ratio_idx, 'Throughput_Docs_Per_Sec']:.2f} docs/sec")
    
    # Memory recommendations
    if 'Memory_Usage_MB' in df.columns:
        memory_efficient = df.loc[df['Memory_Usage_MB'].idxmin()]
        recommendations.append(f"\n💾 Most Memory Efficient: {memory_efficient['Configuration']}")
        recommendations.append(f"   - Memory usage: {memory_efficient['Memory_Usage_MB']:.1f} MB")
        recommendations.append(f"   - Throughput: {memory_efficient['Throughput_Docs_Per_Sec']:.2f} docs/sec")
    
    # Reliability recommendations
    success_rates = df.groupby('Technology').apply(
        lambda x: (x['Notes'] == 'Success').sum() / len(x) * 100
    )
    most_reliable = success_rates.idxmax()
    recommendations.append(f"\n🛡️  Most Reliable Technology: {most_reliable}")
    recommendations.append(f"   - Success rate: {success_rates.max():.1f}%")
    
    # Print recommendations
    for rec in recommendations:
        print(rec)
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description='Analyze Super Hybrid Search Engine benchmark results')
    parser.add_argument('csv_file', help='Path to the benchmark CSV file')
    parser.add_argument('--output-dir', default='./benchmark_analysis', 
                       help='Directory to save analysis outputs')
    parser.add_argument('--no-viz', action='store_true', 
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f" Error: CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    print(" SUPER HYBRID SEARCH ENGINE BENCHMARK ANALYSIS")
    print("="*80)
    print(f"Analyzing: {args.csv_file}")
    print(f"Output directory: {args.output_dir}")
    
    # Load and clean data
    df_all, df_success = load_and_clean_data(args.csv_file)
    
    if df_all is None or df_success.empty:
        print(" Error: Could not load or no successful benchmark data found")
        sys.exit(1)
    
    print(f" Data loaded: {len(df_all)} total runs, {len(df_success)} successful runs")
    
    # Perform analysis
    df_analyzed = analyze_technology_performance(df_success)
    analyze_scalability(df_success)
    
    # Generate visualizations
    if not args.no_viz:
        try:
            create_visualizations(df_success, args.output_dir)
        except Exception as e:
            print(f"️  Warning: Could not generate visualizations: {e}")
    
    # Generate recommendations
    recommendations = generate_recommendations(df_success)
    
    # Save detailed report
    os.makedirs(args.output_dir, exist_ok=True)
    report_file = os.path.join(args.output_dir, 'analysis_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("SUPER HYBRID SEARCH ENGINE BENCHMARK ANALYSIS REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {args.csv_file}\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        f.write("-"*40 + "\n")
        for rec in recommendations:
            f.write(rec + "\n")
        
        f.write(f"\nDetailed statistics and visualizations available in: {args.output_dir}\n")
    
    print(f"\n Analysis completed!")
    print(f" Detailed report saved to: {report_file}")
    
    if not args.no_viz:
        print(f" Visualizations saved to: {args.output_dir}/")
    
    print("\n Quick Summary:")
    if len(df_success) > 0:
        best_config = df_success.loc[df_success['Throughput_Docs_Per_Sec'].idxmax()]
        print(f"   Best Configuration: {best_config['Configuration']}")
        print(f"   Best Throughput: {best_config['Throughput_Docs_Per_Sec']:.2f} docs/sec")
        print(f"   Success Rate: {len(df_success)/len(df_all)*100:.1f}%")

if __name__ == "__main__":
    main()
