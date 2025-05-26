from utils.datautils import Readdataset, Splitview, calculate_dataset_metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys
def convert_to_wide_format(x_array, y_array, T):
    df = pd.DataFrame(x_array, columns = ['t' + str(i+1) for i in range(T)])
    df['label'] = y_array
    df['label'] = df['label'].astype('category')
    df['signal_id'] = df.index
    return df

def convert_to_long_format(df, T):
    df_long = pd.melt(df, id_vars = ['label', 'signal_id'], value_vars = ['t' + str(i+1) for i in range(T)], var_name = 'timestep', value_name = 'value')
    df_long['timestep'] = df_long['timestep'].str.replace('t', '').astype(int)
    return df_long

import plotly.express as px

def plot_time_series(df_long, title):
    fig = px.line(
        df_long,
        x='timestep',
        y='value',
        color='label',
        line_group='signal_id',
        title=title,
        labels={'timestep': 'Timestep', 'value': 'Value', 'label': 'Label'}
    )
    fig.update_layout(legend_title_text='Label')
    return fig

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_time_series_interactive(df, title):
    """
    Creates an interactive plot where:
    - Each row in df is plotted as a time series line
    - Lines are colored by their class label
    - Legend items can be clicked to show/hide all signals of that label
    
    Parameters:
    - df: DataFrame in wide format with class label column and signal values
    - title: Plot title
    
    Returns:
    - Plotly figure object
    """
    # Convert to long format for plotting if it's not already
    if 'timestep' not in df.columns:
        T = len(df.columns) - 2  # Subtract label and signal_id columns
        df_long = convert_to_long_format(df, T)
    else:
        df_long = df
    
    # Get unique labels
    unique_labels = df_long['label'].unique()
    
    # Create figure
    fig = go.Figure()
    
    # Add traces, grouped by label
    for i, label in enumerate(unique_labels):
        df_label = df_long[df_long['label'] == label]
        color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        
        for signal_id in df_label['signal_id'].unique():
            df_signal = df_label[df_label['signal_id'] == signal_id]
            # Convert numpy.bool_ to Python boolean
            show_in_legend = bool(signal_id == df_label['signal_id'].unique()[0])
            
            fig.add_trace(
                go.Scatter(
                    x=df_signal['timestep'],
                    y=df_signal['value'],
                    mode='lines',
                    name=f'Label {label}',  # All signals of same label share name for legend grouping
                    legendgroup=f'Label {label}',
                    showlegend=show_in_legend,  # Explicitly convert to Python bool
                    line=dict(color=color),
                )
            )
    
    # Update layout for better interactivity
    fig.update_layout(
        title=title,
        xaxis_title='Timestep',
        yaxis_title='Value',
        legend_title='Class Labels',
        legend=dict(
            groupclick="toggleitem",  # Click on legend group toggles all traces in that group
        ),
        height=600,
        hovermode="closest"
    )
    
    return fig

def plot_all_data(dataset_name, dataset_path = "../UCRArchive_2018/"):
    Xtrain, ytrain, Xval, yval, Xtest, ytest = Readdataset(dataset_path, dataset_name)
    N, T = calculate_dataset_metrics(Xtrain)
    Xraw, Xfft, Xder = Splitview(Xtrain, T)

    df_raw = convert_to_wide_format(Xraw, ytrain, T)
    df_fft = convert_to_wide_format(Xfft, ytrain, T)
    df_der = convert_to_wide_format(Xder, ytrain, T)

    df_long_raw = convert_to_long_format(df_raw, T)
    df_long_fft = convert_to_long_format(df_fft, T)
    df_long_der = convert_to_long_format(df_der, T)

    # Create the Plots directory and dataset subdirectory
    plots_dir = os.path.join(f"../Seaborn Plots/{dataset_name}")
    os.makedirs(plots_dir, exist_ok=True)

    # Add interactive legend plots
    plt_raw_interactive = plot_time_series_interactive(df_long_raw, 'Raw Data')
    plt_raw_interactive.write_html(os.path.join(plots_dir, "raw_data_interactive_legend.html"))
    
    plt_fft_interactive = plot_time_series_interactive(df_long_fft, 'FFT Data')
    plt_fft_interactive.write_html(os.path.join(plots_dir, "fft_data_interactive_legend.html"))
    
    plt_der_interactive = plot_time_series_interactive(df_long_der, 'Derivative Data')
    plt_der_interactive.write_html(os.path.join(plots_dir, "derivative_data_interactive_legend.html"))
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_plots.py <dataset_name>")
        sys.exit(1)
    dataset_name = sys.argv[1]
    plot_all_data(dataset_name)