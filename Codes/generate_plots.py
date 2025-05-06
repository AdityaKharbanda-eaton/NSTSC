import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from utils.datautils import Readdataset
import sys

dataset_labels = {"Coffee": {0: "Robusta", 1: "Arabica"},
                  "ItalyPowerDemand": {1: "Oct to March", 2: "April to Sept"},
                  "KentuckyUPS": {1: "Sub-Cycle Disturbance", 2: "Sag", 3: "Healthy"}}
def plot_data_comparison(dataset_name, row_index=0):
    # Create a 2x3 subplot figure
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            f"Raw - Original", f"Raw - FFT", f"Raw - Derivative",
            f"Standardized - Original", f"Standardized - FFT", f"Standardized - Derivative"
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.05
    )
    
    # Get raw data
    df = pd.read_csv(f"../UCRArchive_2018/{dataset_name}/{dataset_name}_TRAIN.tsv", sep='\t', header=None)
    X_train = df.iloc[:, 1:].values
    y_train = df.iloc[:, 0].values
    
    # Get standardized data
    Xtrain, ytrain, _, _, _, _ = Readdataset("../UCRArchive_2018/", dataset_name, standalize=True, val=False, shuffle = False)
    T = Xtrain.shape[1] // 3
    Xraw = Xtrain[:, :T]
    Xfft = Xtrain[:, T:2*T]
    Xdif = Xtrain[:, 2*T:]
    
    # Class label
    class_label = dataset_labels[dataset_name][y_train[row_index]] if dataset_name in dataset_labels else f"Class {y_train[row_index]}"
    
    # RAW DATA ROW
    # Original raw data
    sample_raw = X_train[row_index]
    x_raw = np.arange(0, len(sample_raw))
    fig.add_trace(
        go.Scatter(x=x_raw, y=sample_raw, name="Raw", line=dict(color="blue")),
        row=1, col=1
    )
    
    # FFT of raw data
    X_fft = np.abs(np.fft.fft(X_train, axis=1))
    sample_fft = X_fft[row_index]
    fig.add_trace(
        go.Scatter(x=x_raw, y=sample_fft, name="Raw FFT", line=dict(color="blue")),
        row=1, col=2
    )
    
    # Derivative of raw data
    X_dif = X_train[:,1:] - X_train[:,:-1]
    X_dif = np.concatenate((X_dif[:,0].reshape([-1,1]),X_dif),1)
    # X_dif = np.diff(X_train, axis=1, prepend=X_train[:, 0].reshape(-1, 1))
    sample_dif = X_dif[row_index]
    fig.add_trace(
        go.Scatter(x=x_raw, y=sample_dif, name="Raw Derivative", line=dict(color="blue")),
        row=1, col=3
    )
    
    # STANDARDIZED DATA ROW
    # Standardized original data
    sample_std = Xraw[row_index]
    x_std = np.arange(0, len(sample_std))
    fig.add_trace(
        go.Scatter(x=x_std, y=sample_std, name="Standardized", line=dict(color="red")),
        row=2, col=1
    )
    
    # Standardized FFT
    sample_std_fft = Xfft[row_index]
    fig.add_trace(
        go.Scatter(x=x_std, y=sample_std_fft, name="Standardized FFT", line=dict(color="red")),
        row=2, col=2
    )
    
    # Standardized derivative
    sample_std_dif = Xdif[row_index]
    fig.add_trace(
        go.Scatter(x=x_std, y=sample_std_dif, name="Standardized Derivative", line=dict(color="red")),
        row=2, col=3
    )
    
    # Update layout and axis labels
    fig.update_layout(
        title_text=f"{dataset_name} - {class_label} (Sample Index: {row_index})",
        height=1500,
        width=3500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    
    # Update x-axis titles
    for i in range(1, 3):
        for j in range(1, 4):
            x_title = "Timestamps"
            if j == 2:
                x_title = "Timestamps"
            fig.update_xaxes(title_text=x_title, row=i, col=j)
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Raw Waveform Value", row=1, col=1)
    fig.update_yaxes(title_text="Raw FFT Value", row=1, col=2)
    fig.update_yaxes(title_text="Raw Differences", row=1, col=3)
    fig.update_yaxes(title_text="Standardised Waveform Value", row=2, col=1)
    fig.update_yaxes(title_text="Standardised FFT Values", row=2, col=2)
    fig.update_yaxes(title_text="Standardised Differences", row=2, col=3)
    
    return fig
if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "Coffee"
    
    # Load the dataset to get its length
    df = pd.read_csv(f"../UCRArchive_2018/{dataset_name}/{dataset_name}_TRAIN.tsv", sep='\t', header=None)
    num_samples = len(df)
    
    # Loop through each sample in the dataset
    for row_index in range(num_samples):
        print(f"Generating plot for sample {row_index+1}/{num_samples}")
        fig = plot_data_comparison(dataset_name, row_index)
        fig.write_html(f"../Plotly_plots/{dataset_name}/{dataset_name}_TRAIN_sample_{row_index}.html")  # Save as HTML file