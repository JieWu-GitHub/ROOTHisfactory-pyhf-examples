'''
Author       : Jie Wu j.wu@cern.ch
Date         : 2025-02-12 02:58:55 +0100
LastEditors  : Jie Wu j.wu@cern.ch
LastEditTime : 2025-02-12 05:07:56 +0100
FilePath     : utilities_pyhf.py
Description  : 

Copyright (c) 2025 by everyone, All Rights Reserved. 
'''

import os
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pyhf
import cabinetry
import uproot
import yaml
from typing import Dict, List, Tuple, Optional, Union
import json
import hist

from rich import print as rprint

# Use rich backend for logging
import logging
from rich.logging import RichHandler

# Configure rich logging globally
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%x %X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)  # or "DEBUG"/"WARNING" as needed
logger = logging.getLogger(__name__)


# Set verbosity to DEBUG for detailed output
# cabinetry.set_logging()


########################### maniplate with the histogram #########################################


# Check if it is a histogram
def is_histogram(obj, key: str) -> bool:
    """Check if the key is a histogram"""

    return issubclass(type(obj), uproot.behaviors.TH1.Histogram)


def flatten_and_save_histogram(
    input_file: Union[str, Path],
    hist_name: str,
    output_file: Union[str, Path],
    output_hist_name: Optional[str] = None,
    option: str = 'recreate',
) -> hist.Hist:
    """Flatten a multi-dimensional histogram and save it as 1D.

    Args:
        input_file: Path to input ROOT file
        hist_name: Name of histogram in input file
        output_file: Path to output ROOT file
        output_hist_name: Name for output histogram (defaults to input name)
    """
    # Convert paths to Path objects
    input_file = Path(input_file).resolve()
    output_file = Path(output_file).resolve()

    # Use input name if output name not specified
    output_hist_name = output_hist_name or hist_name

    with uproot.open(input_file) as f:
        orig_hist = f[hist_name]
        if not is_histogram(orig_hist, hist_name):
            logger.warning(f"The object {hist_name} is not a histogram, skipping")
            return {}

        # Flatten the histogram values and errors
        # Flatten the histogram values and errors
        values = orig_hist.values().flatten()
        errors = np.sqrt(orig_hist.variances().flatten())  # Convert variances to errors

        # Create 1D histogram with nbins equal to flattened length
        h1d = hist.Hist(
            hist.axis.Regular(
                bins=len(values),  # Number of bins equals length of flattened array
                start=0.5,  # Start from 0.5 to center bins on integers
                stop=len(values) + 0.5,  # End at n+0.5 to center bins
                name="bin_number",
            ),
            storage=hist.storage.Weight(),  # Enable error storage
        )

        # Fill histogram with values and errors
        bin_centers = np.arange(1, len(values) + 1)  # 1-based bin numbering
        h1d.fill(bin_centers, weight=values)

        # Set variances after filling
        h1d.view().variance = errors**2

        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save as 1D histogram
        if option == 'recreate':
            with uproot.recreate(output_file) as f:
                f[output_hist_name] = h1d
                logger.info(f"Flattened histogram {output_hist_name} saved to {output_file}")
        else:
            with uproot.update(output_file) as f:
                f[output_hist_name] = h1d
                logger.info(f"Flattened histogram {output_hist_name} saved to {output_file}")

    return h1d


def flatten_all_histograms(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
) -> None:
    """Flatten all histograms in a ROOT file and save them as 1D.

    Args:
        input_file: Path to input ROOT file
        output_file: Path to output ROOT file
    """

    input_file = Path(input_file)
    output_file = Path(output_file)

    with uproot.open(input_file) as f:
        # Get all histogram names
        hist_names = [key.split(';')[0] for key in f.keys() if is_histogram(f[key], key)]

    # Process each histogram
    for i, hist_name in enumerate(hist_names):
        flatten_and_save_histogram(input_file, hist_name, output_file, hist_name, option='recreate' if i == 0 else 'update')


def load_flattened_histograms(
    filename: str,
    hist_names: Union[str, List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Load histograms from flattened ROOT file"""
    histograms = {}

    # get the hist_names to be loaded
    if hist_names is None:
        hist_names = []
        with uproot.open(filename) as f:
            for key in f.keys():
                if is_histogram(f[key], key):
                    logger.info(f"Loading histogram {key}")
                    hist_names.append(key.split(';')[0])
                else:
                    logger.warning(f"The object {key} is not a histogram, skipping")

    # load the histograms
    with uproot.open(filename) as f:
        for hist_name in hist_names:
            histograms[hist_name] = f[hist_name].values()
            histograms[f'{hist_name}_err'] = np.sqrt(f[hist_name].variances())

    return histograms


# Get the model and data in each bin
def get_model_and_data_in_each_bin(
    model_pred: cabinetry.model_utils.ModelPrediction,
    data: List[float],
    bin_index: int,
) -> Tuple[cabinetry.model_utils.ModelPrediction, List[float]]:
    """Get the model and data in each bin"""

    selected_yields: List[List[List[float]]] = []
    selected_uncertainties: List[List[List[float]]] = []

    # Loop over channels
    for channel_yields, channel_uncs in zip(model_pred.model_yields, model_pred.total_stdev_model_bins):
        channel_selected_yields = []
        channel_selected_uncertainties = []

        # Loop over samples
        for sample_yields in channel_yields:
            channel_selected_yields.append([sample_yields[bin_index]])

        for sample_uncs in channel_uncs:
            channel_selected_uncertainties.append([sample_uncs[bin_index]])

        selected_yields.append(channel_selected_yields)
        selected_uncertainties.append(channel_selected_uncertainties)

    # Create new ModelPrediction object for the projection
    selected_pred = cabinetry.model_utils.ModelPrediction(
        model=model_pred.model,
        model_yields=selected_yields,
        total_stdev_model_bins=selected_uncertainties,
        total_stdev_model_channels=model_pred.total_stdev_model_channels,
        label=model_pred.label,
    )

    return selected_pred, [data[bin_index]]


def create_projection(
    model_pred: cabinetry.model_utils.ModelPrediction,
    data: List[float],
    axis: int,
    shape: Tuple[int, ...],
) -> Tuple[cabinetry.model_utils.ModelPrediction, List[float]]:
    """Create projection of model prediction and data onto specified axis

    Args:
        model_pred: ModelPrediction object
        data: list of data values
        axis: axis to project onto (0 to ndim-1)
        shape: original histogram shape (e.g., (40, 32, 4) for 3D)
    """

    # Convert data list to numpy array and reshape
    data_array: np.ndarray = np.array(data)
    data_nd: np.ndarray = data_array.reshape(shape)
    ndim: int = len(shape)
    axes_to_sum: Tuple[int, ...] = tuple(i for i in range(ndim) if i != axis)
    data_proj: np.ndarray = data_nd.sum(axis=axes_to_sum)

    # Project model predictions for each sample (channel, sample, bin)
    projected_yields: List[List[List[float]]] = []
    projected_uncertainties: List[List[List[float]]] = []

    # Loop over channels
    for channel_yields, channel_uncs in zip(model_pred.model_yields, model_pred.total_stdev_model_bins):
        channel_proj_yields = []
        channel_proj_uncs = []

        # Loop over samples in the channel
        for sample_yields in channel_yields:
            sample_array = np.array(sample_yields).reshape(shape)
            sample_proj = sample_array.sum(axis=axes_to_sum)
            channel_proj_yields.append(sample_proj.tolist())

        # Handle uncertainties for this channel
        for sample_unc in channel_uncs:
            unc_array = np.array(sample_unc).reshape(shape)
            unc_proj = np.sqrt(np.sum(unc_array**2, axis=axes_to_sum))
            channel_proj_uncs.append(unc_proj.tolist())

        projected_yields.append(channel_proj_yields)
        projected_uncertainties.append(channel_proj_uncs)

    # Create new ModelPrediction object for the projection
    proj_pred = cabinetry.model_utils.ModelPrediction(
        model=model_pred.model,
        model_yields=projected_yields,  # List of channels, each containing list of samples
        total_stdev_model_bins=projected_uncertainties,  # Same structure as model_yields
        total_stdev_model_channels=model_pred.total_stdev_model_channels,
        label=model_pred.label,
    )

    return proj_pred, data_proj.tolist()


def perform_fit(
    workspace: pyhf.Workspace,
) -> cabinetry.fit.FitResults:
    """Perform fit using cabinetry"""
    model, data = cabinetry.model_utils.model_and_data(workspace)

    init_pars = model.config.suggested_init()
    par_bounds = model.config.suggested_bounds()
    par_fixed = model.config.suggested_fixed()

    fit_results = cabinetry.fit.fit(
        model,
        data,
        init_pars=init_pars,
        par_bounds=par_bounds,
        fix_pars=par_fixed,
        strategy=2,
    )

    # Print the fit results
    rprint(fit_results)

    for name, value, error in zip(fit_results.labels, fit_results.bestfit, fit_results.uncertainty):
        rprint(f"{name}: {value:.4f} ± {error:.4f}")

    return fit_results


def visualise_fit_results(
    fit_results: cabinetry.fit.FitResults,
    workspace: pyhf.Workspace,
    output_folder: str,
):

    # get the model and data from the workspace
    model, data = cabinetry.model_utils.model_and_data(workspace)

    # visualize the fit results
    output_folder = Path(output_folder).resolve().as_posix()
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # save the correlation matrix
    cabinetry.visualize.correlation_matrix(fit_results, figure_folder=output_folder, pruning_threshold=0)

    # visualize pulls
    cabinetry.visualize.pulls(fit_results, figure_folder=output_folder)

    # # ranking the parameters by significance
    # ranking_results = cabinetry.fit.ranking(model, data)
    # cabinetry.visualize.ranking(ranking_results)

    # # perform the scan over the parameter of interest
    # scan_results = cabinetry.fit.scan(model, data, par_name='RawRDst')
    # cabinetry.visualize.scan(scan_results)

    # # limit scan results
    # limit_results = cabinetry.fit.limit(model, data)
    # cabinetry.visualize.limit(limit_results)

    # # significance results
    # significance_results = cabinetry.fit.significance(model, data)
    # rprint(significance_results)

    # exit(1)

    # visualize the distribution of the data and the fit
    data_no_aux = workspace.data(model, include_auxdata=False)
    model_pred_prefit = cabinetry.model_utils.prediction(model)

    # for bin_index in range(len(data_no_aux)):
    for bin_index in range(0, 2):

        selected_pred, selected_data = get_model_and_data_in_each_bin(model_pred_prefit, data_no_aux, bin_index)

        cabinetry.visualize.data_mc(
            selected_pred,
            selected_data,
            figure_folder=f'{output_folder}/data-model-comparison/prefit/bin_{bin_index}',
            close_figure=True,
            log_scale=True,
        )

        cabinetry.visualize.data_mc(
            selected_pred,
            selected_data,
            figure_folder=f'{output_folder}/data-model-comparison/prefit/bin_{bin_index}',
            close_figure=True,
            log_scale=False,
        )

    # # Create a subset of the prediction for the first 10 bins
    # subset_pred = cabinetry.model_utils.ModelPrediction(
    #     model=model_pred_prefit.model,
    #     model_yields=[[yields[:10] for yields in channel] for channel in model_pred_prefit.model_yields],
    #     total_stdev_model_bins=[[stdev[:10] for stdev in channel] for channel in model_pred_prefit.total_stdev_model_bins],
    #     total_stdev_model_channels=model_pred_prefit.total_stdev_model_channels,
    #     label=model_pred_prefit.label,
    # )

    # # Create subset of data for first 10 bins
    # subset_data = data[:10]

    # # Visualize the fit results for first 10 bins
    # cabinetry.visualize.data_mc(
    #     subset_pred,
    #     subset_data,
    #     figure_folder="figures/first_10_bins",
    # )

    # Create projections for each axis
    axis_labels = [
        (r"$m^2_{miss}$ [GeV$^2$]", "Events"),
        (r"$E_{\mu}$ [MeV]", "Events"),
        (r"$q^2$ [MeV$^2$]", "Events"),
    ]

    shape = (40, 32, 4)
    # model - data comparison before fit
    for axis, (xlabel, ylabel) in enumerate(axis_labels):
        proj_pred, proj_data = create_projection(model_pred_prefit, data_no_aux, axis=axis, shape=shape)

        # Visualize the projection
        cabinetry.visualize.data_mc(
            proj_pred,
            proj_data,
            figure_folder=f'{output_folder}/data-model-comparison/prefit/projection_axis_{axis}',
            close_figure=True,
            log_scale=True,
        )

        cabinetry.visualize.data_mc(
            proj_pred,
            proj_data,
            figure_folder=f'{output_folder}/data-model-comparison/prefit/projection_axis_{axis}',
            close_figure=True,
            log_scale=False,
        )

    # model - data comparison after fit
    model_pred_postfit = cabinetry.model_utils.prediction(model, fit_results=fit_results)
    for axis, (xlabel, ylabel) in enumerate(axis_labels):
        proj_pred, proj_data = create_projection(model_pred_postfit, data_no_aux, axis=axis, shape=shape)

        # Visualize the projection
        cabinetry.visualize.data_mc(
            proj_pred,
            proj_data,
            figure_folder=f'{output_folder}/data-model-comparison/postfit/projection_axis_{axis}',
            close_figure=True,
            log_scale=True,
        )

        fig = cabinetry.visualize.data_mc(
            proj_pred,
            proj_data,
            figure_folder=f'{output_folder}/data-model-comparison/postfit/projection_axis_{axis}',
            # close_figure=True,
            close_figure=False,
            log_scale=False,
        )

    # # Visualize the data and model comparison
    # _fig = cabinetry.visualize.data_mc(
    #     model_pred_postfit,
    #     data_no_aux,
    #     figure_folder=f"{output_folder}/data-model-comparison/postfit/data-model-comparison",
    #     close_figure=False,
    #     log_scale=False,
    # )
    # # set the figure width to 1000
    # for i in _fig:

    #     fig = i['figure']
    #     # Get the current size in inches (width, height)
    #     current_width, current_height = fig.get_size_inches()

    #     # Apply the new width while keeping the same height
    #     fig.set_size_inches(current_width * 2, current_height)

    #     # Finally, save the figure to a file
    #     fig.savefig(f"{output_folder}/data-model-comparison/postfit/data-model-comparison-wide.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    input_root_file = 'input/DemoHistos.root'
    output_root_file = 'input/DemoHistos_flattened.root'

    input_file = Path(input_root_file)
    output_file = Path(output_root_file)

    flatten_all_histograms(input_root_file, output_root_file)
