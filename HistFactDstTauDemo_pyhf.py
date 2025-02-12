'''
Author       : Jie Wu j.wu@cern.ch
Date         : 2025-02-09 07:31:44 +0100
LastEditors  : Jie Wu j.wu@cern.ch
LastEditTime : 2025-02-12 02:09:12 +0100
FilePath     : HistFactDstTauDemo_pyhf.py
Description  : Converted to use pyhf and cabinetry instead of ROOT

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


@dataclass
class AnalysisConfig:
    """Analysis configuration structure"""

    constrainDstst: bool
    useMinos: bool
    useMuShapeUncerts: bool
    useTauShapeUncerts: bool
    useDststShapeUncerts: bool
    fixshapes: bool
    fixshapesdstst: bool
    doFit: bool
    fitfirst: bool
    slowplots: bool
    BBon3d: bool
    expTau: float
    expMu: float
    relLumi: float

    def __init__(self):
        # Steering flags
        self.constrainDstst = True
        self.useMinos = True
        self.useMuShapeUncerts = True
        self.useTauShapeUncerts = True
        self.useDststShapeUncerts = True
        self.fixshapes = False
        self.fixshapesdstst = False
        self.doFit = True
        self.fitfirst = False
        self.slowplots = True
        self.BBon3d = False

        # Physics parameters
        self.expTau = 0.252 * 0.1742 * 0.781 / 0.85  # = 0.04033488282
        self.expMu = 50e3
        self.relLumi = 1.0


def flatten_nd_hist(hist: uproot.behaviors.TH3.TH3) -> np.ndarray:
    """Flatten a 3D histogram into 1D array"""
    hist_sum = hist.values().sum()

    return hist.values().flatten()


def reshape_nd_array(array: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reshape a 1D array into a multi-dimensional histogram"""
    return array.reshape(shape)


def load_histograms() -> Dict[str, np.ndarray]:
    """Load and flatten histograms from ROOT file"""
    with uproot.open("DemoHistos.root") as f:
        histograms: Dict[str, np.ndarray] = {
            "data": flatten_nd_hist(f["h_data"]),
            "sigmu": flatten_nd_hist(f["h_sigmu"]),
            "sigtau": flatten_nd_hist(f["h_sigtau"]),
            "D1": flatten_nd_hist(f["h_D1"]),
            "misID": flatten_nd_hist(f["h_misID"]),
        }

        # Load systematic variations if they exist
        if "h_sigmu_v1p" in f:
            histograms.update(
                {
                    "sigmu_v1p": flatten_nd_hist(f["h_sigmu_v1p"]),
                    "sigmu_v1m": flatten_nd_hist(f["h_sigmu_v1m"]),
                    "sigmu_v2p": flatten_nd_hist(f["h_sigmu_v2p"]),
                    "sigmu_v2m": flatten_nd_hist(f["h_sigmu_v2m"]),
                    "sigmu_v3p": flatten_nd_hist(f["h_sigmu_v3p"]),
                    "sigmu_v3m": flatten_nd_hist(f["h_sigmu_v3m"]),
                }
            )

        if "h_sigtau_v1p" in f:
            histograms.update(
                {
                    "sigtau_v1p": flatten_nd_hist(f["h_sigtau_v1p"]),
                    "sigtau_v1m": flatten_nd_hist(f["h_sigtau_v1m"]),
                    "sigtau_v2p": flatten_nd_hist(f["h_sigtau_v2p"]),
                    "sigtau_v2m": flatten_nd_hist(f["h_sigtau_v2m"]),
                    "sigtau_v3p": flatten_nd_hist(f["h_sigtau_v3p"]),
                    "sigtau_v3m": flatten_nd_hist(f["h_sigtau_v3m"]),
                    "sigtau_v4p": flatten_nd_hist(f["h_sigtau_v4p"]),
                    "sigtau_v4m": flatten_nd_hist(f["h_sigtau_v4m"]),
                }
            )

        if "h_D1IWp" in f:
            histograms.update(
                {
                    "D1_IWp": flatten_nd_hist(f["h_D1IWp"]),
                    "D1_IWm": flatten_nd_hist(f["h_D1IWm"]),
                }
            )

    return histograms


def create_pyhf_workspace(
    config: AnalysisConfig,
    histograms: Dict[str, np.ndarray],
) -> pyhf.Workspace:
    """Create pyhf workspace directly"""

    # Calculate MC normalizations
    mc_norm_sigmu: float = 1.0 / np.sum(histograms["sigmu"])
    mc_norm_sigtau: float = 1.0 / np.sum(histograms["sigtau"])
    mc_norm_D1: float = 1.0 / np.sum(histograms["D1"])
    # mc_norm_misID: float = 1.0 / np.sum(histograms["misID"])

    spec = {
        'channels': [
            {
                'name': 'Dstmu_kinematic',
                'samples': [
                    {
                        'name': 'sigmu',
                        'data': histograms['sigmu'].tolist(),
                        'modifiers': [
                            {'name': 'mcNorm_sigmu', 'type': 'normfactor', 'data': None},
                            {'name': 'Nmu', 'type': 'normfactor', 'data': None},
                        ],
                    },
                    {
                        'name': 'sigtau',
                        'data': histograms['sigtau'].tolist(),
                        'modifiers': [
                            {'name': 'mcNorm_sigtau', 'type': 'normfactor', 'data': None},
                            {'name': 'Nmu', 'type': 'normfactor', 'data': None},
                            {'name': 'RawRDst', 'type': 'normfactor', 'data': None},
                        ],
                    },
                    {
                        'name': 'D1',
                        'data': histograms['D1'].tolist(),
                        'modifiers': [
                            {'name': 'mcNorm_D1', 'type': 'normfactor', 'data': None},
                            {'name': 'Nmu', 'type': 'normfactor', 'data': None},
                            {'name': 'NDstst0', 'type': 'normfactor', 'data': None},
                            {'name': 'fD1', 'type': 'normfactor', 'data': None},
                            {'name': 'BFD1', 'type': 'normsys', 'data': {'hi': 1.1, 'lo': 0.9}},
                        ],
                    },
                    {
                        'name': 'misID',
                        'data': histograms['misID'].tolist(),
                        'modifiers': [
                            # {'name': 'mcNorm_misID', 'type': 'normfactor', 'data': None},
                            {'name': 'NmisID', 'type': 'normfactor', 'data': None},
                        ],
                    },
                ],
            }
        ],
        # The data
        'observations': [
            {'name': 'Dstmu_kinematic', 'data': histograms['data'].tolist()},
        ],
        # The measurement
        'measurements': [
            {
                'name': 'DstTau',
                'config': {
                    'parameters': [
                        # Fixed MC normalization factors - keep these exactly as is
                        {'name': 'mcNorm_sigmu', 'bounds': [[mc_norm_sigmu, mc_norm_sigmu]], 'inits': [mc_norm_sigmu], 'fixed': True},
                        {'name': 'mcNorm_sigtau', 'bounds': [[mc_norm_sigtau, mc_norm_sigtau]], 'inits': [mc_norm_sigtau], 'fixed': True},
                        {'name': 'mcNorm_D1', 'bounds': [[mc_norm_D1, mc_norm_D1]], 'inits': [mc_norm_D1], 'fixed': True},
                        # {'name': 'mcNorm_misID', 'bounds': [[mc_norm_misID, mc_norm_misID]], 'inits': [mc_norm_misID], 'fixed': True},
                        # Signal parameters
                        {'name': 'Nmu', 'bounds': [[1e-6, 1e6]], 'inits': [config.expMu]},
                        {'name': 'RawRDst', 'bounds': [[1e-6, 0.2]], 'inits': [config.expTau]},
                        # Background parameters
                        {'name': 'NDstst0', 'bounds': [[0.102, 0.102]], 'inits': [0.102], 'fixed': True},  # Fixed
                        {'name': 'fD1', 'bounds': [[3.2, 3.2]], 'inits': [3.2], 'fixed': True},  # Fixed
                        {'name': 'BFD1', 'bounds': [[-3, 3]], 'inits': [0.0]},
                        {'name': 'NmisID', 'bounds': [[1.0, 1.0]], 'inits': [1.0], 'fixed': True},  # Fixed
                    ],
                    'poi': 'RawRDst',
                },
            }
        ],
        'version': '1.0.0',
    }

    # Add systematics if enabled
    if config.useMuShapeUncerts:
        for sample in spec['channels'][0]['samples']:
            if sample['name'] == 'sigmu':
                sample['modifiers'].extend(
                    [
                        # histosys
                        {'name': 'mu_shape_v1', 'type': 'histosys', 'data': {'hi_data': histograms['sigmu_v1p'].tolist(), 'lo_data': histograms['sigmu_v1m'].tolist()}},
                        {'name': 'mu_shape_v2', 'type': 'histosys', 'data': {'hi_data': histograms['sigmu_v2p'].tolist(), 'lo_data': histograms['sigmu_v2m'].tolist()}},
                        {'name': 'mu_shape_v3', 'type': 'histosys', 'data': {'hi_data': histograms['sigmu_v3p'].tolist(), 'lo_data': histograms['sigmu_v3m'].tolist()}},
                    ]
                )
                # Add parameters for shape systematics

                spec['measurements'][0]['config']['parameters'].extend(
                    [
                        {'name': 'mu_shape_v1', 'bounds': [[-8, 8]], 'inits': [0.0]},
                        {'name': 'mu_shape_v2', 'bounds': [[-12, 12]], 'inits': [0.0]},
                        {'name': 'mu_shape_v3', 'bounds': [[-8, 8]], 'inits': [0.0]},
                    ]
                )

    if config.useTauShapeUncerts:
        for sample in spec['channels'][0]['samples']:
            if sample['name'] == 'sigtau':
                sample['modifiers'].extend(
                    [
                        # shared with sigmu
                        {'name': 'mu_shape_v1', 'type': 'histosys', 'data': {'hi_data': histograms['sigtau_v1p'].tolist(), 'lo_data': histograms['sigtau_v1m'].tolist()}},
                        {'name': 'mu_shape_v2', 'type': 'histosys', 'data': {'hi_data': histograms['sigtau_v2p'].tolist(), 'lo_data': histograms['sigtau_v2m'].tolist()}},
                        {'name': 'mu_shape_v3', 'type': 'histosys', 'data': {'hi_data': histograms['sigtau_v3p'].tolist(), 'lo_data': histograms['sigtau_v3m'].tolist()}},
                        # for tau shape
                        {'name': 'tau_shape_v4', 'type': 'histosys', 'data': {'hi_data': histograms['sigtau_v4p'].tolist(), 'lo_data': histograms['sigtau_v4m'].tolist()}},
                    ]
                )
                # Add parameter for tau shape systematic
                spec['measurements'][0]['config']['parameters'].extend(
                    [
                        {'name': 'tau_shape_v4', 'bounds': [[-5, 5]], 'inits': [0.0]},
                    ]
                )

    if config.useDststShapeUncerts:
        for sample in spec['channels'][0]['samples']:
            if sample['name'] == 'D1':
                sample['modifiers'].extend(
                    [
                        {'name': 'IW', 'type': 'histosys', 'data': {'hi_data': histograms['D1_IWp'].tolist(), 'lo_data': histograms['D1_IWm'].tolist()}},
                    ]
                )
                # Add parameter for IW systematic
                spec['measurements'][0]['config']['parameters'].extend(
                    [
                        {'name': 'IW', 'bounds': [[-3, 3]], 'inits': [0.0]},
                    ]
                )

    return pyhf.Workspace(spec)


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
    config: AnalysisConfig,
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

    # Visualize the data and model comparison
    _fig = cabinetry.visualize.data_mc(
        model_pred_postfit,
        data_no_aux,
        figure_folder=f"{output_folder}/data-model-comparison/postfit/data-model-comparison",
        close_figure=False,
        log_scale=False,
    )
    # # set the figure width to 1000
    # for i in _fig:

    #     fig = i['figure']
    #     # Get the current size in inches (width, height)
    #     current_width, current_height = fig.get_size_inches()

    #     # Apply the new width while keeping the same height
    #     fig.set_size_inches(current_width * 2, current_height)

    #     # Finally, save the figure to a file
    #     fig.savefig(f"{output_folder}/data-model-comparison/postfit/data-model-comparison-wide.png", dpi=300, bbox_inches='tight')


def update_main() -> Tuple[pyhf.Workspace, Optional[List[Tuple[float, float]]]]:
    """Updated main function using pyhf"""
    # Create configuration
    config = AnalysisConfig()

    # Load and flatten histograms
    histograms = load_histograms()

    # Create pyhf workspace directly
    workspace = create_pyhf_workspace(config, histograms)

    # Check the workspace by using cabinetry
    model, data = cabinetry.model_utils.model_and_data(workspace)
    cabinetry.visualize.modifier_grid(model, split_by_sample=True)
    # cabinetry.visualize.modifier_grid(model, split_by_sample=False)

    # Perform fit if requested
    result = None
    if config.doFit:
        result = perform_fit(workspace, config)

        if result is not None:

            # Generate plots
            visualise_fit_results(result, workspace, "results-pyhf")
            # generate_plots(workspace, result, config)
            pass
        else:
            print("Fit failed.")

    return workspace, result


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Run the main function
    workspace, result = update_main()
