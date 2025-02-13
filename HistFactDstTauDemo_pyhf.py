'''
Author       : Jie Wu j.wu@cern.ch
Date         : 2025-02-09 07:31:44 +0100
LastEditors  : Jie Wu j.wu@cern.ch
LastEditTime : 2025-02-13 09:09:01 +0100
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


from utilities_pyhf import perform_fit, visualise_fit_results


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
    with uproot.open("input/DemoHistos.root") as f:
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
        result = perform_fit(workspace)

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
