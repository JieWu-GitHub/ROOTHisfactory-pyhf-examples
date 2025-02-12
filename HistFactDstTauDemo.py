'''
Author       : Jie Wu j.wu@cern.ch
Date         : 2025-02-09 07:31:44 +0100
LastEditors  : Jie Wu j.wu@cern.ch
LastEditTime : 2025-02-10 05:00:16 +0100
FilePath     : HistFactDstTauDemo.py
Description  : 

Copyright (c) 2025 by everyone, All Rights Reserved. 
'''

#########################################################################
# File Name: HistFactDstTauDemo.py
# Author: Jie Wu
# mail: j.wu@cern.ch
# Created Time: Sun 09 Feb 2025 07:31:24 AM CET
#########################################################################


import os
import sys
from pathlib import Path
from dataclasses import dataclass
import ROOT
from ROOT import (
    TRandom3,
    TCanvas,
    TDatime,
    TStopwatch,
    TLegend,
    TIterator,
    TH3,
    TLatex,
    RooChi2Var,
    RooAbsData,
    RooRealSumPdf,
    RooPoisson,
    RooGaussian,
    RooRealVar,
    RooMCStudy,
    RooMinimizer,
    RooCategory,
    RooHistPdf,
    RooSimultaneous,
    RooExtendPdf,
    RooDataSet,
    RooDataHist,
    RooFitResult,
    RooMsgService,
    RooParamHistFunc,
    RooHist,
    RooRandom,
    RooStats,
    gROOT,
    gStyle,
    kTRUE,
    kFALSE,
)

# Enable multi-threading
ROOT.EnableImplicitMT()

# Define whether to unblind
UNBLIND = True


@dataclass
class AnalysisConfig:
    """Analysis configuration structure"""

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


# Global date pointer
global_date = None


def initialize_global_settings():
    """Initialize global settings and style"""
    global global_date

    # Create a global time object to seed the random generator
    global_date = TDatime()

    # Set some graphic style options
    t = TLatex()
    t.SetTextAlign(22)
    t.SetTextSize(0.06)
    t.SetTextFont(132)

    # Set ROOT style parameters
    gStyle.SetLabelFont(132, "xyz")
    gStyle.SetTitleFont(132, "xyz")
    gStyle.SetTitleFont(132, "t")
    gStyle.SetTitleSize(0.08, "t")
    gStyle.SetTitleY(0.970)


def load_normalization_factors():
    """Load normalization factors from DemoHistos.root"""
    with ROOT.TFile("DemoHistos.root") as q:
        mc_norms = {}
        mc_histos = ["sigmu", "sigtau", "D1"]

        for histo in mc_histos:
            h_temp = q.Get(f"h_{histo}")
            assert h_temp is not None, f"Histogram h_{histo} not found"
            mc_norms[histo] = 1.0 / h_temp.Integral()
            print(f"mcNorm_{histo} = {1.0 / mc_norms[histo]}")

    return mc_norms


def get_q2_binning():
    """Get q² binning information from h_sigmu"""
    with ROOT.TFile("DemoHistos.root") as file:
        h_sigmu = file.Get("h_sigmu")
        assert h_sigmu is not None, "h_sigmu histogram not found"

        q2_low = h_sigmu.GetZaxis().GetXmin()
        q2_high = h_sigmu.GetZaxis().GetXmax()
        q2_bins = h_sigmu.GetZaxis().GetNbins()

    return q2_low, q2_high, q2_bins


def add_signal_mu_sample(chan, config, mc_norm_sigmu):
    """Add B0->D*munu (NORM) sample
    Norm = Nmu * mcNorm_sigmu
    """
    sigmu = ROOT.RooStats.HistFactory.Sample("h_sigmu", "h_sigmu", "DemoHistos.root")

    if config.useMuShapeUncerts:
        sigmu.AddHistoSys("v1mu", "h_sigmu_v1m", "DemoHistos.root", "", "h_sigmu_v1p", "DemoHistos.root", "")
        sigmu.AddHistoSys("v2mu", "h_sigmu_v2m", "DemoHistos.root", "", "h_sigmu_v2p", "DemoHistos.root", "")
        sigmu.AddHistoSys("v3mu", "h_sigmu_v3m", "DemoHistos.root", "", "h_sigmu_v3p", "DemoHistos.root", "")

    if config.BBon3d:
        sigmu.ActivateStatError()

    # Normalize by theory (not needed for signal)
    sigmu.SetNormalizeByTheory(kFALSE)
    sigmu.AddNormFactor("Nmu", config.expMu, 1e-6, 1e6)
    sigmu.AddNormFactor("mcNorm_sigmu", mc_norm_sigmu, 1e-9, 1.0)
    chan.AddSample(sigmu)


def add_signal_tau_sample(chan, config, mc_norm_sigtau):
    """Add B0->D*taunu (SIGNAL) sample
    Norm = Nmu * RawRDst * mcNorm_sigtau
    """
    sigtau = ROOT.RooStats.HistFactory.Sample("h_sigtau", "h_sigtau", "DemoHistos.root")

    if config.useTauShapeUncerts:
        sigtau.AddHistoSys("v1mu", "h_sigtau_v1m", "DemoHistos.root", "", "h_sigtau_v1p", "DemoHistos.root", "")
        sigtau.AddHistoSys("v2mu", "h_sigtau_v2m", "DemoHistos.root", "", "h_sigtau_v2p", "DemoHistos.root", "")
        sigtau.AddHistoSys("v3mu", "h_sigtau_v3m", "DemoHistos.root", "", "h_sigtau_v3p", "DemoHistos.root", "")
        sigtau.AddHistoSys("v4tau", "h_sigtau_v4m", "DemoHistos.root", "", "h_sigtau_v4p", "DemoHistos.root", "")

    if config.BBon3d:
        sigtau.ActivateStatError()

    sigtau.SetNormalizeByTheory(kFALSE)
    sigtau.AddNormFactor("Nmu", config.expMu, 1e-6, 1e6)
    sigtau.AddNormFactor("RawRDst", config.expTau, 1e-6, 0.2)
    sigtau.AddNormFactor("mcNorm_sigtau", mc_norm_sigtau, 1e-9, 1.0)
    chan.AddSample(sigtau)


def add_d1_background(chan, config, mc_norm_D1):
    """Add D1 background sample"""
    d1mu = ROOT.RooStats.HistFactory.Sample("h_D1", "h_D1", "DemoHistos.root")

    if config.BBon3d:
        d1mu.ActivateStatError()

    if config.useDststShapeUncerts:
        d1mu.AddHistoSys("IW", "h_D1IWm", "DemoHistos.root", "", "h_D1IWp", "DemoHistos.root", "")

    d1mu.SetNormalizeByTheory(kFALSE)
    d1mu.AddNormFactor("mcNorm_D1", mc_norm_D1, 1e-9, 1.0)

    if not config.constrainDstst:
        d1mu.AddNormFactor("ND1", 1e2, 1e-6, 1e5)
    else:
        d1mu.AddNormFactor("NDstst0", 0.102, 1e-6, 1e0)
        d1mu.AddNormFactor("Nmu", config.expMu, 1e-6, 1e6)
        d1mu.AddNormFactor("fD1", 3.2, 3.2, 3.2)
        d1mu.AddOverallSys("BFD1", 0.9, 1.1)

    chan.AddSample(d1mu)


def add_misid_background(chan, config):
    """Add misID background sample"""
    misID = ROOT.RooStats.HistFactory.Sample("h_misID", "h_misID", "DemoHistos.root")

    if config.BBon3d:
        misID.ActivateStatError()

    misID.SetNormalizeByTheory(kTRUE)
    misID.AddNormFactor("NmisID", config.relLumi, 1e-6, 1e5)
    chan.AddSample(misID)


def build_model(config, mc_norms):
    """Build the HistFactory model and return the RooWorkspace"""

    # Create measurement
    meas = ROOT.RooStats.HistFactory.Measurement("my_measurement", "my measurement")
    meas.SetOutputFilePrefix("results/my_measurement")
    meas.SetExportOnly(kTRUE)
    meas.SetPOI("RawRDst")

    # Set luminosity parameters
    meas.SetLumi(1.0)
    meas.SetLumiRelErr(0.05)
    meas.AddConstantParam("Lumi")

    # Create channel and set data
    chan = ROOT.RooStats.HistFactory.Channel("Dstmu_kinematic")
    chan.SetStatErrorConfig(1e-5, "Poisson")
    chan.SetData("h_data", "DemoHistos.root")

    # Add samples
    add_signal_mu_sample(chan, config, mc_norms['sigmu'])
    add_signal_tau_sample(chan, config, mc_norms['sigtau'])
    add_d1_background(chan, config, mc_norms['D1'])
    add_misid_background(chan, config)

    # Add channel to measurement
    meas.AddChannel(chan)

    # Collect histograms and create workspace
    meas.CollectHistograms()
    w = ROOT.RooStats.HistFactory.MakeModelAndMeasurementFast(meas)

    # Save the measurement as XML
    meas.PrintXML("results/my_measurement")

    return w


def configure_parameters(w, config):
    """Configure miscellaneous model parameters"""
    mc = w.obj("ModelConfig")

    # Fix the normalization factors for Monte Carlo samples
    mc_histos = ["sigmu", "sigtau", "D1"]
    for histo in mc_histos:
        par = mc.GetNuisanceParameters().find(f"mcNorm_{histo}")
        if par:
            par.setConstant(kTRUE)
            print(f"Fixed mcNorm_{histo} = {par.getVal()}")

    # Fix specific nuisance parameters
    ndstst0 = mc.GetNuisanceParameters().find("NDstst0")
    if ndstst0:
        ndstst0.setVal(0.102)
        ndstst0.setConstant(kTRUE)

    fD1 = mc.GetNuisanceParameters().find("fD1")
    if fD1:
        fD1.setConstant(kTRUE)

    nmisID = mc.GetNuisanceParameters().find("NmisID")
    if nmisID:
        nmisID.setConstant(kTRUE)

    # Set ranges for shape systematic parameters
    if config.useDststShapeUncerts:
        alphaIW = mc.GetNuisanceParameters().find("alpha_IW")
        if alphaIW:
            alphaIW.setRange(-3.0, 3.0)

    if config.useMuShapeUncerts:
        alpha_v1mu = mc.GetNuisanceParameters().find("alpha_v1mu")
        if alpha_v1mu:
            alpha_v1mu.setRange(-8, 8)
        alpha_v2mu = mc.GetNuisanceParameters().find("alpha_v2mu")
        if alpha_v2mu:
            alpha_v2mu.setRange(-12, 12)
        alpha_v3mu = mc.GetNuisanceParameters().find("alpha_v3mu")
        if alpha_v3mu:
            alpha_v3mu.setRange(-8, 8)

    alpha_BFD1 = mc.GetNuisanceParameters().find("alpha_BFD1")
    if alpha_BFD1:
        alpha_BFD1.setRange(-3, 3)

    # Optionally fix shape parameters
    if config.fixshapes:
        a1 = mc.GetNuisanceParameters().find("alpha_v1mu")
        if a1:
            a1.setVal(1.06)
            a1.setConstant(kTRUE)
        a2 = mc.GetNuisanceParameters().find("alpha_v2mu")
        if a2:
            a2.setVal(-0.159)
            a2.setConstant(kTRUE)
        a3 = mc.GetNuisanceParameters().find("alpha_v3mu")
        if a3:
            a3.setVal(-1.75)
            a3.setConstant(kTRUE)
        a4 = mc.GetNuisanceParameters().find("alpha_v4tau")
        if a4:
            a4.setVal(0.0002)
            a4.setConstant(kTRUE)

    if config.fixshapesdstst:
        alpha_IW = mc.GetNuisanceParameters().find("alpha_IW")
        if alpha_IW:
            alpha_IW.setVal(-0.005)
            alpha_IW.setConstant(kTRUE)


def perform_fit(w, config):
    """Perform the fit using RooMinimizer and return the result"""
    mc = w.obj("ModelConfig")
    data = w.data("obsData")

    # Use a RooSimultaneous PDF
    model = mc.GetPdf()

    # Create a new PDF for the HF model
    model_hf = ROOT.RooSimultaneous(model)
    nll = model_hf.createNLL(data, ROOT.RooFit.Offset(kTRUE), ROOT.RooFit.NumCPU(8))

    minimizer = ROOT.RooMinimizer(nll)
    minimizer.setErrorLevel(0.5)
    if not UNBLIND:
        minimizer.setPrintLevel(-1)
    minimizer.setStrategy(2)

    tries = 5
    status = minimizer.migrad()
    while status != 0 and tries > 0:
        status = minimizer.migrad()
        tries -= 1

    minimizer.hesse()
    if config.useMinos:
        minimizer.minos(mc.GetParametersOfInterest())

    result = minimizer.save("Result", "Result")

    return result


def generate_plots(w, result, config):
    """Generate plots for the fit results"""
    mc = w.obj("ModelConfig")
    data = w.data("obsData")
    model = mc.GetPdf()

    # Retrieve observables and set titles/units
    obs = mc.GetObservables()
    x = obs.find("obs_x_Dstmu_kinematic")
    y = obs.find("obs_y_Dstmu_kinematic")
    z = obs.find("obs_z_Dstmu_kinematic")

    x.SetTitle("m^{2}_{miss}")
    x.setUnit("GeV^{2}")
    y.SetTitle("E_{#mu}")
    y.setUnit("MeV")
    z.SetTitle("q^{2}")
    z.setUnit("MeV^{2}")

    # The category for simultaneous fits
    idx = obs.find("channelCat")

    # Create one-dimensional frames for each observable
    mm2_frame = x.frame(ROOT.RooFit.Title("m^{2}_{miss}"))
    El_frame = y.frame(ROOT.RooFit.Title("E_{#mu}"))
    q2_frame = z.frame(ROOT.RooFit.Title("q^{2}"))

    # Create the colors for the components
    colors = [ROOT.kRed, ROOT.kBlue + 1, ROOT.kViolet, ROOT.kViolet + 1, ROOT.kViolet + 2, ROOT.kGreen, ROOT.kGreen + 1, ROOT.kOrange + 1, ROOT.kOrange + 2, ROOT.kOrange + 3]

    names = [
        "Data",
        "Total Fit",
        "B #rightarrow D*#mu#nu",
        "B #rightarrow D**#mu#nu",
        "B #rightarrow D**#tau#nu",
        "B #rightarrow D*[D_{q} #rightarrow #mu#nuX]Y",
        "Combinatoric (wrong-sign)",
        "Misidentification BKG",
        "Wrong-sign slow #pi",
    ]

    # Plot data and model
    frames = [mm2_frame, El_frame, q2_frame]
    resids = []

    for frame in frames:
        data.plotOn(frame, ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson), ROOT.RooFit.Cut("channelCat==0"), ROOT.RooFit.MarkerSize(0.4), ROOT.RooFit.DrawOption("ZP"))

        model.plotOn(frame, ROOT.RooFit.Slice(idx), ROOT.RooFit.ProjWData(idx, data), ROOT.RooFit.DrawOption("F"), ROOT.RooFit.FillColor(ROOT.kRed))

        # Grab pulls
        resids.append(frame.pullHist())

        # Plot components
        model.plotOn(frame, ROOT.RooFit.Slice(idx), ROOT.RooFit.ProjWData(idx, data), ROOT.RooFit.DrawOption("F"), ROOT.RooFit.FillColor(ROOT.kViolet), ROOT.RooFit.Components("*misID*,*sigmu*,*D1*"))

        model.plotOn(frame, ROOT.RooFit.Slice(idx), ROOT.RooFit.ProjWData(idx, data), ROOT.RooFit.DrawOption("F"), ROOT.RooFit.FillColor(ROOT.kBlue + 1), ROOT.RooFit.Components("*misID*,*sigmu*"))

        model.plotOn(frame, ROOT.RooFit.Slice(idx), ROOT.RooFit.ProjWData(idx, data), ROOT.RooFit.DrawOption("F"), ROOT.RooFit.FillColor(ROOT.kOrange), ROOT.RooFit.Components("*misID*"))

        data.plotOn(frame, ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson), ROOT.RooFit.Cut("channelCat==0"), ROOT.RooFit.MarkerSize(0.4), ROOT.RooFit.DrawOption("ZP"))

    # Create and save plots
    os.makedirs("plots", exist_ok=True)

    c1 = ROOT.TCanvas("c1", "c1", 1000, 300)
    c1.Divide(3, 1)

    for i, frame in enumerate(frames, 1):
        pad = c1.cd(i)
        pad.SetTickx(1)
        pad.SetTicky(1)
        pad.SetRightMargin(0.02)
        pad.SetLeftMargin(0.20)
        pad.SetTopMargin(0.02)
        pad.SetBottomMargin(0.13)

        frame.SetTitle("")
        frame.GetXaxis().SetLabelSize(0.06)
        frame.GetXaxis().SetTitleSize(0.06)
        frame.GetYaxis().SetLabelSize(0.06)
        frame.GetYaxis().SetTitleSize(0.06)
        frame.GetYaxis().SetTitleOffset(1.75)
        frame.GetXaxis().SetTitleOffset(0.9)

        title = frame.GetYaxis().GetTitle()
        title = title.replace("Events", "Candidates")
        frame.GetYaxis().SetTitle(title)

        frame.Draw()

        t = ROOT.TLatex()
        t.SetTextAlign(22)
        t.SetTextSize(0.06)
        t.SetTextFont(132)

        if i == 1:
            t.DrawLatex(8.7, frame.GetMaximum() * 0.95, "Demo")
        elif i == 2:
            t.DrawLatex(2250, frame.GetMaximum() * 0.95, "Demo")
        else:
            t.DrawLatex(11.1e6, frame.GetMaximum() * 0.95, "Demo")

    c1.SaveAs("plots/HistFactDstTauDemo_nominal.png")

    # Create pull plots
    c3 = ROOT.TCanvas("c3", "c3", 640, 1000)
    c3.Divide(1, 3)

    pull_frames = []
    for i, resid in enumerate(resids, 1):
        pad = c3.cd(i)
        frame = frames[i - 1]
        # Create a unique name for each pull frame to avoid ROOT naming conflicts
        pull_frame = frame.emptyClone(f'pull_for_nominal_frame_{i}')
        pull_frame.addPlotable(resid, "P")

        # Set reasonable y-axis range for pulls
        pull_frame.SetMinimum(-5)
        pull_frame.SetMaximum(5)

        # Draw the pull frame
        pull_frame.Draw()

        # Store the plotable
        pull_frames.append(pull_frame)

    c3.SaveAs("plots/HistFactDstTauDemo_pulls.png")

    if config.slowplots:
        generate_slow_plots(w, data, model, idx, config)


def generate_slow_plots(w, data, model, idx, config):
    """Generate additional plots binned in q²"""
    q2_low, q2_high, q2_bins = get_q2_binning()
    print(f"Generating slow plots with {q2_bins} q² bins.")

    # Create frames for each q² bin
    mm2q2_frame = []
    Elq2_frame = []
    q2frames = []
    q2bframes = []
    mm2q2_pulls = []
    Elq2_pulls = []

    # Get observables
    mc = w.obj("ModelConfig")
    obs = mc.GetObservables()
    x = obs.find("obs_x_Dstmu_kinematic")
    y = obs.find("obs_y_Dstmu_kinematic")
    z = obs.find("obs_z_Dstmu_kinematic")

    # Initialize frames
    for i in range(2 * q2_bins):
        if i < q2_bins:
            mm2q2_frame.append(x.frame())
            q2frames.extend([mm2q2_frame[-1]])
            q2bframes.extend([x.frame()])

        else:
            Elq2_frame.append(y.frame())
            q2frames.extend([Elq2_frame[-1]])
            q2bframes.extend([y.frame()])

    # Prepare range names, labels and cut strings
    rangeNames = []
    rangeLabels = []
    cutStrings = []

    for i in range(q2_bins):
        binlow = q2_low + i * (q2_high - q2_low) / q2_bins
        binhigh = q2_low + (i + 1) * (q2_high - q2_low) / q2_bins

        rangeLabels.append(f"{binlow*1e-6:.2f} < q^{{2}} < {binhigh*1e-6:.2f}")
        rangeNames.append(f"q2bin_{i}")
        cutStrings.append(f"obs_z_Dstmu_kinematic > {binlow} && " f"obs_z_Dstmu_kinematic < {binhigh} && " f"channelCat==0")
        z.setRange(rangeNames[i], binlow, binhigh)

    print("Drawing Slow Plots")
    for i in range(q2_bins):
        # Plot mm2 frame
        data.plotOn(mm2q2_frame[i], ROOT.RooFit.Cut(cutStrings[i]), ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson), ROOT.RooFit.MarkerSize(0.4), ROOT.RooFit.DrawOption("ZP"))

        model.plotOn(
            mm2q2_frame[i], ROOT.RooFit.Slice(idx), ROOT.RooFit.ProjWData(idx, data), ROOT.RooFit.ProjectionRange(rangeNames[i]), ROOT.RooFit.DrawOption("F"), ROOT.RooFit.FillColor(ROOT.kRed)
        )

        # Plot El frame
        data.plotOn(Elq2_frame[i], ROOT.RooFit.Cut(cutStrings[i]), ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson), ROOT.RooFit.MarkerSize(0.4), ROOT.RooFit.DrawOption("ZP"))

        model.plotOn(Elq2_frame[i], ROOT.RooFit.Slice(idx), ROOT.RooFit.ProjWData(idx, data), ROOT.RooFit.ProjectionRange(rangeNames[i]), ROOT.RooFit.DrawOption("F"), ROOT.RooFit.FillColor(ROOT.kRed))

        # Get pulls
        mm2q2_pulls.append(mm2q2_frame[i].pullHist())
        Elq2_pulls.append(Elq2_frame[i].pullHist())

        # Plot components
        for frame, components in [(mm2q2_frame[i], ["*misID*,*sigmu*,*D1*", "*misID*,*sigmu*", "*misID*"]), (Elq2_frame[i], ["*misID*,*sigmu*,*D1*", "*misID*,*sigmu*", "*misID*"])]:
            for comp, color in zip(components, [ROOT.kViolet, ROOT.kBlue + 1, ROOT.kOrange]):
                model.plotOn(
                    frame,
                    ROOT.RooFit.Slice(idx),
                    ROOT.RooFit.ProjWData(idx, data),
                    ROOT.RooFit.ProjectionRange(rangeNames[i]),
                    ROOT.RooFit.DrawOption("F"),
                    ROOT.RooFit.FillColor(color),
                    ROOT.RooFit.Components(comp),
                )

            # Replot data on top
            data.plotOn(frame, ROOT.RooFit.Cut(cutStrings[i]), ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson), ROOT.RooFit.MarkerSize(0.4), ROOT.RooFit.DrawOption("ZP"))

    # Create canvas and draw plots
    c2 = ROOT.TCanvas("c2", "c2", 1200, 600)
    c2.Divide(q2_bins, 2)

    max_scale = 1.05
    max_scale2 = 1.05

    t = ROOT.TLatex()
    t.SetTextAlign(22)
    t.SetTextSize(0.06)
    t.SetTextFont(132)

    for k in range(2 * q2_bins):
        c2.cd(k + 1)

        # Create bottom pad for pulls
        padbottom = ROOT.TPad(f"bottompad_{k}", f"bottompad_{k}", 0.0, 0.0, 1.0, 0.3)
        padbottom.SetFillColor(0)
        padbottom.SetGridy()
        padbottom.SetTickx()
        padbottom.SetTicky()
        padbottom.SetFillStyle(0)
        padbottom.SetLeftMargin(padbottom.GetLeftMargin() + 0.08)
        padbottom.SetTopMargin(0)
        padbottom.SetRightMargin(0.04)
        padbottom.SetBottomMargin(0.5)
        padbottom.Draw()
        padbottom.cd()

        # Draw pulls
        pull = mm2q2_pulls[k] if k < q2_bins else Elq2_pulls[k - q2_bins]
        pull.SetFillColor(ROOT.kBlue)
        pull.SetLineColor(ROOT.kWhite)

        pull_frame = q2bframes[k]
        pull_frame.SetTitle(q2frames[k].GetTitle())
        pull_frame.addPlotable(pull, "B")

        # Configure pull frame
        pull_frame.GetXaxis().SetLabelSize(0.33 * 0.22 / 0.3)
        pull_frame.GetXaxis().SetTitleSize(0.36 * 0.22 / 0.3)
        pull_frame.GetXaxis().SetTickLength(0.10)
        pull_frame.GetYaxis().SetTickLength(0.05)
        pull_frame.SetTitle("")
        pull_frame.GetYaxis().SetTitleSize(0.33 * 0.22 / 0.3)
        pull_frame.GetYaxis().SetTitle("Pulls")
        pull_frame.GetYaxis().SetTitleOffset(0.2)
        pull_frame.GetXaxis().SetTitleOffset(0.78)
        pull_frame.GetYaxis().SetLabelSize(0.33 * 0.22 / 0.3)
        pull_frame.GetYaxis().SetLabelOffset(99)
        pull_frame.GetYaxis().SetNdivisions(205)

        pull_frame.Draw()

        # Draw pull labels
        xloc = -2.25 if k < q2_bins else 50
        t.SetTextSize(0.33 * 0.22 / 0.3)
        t.DrawLatex(xloc, -2, "-2")
        t.DrawLatex(xloc * 0.99, 2, " 2")

        # Create top pad for main plot
        c2.cd(k + 1)
        padtop = ROOT.TPad(f"toppad_{k}", f"toppad_{k}", 0.0, 0.3, 1.0, 1.0)
        padtop.SetLeftMargin(padtop.GetLeftMargin() + 0.08)
        padtop.SetBottomMargin(0)
        padtop.SetTopMargin(0.02)
        padtop.SetRightMargin(0.04)
        padtop.SetFillColor(0)
        padtop.SetFillStyle(0)
        padtop.SetTickx()
        padtop.SetTicky()
        padtop.Draw()
        padtop.cd()

        # Configure and draw main frame
        frame = q2frames[k]
        frame.SetMinimum(1e-4)
        if k < q2_bins:
            frame.SetMaximum(frame.GetMaximum() * max_scale)
        else:
            frame.SetMaximum(frame.GetMaximum() * max_scale2)

        frame.SetTitle(rangeLabels[k % q2_bins])
        frame.SetTitleFont(132, "t")
        frame.GetXaxis().SetLabelSize(0.09 * 0.78 / 0.7)
        frame.GetXaxis().SetTitleSize(0.09 * 0.78 / 0.7)
        frame.GetYaxis().SetTitleSize(0.09 * 0.78 / 0.7)

        title = frame.GetYaxis().GetTitle()
        frame.GetYaxis().SetTitle("")
        frame.GetYaxis().SetLabelSize(0.09 * 0.78 / 0.7)
        frame.GetXaxis().SetTitleOffset(0.95)
        frame.GetYaxis().SetTitleOffset(0.95)
        frame.GetYaxis().SetNdivisions(506)
        frame.Draw()

        # Draw title and labels
        t.SetTextSize(0.07)
        t.SetTextAlign(33)
        t.SetTextAngle(90)
        c2.cd((k < q2_bins) * (2 * k + 1) + (k >= q2_bins) * (2 * (k + 1 - q2_bins)))
        t.DrawLatex(0.01, 0.99, title)

        t.SetTextAlign(22)
        t.SetTextAngle(0)
        padtop.cd()
        t.SetTextSize(0.09 * 0.78 / 0.7)

        # if k >= q2_bins:
        #     t.DrawLatex(2250, frame.GetMaximum() * 0.92, "Demo")
        # else:
        #     t.DrawLatex(8.7, frame.GetMaximum() * 0.92, "Demo")

    # Save the canvas
    c2.SaveAs("plots/HistFactDstTauDemo_in_q2_bins.png")


def update_main():
    """Updated main function including fitting and plotting"""
    # Initialize settings
    initialize_global_settings()

    # Create configuration
    config = AnalysisConfig()

    # Load normalization factors
    mc_norms = load_normalization_factors()

    # Build the model
    workspace = build_model(config, mc_norms)

    # Configure parameters
    configure_parameters(workspace, config)

    # Perform fit if requested
    if config.doFit:
        result = perform_fit(workspace, config)
        if result:
            print("----------------------------------------")
            print(f"Fit Status: {result.status()}")
            print(f"Fit EDM: {result.edm()}")

            # Print results
            print("\nInitial fitted results:")
            result.floatParsInit().Print("v")

            print("\nFinal fitted results:")
            result.floatParsFinal().Print("v")

            print("\nPrint all parameters:")
            print("\nFloat parameters:")
            result.floatParsFinal().Print("v")

            print("\nConstant parameters:")
            result.constPars().Print("v")

            # Generate plots
            generate_plots(workspace, result, config)
        else:
            print("Fit failed.")

    # Save the workspace
    # workspace.writeToFile("results/workspace.root")
    # print("Workspace has been saved to results/workspace.root")

    return workspace, result if config.doFit else None


if __name__ == "__main__":
    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)

    # Run the main function
    workspace, result = update_main()
