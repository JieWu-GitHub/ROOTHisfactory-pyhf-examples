/*
 * @Author       : Jie Wu j.wu@cern.ch
 * @Date         : 2023-11-24 16:13:46 +0100
 * @LastEditors  : Jie Wu j.wu@cern.ch
 * @LastEditTime : 2025-02-09 07:20:33 +0100
 * @FilePath     : HistFactDstTauDemo_new.C
 * @Description  :
 *
 * Copyright (c) 2023 by everyone, All Rights Reserved.
 */
#include <iostream>
#include <stdio.h>

#include "TRandom3.h"
#include "TCanvas.h"
#include "TDatime.h"
#include "TStopwatch.h"
#include "TLegend.h"
#include "TIterator.h"
#include "TH3.h"
#include "TLatex.h"

#include "RooChi2Var.h"
#include "RooAbsData.h"
#include "RooRealSumPdf.h"
#include "RooPoisson.h"
#include "RooGaussian.h"
#include "RooRealVar.h"
#include "RooMCStudy.h"
#include "RooMinimizer.h"
#include "RooCategory.h"
#include "RooHistPdf.h"
#include "RooSimultaneous.h"
#include "RooExtendPdf.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooFitResult.h"
#include "RooMsgService.h"
#include "RooParamHistFunc.h"
#include "RooHist.h"
#include "RooRandom.h"

#include "RooStats/ModelConfig.h"
#include "RooStats/ToyMCSampler.h"
#include "RooStats/MinNLLTestStat.h"

#include "RooStats/HistFactory/FlexibleInterpVar.h"
#include "RooStats/HistFactory/PiecewiseInterpolation.h"
// #include "RooStats/HistFactory/HistFactorySimultaneous.h"
#include "RooStats/HistFactory/Channel.h"
#include "RooStats/HistFactory/MakeModelAndMeasurementsFast.h"
#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/ParamHistFunc.h"

// #include "HistFactoryModelUtils.cxx"
#include "RooStats/HistFactory/HistFactoryModelUtils.h"
#include "RooStats/HistFactory/RooBarlowBeestonLL.h"

// Namespaces
using namespace std;
using namespace RooFit;
using namespace RooStats;
using namespace HistFactory;

#define UNBLIND

// ---------------------------------------------------------
// Analysis configuration structure
// ---------------------------------------------------------
struct AnalysisConfig
{
    // Steering flags
    bool constrainDstst;
    bool useMinos;
    bool useMuShapeUncerts;
    bool useTauShapeUncerts;
    bool useDststShapeUncerts;
    bool fixshapes;
    bool fixshapesdstst;
    bool doFit;
    bool fitfirst;
    bool slowplots;
    bool BBon3d;

    // Physics parameters
    double expTau;
    double expMu;
    double relLumi;

    // Constructor: default values from original script call
    AnalysisConfig() : constrainDstst(true),
                       useMinos(true),
                       useMuShapeUncerts(true),
                       useTauShapeUncerts(true),
                       useDststShapeUncerts(true),
                       fixshapes(false),
                       fixshapesdstst(false),
                       doFit(true),
                       fitfirst(false),
                       slowplots(true),
                       BBon3d(false),

                       // Physics parameters
                       expTau(0.252 * 0.1742 * 0.781 / 0.85), // = 0.04033488282
                       expMu(50e3),
                       relLumi(1.0)
    {
    }
};

// Global date pointer
TDatime *globalDate = nullptr;

// ---------------------------------------------------------
// Global settings and style initialization
// ---------------------------------------------------------
void initializeGlobalSettings()
{
    // Create a global time object to seed the random generator
    globalDate = new TDatime();

    // Set some graphic style options
    TLatex *t = new TLatex();
    t->SetTextAlign(22);
    t->SetTextSize(0.06);
    t->SetTextFont(132);
    gROOT->ProcessLine("gStyle->SetLabelFont(132,\"xyz\");");
    gROOT->ProcessLine("gStyle->SetTitleFont(132,\"xyz\");");
    gROOT->ProcessLine("gStyle->SetTitleFont(132,\"t\");");
    gROOT->ProcessLine("gStyle->SetTitleSize(0.08,\"t\");");
    gROOT->ProcessLine("gStyle->SetTitleY(0.970);");

    delete t; // not needed any longer
}

// ---------------------------------------------------------
// Load normalization factors from DemoHistos.root
// ---------------------------------------------------------
void loadNormalizationFactors(double &mcNorm_sigmu, double &mcNorm_sigtau, double &mcNorm_D1)
{
    TFile q("input/DemoHistos.root");
    TH1 *htemp = nullptr;
    TString mchistos[3] = {"sigmu", "sigtau", "D1"};
    double *mcnorms[3] = {&mcNorm_sigmu, &mcNorm_sigtau, &mcNorm_D1};
    for (int i = 0; i < 3; i++)
    {
        q.GetObject("h_" + mchistos[i], htemp);
        assert(htemp != NULL);
        *(mcnorms[i]) = 1.0 / htemp->Integral();
        cout << "mcNorm_" << mchistos[i] << " = " << 1.0 / *(mcnorms[i]) << endl;
    }
    q.Close();
}

// ---------------------------------------------------------
// Get q² binning information from h_sigmu
// ---------------------------------------------------------
void getQ2Binning(double &q2_low, double &q2_high, int &q2_bins)
{
    TFile file("input/DemoHistos.root");
    TH3 *h_sigmu = nullptr;
    file.GetObject("h_sigmu", h_sigmu);
    assert(h_sigmu != NULL);
    q2_low = h_sigmu->GetZaxis()->GetXmin();
    q2_high = h_sigmu->GetZaxis()->GetXmax();
    q2_bins = h_sigmu->GetZaxis()->GetNbins();
    h_sigmu->SetDirectory(0);
    file.Close();
}

/*********************** B0->D*munu (NORM) *******************************/
// Norm = Nmu * mcNorm_sigmu
void addSignalMuSample(Channel &chan, const AnalysisConfig &config, double mcNorm_sigmu)
{
    Sample sigmu("h_sigmu", "h_sigmu", "input/DemoHistos.root");

    if (config.useMuShapeUncerts)
    {
        sigmu.AddHistoSys("v1mu", "h_sigmu_v1m", "input/DemoHistos.root", "", "h_sigmu_v1p", "input/DemoHistos.root", "");
        sigmu.AddHistoSys("v2mu", "h_sigmu_v2m", "input/DemoHistos.root", "", "h_sigmu_v2p", "input/DemoHistos.root", "");
        sigmu.AddHistoSys("v3mu", "h_sigmu_v3m", "input/DemoHistos.root", "", "h_sigmu_v3p", "input/DemoHistos.root", "");
    }

    if (config.BBon3d)
    {
        sigmu.ActivateStatError();
    }

    // Normalize by theory (not needed for signal)
    sigmu.SetNormalizeByTheory(kFALSE);
    sigmu.AddNormFactor("Nmu", config.expMu, 1e-6, 1e6);
    sigmu.AddNormFactor("mcNorm_sigmu", mcNorm_sigmu, 1e-9, 1.);
    chan.AddSample(sigmu);
}

/************************* B0->D*taunu (SIGNAL) *******************************/
// Norm = Nmu * RawRDst * mcNorm_sigtau
void addSignalTauSample(Channel &chan, const AnalysisConfig &config, double mcNorm_sigtau)
{
    Sample sigtau("h_sigtau", "h_sigtau", "input/DemoHistos.root");
    if (config.useTauShapeUncerts)
    {
        sigtau.AddHistoSys("v1mu", "h_sigtau_v1m", "input/DemoHistos.root", "", "h_sigtau_v1p", "input/DemoHistos.root", "");
        sigtau.AddHistoSys("v2mu", "h_sigtau_v2m", "input/DemoHistos.root", "", "h_sigtau_v2p", "input/DemoHistos.root", "");
        sigtau.AddHistoSys("v3mu", "h_sigtau_v3m", "input/DemoHistos.root", "", "h_sigtau_v3p", "input/DemoHistos.root", "");
        sigtau.AddHistoSys("v4tau", "h_sigtau_v4m", "input/DemoHistos.root", "", "h_sigtau_v4p", "input/DemoHistos.root", "");
    }
    if (config.BBon3d)
        sigtau.ActivateStatError();
    sigtau.SetNormalizeByTheory(kFALSE);
    sigtau.AddNormFactor("Nmu", config.expMu, 1e-6, 1e6);
    sigtau.AddNormFactor("RawRDst", config.expTau, 1e-6, 0.2);
    sigtau.AddNormFactor("mcNorm_sigtau", mcNorm_sigtau, 1e-9, 1.);
    chan.AddSample(sigtau);
}

void addD1Background(Channel &chan, const AnalysisConfig &config, double mcNorm_D1)
{
    Sample d1mu("h_D1", "h_D1", "input/DemoHistos.root");
    if (config.BBon3d)
        d1mu.ActivateStatError();
    if (config.useDststShapeUncerts)
    {
        d1mu.AddHistoSys("IW", "h_D1IWp", "input/DemoHistos.root", "", "h_D1IWm", "input/DemoHistos.root", "");
    }
    d1mu.SetNormalizeByTheory(kFALSE);
    d1mu.AddNormFactor("mcNorm_D1", mcNorm_D1, 1e-9, 1.);
    if (!config.constrainDstst)
    {
        d1mu.AddNormFactor("ND1", 1e2, 1e-6, 1e5);
    }
    else
    {
        d1mu.AddNormFactor("NDstst0", 0.102, 1e-6, 1e0);
        d1mu.AddNormFactor("Nmu", config.expMu, 1e-6, 1e6);
        d1mu.AddNormFactor("fD1", 3.2, 3.2, 3.2);
        d1mu.AddOverallSys("BFD1", 0.9, 1.1);
    }
    chan.AddSample(d1mu);
}

void addMisIDBackground(Channel &chan, const AnalysisConfig &config)
{
    Sample misID("h_misID", "h_misID", "input/DemoHistos.root");
    if (config.BBon3d)
        misID.ActivateStatError();
    misID.SetNormalizeByTheory(kTRUE);
    misID.AddNormFactor("NmisID", config.relLumi, 1e-6, 1e5);
    chan.AddSample(misID);
}

// ---------------------------------------------------------
// Build the HistFactory model and return the RooWorkspace
// ---------------------------------------------------------
RooWorkspace *buildModel(
    const AnalysisConfig &config,
    double mcNorm_sigmu, double mcNorm_sigtau, double mcNorm_D1)
{

    // all output for this measurement

    Measurement meas("my_measurement", "my measurement");
    meas.SetOutputFilePrefix("results/my_measurement");
    meas.SetExportOnly(kTRUE); // Tells histfactory to not run the fit and display
                               // results using its own
    meas.SetPOI("RawRDst");

    // set the lumi for the measurement.
    // only matters for the data-driven (expected yield = lumi * cross section * BR * efficiency)
    // pdfs the way I've set it up. in invfb
    // variable rellumi gives the relative luminosity between the
    // data used to generate the pdfs and the sample
    // we are fitting

    // actually, now this is only used for the misID
    meas.SetLumi(1.0);
    meas.SetLumiRelErr(0.05);
    meas.AddConstantParam("Lumi"); // Add the luminosity parameter to the measurement and fix it

    // Create channel and set data
    Channel chan("Dstmu_kinematic");
    chan.SetStatErrorConfig(1e-5, "Poisson");
    chan.SetData("h_data", "input/DemoHistos.root");

    // Add samples
    addSignalMuSample(chan, config, mcNorm_sigmu);
    addSignalTauSample(chan, config, mcNorm_sigtau);
    addD1Background(chan, config, mcNorm_D1);
    addMisIDBackground(chan, config);

    /****** END SAMPLE CHANNELS *******/

    // Add the channel to the measurement
    meas.AddChannel(chan);

    // Collect the histograms
    meas.CollectHistograms();

    // Make the model and measurement
    RooWorkspace *w = MakeModelAndMeasurementFast(meas);
    return w;
}

// ---------------------------------------------------------
// Configure miscellaneous model parameters
// ---------------------------------------------------------
void configureParameters(RooWorkspace *w, const AnalysisConfig &config)
{
    ModelConfig *mc = (ModelConfig *)w->obj("ModelConfig");
    // Fix the normalization factors for Monte Carlo samples
    TString mchistos[3] = {"sigmu", "sigtau", "D1"};
    for (int i = 0; i < 3; i++)
    {
        RooRealVar *par = (RooRealVar *)(mc->GetNuisanceParameters()->find("mcNorm_" + mchistos[i]));
        if (par != NULL)
        {
            par->setConstant(kTRUE);
            cout << "Fixed mcNorm_" << mchistos[i] << " = " << par->getVal() << endl;
        }
    }
    // Fix specific nuisance parameters
    RooRealVar *ndstst0 = (RooRealVar *)(mc->GetNuisanceParameters()->find("NDstst0"));
    if (ndstst0)
    {
        ndstst0->setVal(0.102);
        ndstst0->setConstant(kTRUE);
    }
    RooRealVar *fD1 = (RooRealVar *)(mc->GetNuisanceParameters()->find("fD1"));
    if (fD1)
        fD1->setConstant(kTRUE);
    RooRealVar *nmisID = (RooRealVar *)(mc->GetNuisanceParameters()->find("NmisID"));
    if (nmisID)
        nmisID->setConstant(kTRUE);

    // Set ranges for shape systematic parameters
    if (config.useDststShapeUncerts)
    {
        RooRealVar *alphaIW = (RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_IW"));
        if (alphaIW)
            alphaIW->setRange(-3.0, 3.0);
    }
    if (config.useMuShapeUncerts)
    {
        RooRealVar *alpha_v1mu = (RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v1mu"));
        if (alpha_v1mu)
            alpha_v1mu->setRange(-8, 8);
        RooRealVar *alpha_v2mu = (RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v2mu"));
        if (alpha_v2mu)
            alpha_v2mu->setRange(-12, 12);
        RooRealVar *alpha_v3mu = (RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v3mu"));
        if (alpha_v3mu)
            alpha_v3mu->setRange(-8, 8);
    }
    RooRealVar *alpha_BFD1 = (RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_BFD1"));
    if (alpha_BFD1)
        alpha_BFD1->setRange(-3, 3);

    // Optionally fix shape parameters
    if (config.fixshapes)
    {
        RooRealVar *a1 = (RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v1mu"));
        if (a1)
        {
            a1->setVal(1.06);
            a1->setConstant(kTRUE);
        }
        RooRealVar *a2 = (RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v2mu"));
        if (a2)
        {
            a2->setVal(-0.159);
            a2->setConstant(kTRUE);
        }
        RooRealVar *a3 = (RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v3mu"));
        if (a3)
        {
            a3->setVal(-1.75);
            a3->setConstant(kTRUE);
        }
        RooRealVar *a4 = (RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v4tau"));
        if (a4)
        {
            a4->setVal(0.0002);
            a4->setConstant(kTRUE);
        }
    }
    if (config.fixshapesdstst)
    {
        RooRealVar *alpha_IW = (RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_IW"));
        if (alpha_IW)
        {
            alpha_IW->setVal(-0.005);
            alpha_IW->setConstant(kTRUE);
        }
    }
}

// ---------------------------------------------------------
// Perform the fit using RooMinimizer and return the result
// ---------------------------------------------------------
RooFitResult *performFit(RooWorkspace *w, const AnalysisConfig &config)
{
    ModelConfig *mc = (ModelConfig *)w->obj("ModelConfig");
    RooAbsData *data = w->data("obsData");
    // Use a RooSimultaneous PDF
    RooSimultaneous *model = (RooSimultaneous *)mc->GetPdf();

    // Create a new PDF for the HF model (can be replaced by HistFactorySimultaneous if needed)
    RooSimultaneous *model_hf = new RooSimultaneous(*model);
    RooAbsReal *nll = model_hf->createNLL(*data, Offset(kTRUE), NumCPU(8));

    RooMinimizer minimizer(*nll);
    minimizer.setErrorLevel(0.5);
#ifndef UNBLIND
    minimizer.setPrintLevel(-1);
#endif
    minimizer.setStrategy(2);

    int Tries = 5;
    int status = minimizer.migrad();
    while (status != 0 && Tries > 0)
    {
        status = minimizer.migrad();
        Tries--;
    }
    minimizer.hesse();
    if (config.useMinos)
        minimizer.minos(*mc->GetParametersOfInterest());

    RooFitResult *result = minimizer.save("Result", "Result");

    delete nll;
    delete model_hf;

    return result;
}

// ---------------------------------------------------------
// Generate plots for the fit results
// ---------------------------------------------------------
void generatePlots(RooWorkspace *w, RooFitResult *result, const AnalysisConfig &config)
{
    ModelConfig *mc = (ModelConfig *)w->obj("ModelConfig");
    RooAbsData *data = w->data("obsData");
    RooSimultaneous *model = (RooSimultaneous *)mc->GetPdf();

    // Retrieve observables and set titles/units.
    RooArgSet *obs = (RooArgSet *)mc->GetObservables();
    RooRealVar *x = (RooRealVar *)obs->find("obs_x_Dstmu_kinematic");
    RooRealVar *y = (RooRealVar *)obs->find("obs_y_Dstmu_kinematic");
    RooRealVar *z = (RooRealVar *)obs->find("obs_z_Dstmu_kinematic");
    x->SetTitle("m^{2}_{miss}");
    x->setUnit("GeV^{2}");
    y->SetTitle("E_{#mu}");
    y->setUnit("MeV");
    z->SetTitle("q^{2}");
    z->setUnit("MeV^{2}");

    // The category for simultaneous fits
    RooCategory *idx = (RooCategory *)obs->find("channelCat");

    // Create one-dimensional frames for each observable.
    RooPlot *mm2_frame = x->frame(Title("m^{2}_{miss}"));
    RooPlot *El_frame = y->frame(Title("E_{#mu}"));
    RooPlot *q2_frame = z->frame(Title("q^{2}"));

    // Create the colors for the components
    const int ncomps = 10;
    int colors[ncomps] = {kRed, kBlue + 1, kViolet, kViolet + 1, kViolet + 2, kGreen, kGreen + 1, kOrange + 1, kOrange + 2, kOrange + 3};
    const int ncomps2 = 8;
    // Create the names for the components
    TString names[ncomps2 + 1] = {"Data",
                                  "Total Fit",
                                  "B #rightarrow D*#mu#nu",
                                  "B #rightarrow D**#mu#nu",
                                  "B #rightarrow D**#tau#nu",
                                  "B #rightarrow D*[D_{q} #rightarrow #mu#nuX]Y",
                                  "Combinatoric (wrong-sign)",
                                  "Misidentification BKG",
                                  "Wrong-sign slow #pi"};

    // Create the residuals
    const int nframes = 3;
    RooPlot *drawframes[nframes] = {mm2_frame, El_frame, q2_frame};

    RooHist *resids[nframes]; // to store the residuals

    // Plot the data and the model
    for (int i = 0; i < nframes; i++)
    {
        data->plotOn(drawframes[i], DataError(RooAbsData::Poisson), Cut("channelCat==0"), MarkerSize(0.4), DrawOption("ZP"));
        model->plotOn(drawframes[i], Slice(*idx), ProjWData(*idx, *data), DrawOption("F"), FillColor(kRed));

        // Grab pulls
        resids[i] = drawframes[i]->pullHist();

        // Plot the components
        model->plotOn(drawframes[i], Slice(*idx), ProjWData(*idx, *data), DrawOption("F"), FillColor(kViolet), Components("*misID*,*sigmu*,*D1*"));
        model->plotOn(drawframes[i], Slice(*idx), ProjWData(*idx, *data), DrawOption("F"), FillColor(kBlue + 1), Components("*misID*,*sigmu*"));
        model->plotOn(drawframes[i], Slice(*idx), ProjWData(*idx, *data), DrawOption("F"), FillColor(kOrange), Components("*misID*"));
        data->plotOn(drawframes[i], DataError(RooAbsData::Poisson), Cut("channelCat==0"), MarkerSize(0.4), DrawOption("ZP"));
    }

    TLatex *t = new TLatex();
    t->SetTextAlign(22);
    t->SetTextSize(0.06);
    t->SetTextFont(132);

    // Create the canvas for the plots
    TCanvas *c1 = new TCanvas("c1", "c1", 1000, 300);
    c1->SetTickx();
    c1->SetTicky();
    c1->Divide(3, 1);
    TVirtualPad *curpad;
    curpad = c1->cd(1);
    curpad->SetTickx();
    curpad->SetTicky();
    curpad->SetRightMargin(0.02);
    curpad->SetLeftMargin(0.20);
    curpad->SetTopMargin(0.02);
    curpad->SetBottomMargin(0.13);
    mm2_frame->SetTitle("");
    mm2_frame->GetXaxis()->SetLabelSize(0.06);
    mm2_frame->GetXaxis()->SetTitleSize(0.06);
    mm2_frame->GetYaxis()->SetLabelSize(0.06);
    mm2_frame->GetYaxis()->SetTitleSize(0.06);
    mm2_frame->GetYaxis()->SetTitleOffset(1.75);
    mm2_frame->GetXaxis()->SetTitleOffset(0.9);
    TString thetitle = mm2_frame->GetYaxis()->GetTitle();
    thetitle.Replace(0, 6, "Candidates");
    mm2_frame->GetYaxis()->SetTitle(thetitle);
    mm2_frame->Draw();
    t->DrawLatex(8.7, mm2_frame->GetMaximum() * 0.95, "Demo");
    curpad = c1->cd(2);
    curpad->SetTickx();
    curpad->SetTicky();
    curpad->SetRightMargin(0.02);
    curpad->SetLeftMargin(0.20);
    curpad->SetTopMargin(0.02);
    curpad->SetBottomMargin(0.13);
    El_frame->SetTitle("");
    El_frame->GetXaxis()->SetLabelSize(0.06);
    El_frame->GetXaxis()->SetTitleSize(0.06);
    El_frame->GetYaxis()->SetLabelSize(0.06);
    El_frame->GetYaxis()->SetTitleSize(0.06);
    El_frame->GetYaxis()->SetTitleOffset(1.75);
    El_frame->GetXaxis()->SetTitleOffset(0.9);
    thetitle = El_frame->GetYaxis()->GetTitle();
    thetitle.Replace(0, 6, "Candidates");
    El_frame->GetYaxis()->SetTitle(thetitle);
    El_frame->Draw();
    t->DrawLatex(2250, El_frame->GetMaximum() * 0.95, "Demo");
    curpad = c1->cd(3);
    curpad->SetTickx();
    curpad->SetTicky();
    curpad->SetRightMargin(0.02);
    curpad->SetLeftMargin(0.20);
    curpad->SetTopMargin(0.02);
    curpad->SetBottomMargin(0.13);
    q2_frame->SetTitle("");
    q2_frame->GetXaxis()->SetLabelSize(0.06);
    q2_frame->GetXaxis()->SetTitleSize(0.06);
    q2_frame->GetYaxis()->SetLabelSize(0.06);
    q2_frame->GetYaxis()->SetTitleSize(0.06);
    q2_frame->GetYaxis()->SetTitleOffset(1.75);
    q2_frame->GetXaxis()->SetTitleOffset(0.9);
    thetitle = q2_frame->GetYaxis()->GetTitle();
    thetitle.Replace(0, 6, "Candidates");
    q2_frame->GetYaxis()->SetTitle(thetitle);
    q2_frame->Draw();
    t->DrawLatex(11.1e6, q2_frame->GetMaximum() * 0.95, "Demo");

    // Create the directory for the plots
    system("mkdir -p plots");

    // Save the canvas
    c1->SaveAs("plots/HistFactDstTauDemo_nominal.png");

    // For pull plots
    RooPlot *mm2_resid_frame = x->frame(Title("mm2"));
    RooPlot *El_resid_frame = y->frame(Title("El"));
    RooPlot *q2_resid_frame = z->frame(Title("q2"));

    //   cerr << __LINE__ << endl;
    mm2_resid_frame->addPlotable(resids[0], "P");
    //   cerr << __LINE__ << endl;
    El_resid_frame->addPlotable(resids[1], "P");
    //   cerr << __LINE__ << endl;
    q2_resid_frame->addPlotable(resids[2], "P");
    //   cerr << __LINE__ << endl;

    TCanvas *c3 = new TCanvas("c3", "c3", 640, 1000);
    c3->Divide(1, 3);
    c3->cd(1);
    mm2_resid_frame->Draw();
    c3->cd(2);
    El_resid_frame->Draw();
    c3->cd(3);
    q2_resid_frame->Draw();

    // Save the canvas
    c3->SaveAs("plots/HistFactDstTauDemo_pulls.png");

    // Optionally, produce additional "slow" plots binned in q^2.
    if (config.slowplots)
    {

        double q2_low, q2_high;
        int q2_bins;
        getQ2Binning(q2_low, q2_high, q2_bins);
        cout << "Generating slow plots with " << q2_bins << " q^2 bins." << endl;

        // Prepare the strings for the cuts and ranges
        char cutstrings[q2_bins][128];
        char rangenames[q2_bins][32];
        char rangelabels[q2_bins][128];
        RooHist *mm2q2_pulls[q2_bins];
        RooHist *Elq2_pulls[q2_bins];

        vector<RooPlot *> mm2q2_frame(q2_bins), Elq2_frame(q2_bins);
        vector<TString> rangeNames(q2_bins), rangeLabels(q2_bins), cutStrings(q2_bins);

        RooPlot *q2frames[2 * q2_bins];
        RooPlot *q2bframes[2 * q2_bins];

        for (int i = 0; i < q2_bins; i++)
        {
            mm2q2_frame[i] = x->frame();
            Elq2_frame[i] = y->frame();

            // For simple access by loop index
            q2frames[i] = mm2q2_frame[i];
            q2frames[i + q2_bins] = Elq2_frame[i];
            q2bframes[i] = x->frame();
            q2bframes[i + q2_bins] = y->frame();
        }

        // Fill the strings for the cuts and ranges
        for (int i = 0; i < q2_bins; i++)
        {
            double binlow = q2_low + i * (q2_high - q2_low) / q2_bins;
            double binhigh = q2_low + (i + 1) * (q2_high - q2_low) / q2_bins;
            rangeLabels[i] = Form("%.2f < q^{2} < %.2f", binlow * 1e-6, binhigh * 1e-6);
            rangeNames[i] = Form("q2bin_%d", i);
            cutStrings[i] = Form("obs_z_Dstmu_kinematic > %f && obs_z_Dstmu_kinematic < %f && channelCat==0",
                                 binlow, binhigh);
            z->setRange(rangeNames[i], binlow, binhigh);
        }

        cout << "Drawing Slow Plots" << endl;
        for (int i = 0; i < q2_bins; i++)
        {

            data->plotOn(mm2q2_frame[i], Cut(cutStrings[i]), DataError(RooAbsData::Poisson), MarkerSize(0.4), DrawOption("ZP"));
            model->plotOn(mm2q2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangeNames[i]), DrawOption("F"), FillColor(kRed));
            // data->plotOn(mm2q2_frame[i], Cut(cutStrings[i]), DataError(RooAbsData::Poisson), MarkerSize(0.4), DrawOption("ZP"));

            data->plotOn(Elq2_frame[i], Cut(cutStrings[i]), DataError(RooAbsData::Poisson), MarkerSize(0.4), DrawOption("ZP"));
            model->plotOn(Elq2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangeNames[i]), DrawOption("F"), FillColor(kRed));
            // data->plotOn(Elq2_frame[i], Cut(cutStrings[i]), DataError(RooAbsData::Poisson), MarkerSize(0.4), DrawOption("ZP"));

            // Grab pulls
            mm2q2_pulls[i] = mm2q2_frame[i]->pullHist();
            Elq2_pulls[i] = Elq2_frame[i]->pullHist();

            // Plot the components
            model->plotOn(mm2q2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangeNames[i]), DrawOption("F"), FillColor(kViolet), Components("*misID*,*sigmu*,*D1*"));
            model->plotOn(Elq2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangeNames[i]), DrawOption("F"), FillColor(kViolet), Components("*misID*,*sigmu*,*D1*"));
            model->plotOn(mm2q2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangeNames[i]), DrawOption("F"), FillColor(kBlue + 1), Components("*misID*,*sigmu*"));
            model->plotOn(Elq2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangeNames[i]), DrawOption("F"), FillColor(kBlue + 1), Components("*misID*,*sigmu*"));
            model->plotOn(mm2q2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangeNames[i]), DrawOption("F"), FillColor(kOrange), Components("*misID*"));
            model->plotOn(Elq2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangeNames[i]), DrawOption("F"), FillColor(kOrange), Components("*misID*"));
            data->plotOn(mm2q2_frame[i], Cut(cutStrings[i]), DataError(RooAbsData::Poisson), MarkerSize(0.4), DrawOption("ZP"));
            data->plotOn(Elq2_frame[i], Cut(cutStrings[i]), DataError(RooAbsData::Poisson), MarkerSize(0.4), DrawOption("ZP"));
        }
        //////////////

        TCanvas *c2 = new TCanvas("c2", "c2", 1200, 600);
        c2->Divide(q2_bins, 2);
        double max_scale = 1.05;
        double max_scale2 = 1.05;
        char thename[32];
        for (int k = 0; k < q2_bins * 2; k++)

        {
            c2->cd(k + 1);
            /*
            q2frames[k]->SetTitle(rangelabels[(k % q2_bins)]);
            q2frames[k]->Draw();*/
            sprintf(thename, "bottompad_%d", k);
            // c2->cd((k<q2_bins)*(2*k+1)+(k>=q2_bins)*(2*(k+1-q2_bins)));
            TPad *padbottom = new TPad(thename, thename, 0., 0., 1., 0.3);

            padbottom->SetFillColor(0);
            padbottom->SetGridy();
            padbottom->SetTickx();
            padbottom->SetTicky();
            padbottom->SetFillStyle(0);
            padbottom->Draw();
            padbottom->cd();
            padbottom->SetLeftMargin(padbottom->GetLeftMargin() + 0.08);
            padbottom->SetTopMargin(0); // 0.01);
            padbottom->SetRightMargin(0.04);
            // padbottom->SetBottomMargin(padbottom->GetBottomMargin()+0.23);
            padbottom->SetBottomMargin(0.5);

            // c2b->cd(k+1);
            TH1 *temphist2lo, *temphist2, *tempdathist;
            RooHist *temphist;
            if (k < q2_bins)
            {
                temphist = mm2q2_pulls[k];
            }
            else
            {
                temphist = Elq2_pulls[k - q2_bins];
            }
            temphist->SetFillColor(kBlue);
            temphist->SetLineColor(kWhite);
            q2bframes[k]->SetTitle(q2frames[k]->GetTitle());
            q2bframes[k]->addPlotable(temphist, "B");
            q2bframes[k]->GetXaxis()->SetLabelSize(0.33 * 0.22 / 0.3);
            q2bframes[k]->GetXaxis()->SetTitleSize(0.36 * 0.22 / 0.3);
            // q2bframes[k]->GetXaxis()->SetTitle("");
            q2bframes[k]->GetXaxis()->SetTickLength(0.10);
            q2bframes[k]->GetYaxis()->SetTickLength(0.05);
            q2bframes[k]->SetTitle("");
            q2bframes[k]->GetYaxis()->SetTitleSize(0.33 * 0.22 / 0.3);
            q2bframes[k]->GetYaxis()->SetTitle("Pulls");
            q2bframes[k]->GetYaxis()->SetTitleOffset(0.2);
            q2bframes[k]->GetXaxis()->SetTitleOffset(0.78);
            q2bframes[k]->GetYaxis()->SetLabelSize(0.33 * 0.22 / 0.3);
            q2bframes[k]->GetYaxis()->SetLabelOffset(99);
            q2bframes[k]->GetYaxis()->SetNdivisions(205);
            q2bframes[k]->Draw();
            q2bframes[k]->Draw();
            double xloc = -2.25;
            if (k >= q2_bins)
                xloc = 50;
            t->SetTextSize(0.33 * 0.22 / 0.3);
            t->DrawLatex(xloc, -2, "-2");
            // t->DrawLatex(xloc,0," 0");
            t->DrawLatex(xloc * 0.99, 2, " 2");

            c2->cd(k + 1);
            // c2->cd((k<q2_bins)*(2*k+1)+(k>=q2_bins)*(2*(k+1-q2_bins)));
            sprintf(thename, "toppad_%d", k);
            TPad *padtop = new TPad(thename, thename, 0., 0.3, 1., 1.);
            padtop->SetLeftMargin(padtop->GetLeftMargin() + 0.08);
            padtop->SetBottomMargin(0); // padtop->GetBottomMargin()+0.08);
            padtop->SetTopMargin(0.02); // padtop->GetBottomMargin()+0.08);
            padtop->SetRightMargin(0.04);
            padtop->SetFillColor(0);
            padtop->SetFillStyle(0);
            padtop->SetTickx();
            padtop->SetTicky();
            padtop->Draw();
            padtop->cd();
            q2frames[k]->SetMinimum(1e-4);
            if (k < q2_bins)
                q2frames[k]->SetMaximum(q2frames[k]->GetMaximum() * max_scale);
            if (k >= q2_bins)
                q2frames[k]->SetMaximum(q2frames[k]->GetMaximum() * max_scale2);
            // q2frames[k]->SetMaximum(1.05*q2frames[k]->GetMaximum());
            q2frames[k]->SetTitle(rangelabels[(k % q2_bins)]);
            q2frames[k]->SetTitleFont(132, "t");
            q2frames[k]->GetXaxis()->SetLabelSize(0.09 * 0.78 / 0.7);
            q2frames[k]->GetXaxis()->SetTitleSize(0.09 * 0.78 / 0.7);
            q2frames[k]->GetYaxis()->SetTitleSize(0.09 * 0.78 / 0.7);
            TString thetitle = q2frames[k]->GetYaxis()->GetTitle();
            /*thetitle.Replace(10,1,"");
            if(k < q2_bins)thetitle.Replace(27,1,"");
            if(k >= q2_bins)thetitle.Replace(16,1,"");
            thitle.Replace(0,6,"Candidates");*/
            q2frames[k]->GetYaxis()->SetTitle("");
            q2frames[k]->GetYaxis()->SetLabelSize(0.09 * 0.78 / 0.7);
            q2frames[k]->GetXaxis()->SetTitleOffset(0.95);
            q2frames[k]->GetYaxis()->SetTitleOffset(0.95);
            q2frames[k]->GetYaxis()->SetNdivisions(506);
            q2frames[k]->Draw();
            t->SetTextSize(0.07);
            t->SetTextAlign(33);
            t->SetTextAngle(90);
            c2->cd((k < q2_bins) * (2 * k + 1) + (k >= q2_bins) * (2 * (k + 1 - q2_bins)));
            if (k >= q2_bins)
            {
                t->DrawLatex(0.01, 0.99, thetitle);
            }
            if (k < q2_bins)
            {
                t->DrawLatex(0.01, 0.99, thetitle);
            }
            t->SetTextAlign(22);
            t->SetTextAngle(0);
            padtop->cd();
            t->SetTextSize(0.09 * 0.78 / 0.7);
            if (k >= q2_bins)
            {
                t->DrawLatex(2250, q2frames[k]->GetMaximum() * 0.92, "Demo");
            }
            if (k < q2_bins)
            {
                t->DrawLatex(8.7, q2frames[k]->GetMaximum() * 0.92, "Demo");
            }
        }

        // Save the canvas
        c2->SaveAs("plots/HistFactDstTauDemo_in_q2_bins.png");
    }
}

// ---------------------------------------------------------
// Clean up allocated global resources.
// ---------------------------------------------------------
void cleanupResources()
{
    if (globalDate)
    {
        delete globalDate;
        globalDate = nullptr;
    }
}

// ---------------------------------------------------------
// Main analysis function that puts everything together
// ---------------------------------------------------------
void main_HistFactDstTauDemo()
{
    // Create analysis configuration (modify these to change steering options)
    AnalysisConfig config;

    // Initialization and style settings
    initializeGlobalSettings();

    // Load normalization factors from the input histogram file
    double mcNorm_sigmu, mcNorm_sigtau, mcNorm_D1;
    loadNormalizationFactors(mcNorm_sigmu, mcNorm_sigtau, mcNorm_D1);

    // Build the HistFactory model and obtain the RooWorkspace
    RooWorkspace *w = buildModel(config, mcNorm_sigmu, mcNorm_sigtau, mcNorm_D1);

    // Configure nuisance parameters and fix constants
    configureParameters(w, config);

    // Perform fit if requested
    RooFitResult *result = 0;
    if (config.doFit)
    {

        result = performFit(w, config);
        if (result)
        {
            cout << "----------------------------------------" << endl;
            cout << "Fit Status: " << result->status() << endl;
            cout << "Fit EDM: " << result->edm() << endl;

            // print the initial fitted results
            cout << "\nInitial fitted results:" << endl;
            result->floatParsInit().Print("v");

            // print the final fitted results
            cout << "\nFinal fitted results:" << endl;
            result->floatParsFinal().Print("v");

            // Get the parameter of interest
            ModelConfig *mc = (ModelConfig *)w->obj("ModelConfig"); // Get model manually
            RooRealVar *poi = dynamic_cast<RooRealVar *>(mc->GetParametersOfInterest()->first());

            int final_par_counter = 0;
            for (const auto &arg : result->floatParsFinal())
            {
                RooRealVar *param = dynamic_cast<RooRealVar *>(arg);
                if (param && !param->isConstant())
                {
                    if (TString(param->GetName()) != poi->GetName())
                    {
                        cout << final_par_counter << ": "
                             << param->GetName() << "\t\t\t = "
                             << param->getVal()
                             << " +/- "
                             << param->getError() << endl;
                    }
                }
                final_par_counter++;
            }
            result->correlationMatrix().Print();
        }
        else
        {
            cout << "Fit failed." << endl;
        }
    }

    // Generate plots of data and fit projections
    generatePlots(w, result, config);

    // Cleanup allocated objects
    cleanupResources();
}

// ---------------------------------------------------------
// User-callable function
// ---------------------------------------------------------
void HistFactDstTauDemo_new()
{
    main_HistFactDstTauDemo();
}
