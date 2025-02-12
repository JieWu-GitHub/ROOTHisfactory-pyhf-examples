/*
 * @Author       : Jie Wu j.wu@cern.ch
 * @Date         : 2023-11-24 16:13:46 +0100
 * @LastEditors  : Jie Wu j.wu@cern.ch
 * @LastEditTime : 2025-02-09 07:27:29 +0100
 * @FilePath     : HistFactDstTauDemo.C
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

TDatime *date = new TDatime();

void main_HistFactDstTauDemo(
    // Many many flags for steering
    /* STEERING OPTIONS */
    const bool constrainDstst,
    const bool useMinos,
    const bool useMuShapeUncerts,
    const bool useTauShapeUncerts,
    const bool useDststShapeUncerts,
    const bool fixshapes,
    const bool fixshapesdstst,
    const bool dofit,

    const bool fitfirst,
    const bool slowplots,
    const bool BBon3d
    // Should allow easy comparison of fit errors with and
    // without the technique. 3d or not is legacy from an old
    //(3+1)d fit configuration
)

{
    // Initialize the analysis configuration
    TLatex *t = new TLatex();
    t->SetTextAlign(22);
    t->SetTextSize(0.06);
    t->SetTextFont(132);
    gROOT->ProcessLine("gStyle->SetLabelFont(132,\"xyz\");");
    gROOT->ProcessLine("gStyle->SetTitleFont(132,\"xyz\");");
    gROOT->ProcessLine("gStyle->SetTitleFont(132,\"t\");");
    gROOT->ProcessLine("gStyle->SetTitleSize(0.08,\"t\");");
    gROOT->ProcessLine("gStyle->SetTitleY(0.970);");
    char substr[128];

    // RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING); // avoid accidental unblinding!

    // Below: Read histogram file to generate normalization constants required to make
    // each histo normalized to unity. Not totally necessary here, but convenient

    // Open the histogram file
    TFile q("DemoHistos.root");
    TH1 *htemp;
    TString mchistos[3] = {"sigmu", "sigtau", "D1"};
    double mcN_sigmu, mcN_sigtau, mcN_D1;
    double *mcnorms[3] = {&mcN_sigmu, &mcN_sigtau, &mcN_D1};
    for (int i = 0; i < 3; i++)
    {
        // Get the histogram object
        q.GetObject("h_" + mchistos[i], htemp);
        assert(htemp != NULL);

        // Calculate the normalization factor
        *(mcnorms[i]) = 1. / htemp->Integral();
        cout << "mcN_" + mchistos[i] + " = " << 1. / *(mcnorms[i]) << endl;
    }

    // Useful later to have the bin max and min for drawing
    // Get the histogram object
    TH3 *JUNK;
    q.GetObject("h_sigmu", JUNK);
    double q2_low = JUNK->GetZaxis()->GetXmin();
    double q2_high = JUNK->GetZaxis()->GetXmax();
    const int q2_bins = JUNK->GetZaxis()->GetNbins();
    JUNK->SetDirectory(0);

    // Close the histogram file
    q.Close();

    // Start the timer
    TStopwatch sw, sw2;

    // Initialize the random number generator
    TRandom *r3 = new TRandom3(date->Get());

    // Set the prefix that will appear before

    // all output for this measurement
    RooStats::HistFactory::Measurement meas("my_measurement", "my measurement");
    meas.SetOutputFilePrefix("results/my_measurement");
    meas.SetExportOnly(kTRUE); // Tells histfactory to not run the fit and display
                               // results using its own

    meas.SetPOI("RawRDst"); // Set the parameter of interest

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

    /******* Fit starting constants ***********/

    // ISOLATED FULL RANGE NONN
    //*
    const double expTau = 0.252 * 0.1742 * 0.781 / 0.85;
    const double e_iso = 0.314;
    double expMu = 50e3;
    // 8*/

    double RelLumi = 1.00;

    // Create the channel
    RooStats::HistFactory::Channel chan("Dstmu_kinematic");
    chan.SetStatErrorConfig(1e-5, "Poisson");

    // tell histfactory what data to use
    chan.SetData("h_data", "DemoHistos.root");

    // Now that data is set up, start creating our samples
    // describing the processes to model the data

    /*********************** B0->D*munu (NORM) *******************************/
    // Norm = Nmu * mcNorm_sigmu

    RooStats::HistFactory::Sample sigmu("h_sigmu", "h_sigmu", "DemoHistos.root");
    if (useMuShapeUncerts) // Add the shape uncertainties (shape variations)
    {
        sigmu.AddHistoSys("v1mu", "h_sigmu_v1m", "DemoHistos.root", "", "h_sigmu_v1p", "DemoHistos.root", "");
        sigmu.AddHistoSys("v2mu", "h_sigmu_v2m", "DemoHistos.root", "", "h_sigmu_v2p", "DemoHistos.root", "");
        sigmu.AddHistoSys("v3mu", "h_sigmu_v3m", "DemoHistos.root", "", "h_sigmu_v3p", "DemoHistos.root", "");
    }
    if (BBon3d) // Add the statistical errors
    {
        sigmu.ActivateStatError();
    }
    // Normalize by theory (not needed for signal)
    sigmu.SetNormalizeByTheory(kFALSE);
    sigmu.AddNormFactor("Nmu", expMu, 1e-6, 1e6);
    sigmu.AddNormFactor("mcNorm_sigmu", mcN_sigmu, 1e-9, 1.);
    chan.AddSample(sigmu);

    /************************* B0->D*taunu (SIGNAL) *******************************/
    // Norm = Nmu * RawRDst * mcNorm_sigtau

    RooStats::HistFactory::Sample sigtau("h_sigtau", "h_sigtau", "DemoHistos.root");
    if (useTauShapeUncerts) // Add the shape uncertainties (shape variations)
    {
        sigtau.AddHistoSys("v1mu", "h_sigtau_v1m", "DemoHistos.root", "", "h_sigtau_v1p", "DemoHistos.root", "");
        sigtau.AddHistoSys("v2mu", "h_sigtau_v2m", "DemoHistos.root", "", "h_sigtau_v2p", "DemoHistos.root", "");
        sigtau.AddHistoSys("v3mu", "h_sigtau_v3m", "DemoHistos.root", "", "h_sigtau_v3p", "DemoHistos.root", "");
        sigtau.AddHistoSys("v4tau", "h_sigtau_v4m", "DemoHistos.root", "", "h_sigtau_v4p", "DemoHistos.root", "");
    }
    if (BBon3d) // Add the statistical errors
    {
        sigtau.ActivateStatError();
    }
    // Normalize by theory (not needed for signal)
    sigtau.SetNormalizeByTheory(kFALSE);
    sigtau.AddNormFactor("Nmu", expMu, 1e-6, 1e6);
    // Add the relative normalization factor of tau
    sigtau.AddNormFactor("RawRDst", expTau, 1e-6, 0.2);
    // Add the normalization factor of tau MC
    sigtau.AddNormFactor("mcNorm_sigtau", mcN_sigtau, 1e-9, 1.);
    chan.AddSample(sigtau);

    /************************* B0->D1munu **************************************/
    // Norm = mcNorm_D1 * ND1 (in the case of no constraint)
    // Norm = mcNorm_D1 * Nmu * NDstst0 * fD1 * BFD1 (in the case of constraint)
    RooStats::HistFactory::Sample d1mu("h_D1", "h_D1", "DemoHistos.root");
    if (BBon3d) // Add the statistical errors
    {
        d1mu.ActivateStatError();
    }
    // Add the shape uncertainties (shape variations)
    if (useDststShapeUncerts)
    {
        d1mu.AddHistoSys("IW", "h_D1IWp", "DemoHistos.root", "", "h_D1IWm", "DemoHistos.root", "");
    }
    // Normalize by theory (not needed for signal)
    d1mu.SetNormalizeByTheory(kFALSE);
    // Add the normalization factor of D1 MC
    d1mu.AddNormFactor("mcNorm_D1", mcN_D1, 1e-9, 1.);

    // Add the relative normalization factor of D1
    if (!constrainDstst)
    {
        d1mu.AddNormFactor("ND1", 1e2, 1e-6, 1e5);
    }
    else
    {
        d1mu.AddNormFactor("NDstst0", 0.102, 1e-6, 1e0);
        d1mu.AddNormFactor("Nmu", expMu, 1e-6, 1e6);
        d1mu.AddNormFactor("fD1", 3.2, 3.2, 3.2);
        d1mu.AddOverallSys("BFD1", 0.9, 1.1);
    }
    chan.AddSample(d1mu);
    /*********************** MisID BKG (FROM DATA)  *******************************/
    // Norm = NmisID

    RooStats::HistFactory::Sample misID("h_misID", "h_misID", "DemoHistos.root");
    if (BBon3d)
        misID.ActivateStatError();

    misID.SetNormalizeByTheory(kTRUE);
    misID.AddNormFactor("NmisID", RelLumi, 1e-6, 1e5);
    chan.AddSample(misID);

    /****** END SAMPLE CHANNELS *******/

    // Add the channel to the measurement
    meas.AddChannel(chan);
    // Collect the histograms
    meas.CollectHistograms();

    // Set the nuisance parameters to be fixed
    // meas.AddConstantParam("mcNorm_sigmu");
    // meas.AddConstantParam("mcNorm_sigtau");
    // meas.AddConstantParam("mcNorm_D1");
    // meas.AddConstantParam("fD1");
    // meas.AddConstantParam("NDstst0");
    // meas.AddConstantParam("NmisID");

    // Make the model and measurement
    RooWorkspace *w;
    w = RooStats::HistFactory::MakeModelAndMeasurementFast(meas);

    // ------ DEBUG ------
    cout << "Check point 2: The model and measurement are built" << endl;
    //    exit(0);

    // Get the model config and the model
    ModelConfig *mc = (ModelConfig *)w->obj("ModelConfig"); // Get model manually
    RooSimultaneous *model = (RooSimultaneous *)mc->GetPdf();

    // Get the parameter of interest
    RooRealVar *poi = dynamic_cast<RooRealVar *>(mc->GetParametersOfInterest()->first());

    std::cout << "Param of Interest: " << poi->GetName() << std::endl;

    // Lets tell roofit the right names for our histogram variables //
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

    // For simultaneous fits, this is the category histfactory uses to sort the channels
    RooCategory *idx = (RooCategory *)obs->find("channelCat");
    RooAbsData *data = (RooAbsData *)w->data("obsData");

    /* FIX SOME MODEL PARAMS */
    for (int i = 0; i < 3; i++)
    {
        if (((RooRealVar *)(mc->GetNuisanceParameters()->find("mcNorm_" + mchistos[i]))) != NULL)
        {
            ((RooRealVar *)(mc->GetNuisanceParameters()->find("mcNorm_" + mchistos[i])))->setConstant(kTRUE);
            cout << "mcNorm_" + mchistos[i] + " = " << ((RooRealVar *)(mc->GetNuisanceParameters()->find("mcNorm_" + mchistos[i])))->getVal() << endl;
        }
    }

    ((RooRealVar *)(mc->GetNuisanceParameters()->find("NDstst0")))->setVal(0.102);
    ((RooRealVar *)(mc->GetNuisanceParameters()->find("NDstst0")))->setConstant(kTRUE);
    ((RooRealVar *)(mc->GetNuisanceParameters()->find("fD1")))->setConstant(kTRUE);
    ((RooRealVar *)(mc->GetNuisanceParameters()->find("NmisID")))->setConstant(kTRUE);

    // Adjust the range of the shape uncertainties
    if (useDststShapeUncerts)
        ((RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_IW")))->setRange(-3.0, 3.0);
    if (useMuShapeUncerts)
        ((RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v1mu")))->setRange(-8, 8);
    if (useMuShapeUncerts)
        ((RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v2mu")))->setRange(-12, 12);
    if (useMuShapeUncerts)
        ((RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v3mu")))->setRange(-8, 8);
    ((RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_BFD1")))->setRange(-3, 3);

    // Fix the shape uncertainties
    if (fixshapes)
    {
        ((RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v1mu")))->setVal(1.06);
        ((RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v1mu")))->setConstant(kTRUE);
        ((RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v2mu")))->setVal(-0.159);
        ((RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v2mu")))->setConstant(kTRUE);
        ((RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v3mu")))->setVal(-1.75);
        ((RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v3mu")))->setConstant(kTRUE);
        ((RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v4tau")))->setVal(0.0002);
        ((RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_v4tau")))->setConstant(kTRUE);
    }
    if (fixshapesdstst)
    {
        ((RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_IW")))->setVal(-0.005); //-2.187);
        ((RooRealVar *)(mc->GetNuisanceParameters()->find("alpha_IW")))->setConstant(kTRUE);
    }

    // This switches the model to a class written to handle analytic Barlow-Beeston lite.
    // Otherwise, every bin gets a minuit variable to minimize over!
    // This class, on the other hand, allows a likelihood where the bin parameters
    // are analyitically minimized at each step
    //    HistFactorySimultaneous *model_hf = new HistFactorySimultaneous(*model);
    RooSimultaneous *model_hf = new RooSimultaneous(*model);

    RooAbsReal *nll_hf;

    RooFitResult *result, *result2;

    cerr << "Saving PDF snapshot" << endl;
    RooArgSet *allpars;
    allpars = (RooArgSet *)((RooArgSet *)mc->GetNuisanceParameters())->Clone();
    allpars->add(*poi);
    RooArgSet *constraints;
    constraints = (RooArgSet *)mc->GetConstraintParameters();
    if (constraints != NULL)
        allpars->add(*constraints);
    w->saveSnapshot("TMCPARS", *allpars, kTRUE);

    // TIterator *iter = allpars->createIterator();
    // RooAbsArg *tempvar;
    const RooArgSet &parSet = *allpars;

    if (dofit)
    { // return;
        // Create the NLL
        nll_hf = model_hf->createNLL(*data, RooFit::Offset(kTRUE), RooFit::NumCPU(8));

        // RooMinuit *minuit_hf = new RooMinuit(*nll_hf);
        RooMinimizer *minuit_hf = new RooMinimizer(*nll_hf);
        RooArgSet *temp = new RooArgSet();
        nll_hf->getParameters(temp)->Print("V");
        cout << "**********************************************************************" << endl;
        minuit_hf->setErrorLevel(0.5);
#ifndef UNBLIND
        minuit_hf->setPrintLevel(-1);
#endif

        std::cout << "Minimizing the Minuit (Migrad)" << std::endl;

        w->saveSnapshot("TMCPARS", *allpars, kTRUE);

        sw.Reset();
        sw.Start();
        minuit_hf->setStrategy(2);
        // minuit_hf->fit("smh");

        // Migrad the NLL
        minuit_hf->migrad();

        int Tries = 5;
        minuit_hf->setStrategy(2);
        int statusVal_migrad = minuit_hf->migrad();

        cout << "MIGRAD STATUS: " << statusVal_migrad << endl;
        // exit(1);

        // Retry the fit if it failed
        while ((statusVal_migrad != 0) && (Tries > 0))
        {
            cout << "MIGRAD FAILED. RETRYING..." << endl;
            statusVal_migrad = minuit_hf->migrad();
            Tries -= 1;
        }

        // Calculate the errors
        minuit_hf->hesse();

        // Save the result
        RooFitResult *tempResult = minuit_hf->save("TempResult", "TempResult");

        cout << tempResult->edm() << endl;
        if (useMinos)
            minuit_hf->minos(RooArgSet(*poi));
        sw.Stop();
        result = minuit_hf->save("Result", "Result");
    }

    // Create the frames for the plots
    RooPlot *mm2_frame = x->frame(Title("m^{2}_{miss}"));
    RooPlot *El_frame = y->frame(Title("E_{#mu}"));
    RooPlot *q2_frame = z->frame(Title("q^{2}"));
    RooPlot *mm2q2_frame[q2_bins];
    RooPlot *Elq2_frame[q2_bins];

    // Prepare the frames for the plots
    const int nframes = 3;
    RooPlot *drawframes[nframes] = {mm2_frame, El_frame, q2_frame};
    RooPlot *q2frames[2 * q2_bins];
    RooPlot *q2bframes[2 * q2_bins];

    for (int i = 0; i < q2_bins; i++)
    {
        mm2q2_frame[i] = x->frame();
        Elq2_frame[i] = y->frame();
        q2frames[i] = mm2q2_frame[i];
        q2frames[i + q2_bins] = Elq2_frame[i];
        q2bframes[i] = x->frame();
        q2bframes[i + q2_bins] = y->frame();
    }

    const int ncomps = 10;

    if (result != NULL)
    {
        printf("Fit ran with status %d\n", result->status());

        printf("Stat error on R(D*) is %f\n", poi->getError());

        printf("EDM at end was %f\n", result->edm());

        result->floatParsInit().Print();

        cout << "CURRENT NUISANCE PARAMETERS:" << endl;

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

        if (dofit)
            printf("Stopwatch: fit ran in %f seconds\n", sw.RealTime());
    }

    // Create the colors for the components
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
    RooHist *mm2resid; // = mm2_frame->pullHist() ;
    RooHist *Elresid;  // = El_frame->pullHist() ;
    RooHist *q2resid;  // = q2_frame->pullHist() ;

    RooHist *resids[nframes];

    // Plot the data and the model
    for (int i = 0; i < nframes; i++)
    {
        data->plotOn(drawframes[i], DataError(RooAbsData::Poisson), Cut("channelCat==0"), MarkerSize(0.4), DrawOption("ZP"));
        model->plotOn(drawframes[i], Slice(*idx), ProjWData(*idx, *data), DrawOption("F"), FillColor(kRed));
        model->plotOn(drawframes[i], Slice(*idx), ProjWData(*idx, *data), DrawOption("F"), FillColor(kViolet), Components("*misID*,*sigmu*,*D1*"));
        model->plotOn(drawframes[i], Slice(*idx), ProjWData(*idx, *data), DrawOption("F"), FillColor(kBlue + 1), Components("*misID*,*sigmu*"));
        model->plotOn(drawframes[i], Slice(*idx), ProjWData(*idx, *data), DrawOption("F"), FillColor(kOrange), Components("*misID*"));
        resids[i] = drawframes[i]->pullHist();
        data->plotOn(drawframes[i], DataError(RooAbsData::Poisson), Cut("channelCat==0"), MarkerSize(0.4), DrawOption("ZP"));
    }

    // Create the residuals
    mm2resid = resids[0];
    Elresid = resids[1];
    q2resid = resids[2];

    // Prepare the strings for the cuts and ranges
    char cutstrings[q2_bins][128];
    char rangenames[q2_bins][32];
    char rangelabels[q2_bins][128];
    RooHist *mm2q2_pulls[q2_bins];
    RooHist *Elq2_pulls[q2_bins];

    // Fill the strings for the cuts and ranges
    for (int i = 0; i < q2_bins; i++)
    {
        double binlow = q2_low + i * (q2_high - q2_low) / q2_bins;
        double binhigh = q2_low + (i + 1) * (q2_high - q2_low) / q2_bins;
        sprintf(rangelabels[i], "%.2f < q^{2} < %.2f", binlow * 1e-6, binhigh * 1e-6);
        sprintf(cutstrings[i], "obs_z_Dstmu_kinematic > %f && obs_z_Dstmu_kinematic < %f && channelCat==0", q2_low + i * (q2_high - q2_low) / q2_bins, q2_low + (i + 1) * (q2_high - q2_low) / q2_bins);
        sprintf(rangenames[i], "q2bin_%d", i);
        z->setRange(rangenames[i], binlow, binhigh);
    }

    if (slowplots == true) // Draw plots for each q2 bin
    {
        cout << "Drawing Slow Plots" << endl;
        for (int i = 0; i < q2_bins; i++)
        {
            data->plotOn(mm2q2_frame[i], Cut(cutstrings[i]), DataError(RooAbsData::Poisson), MarkerSize(0.4), DrawOption("ZP"));
            model->plotOn(mm2q2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangenames[i]), DrawOption("F"), FillColor(kRed));

            data->plotOn(Elq2_frame[i], Cut(cutstrings[i]), DataError(RooAbsData::Poisson), MarkerSize(0.4), DrawOption("ZP"));
            model->plotOn(Elq2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangenames[i]), DrawOption("F"), FillColor(kRed));

            // Grab pulls
            mm2q2_pulls[i] = mm2q2_frame[i]->pullHist();
            Elq2_pulls[i] = Elq2_frame[i]->pullHist();

            // Plot the components
            model->plotOn(mm2q2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangenames[i]), DrawOption("F"), FillColor(kViolet), Components("*sigmu*,*D1*,*misID*"));
            model->plotOn(Elq2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangenames[i]), DrawOption("F"), FillColor(kViolet), Components("*sigmu*,*D1*,*misID*"));
            model->plotOn(mm2q2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangenames[i]), DrawOption("F"), FillColor(kBlue + 1), Components("*sigmu*,*misID*"));
            model->plotOn(Elq2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangenames[i]), DrawOption("F"), FillColor(kBlue + 1), Components("*sigmu*,*misID*"));
            model->plotOn(mm2q2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangenames[i]), DrawOption("F"), FillColor(kOrange), Components("*misID*"));
            model->plotOn(Elq2_frame[i], Slice(*idx), ProjWData(*idx, *data), ProjectionRange(rangenames[i]), DrawOption("F"), FillColor(kOrange), Components("*misID*"));
            data->plotOn(mm2q2_frame[i], Cut(cutstrings[i]), DataError(RooAbsData::Poisson), MarkerSize(0.4), DrawOption("ZP"));
            data->plotOn(Elq2_frame[i], Cut(cutstrings[i]), DataError(RooAbsData::Poisson), MarkerSize(0.4), DrawOption("ZP"));
        }
    }

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

    RooPlot *DOCA_resid_frame;
    /*
          cerr << __LINE__ << endl;
          mm2_resid_frame->addPlotable(mm2resid,"P");
          cerr << __LINE__ << endl;
          El_resid_frame->addPlotable(Elresid,"P");
          cerr << __LINE__ << endl;
          q2_resid_frame->addPlotable(q2resid,"P");
          cerr << __LINE__ << endl;

          TCanvas *c3 = new TCanvas("c3","c3",640,1000);
          c3->Divide(1,3);
          c3->cd(1);
          mm2_resid_frame->Draw();
          c3->cd(2);
          El_resid_frame->Draw();
          c3->cd(3);
          q2_resid_frame->Draw();
          */

    // Create the canvas for the plots in q2 bins
    TCanvas *c2;
    if (slowplots == true)
    {

        c2 = new TCanvas("c2", "c2", 1200, 600);
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
    }

    // Save the canvas
    c2->SaveAs("plots/HistFactDstTauDemo_in_q2_bins.png");
}

void HistFactDstTauDemo()
{
    main_HistFactDstTauDemo(
        /* STEERING OPTIONS */
        true,  // constrainDstst
        true,  // useMinos
        true,  // useMuShapeUncerts
        true,  // useTauShapeUncerts
        true,  // useDststShapeUncerts
        false, // fixshapes
        false, // fixshapesdstst
        true,  // dofit
        false, // fitfirst
        true,  // slowplots
        false  // BBon3d
    );
}