# Task 01 : Build a hybrid model for learning WROASTED/LUE model parameters from synthetic FLUXNET data

Author: Xu Shan, Sujan Koirala, and Nuno Carvalhais

In this tutorial, we will build a hybrid model to learn the WROASTED/LUE model parameters from synthetic FLUXNET data using a machine learning model in SINDBAD.MachineLearning. We will explore different ML architectures (e.g., feedforward neural networks, recurrent neural networks) to improve the generalizability of the model.

This script is written in the form of md file because of some of the commands cannot be run in the `.ipynb` notebooks, but need to be run in the REPL/terminal. You can copy and paste the commands into the terminal, or run the script directly in the terminal with `julia`.

## Step 01: Understand, test, and run the existing code for learning WROASTED/LUE model parameters from synthetic FLUXNET data

In this step, we will understand, test, and run the existing code for learning WROASTED/LUE model parameters from synthetic FLUXNET data using a machine learning model in SINDBAD. We will test and run the notebook `Task01_LUE_inversion_XuShan.ipynb` for learning LUE model parameters from synthetic FLUXNET data. The purpose of this step is to understand the existing code and results for learning WROASTED/LUE model parameters from synthetic FLUXNET data, and to make sure that you can run the existing code successfully before we start building the hybrid model in the next steps. You can also explore the code and results in the notebooks, and try to understand how the ML model is trained and evaluated, and how it is linked to the Sindbad model. This will help you to build a better hybrid model in the next steps.

## Step 02: Try to use a different set of covariates to learn the LUE parameters from synthetic FLUXNET data
In this step, we will try to use a different set of covariates to learn the LUE parameters from synthetic FLUXNET data using a machine learning model in SINDBAD. We will modify the existing code for learning LUE model parameters from synthetic FLUXNET data to use a different set of covariates, and evaluate the performance of the ML model in learning the LUE parameters with the new set of covariates. The purpose of this step is to explore how different sets of covariates can affect the performance of the ML model in learning the LUE parameters, and to understand the importance of covariate selection in building a hybrid model for learning WROASTED/LUE model parameters from synthetic FLUXNET data. You can also try to use different sets of covariates, and compare their performance in learning the LUE parameters from synthetic FLUXNET data. This will help you to build a better hybrid model in the next steps.

The way we use a different set of covairates relies on the extension of the existing function `loadCovariates` in the `Step02_load_covariates.jl` script, and the corresponding modification of the `trainML` function. We define a new `Sindbad` type within `Sindbad.Setup` and use it for this excercise. The way we do it is to import the existing `Sindbad` type and the `trainML` function, and then extend the `loadCovariates` function for the new `Sindbad` type to load a different set of covariates from the synthetic FLUXNET data, and extend the `trainML` function for the new `Sindbad` type to use the new set of covariates to train the ML model to learn the LUE parameters from synthetic FLUXNET data. Please refer to the `Step02_load_covariates.jl` and `Step03_use_pca.jl` scripts for the implementation of using a different set of covariates to learn the LUE parameters from synthetic FLUXNET data.

```julia
using SindbadTutorials
using SindbadTutorials.Sindbad
import SindbadTutorials.Sindbad: purpose
import SindbadTutorials.Sindbad.MachineLearning: trainML, loadCovariates
import SindbadTutorials.Sindbad.MachineLearning: lcKAoneHotbatch, vegKAoneHotbatch
import SindbadTutorials.Sindbad.DataLoaders: Cube, At, yaxCubeToKeyedArray
```

### Redefine and extend the `Sindbad` type and the `trainML` and `loadCovariates` functions

Then we can define a new `Sindbad` type within `Sindbad.Setup` and use it for this excercise. We will extend the `loadCovariates` function for the new `Sindbad` type to load a different set of covariates from the synthetic FLUXNET data, and extend the `trainML` function for the new `Sindbad` type to use the new set of covariates to train the ML model to learn the LUE parameters from synthetic FLUXNET data. Please refer to the `Step02_load_covariates.jl` and `Step03_use_pca.jl` scripts for the implementation of using a different set of covariates to learn the LUE parameters from synthetic FLUXNET data.

```julia

struct FluxnetParameterLearningWROASTEDNewCovariates <: MachineLearningExperimentType end
purpose(::Type{FluxnetParameterLearningWROASTEDNewCovariates}) = "Experiment type for learning spatial variability of WROASTED model parameters from FLUXNET data using a machine learning model in SINDBAD.MachineLearning"

@eval SindbadTutorials.Sindbad.Setup begin
    struct FluxnetParameterLearningWROASTEDNewCovariates <: MachineLearningExperimentType end
end

purpose(::Type{SindbadTutorials.Sindbad.Setup.FluxnetParameterLearningWROASTEDNewCovariates}) =
    "Experiment type for learning spatial variability of WROASTED model parameters from FLUXNET data using a machine learning model in SINDBAD.MachineLearning"

function trainML(hybrid_helpers, ::Sindbad.Setup.FluxnetParameterLearningWROASTEDNewCovariates)
    trainML(hybrid_helpers, MixedGradient())
end


function loadCovariates(
    ::Sindbad.Setup.FluxnetParameterLearningWROASTEDNewCovariates,
    sites_forcing,
    covariate_path,
    covariate_options,
)
    @info "using custom loadCovariates"
    c_read = Cube(covariate_path)
    kind = covariate_options.kind
    # select features, do only nor
    only_nor = occursin.(r"nor", c_read.features)
    nor_sel = c_read.features[only_nor].val
    nor_sel = [string.(s) for s in nor_sel] |> sort
    # select only normalized continuous variables
    ds_nor = c_read[features = At(nor_sel)]
    xfeat_nor = yaxCubeToKeyedArray(ds_nor)
    # apply PCA to xfeat_nor if needed
    # ? where is age?
    kg_data = c_read[features=At("KG")][:].data
    oneHot_KG = lcKAoneHotbatch(kg_data, 32, "KG", string.(c_read.site))
    pft_data = c_read[features=At("PFT")][:].data
    oneHot_pft = lcKAoneHotbatch(pft_data, 17, "PFT", string.(c_read.site))
    oneHot_veg = vegKAoneHotbatch(pft_data, string.(c_read.site))

    stackedFeatures = if kind=="all" 
            reduce(vcat, [oneHot_KG, oneHot_pft, xfeat_nor])
        elseif  kind=="PFT"
            reduce(vcat, [oneHot_pft])
        elseif kind=="KG"
            reduce(vcat, [oneHot_KG])
        elseif kind=="KG_PFT"
            reduce(vcat, [oneHot_KG, oneHot_pft])
        elseif kind=="PFT_ABCNOPSWB"
            reduce(vcat, [oneHot_pft, xfeat_nor])
        elseif kind=="KG_ABCNOPSWB"
            reduce(vcat, [oneHot_KG, xfeat_nor])
        elseif kind=="ABCNOPSWB"
            reduce(vcat, [xfeat_nor])
        elseif kind =="veg_all"
            reduce(vcat, [oneHot_KG, oneHot_veg, xfeat_nor])
        elseif kind=="veg"
            reduce(vcat, [oneHot_veg])
        elseif kind=="KG_veg"
            reduce(vcat, [oneHot_KG, oneHot_veg])
        elseif kind=="veg_ABCNOPSWB"
            reduce(vcat, [oneHot_veg, xfeat_nor])
        end
    # remove sites (with NaNs and duplicates)
    to_remove = [
        "CA-NS3",
        # "CA-NS4",
        "IT-CA1",
        # "IT-CA2",
        "IT-SR2",
        # "IT-SRo",
        "US-ARb",
        # "US-ARc",
        "US-GBT",
        # "US-GLE",
        "US-Tw1",
        # "US-Tw2"
        ]
    not_these = ["RU-Tks", "US-Atq", "US-UMd"] # NaNs
    not_these = vcat(not_these, to_remove)
    new_sites = setdiff(c_read.site, not_these)
    stackedFeatures = stackedFeatures(; site=new_sites)
    # get common sites between names in forcing and covariates
    sites_feature_all = [s for s in stackedFeatures.site]
    sites_common = intersect(sites_feature_all, sites_forcing)
    xfeatures = Float32.(stackedFeatures(; site=sites_common))
    @info "features loaded with size: ", size(xfeatures)
    return xfeatures
end
```

Note that here we directly copy paste the original `loadCovariates` function and did not modify it at all. This is to leave for the students to explore and modify the `loadCovariates` function to use a different set of covariates from the synthetic FLUXNET data. 

### Run and test the new `loadCovariates` and `trainML` functions
After we extend the `loadCovariates` and `trainML` functions for the new `Sindbad` type, we can run and test the new `loadCovariates` and `trainML` functions to see how they work with the new set of covariates. We can run the `Step02_load_covariates.jl` script to test the new `loadCovariates` function, or simply copy paste the `julia` command here to the terminal for the running:
```julia

####
# test the new loadCovariates functions
# ================================== using tools ==================================================
# some of the things that will be using... Julia tools, SINDBAD tools, local codes...
using Revise
using Flux
using SindbadTutorials
using Sindbad.MachineLearning
using Sindbad.MachineLearning.Random
using SindbadTutorials.Plots

# include("tutorial_helpers.jl")

## get the sites to run experiment on
selected_site_indices = getSiteIndicesForHybrid();
do_random = 0# set to integer values larger than zero to use random selection of #do_random sites
if do_random > 0
    Random.seed!(1234)
    selected_site_indices = first(shuffle(selected_site_indices), do_random)
end

path_output         = "";

# this one takes a hugh amount of time, leave it here for reference
# ================================== setting up the experiment ====================================
# experiment is all set up according to a (collection of) json file(s)
path_experiment_json    = joinpath(@__DIR__,"..","setups","LUE","experiment_hybrid.json");
path_training_folds     = "";#joinpath(@__DIR__,"..","setups","WROASTED_HB","nfolds_sites_indices.jld2");

replace_info = Dict(
    "forcing.subset.site" => selected_site_indices,
    "optimization.optimization_cost_threaded" => false,
    "optimization.optimization_parameter_scaling" => nothing,
    "hybrid.ml_training.fold_path" => nothing,
    "experiment.basics.config_files.hybrid" => "parameter_learning_LUE_newCovariates.json",
    );

# generate the info and other helpers
info            = getExperimentInfo(path_experiment_json; replace_info=deepcopy(replace_info));
forcing         = getForcing(info);
observations    = getObservation(info, forcing.helpers);
sites_forcing   = forcing.data[1].site;
hybrid_helpers  = prepHybrid(forcing, observations, info, info.hybrid.ml_training.method);

# # train the model
trainML(hybrid_helpers, info.hybrid.ml_training.method);

```
## Step 03: Try to use PCA to reduce the dimensionality of the covariates for learning the LUE parameters from synthetic FLUXNET data
In this step, we will try to use PCA to reduce the dimensionality of the covariates for learning the LUE parameters from synthetic FLUXNET data using a machine learning model in SINDBAD. We will modify the existing code for learning LUE model parameters from synthetic FLUXNET data to use PCA to reduce the dimensionality of the covariates, and evaluate the performance of the ML model in learning the LUE parameters with PCA-reduced covariates. The purpose of this step is to explore how dimensionality reduction techniques like PCA can affect the performance of the ML model in learning the LUE parameters, and to understand the importance of dimensionality reduction in building a hybrid model for learning WROASTED/LUE model parameters from synthetic FLUXNET data. You can also try to use different dimensionality reduction techniques, and compare their performance in learning the LUE parameters from synthetic FLUXNET data. This will help you to build a better hybrid model in the next steps.

The way we use PCA to reduce the dimensionality of the covariates relies on the extension of the existing function `trainML` in the `Step03_use_pca.jl` script. We will extend the `trainML` function for the new `Sindbad` type to apply PCA to the covariates before training the ML model to learn the LUE parameters from synthetic FLUXNET data. Please refer to the `Step03_use_pca.jl` script for the implementation of using PCA to reduce the dimensionality of the covariates for learning the LUE parameters from synthetic FLUXNET data.

### Extend new `loadCovariates` and `trainML` functions to use PCA for dimensionality reduction

```julia
using LinearAlgebra
using Statistics
using AxisKeys: KeyedArray
using SindbadTutorials
using SindbadTutorials.Sindbad
import SindbadTutorials.Sindbad: purpose
import SindbadTutorials.Sindbad.MachineLearning: trainML, loadCovariates
import SindbadTutorials.Sindbad.MachineLearning: lcKAoneHotbatch, vegKAoneHotbatch
import SindbadTutorials.Sindbad.DataLoaders: Cube, At, yaxCubeToKeyedArray


struct FluxnetParameterLearningWROASTEDNewCovariates <: MachineLearningExperimentType end
purpose(::Type{FluxnetParameterLearningWROASTEDNewCovariates}) = "Experiment type for learning spatial variability of WROASTED model parameters from FLUXNET data using a machine learning model in SINDBAD.MachineLearning"

@eval SindbadTutorials.Sindbad.Setup begin
    struct FluxnetParameterLearningWROASTEDNewCovariates <: MachineLearningExperimentType end
end

purpose(::Type{SindbadTutorials.Sindbad.Setup.FluxnetParameterLearningWROASTEDNewCovariates}) =
    "Experiment type for learning spatial variability of WROASTED model parameters from FLUXNET data using a machine learning model in SINDBAD.MachineLearning"

function trainML(hybrid_helpers, ::Sindbad.Setup.FluxnetParameterLearningWROASTEDNewCovariates)
    trainML(hybrid_helpers, MixedGradient())
end

```

Then we could test the PCA function with following codes:
```julia
function pcaKeyedArray(xfeat_nor; n_components=5)
    X = Float32.(Array(xfeat_nor))   # size: n_features x n_sites

    # center each feature across sites
    mu = mean(X; dims=2)
    Xc = X .- mu

    # PCA by SVD, keeping sites as columns
    F = svd(Xc)
    k = min(n_components, size(X, 1), size(X, 2))

    # principal-component scores for each site: k x n_sites
    scores = Diagonal(F.S[1:k]) * F.Vt[1:k, :]

    pc_labels = ["PC$(i)" for i in 1:k]
    site_labels = collect(xfeat_nor.site)

    return KeyedArray(Float32.(scores); features=pc_labels, site=site_labels)
end

function loadCovariates(
    ::Sindbad.Setup.FluxnetParameterLearningWROASTEDNewCovariates,
    sites_forcing,
    covariate_path,
    covariate_options,
)
    @info "using custom loadCovariates, with PCA and site filtering"

    c_read = Cube(covariate_path)
    kind = covariate_options.kind
    n_components = hasproperty(covariate_options, :n_components) ? covariate_options.n_components : 5

    only_nor = occursin.(r"nor", c_read.features)
    nor_sel = c_read.features[only_nor].val
    nor_sel = [string.(s) for s in nor_sel] |> sort

    ds_nor = c_read[features = At(nor_sel)]
    xfeat_nor = yaxCubeToKeyedArray(ds_nor)

    kg_data = c_read[features=At("KG")][:].data
    oneHot_KG = lcKAoneHotbatch(kg_data, 32, "KG", string.(c_read.site))

    pft_data = c_read[features=At("PFT")][:].data
    oneHot_pft = lcKAoneHotbatch(pft_data, 17, "PFT", string.(c_read.site))
    oneHot_veg = vegKAoneHotbatch(pft_data, string.(c_read.site))

    # remove problematic sites before PCA
    to_remove = [
        "CA-NS3",
        "IT-CA1",
        "IT-SR2",
        "US-ARb",
        "US-GBT",
        "US-Tw1",
    ]
    not_these = vcat(["RU-Tks", "US-Atq", "US-UMd"], to_remove)
    new_sites = setdiff(c_read.site, not_these)

    xfeat_nor = xfeat_nor(; site=new_sites)
    oneHot_KG = oneHot_KG(; site=new_sites)
    oneHot_pft = oneHot_pft(; site=new_sites)
    oneHot_veg = oneHot_veg(; site=new_sites)

    xfeat_pca = pcaKeyedArray(xfeat_nor; n_components=n_components)
    @info "PCA completed, keeping $n_components components"

    stackedFeatures = if kind == "all"
        reduce(vcat, [oneHot_KG, oneHot_pft, xfeat_pca])
    elseif kind == "PFT"
        oneHot_pft
    elseif kind == "KG"
        oneHot_KG
    elseif kind == "KG_PFT"
        reduce(vcat, [oneHot_KG, oneHot_pft])
    elseif kind == "PFT_PCA"
        reduce(vcat, [oneHot_pft, xfeat_pca])
    elseif kind == "KG_PCA"
        reduce(vcat, [oneHot_KG, xfeat_pca])
    elseif kind == "PCA"
        xfeat_pca
    elseif kind == "veg_all"
        reduce(vcat, [oneHot_KG, oneHot_veg, xfeat_pca])
    elseif kind == "veg"
        oneHot_veg
    elseif kind == "KG_veg"
        reduce(vcat, [oneHot_KG, oneHot_veg])
    elseif kind == "veg_PCA"
        reduce(vcat, [oneHot_veg, xfeat_pca])
    else
        error("Unknown covariate kind: $kind")
    end

    sites_feature_all = collect(stackedFeatures.site)
    sites_common = intersect(sites_feature_all, sites_forcing)
    xfeatures = Float32.(stackedFeatures(; site=sites_common))

    return xfeatures
end
```

## Step 04: Evaluate the performance of the ML model with different sets of covariates and with/without PCA for learning the LUE parameters from synthetic FLUXNET data
In this step, we will evaluate the performance of the ML model with different sets of covariates and with/without PCA for learning the LUE parameters from synthetic FLUXNET data using a machine learning model in SINDBAD. We will run the `Step02_load_covariates.jl` and `Step03_use_pca.jl` scripts with different configurations of covariates and PCA, and evaluate the performance of the ML model in learning the LUE parameters from synthetic FLUXNET data. The purpose of this step is to understand how different sets of covariates and dimensionality reduction techniques like PCA can affect the performance of the ML model in learning the LUE parameters, and to identify the best configuration of covariates and PCA for building a hybrid model for learning WROASTED/LUE model parameters from synthetic FLUXNET data.

This is left for the students to explore. You can try the example in the `Task01_LUE_inversion_XuShan.ipynb` notebook, and modify the code to run with different configurations of covariates and PCA, and evaluate the performance of the ML model in learning the LUE parameters from synthetic FLUXNET data. You can also visualize the results to better understand the performance of the ML model with different configurations of covariates and PCA. This will help you to build a better hybrid model in the next steps.