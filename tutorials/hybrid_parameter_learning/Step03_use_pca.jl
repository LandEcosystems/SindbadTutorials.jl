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
    "experiment.basics.config_files.hybrid" => "../LUE_newCovariates/parameter_learning_LUE_newCovariates.json",
    );

# generate the info and other helpers
info            = getExperimentInfo(path_experiment_json; replace_info=deepcopy(replace_info));
forcing         = getForcing(info);
observations    = getObservation(info, forcing.helpers);
sites_forcing   = forcing.data[1].site;
hybrid_helpers  = prepHybrid(forcing, observations, info, info.hybrid.ml_training.method);

# # train the model
trainML(hybrid_helpers, info.hybrid.ml_training.method);
# ## check the docs for output at: http://sindbad-mdi.org/pages/develop/hybrid_modeling.html and http://sindbad-mdi.org/pages/develop/sindbad_outputs.html
