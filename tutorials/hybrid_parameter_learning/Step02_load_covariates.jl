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
# trainML(hybrid_helpers, info.hybrid.ml_training.method)
# ## check the docs for output at: http://sindbad-mdi.org/pages/develop/hybrid_modeling.html and http://sindbad-mdi.org/pages/develop/sindbad_outputs.html


#################################################################################################################
# add PCA into the covariates?
