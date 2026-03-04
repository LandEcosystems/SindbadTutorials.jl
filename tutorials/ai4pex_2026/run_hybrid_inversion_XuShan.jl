# ================================== using tools ==================================================
# some of the things that will be using... Julia tools, SINDBAD tools, local codes...
import Pkg
Pkg.activate(".")
Pkg.instantiate()

using Revise
using SindbadTutorials
using SindbadML
using SindbadML.Random
using SindbadTutorials.Plots

include("tutorial_helpers.jl")

## get the sites to run experiment on
selected_site_indices = getSiteIndicesForHybrid();
do_random = 0# set to integer values larger than zero to use random selection of #do_random sites
if do_random > 0
    Random.seed!(1234)
    selected_site_indices = first(shuffle(selected_site_indices), do_random)
end

# ================================== get data / set paths ========================================= 
# data to be used can be found here: https://nextcloud.bgc-jena.mpg.de/s/w2mbH59W4nF3Tcd
# organizing the paths of data sources and outputs for this experiment
path_input_dir      = getSindbadDataDepot(; env_data_depot_var="SINDBAD_DATA_DEPOT", 
                    local_data_depot=joinpath("/home/jovyan/data/ellis_jena_2025")); # for convenience, the data file is set within the SINDBAD-Tutorials path; this needs to be changed otherwise.
path_input          = joinpath("$(path_input_dir)","FLUXNET_v2023_12_1D_REPLACED_Noise003_v1.zarr"); # zarr data source containing all the data for site level runs
path_observation    = path_input; # observations (synthetic or otherwise) are included in the same file
path_covariates     = joinpath("$(path_input_dir)","CovariatesFLUXNET.zarr"); # zarr data source containing all the covariates
path_output         = "";

#= this one takes a hugh amount of time, leave it here for reference
# ================================== setting up the experiment ====================================
# experiment is all set up according to a (collection of) json file(s)
path_experiment_json    = joinpath(@__DIR__,"..","ellis_jena_2025","settings_WROASTED_HB","experiment_hybrid.json");
path_training_folds     = "";#joinpath(@__DIR__,"..","ellis_jena_2025","settings_WROASTED_HB","nfolds_sites_indices.jld2");

replace_info = Dict(
    "forcing.default_forcing.data_path" => path_input,
    "forcing.subset.site" => selected_site_indices,
    "optimization.observations.default_observation.data_path" => path_observation,
    "optimization.optimization_cost_threaded" => false,
    "optimization.optimization_parameter_scaling" => nothing,
    "hybrid.ml_training.fold_path" => nothing,
    "hybrid.covariates.path" => path_covariates,
);

# generate the info and other helpers
info            = getExperimentInfo(path_experiment_json; replace_info=deepcopy(replace_info));
forcing         = getForcing(info);
observations    = getObservation(info, forcing.helpers);
sites_forcing   = forcing.data[1].site;
hybrid_helpers  = prepHybrid(forcing, observations, info, info.hybrid.ml_training.method);

# ================================== train the hybrid model =======================================
trainML(hybrid_helpers, info.hybrid.ml_training.method)
=#

# ================================== change setup to WROASTED ==========================================
# same as before, but for a slower / more complicated WROASTED model
path_experiment_json    = joinpath(@__DIR__,"..","ellis_jena_2025","settings_LUE","experiment_hybrid.json");
path_training_folds     = "";#joinpath(@__DIR__,"..","ellis_jena_2025","settings_WROASTED_HB","nfolds_sites_indices.jld2");

replace_info = Dict(
    "forcing.default_forcing.data_path" => path_input,
    "forcing.subset.site" => selected_site_indices,
    "optimization.observations.default_observation.data_path" => path_observation,
    "optimization.optimization_cost_threaded" => false,
    "optimization.optimization_parameter_scaling" => nothing,
    "hybrid.ml_training.fold_path" => nothing,
    "hybrid.covariates.path" => path_covariates,
);

# generate the info and other helpers
info            = getExperimentInfo(path_experiment_json; replace_info=deepcopy(replace_info));
forcing         = getForcing(info);
observations    = getObservation(info, forcing.helpers);
sites_forcing   = forcing.data[1].site;
hybrid_helpers  = prepHybrid(forcing, observations, info, info.hybrid.ml_training.method);

# train the model
trainML(hybrid_helpers, info.hybrid.ml_training.method)

## check the docs for output at: http://sindbad-mdi.org/pages/develop/hybrid_modeling.html and http://sindbad-mdi.org/pages/develop/sindbad_outputs.html

# hybrid_helpers_cp = deepcopy(hybrid_helpers);
# using JLD2
# @load "/Users/xshan/Research/MPI/hybrid_helpers" hybrid_helpers;

# ================================== posterior run diagnostics =====================================
# retrieve the inversed parameters
sites_forcing = forcing.data[1].site;
ml_model = hybrid_helpers.ml_model;
xfeatures = hybrid_helpers.features.data;
loss_functions = hybrid_helpers.loss_functions;
loss_component_functions = hybrid_helpers.loss_component_functions;

params_sites = ml_model(xfeatures)
@info "params_sites: [$(minimum(params_sites)), $(maximum(params_sites))]"

scaled_params_sites = getParamsAct(params_sites, info.optimization.parameter_table)
@info "scaled_params_sites: [$(minimum(scaled_params_sites)), $(maximum(scaled_params_sites))]"

# select a site
# idx = findfirst(x -> x == "DE-Hai", sites_forcing)
site_index = 67
site_name = sites_forcing[site_index]

loc_params = scaled_params_sites(site=site_name).data.data
loss_f_site = loss_functions(site=site_name);
loss_vector_f_site = loss_component_functions(site=site_name);
@time loss_f_site(loc_params)
loss_vector_f_site(loc_params)


# run the model for the site with the default parameters
function run_model_param(info, forcing, observations, site_index, loc_params)
    # info = @set info.helpers.run.land_output_type = PreAllocArrayAll();
    run_helpers = prepTEM(info.models.forward, forcing, observations, info);
    # output all variables
    # info = @set info.output.variables = run_helpers.output_vars;

    params = loc_params;
    selected_models = info.models.forward;
    parameter_scaling_type = info.optimization.run_options.parameter_scaling;
    tbl_params = info.optimization.parameter_table;
    param_to_index = getParameterIndices(selected_models, tbl_params);

    models = updateModels(params, param_to_index, parameter_scaling_type, selected_models);

    loc_forcing = run_helpers.space_forcing[site_index];
    loc_spinup_forcing = run_helpers.space_spinup_forcing[site_index];
    loc_forcing_t = run_helpers.loc_forcing_t;
    loc_output = getCacheFromOutput(run_helpers.space_output[site_index], info.hybrid.ml_gradient.method);
    gradient_lib = info.hybrid.ml_gradient.method;
    loc_output_from_cache = getOutputFromCache(loc_output, params, gradient_lib);
    land_init = deepcopy(run_helpers.loc_land);
    tem_info = run_helpers.tem_info;
    loc_obs = run_helpers.space_observation[site_index];
    loc_cost_option = prepCostOptions(loc_obs, info.optimization.cost_options);
    constraint_method = info.optimization.run_options.multi_constraint_method;
    coreTEM!(
            models,
            loc_forcing,
            loc_spinup_forcing,
            loc_forcing_t,
            loc_output_from_cache,
            land_init,
            tem_info);
    forward_output = (; Pair.(getUniqueVarNames(info.output.variables), loc_output_from_cache)...)
    loss_vector = SindbadTutorials.metricVector(loc_output_from_cache, loc_obs, loc_cost_option);
    t_loss = combineMetric(loss_vector, constraint_method);
    return forward_output, loss_vector, t_loss
end
output_default_site, _, _ = run_model_param(info, forcing, observations, 
                                            site_index, 
                                            info.optimization.parameter_table.default);



# run the model for the site with the parameters from the hybrid 
output_optimized_site, _, _ = run_model_param(info, forcing, observations, 
                                                site_index, 
                                                loc_params);
# check loss_vector, t_loss have the same values with loss_vector_f_site(loc_params)
# julia> loss_vector_opt, t_loss_opt
# (Float32[0.3475957, 0.32737154, 0.88894814, 0.09142125, 0.090762675, 0.14379889, 0.59722495], 2.487123f0)

# julia> loss_vector_f_site(loc_params)
# (2.487123f0, Float32[0.3475957, 0.32737154, 0.88894814, 0.09142125, 0.090762675, 0.14379889, 0.59722495], [1, 2, 3, 4, 5, 6, 7])


# do plots, compute some simple statistics e.g. NSE
def_dat = output_default_site;
opt_dat = output_optimized_site;
loc_observation = [Array(o[:, site_index]) for o in observations.data];
costOpt = prepCostOptions(loc_observation, info.optimization.cost_options);
default(titlefont=(20, "times"), legendfontsize=18, tickfont=(15, :blue))
foreach(costOpt) do var_row
    v = var_row.variable
    println("plot obs::", v)
    v = (var_row.mod_field, var_row.mod_subfield)
    vinfo = getVariableInfo(v, info.experiment.basics.temporal_resolution)
    v = vinfo["standard_name"]
    lossMetric = var_row.cost_metric
    loss_name = nameof(typeof(lossMetric))
    if loss_name in (:NNSEInv, :NSEInv)
        lossMetric = NSE()
    end
    (obs_var, obs_σ, def_var) = getData(def_dat, loc_observation, var_row)
    (_, _, opt_var) = getData(opt_dat, loc_observation, var_row)
    obs_var_TMP = obs_var[:, 1, 1, 1]
    non_nan_index = findall(x -> !isnan(x), obs_var_TMP)
    if length(non_nan_index) < 2
        tspan = 1:length(obs_var_TMP)
    else
        tspan = first(non_nan_index):last(non_nan_index)
    end
    obs_σ = obs_σ[tspan]
    obs_var = obs_var[tspan, 1, 1, 1]
    def_var = def_var[tspan, 1, 1, 1]
    opt_var = opt_var[tspan, 1, 1, 1]

    xdata = [info.helpers.dates.range[tspan]...]
    obs_var_n, obs_σ_n, def_var_n = getDataWithoutNaN(obs_var, obs_σ, def_var)
    obs_var_n, obs_σ_n, opt_var_n = getDataWithoutNaN(obs_var, obs_σ, opt_var)
    metr_def = metric(obs_var_n, obs_σ_n, def_var_n, lossMetric)
    metr_opt = metric(obs_var_n, obs_σ_n, opt_var_n, lossMetric)
    plot(xdata, obs_var; label="obs", seriestype=:scatter, mc=:black, ms=4, lw=0, ma=0.65, left_margin=1Plots.cm)
    plot!(xdata, def_var, color=:steelblue2, lw=1.5, ls=:dash, left_margin=1Plots.cm, legend=:outerbottom, legendcolumns=3, label="def ($(round(metr_def, digits=2)))", size=(2000, 1000), title="$(vinfo["long_name"]) ($(vinfo["units"])) -> $(nameof(typeof(lossMetric)))")
    plot!(xdata, opt_var; color=:seagreen3, label="opt ($(round(metr_opt, digits=2)))", lw=1.5, ls=:dash)
    savefig(joinpath(info.output.dirs.figure, "wroasted_$(site_name)_$(v).png"))
end

# run the model for each site with default model parameterization, compute the NSE for each site
# for site_index in 1:length(sites_forcing)
#     _, loss_vector_def, t_loss_def = run_model_param(info, forcing, observations, 
#                                                     site_index, 
#                                                     info.optimization.parameter_table.default);
#     _, loss_vector_opt, t_loss_opt = run_model_param(info, forcing, observations, 
#                                                 site_index, 
#                                                 loc_params);
# end

# preallocate
n_sites = length(sites_forcing);
t_loss_def = Vector{Float32}(undef, n_sites);
t_loss_opt = Vector{Float32}(undef, n_sites);

for (i, site_index) in enumerate(1:n_sites)
    @info site_index
    @info sites_forcing[site_index]
    @info "running using default parameters..."
    # # the following way could be faster to run...
    # loss_f_site = loss_functions(site=site_name);
    # loss_vector_f_site = loss_component_functions(site=site_name);
    # loss_f_site(loc_params)
    # loss_vector_f_site(loc_params)

    # run default
    _, lv_def, tl_def = run_model_param(
       info, forcing, observations, site_index,
       info.optimization.parameter_table.default
    )
    # run optimized
    @info "running using optimized parameters..."
    _, lv_opt, tl_opt = run_model_param(
       info, forcing, observations, site_index,
       scaled_params_sites(site=sites_forcing[site_index]).data.data
    )
    # collect
    t_loss_def[i]     = tl_def
    t_loss_opt[i]     = tl_opt
end

# now plot histogram of scalar losses
histogram(
  t_loss_def,
  bins = 50,
  alpha = 0.5,
  label = "default",
  xlabel = "NSE",
  ylabel = "Count",
  title = "NSE: default vs optimized"
)
histogram!(
  t_loss_opt,
  bins = 50,
  alpha = 0.5,
  label = "optimized"
)
savefig(joinpath(info.output.dirs.figure, "loss_histogram_scalar.png"))



#= useful code snipets
## access and check the loss values
array_loss = hybrid_helpers.array_loss;
array_loss_components = hybrid_helpers.array_loss_components;


## play around with gradient for sites and batch to understand internal workings
sites_forcing = forcing.data[1].site;
ml_model = hybrid_helpers.ml_model;
xfeatures = hybrid_helpers.features.data;
loss_functions = hybrid_helpers.loss_functions;
loss_component_functions = hybrid_helpers.loss_component_functions;

params_sites = ml_model(xfeatures)
@info "params_sites: [$(minimum(params_sites)), $(maximum(params_sites))]"

scaled_params_sites = getParamsAct(params_sites, info.optimization.parameter_table)
@info "scaled_params_sites: [$(minimum(scaled_params_sites)), $(maximum(scaled_params_sites))]"


## try for a site
site_index = 1
site_name = sites_forcing[site_index]

loc_params = scaled_params_sites(site=site_name).data.data
loss_f_site = loss_functions(site=site_name);
loss_vector_f_site = loss_component_functions(site=site_name);
@time loss_f_site(loc_params)
loss_vector_f_site(loc_params)

@time g_site = gradientSite(info.hybrid.ml_gradient.method, loc_params, info.hybrid.ml_gradient.options, loss_functions(site=site_name))

## try for a batch
sites_batch = hybrid_helpers.sites.training[1:info.hybrid.ml_training.options.batch_size]
scaled_params_batch = scaled_params_sites(; site=sites_batch)
grads_batch = zeros(Float32, size(scaled_params_batch, 1), length(sites_batch));

g_batch = gradientBatch!(info.hybrid.ml_gradient.method, grads_batch, info.hybrid.ml_gradient.options, loss_functions, scaled_params_batch, sites_batch; showprog=true)
=#
