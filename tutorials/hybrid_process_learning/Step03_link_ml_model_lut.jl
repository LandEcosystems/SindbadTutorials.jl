using Revise
using Flux
using JLD2
using SindbadTutorials
using SindbadTutorials.SindbadTEM          # brings gppAirT_externalNN into scope (via @reexport)
import SindbadTutorials.SindbadTEM.Processes: define, precompute, compute  # import allows adding new methods

# Extend define for gppAirT_externalNN
# define runs once to fix the type/shape of all land NamedTuple fields.
function define(params::gppAirT_externalNN, forcing, land, helpers)
    @unpack_nt o_one ⇐ land.constants          # o_one = 1.0 scalar
    gpp_f_airT      = o_one                    # initialise to 1 (no stress)
    tair_range_raw  = zeros(Float32, 200)      # placeholder lookup axis
    stress_function = zeros(Float32, 200)      # placeholder stress values
    @pack_nt gpp_f_airT     ⇒ land.diagnostics
    @pack_nt tair_range_raw  ⇒ land.gppAirT
    @pack_nt stress_function ⇒ land.gppAirT
    return land
end

# Extend precompute for gppAirT_externalNN
# precompute runs once before the time loop — ideal for file I/O.
function precompute(params::gppAirT_externalNN, forcing, land, helpers)
    checkpoint      = load(joinpath(@__DIR__, "../../gpp_model.jld2"))
    tair_range_raw  = Float32.(checkpoint["tair_range_raw"])  # Tair sweep, length=200, °C
    stress_function = Float32.(checkpoint["stress_function"])  # stress ∈ [0,1], length=200
    @pack_nt tair_range_raw  ⇒ land.gppAirT
    @pack_nt stress_function ⇒ land.gppAirT
    return land
end

# Extend compute for gppAirT_externalNN
# compute runs every timestep.
# Unpack the lookup tables from land.gppAirT (set in precompute),
# linearly interpolate at the current daytime air temperature,
# clamp to [0,1], and pack the result as gpp_f_airT.
function compute(params::gppAirT_externalNN, forcing, land, helpers)
    @unpack_nt f_airT_day    ⇐ forcing
    @unpack_nt tair_range_raw  ⇐ land.gppAirT
    @unpack_nt stress_function ⇐ land.gppAirT
    @unpack_nt (o_one, z_zero) ⇐ land.constants
    t = f_airT_day
    gpp_f_airT = if t ≤ tair_range_raw[1]
        stress_function[1]                  # below range → clamp to lowest
    elseif t ≥ tair_range_raw[end]
        stress_function[end]                # above range → clamp to highest
    else
        i = searchsortedfirst(tair_range_raw, t)
        α = (t - tair_range_raw[i-1]) / (tair_range_raw[i] - tair_range_raw[i-1])
        clamp(stress_function[i-1] * (1f0 - α) + stress_function[i] * α, z_zero, o_one)
    end
    @info "using gpp_f_airT from external NN"
    @pack_nt gpp_f_airT ⇒ land.diagnostics
    return land
end

## test the compute function
using Revise
using SindbadTutorials
using Sindbad.MachineLearning
using Sindbad.MachineLearning.Random
using SindbadTutorials.Plots
using Flux, PreallocationTools, FiniteDiff, FiniteDifferences, ForwardDiff, Optimisers

## get the sites to run experiment on
selected_site_indices = getSiteIndicesForHybrid();
do_random = 0# set to integer values larger than zero to use random selection of #do_random sites
if do_random > 0
    Random.seed!(1234)
    selected_site_indices = first(shuffle(selected_site_indices), do_random)
end

# ================================== get data / set paths ========================================= 
path_output         = "";

# this one takes a hugh amount of time, leave it here for reference
# ================================== setting up the experiment ====================================
# experiment is all set up according to a (collection of) json file(s)
path_experiment_json    = joinpath(@__DIR__,"..","setups","WROASTED_HB","experiment_hybrid.json");
path_training_folds     = "";#joinpath(@__DIR__,"..","setups","WROASTED_HB","nfolds_sites_indices.jld2");

replace_info = Dict(
    "forcing.subset.site" => selected_site_indices,
    "experiment.basics.config_files.model_structure" => joinpath(@__DIR__,"..","setups","WROASTED_HB","model_structure_externalNN.json"),
    "experiment.basics.config_files.optimization" => joinpath(@__DIR__,"..","setups","WROASTED_HB","optimization_externalNN.json"),
    "optimization.optimization_cost_threaded" => false,
    "optimization.optimization_parameter_scaling" => nothing,
    "hybrid.ml_training.fold_path" => nothing,
);

# generate the info and other helpers
info            = getExperimentInfo(path_experiment_json; replace_info=deepcopy(replace_info));
forcing         = getForcing(info);
observations    = getObservation(info, forcing.helpers);
sites_forcing   = forcing.data[1].site;
hybrid_helpers  = prepHybrid(forcing, observations, info, info.hybrid.ml_training.method);

# run the experiment
# runExperiment(info, forcing, observations, hybrid_helpers; path_output=path_output)
# output =runExperimentForward(path_experiment_json; replace_info=replace_info);
# out_dflt  = runExperimentForward(path_experiment_json; replace_info=deepcopy(replace_info)); # full default model

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
site_index = 1
output_default_site, _, _ = run_model_param(info, forcing, observations, 
                                            site_index, 
                                            info.optimization.parameter_table.default);
