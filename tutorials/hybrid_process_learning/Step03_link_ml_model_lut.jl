using Revise
using Flux
using JLD2
using SindbadTutorials
using SindbadTutorials.SindbadTEM          # brings gppAirT_externalNN_LUT into scope (via @reexport)
import SindbadTutorials.SindbadTEM.Processes: define, precompute, compute  # import allows adding new methods
using Sindbad.MachineLearning
using Sindbad.MachineLearning.Random
using SindbadTutorials.Plots
using Flux, PreallocationTools, FiniteDiff, FiniteDifferences, ForwardDiff, Optimisers

# Extend define for gppAirT_externalNN_LUT
# define runs once to fix the type/shape of all land NamedTuple fields.
function define(params::gppAirT_externalNN_LUT, forcing, land, helpers)
    @unpack_nt o_one ⇐ land.constants          # o_one = 1.0 scalar
    gpp_f_airT      = o_one                    # initialise to 1 (no stress)
    tair_range_raw  = zeros(Float32, 200)      # placeholder lookup axis
    stress_function = zeros(Float32, 200)      # placeholder stress values
    @pack_nt gpp_f_airT     ⇒ land.diagnostics
    @pack_nt tair_range_raw  ⇒ land.gppAirT
    @pack_nt stress_function ⇒ land.gppAirT
    return land
end

# Extend precompute for gppAirT_externalNN_LUT
# precompute runs once before the time loop — ideal for file I/O.
function precompute(params::gppAirT_externalNN_LUT, forcing, land, helpers)
    checkpoint      = load(joinpath(@__DIR__, "../../gpp_model.jld2"))
    tair_range_raw  = Float32.(checkpoint["tair_range_raw"])  # Tair sweep, length=200, °C
    stress_function = Float32.(checkpoint["stress_function"])  # stress ∈ [0,1], length=200
    @pack_nt tair_range_raw  ⇒ land.gppAirT
    @pack_nt stress_function ⇒ land.gppAirT
    return land
end

# Extend compute for gppAirT_externalNN_LUT
# compute runs every timestep.
# Unpack the lookup tables from land.gppAirT (set in precompute),
# linearly interpolate at the current daytime air temperature,
# clamp to [0,1], and pack the result as gpp_f_airT.
function compute(params::gppAirT_externalNN_LUT, forcing, land, helpers)
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
    @info "using gpp_f_airT from external NN Lookup Table"
    @pack_nt gpp_f_airT ⇒ land.diagnostics
    return land
end

## test the compute function

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
# choose on of the model setups
model_setup = "LUE_NN";
# model_setup = "WROASTED_HB";

# set path to experiment json file
path_experiment_json    = joinpath(@__DIR__,"..","setups",model_setup,"experiment_hybrid.json");
path_training_folds     = "";#joinpath(@__DIR__,"..","setups",model_setup,"nfolds_sites_indices.jld2");

replace_info = Dict(
    "forcing.subset.site" => getSiteNames(selected_site_indices),
    "experiment.basics.config_files.optimization" => joinpath(@__DIR__,"..","setups",model_setup,"optimization_externalNN.json"),
    "model_structure.models.gppAirT.approach" => "externalNN_LUT",
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

# run the model for the site with the default parameters
run_helpers     = prepTEM(info.models.forward, forcing, observations, info);
site_index      = 1;
site_default    = runSiteWithParams(run_helpers, info, site_index, info.optimization.parameter_table.default);

# do plots, compute some simple statistics e.g. NSE
plotSiteComparison(info, run_helpers.space_observation[site_index], (; default=site_default.output);
    file_prefix="$(model_setup)_$(sites_forcing[site_index])");
println("Outputs of plotting will be here: " * info.output.dirs.figure);
