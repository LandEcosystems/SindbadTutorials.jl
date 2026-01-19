
# ================================== using tools ==================================================
# some of the things that will be using... Julia tools, SINDBAD tools, local codes...
using Revise
using SindbadTutorials
using SindbadTutorials.Dates
using SindbadTutorials.Plots
using SindbadTutorials.SindbadVisuals
toggle_type_abbrev_in_stacktrace()
include("tutorial_helpers.jl")

# ================================== get data / set paths ========================================= 
# data to be used can be found here: https://nextcloud.bgc-jena.mpg.de/s/w2mbH59W4nF3Tcd
# organizing the paths of data sources and outputs for this experiment
path_input_dir      = getSindbadDataDepot(; env_data_depot_var="SINDBAD_DATA_DEPOT", 
                    local_data_depot="/Users/skoirala/research/RnD/data/SindbadTutorialsData"); # for convenience, the data file is set within the SINDBAD-Tutorials path; this needs to be changed otherwise.
path_input          = joinpath("$(path_input_dir)","FLUXNET_v2023_12_1D_REPLACED_Noise003_v1.zarr"); # zarr data source containing all the data necessary for the exercise
path_observation    = path_input; # observations (synthetic or otherwise) are included in the same file
path_output         = "";

# ================================== selecting a site =============================================
# there is a collection of several sites in the data files site info; #68 is DE-Hai
site_index      = 68;
domain, y_dist  = getSiteInfo(site_index);

# ================================== setting up the experiment ====================================
# experiment is all set up according to a (collection of) json file(s)
experiment_json     = joinpath(@__DIR__,"settings_WROASTED_HB","experiment_insitu.json");
experiment_name     = "WROASTED_inversion_CMAES";
begin_year          = 1979;
end_year            = 2017;
run_optimization    = true;
isfile(experiment_json) ? nothing : println("Hmmm... does not exist : $(experiment_json)");

# setting up the model spinup sequence : can change according to the site...
spinup_sequence = getSpinupSequenceSite(y_dist, begin_year);

# default setting in experiment_json will be replaced by the "replace_info"
replace_info = Dict("experiment.basics.time.date_begin" => "$(begin_year)-01-01",
    "experiment.basics.domain" => domain,
    "experiment.basics.name" => experiment_name,
    "experiment.basics.time.date_end" => "$(end_year)-12-31",
    "experiment.flags.run_optimization" => run_optimization,
    "experiment.model_spinup.sequence" => spinup_sequence,
    "forcing.default_forcing.data_path" => path_input,
    "forcing.subset.site" => [site_index],
    "experiment.model_output.path" => path_output,
    "optimization.observations.default_observation.data_path" => path_observation,
    );

# ================================== forward run ================================================== 
# before running the optimization, check a forward run 
@time out_dflt  = runExperimentForward(experiment_json; replace_info=deepcopy(replace_info)); # full default model

## check the docs for output at: http://sindbad-mdi.org/pages/develop/hybrid_modeling.html and http://sindbad-mdi.org/pages/develop/sindbad_outputs.html

# access some of the internals to do some plots with the forward runs...
info            = getExperimentInfo(experiment_json; replace_info=deepcopy(replace_info)); # note that this will modify information from json with the replace_info
forcing         = getForcing(info); 
run_helpers     = prepTEM(forcing, info); # not needed now
observations    = getObservation(info, forcing.helpers);
obs_array       = [Array(_o) for _o in observations.data]; 
cost_options    = prepCostOptions(obs_array, info.optimization.cost_options);

# plot the default simulations
plotTimeSeriesWithObs(out_dflt,obs_array,cost_options);
println("Outputs of plotting will be here: " * info.output.dirs.figure);

# ================================== optimization ================================================= 
# run the optimization according to the settings above... can take some time...
@time out_opti  = runExperimentOpti(experiment_json; replace_info=deepcopy(replace_info), log_level=:info);

# plot the results
plotTimeSeriesWithObs(out_opti);
plotTimeSeriesDebug(out_opti.info, out_opti.output.optimized, out_opti.output.default);
println("Outputs of plotting will be here: " * info.output.dirs.figure);

# ================================== another model ================================================ 
# all of the above with another model...
# only spin up the moisture pools
spinup_sequence = getSpinupSequenceSite();

# just change the model setup and experiment name
experiment_json = joinpath(@__DIR__,"settings_LUE","experiment.json");
experiment_name = "LUE_inversion_CMAES";
replace_info    = Dict("experiment.basics.time.date_begin" => "$(begin_year)-01-01",
    "experiment.basics.domain" => domain,
    "experiment.basics.name" => experiment_name,
    "experiment.basics.time.date_end" => "$(end_year)-12-31",
    "experiment.flags.run_optimization" => run_optimization,
    "experiment.model_spinup.sequence" => spinup_sequence,
    "forcing.default_forcing.data_path" => path_input,
    "forcing.subset.site" => [site_index],
    "experiment.model_output.path" => path_output,
    "optimization.observations.default_observation.data_path" => path_observation,
    );

# #=
@time out_dflt_lue  = runExperimentForward(experiment_json; replace_info=deepcopy(replace_info)); # full default model
# access some of the internals to do some plots with the forward runs...
info            = getExperimentInfo(experiment_json; replace_info=deepcopy(replace_info)); # note that this will modify information from json with the replace_info
forcing         = getForcing(info); 
run_helpers     = prepTEM(forcing, info); # not needed now
observations    = getObservation(info, forcing.helpers);
obs_array       = [Array(_o) for _o in observations.data]; 
cost_options    = prepCostOptions(obs_array, info.optimization.cost_options);
# =#

# plot the default simulations
plotTimeSeriesWithObs(out_dflt_lue,obs_array,cost_options);
println("Outputs of plotting will be here: " * info.output.dirs.figure);

# run the optimization
@time out_lue_opti  = runExperimentOpti(experiment_json; replace_info=deepcopy(replace_info), log_level=:info);

# plot the results
plotTimeSeriesWithObs(out_lue_opti);
println("Outputs of plotting will be here: " * out_lue_opti.info.output.dirs.figure);

# ================================== time for discussion ========================================== 
