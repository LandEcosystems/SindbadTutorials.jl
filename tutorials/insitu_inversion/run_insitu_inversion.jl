
# ================================== using tools ==================================================
# some of the things that will be using... Julia tools, SINDBAD tools, local codes...
using Revise
using SindbadTutorials
using SindbadTutorials.Sindbad
using SindbadTutorials.Sindbad.Setup.Dates
using Plots
using SindbadTutorials.Sindbad.Visualization
using CMAEvolutionStrategy
toggle_type_abbrev_in_stacktrace()

# ================================== get data / set paths ========================================= 
path_output         = "./";

# ================================== selecting a site =============================================
# there is a collection of several sites in the data files site info; #68 is DE-Hai
site_index      = 68;
domain, y_dist  = getSiteInfo(site_index);

# ================================== setting up the experiment ====================================
# experiment is all set up according to a (collection of) json file(s)
experiment_json     = joinpath(@__DIR__,"../setups/WROASTED_HB","experiment_insitu.json");
experiment_name     = "WROASTED_inversion_CMAES";
begin_year          = 1979;
end_year            = 2017;
run_optimization    = true;
isfile(experiment_json) ? nothing : println("Hmmm... does not exist : $(experiment_json)");

# setting up the model spinup sequence : can change according to the site...
# spinup_sequence = getSpinupSequenceSite(y_dist, begin_year);

# default setting in experiment_json will be replaced by the "replace_info"
replace_info = Dict("experiment.basics.time.date_begin" => "$(begin_year)-01-01",
    "experiment.basics.domain" => domain,
    "experiment.basics.name" => experiment_name,
    "experiment.basics.time.date_end" => "$(end_year)-12-31",
    "experiment.flags.run_optimization" => run_optimization,
    # "experiment.spinup_sequence" => spinup_sequence,
    "forcing.subset.site" => [domain],
    "experiment.model_output.path" => path_output,
    );

# ================================== forward run ================================================== 
# before running the optimization, check a forward run with the default parameters.
# the run saves the output and plots the forcing and the simulated time series.
@time out_dflt  = runExperimentForward(experiment_json; replace_info=deepcopy(replace_info));

## check the docs for output at: http://sindbad-mdi.org/pages/develop/hybrid_modeling.html and http://sindbad-mdi.org/pages/develop/sindbad_outputs.html

# a forward run does not read the observations, so read them to compare the default
# simulations with the observations
info            = getExperimentInfo(experiment_json; replace_info=deepcopy(replace_info));
forcing         = getForcing(info);
observations    = getObservation(info, forcing.helpers);
obs_array       = [Array(_o) for _o in observations.data];
cost_options    = prepCostOptions(obs_array, info.optimization.cost_options);
plotTimeSeries((; info, forcing, observation=obs_array, output=out_dflt.output), cost_options);
println("Outputs of plotting will be here: " * info.output.dirs.figure);

# ================================== optimization =================================================
# run the optimization according to the settings above... can take some time...
# the run plots the default and optimized simulations against the observations and
# the performance histograms by itself
@time out_opti  = runExperimentOpti(experiment_json; replace_info=deepcopy(replace_info), display_level=:info);
println("Outputs of plotting will be here: " * out_opti.info.output.dirs.figure);

# ================================== another model ================================================ 
# all of the above with another model...
# only spin up the moisture pools
spinup_sequence = getSpinupSequenceSite();

# just change the model setup and experiment name
experiment_json = joinpath(@__DIR__,"../setups/LUE","experiment.json");
experiment_name = "LUE_inversion_CMAES";
replace_info    = Dict("experiment.basics.time.date_begin" => "$(begin_year)-01-01",
    "experiment.basics.domain" => domain,
    "experiment.basics.name" => experiment_name,
    "experiment.basics.time.date_end" => "$(end_year)-12-31",
    "experiment.flags.run_optimization" => run_optimization,
    "experiment.spinup_sequence" => spinup_sequence,
    "forcing.subset.site" => [domain],
    "experiment.model_output.path" => path_output,
    );

@time out_dflt_lue  = runExperimentForward(experiment_json; replace_info=deepcopy(replace_info));

# plot the default simulations against the observations
info            = getExperimentInfo(experiment_json; replace_info=deepcopy(replace_info));
forcing         = getForcing(info);
observations    = getObservation(info, forcing.helpers);
obs_array       = [Array(_o) for _o in observations.data];
cost_options    = prepCostOptions(obs_array, info.optimization.cost_options);
plotTimeSeries((; info, forcing, observation=obs_array, output=out_dflt_lue.output), cost_options);
println("Outputs of plotting will be here: " * info.output.dirs.figure);

# run the optimization, which makes its own plots
@time out_lue_opti  = runExperimentOpti(experiment_json; replace_info=deepcopy(replace_info), display_level=:info);
println("Outputs of plotting will be here: " * out_lue_opti.info.output.dirs.figure);

# ================================== time for discussion ========================================== 
