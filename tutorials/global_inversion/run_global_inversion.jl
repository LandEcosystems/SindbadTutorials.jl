using Revise
using SindbadTutorials
using SindbadTutorials.Dates
using SindbadTutorials.Plots
using CMAEvolutionStrategy
toggle_type_abbrev_in_stacktrace()
# site_index = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

# site info

pft = [] # for all sites
pft=[2] # for sites with PFT 1 and 2
site_indices = getSiteIndicesForPFT(pft=pft)

domain="FLUXNET"
if !isempty(pft)
    domain = "$(domain)_PFT_$(join(pft, "_"))"
end
# experiment info
experiment_json = "tutorials/setups/WROASTED_HB/experiment_insitu.json"
experiment_name = "WROASTED_global_inversion_CMAES"
begin_year = 1979
end_year = 2017
run_optimization = true

# experiment paths
path_output = ""

spinup_sequence = getSpinupSequenceSite(2000, begin_year)

replace_info = Dict("experiment.basics.time.date_begin" => "$(begin_year)-01-01",
    "experiment.basics.domain" => domain,
    "experiment.basics.name" => experiment_name,
    "experiment.basics.time.date_end" => "$(end_year)-12-31",
    "experiment.flags.run_optimization" => run_optimization,
    "experiment.spinup_sequence" => spinup_sequence,
    "forcing.subset.site" => getSiteNames(site_indices),
    "optimization.optimization_cost_method" => "CostModelObs",
    "optimization.optimization_cost_threaded"  => false,
    "optimization.algorithm_optimization" => "CMAEvolutionStrategy_CMAES_fn_global.json",

    "experiment.model_output.path" => path_output,)


# the run plots the default and optimized simulations against the observations and
# the performance histograms by itself
@time out_opti = runExperimentOpti(experiment_json; replace_info=replace_info, display_level=:info);
println("Outputs of plotting will be here: " * out_opti.info.output.dirs.figure);

## in case inner objects are needed
info = getExperimentInfo(experiment_json; replace_info=replace_info);
forcing = getForcing(info);
run_helpers = prepTEM(forcing, info);
@time runTEM!(run_helpers.space_selected_models, run_helpers.space_forcing, run_helpers.space_spinup, run_helpers.loc_forcing_t, run_helpers.space_output, run_helpers.space_land, run_helpers.tem_info)
observations = getObservation(info, forcing.helpers);
obs_array = [Array(_o) for _o in observations.data]; 

cost_options = prepCostOptions(obs_array, info.optimization.cost_options) 

metricVector(run_helpers.output_array, obs_array, cost_options) # |> sum