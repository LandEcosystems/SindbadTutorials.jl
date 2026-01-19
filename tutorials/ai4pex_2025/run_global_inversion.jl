using Revise
using SindbadTutorials
using SindbadTutorials.Dates
using SindbadTutorials.Plots
toggle_type_abbrev_in_stacktrace()
include("tutorial_helpers.jl")
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
experiment_json = "../ai4pex_2025/settings_WROASTED_HB/experiment_insitu.json"
experiment_name = "WROASTED_global_inversion_CMAES"
begin_year = 1979
end_year = 2017
run_optimization = true

# experiment paths
path_input = "$(getSindbadDataDepot())/FLUXNET_v2023_12_1D.zarr"
path_observation = path_input;
path_output = ""

spinup_sequence = getSpinupSequenceSite(2000, begin_year)

replace_info = Dict("experiment.basics.time.date_begin" => "$(begin_year)-01-01",
    "experiment.basics.domain" => domain,
    "experiment.basics.name" => experiment_name,
    "experiment.basics.time.date_end" => "$(end_year)-12-31",
    "experiment.flags.run_optimization" => run_optimization,
    "experiment.model_spinup.sequence" => spinup_sequence,
    "forcing.default_forcing.data_path" => path_input,
    "forcing.subset.site" => collect(site_indices),
    "optimization.optimization_cost_method" => "CostModelObs",
    "optimization.optimization_cost_threaded"  => false,
    "optimization.algorithm_optimization" => "CMAEvolutionStrategy_CMAES_fn_global.json",

    "experiment.model_output.path" => path_output,
    "optimization.observations.default_observation.data_path" => path_observation,)


@time out_opti = runExperimentOpti(experiment_json; replace_info=replace_info, log_level=:info);
## check the docs for output at: http://sindbad-mdi.org/pages/develop/hybrid_modeling.html and http://sindbad-mdi.org/pages/develop/sindbad_outputs.html


plotTimeSeriesWithObs(out_opti)
plotPerformanceHistograms(out_opti)
plotTimeSeriesDebug(out_opti.info, out_opti.output.optimized, out_opti.output.default)

## in case inner objects are needed
info = getExperimentInfo(experiment_json; replace_info=replace_info);
forcing = getForcing(info);
run_helpers = prepTEM(forcing, info);
@time runTEM!(info.models.forward, run_helpers.space_forcing, run_helpers.space_spinup_forcing, run_helpers.loc_forcing_t, run_helpers.space_output, run_helpers.space_land, run_helpers.tem_info)
observations = getObservation(info, forcing.helpers);
obs_array = [Array(_o) for _o in observations.data]; 

cost_options = prepCostOptions(obs_array, info.optimization.cost_options) 

metricVector(run_helpers.output_array, obs_array, cost_options) # |> sum