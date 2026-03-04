using Revise
using SindbadTutorials
toggle_type_abbrev_in_stacktrace()
# site_index = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

# site info
site_index_begin = 1
site_index_end = 205

domain = "FLUXNET"
# experiment info
experiment_json = "../ellis_jena_2025/settings_WROASTED_HB/experiment_insitu.json"
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
    "forcing.subset.site" => collect(site_index_begin:site_index_end),
    "optimization.optimization_cost_method" => "CostModelObs",
    "optimization.optimization_cost_threaded" => false,
    "optimization.algorithm_optimization" => "CMAEvolutionStrategy_CMAES_fn_global.json", "experiment.model_output.path" => path_output,
    "optimization.observations.default_observation.data_path" => path_observation,)

info = getExperimentInfo(experiment_json; replace_info=replace_info);
forcing = getForcing(info);

pfts = forcing.data[findall(x -> x == :f_pft, forcing.variables)[1]]
pft_sites = pfts.site
site_info = CSV.File(joinpath(@__DIR__,
        "settings_WROASTED_HB/site_names_disturbance.csv");
    header=true)

header = "site_name,disturbance,pft\n"
open(joinpath(@__DIR__, "settings_WROASTED_HB/site_names_disturbance_pft.csv"), "w") do io
    write(io, header)
    foreach(site_info) do site
        site_name = string(site[1])
        disturbance = string(site[2])
        site_line = "$site_name,$disturbance,20"
        if site_name in pft_sites
            site_index_pft = findfirst(x -> x == site_name, pft_sites)
            site_pft = pfts[site_index_pft]
            site_line = "$site_name,$disturbance,$site_pft\n"
            write(io, site_line)
            println("$site_name $site_pft")
        end
    end

end

