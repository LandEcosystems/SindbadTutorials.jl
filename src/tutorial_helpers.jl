
export getSiteInfo, getSiteNames, getSiteIndicesForHybrid,
getSpinupSequenceSite, getSiteIndicesForPFT, getSiteIndicesForHybridML, runSiteWithParams, plotSiteComparison
using SindbadTutorials

const missing_sites_in_data = [ "FI-Lom", "FI-Sod", "GL-ZaF", "GL-ZaH", "RU-Che", "RU-Cok", "RU-Sam", "RU-Tks", "RU-Vrk", "SE-St1", "SJ-Adv", "SJ-Blv", "US-Atq", "US-Ivo", "CA-NS3", "IT-CA1", "IT-SR2", "US-ARb", "US-GBT", "US-Tw1", "US-UMd" ]

function getSiteInfo(site_index)
    site_info = CSV.File(joinpath(@__DIR__, "../tutorials/setups/", "WROASTED_HB/site_names_disturbance.csv"); header=true)
    domain = string(site_info[site_index][1])
    y_dist = string(site_info[site_index][2])
    return domain, y_dist
end

"""
    getSiteNames(site_indices)

Returns the names of the sites at the given positions of the site list. Sindbad subsets
the `site` dimension by value, so `forcing.subset.site` needs these names, not indices.
"""
function getSiteNames(site_indices)
    site_info = CSV.File(joinpath(@__DIR__, "../tutorials/setups/", "WROASTED_HB/site_names_disturbance.csv"); header=true)
    return [string(site_info[i][1]) for i in site_indices]
end

function getSiteIndicesForHybrid(; exclude_sites=missing_sites_in_data)
    site_info = CSV.File(joinpath(@__DIR__, "../tutorials/setups/", "WROASTED_HB/site_names_disturbance_pft.csv"); header=true)
    site_indices = 1:length(site_info)
    if !isempty(exclude_sites)
        site_indices = findall(∉(exclude_sites), site_info.site_name)
    end
    return site_indices
end

function getSiteIndicesForPFT(; pft=[])
    site_info = CSV.File(joinpath(@__DIR__, "../tutorials/setups/", "WROASTED_HB/site_names_disturbance_pft.csv"); header=true)
    site_indices = 1:length(site_info)
    if !isempty(pft)
        site_indices = findall(in(pft), site_info.pft)
    end
    return site_indices
end

function getSpinupSequenceSite(y_dist, begin_year; nrepeat=200)
    nrepeat_d = nothing
    if y_dist != "undisturbed"
        y_disturb = year(Date(y_dist))
        y_start = begin_year
        nrepeat_d = y_start - y_disturb
    end
    if isnothing(nrepeat_d) || nrepeat_d < 0
        sequence = [
            Dict("spinup_mode" => "loop_TEM", "models" => "spinup", "forcing" => "all_years", "n_repeat" => 1),
            Dict("spinup_mode" => "loop_TEM", "models" => "spinup", "forcing" => "day_MSC", "n_repeat" => nrepeat),
            Dict("spinup_mode" => "eta_scale_AHCWD", "forcing" => "day_MSC", "n_repeat" => 1),
        ]
    elseif nrepeat_d == 0
        sequence = [
            Dict("spinup_mode" => "loop_TEM", "models" => "spinup", "forcing" => "all_years", "n_repeat" => 1),
            Dict("spinup_mode" => "loop_TEM", "models" => "spinup", "forcing" => "day_MSC", "n_repeat" => nrepeat),
            Dict("spinup_mode" => "eta_scale_A0HCWD", "forcing" => "day_MSC", "n_repeat" => 1),
        ]
    elseif nrepeat_d > 0
        sequence = [
            Dict("spinup_mode" => "loop_TEM", "models" => "spinup", "forcing" => "all_years", "n_repeat" => 1),
            Dict("spinup_mode" => "loop_TEM", "models" => "spinup", "forcing" => "day_MSC", "n_repeat" => nrepeat),
            Dict("spinup_mode" => "eta_scale_A0HCWD", "forcing" => "day_MSC", "n_repeat" => 1),
            Dict("spinup_mode" => "loop_TEM", "models" => "spinup", "forcing" => "day_MSC", "n_repeat" => nrepeat_d),
        ]
    else
        error("cannot determine the repeat for disturbance")
    end
    return sequence
end

function getSpinupSequenceSite(;nrepeat=5)
    sequence = [
        Dict("spinup_mode" => "loop_TEM", "models" => "spinup", "forcing" => "all_years", "n_repeat" => 1),
        Dict("spinup_mode" => "loop_TEM", "models" => "spinup", "forcing" => "day_MSC", "n_repeat" => nrepeat),
    ]
    return sequence
end

"""
    runSiteWithParams(run_helpers, info, site_index, params)

Runs the model for one site with the given parameter vector and returns its output and
loss. It does the same as one site in the loss functions of `prepHybrid`, so a hybrid
experiment can be checked site by site with the default or the learned parameters.

# Arguments
- `run_helpers`: from `prepTEM(info.models.forward, forcing, observations, info)`
- `info`: the experiment info
- `site_index`: the position of the site in the forcing
- `params`: the parameter vector, in the order of `info.optimization.parameter_table`

# Returns
- a NamedTuple with the `output` of every output variable, the `loss_vector` with one
  loss per observed variable and the combined `loss`

# Notes
- The output is copied, so a later run of the same site does not overwrite it.
"""
function runSiteWithParams(run_helpers, info, site_index, params)
    models = info.models.forward
    param_to_index = getParameterIndices(models, info.optimization.parameter_table)
    models = updateModels(params, param_to_index, info.optimization.run_options.parameter_scaling, models)
    gradient_lib = info.hybrid.ml_gradient.method
    loc_output = Sindbad.MachineLearning.getCacheFromOutput(run_helpers.space_output[site_index], gradient_lib)
    loc_output = Sindbad.MachineLearning.getOutputFromCache(loc_output, params, gradient_lib)
    loc_obs = run_helpers.space_observation[site_index]
    cost_options = prepCostOptions(loc_obs, info.optimization.cost_options)
    coreTEM!(models,
        run_helpers.space_forcing[site_index],
        run_helpers.space_spinup[site_index],
        run_helpers.loc_forcing_t,
        loc_output,
        deepcopy(run_helpers.loc_land),
        run_helpers.tem_info)
    loss_vector = metricVector(loc_output, loc_obs, cost_options)
    loss = combineMetric(loss_vector, info.optimization.run_options.multi_constraint_method)
    output = (; Pair.(getUniqueVarNames(info.output.variables), deepcopy.(loc_output))...)
    return (; output, loss_vector, loss)
end

"""
    plotSiteComparison(info, loc_obs, outputs; file_prefix="site")

Plots the observations of one site with one or more model runs of that site, one figure
per observed variable, and saves them in `info.output.dirs.figure`.

# Arguments
- `info`: the experiment info
- `loc_obs`: the observations of the site, e.g. `run_helpers.space_observation[site_index]`
- `outputs`: a NamedTuple of the model outputs to compare, with the labels as keys, e.g.
  `(; default=site_default.output, optimized=site_optimized.output)`
- `file_prefix`: the start of the names of the figure files

# Notes
- The legend shows the metric of every run over the period with observations. NNSE and
  its inverse are shown as NSE, which is easier to read.
"""
function plotSiteComparison(info, loc_obs, outputs::NamedTuple; file_prefix="site")
    first_series(x) = x isa AbstractVector ? x : x[:, 1]
    colors = (:steelblue2, :seagreen3, :firebrick, :darkorange)
    cost_options = prepCostOptions(loc_obs, info.optimization.cost_options)
    foreach(cost_options) do var_row
        vinfo = getVariableInfo((var_row.mod_field, var_row.mod_subfield), info.experiment.basics.temporal_resolution)
        println("plot obs::", var_row.variable)
        loss_metric = var_row.cost_metric
        if nameof(typeof(loss_metric)) in (:NNSE, :NNSEInv, :NSEInv)
            loss_metric = NSE()
        end
        y, yσ, _ = getData(first(outputs), loc_obs, var_row)
        y, yσ = first_series(y), first_series(yσ)
        valid = findall(!isnan, y)
        tspan = length(valid) < 2 ? eachindex(y) : (first(valid):last(valid))
        xdata = info.helpers.dates.range[tspan]
        title = "$(vinfo["long_name"]) ($(vinfo["units"])) -> $(nameof(typeof(loss_metric)))"
        p = plot(xdata, y[tspan]; label="obs", seriestype=:scatter, mc=:black, ms=4, lw=0, ma=0.65,
            left_margin=1Plots.cm, legend=:outerbottom, legendcolumns=length(outputs) + 1,
            size=(2000, 1000), title=title)
        for (i, (label, output)) in enumerate(pairs(outputs))
            ŷ = first_series(getData(output, loc_obs, var_row)[3])
            y_n, yσ_n, ŷ_n = getDataWithoutNaN(y[tspan], yσ[tspan], ŷ[tspan])
            m = metric(loss_metric, ŷ_n, y_n, yσ_n)
            plot!(p, xdata, ŷ[tspan]; color=colors[mod1(i, length(colors))], lw=1.5, ls=:dash,
                label="$(label) ($(round(m, digits=2)))")
        end
        savefig(p, joinpath(info.output.dirs.figure, "$(file_prefix)_$(vinfo["standard_name"]).png"))
    end
    return nothing
end
