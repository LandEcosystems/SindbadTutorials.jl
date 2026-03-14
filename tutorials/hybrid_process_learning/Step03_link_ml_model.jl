using Revise
using Flux
using JLD2
using ForwardDiff
using SindbadTutorials
using SindbadTutorials.SindbadTEM
using SindbadTutorials.Plots
import SindbadTutorials.SindbadTEM.Processes: define, precompute, compute

# Extend define for gppAirT_externalNN
function define(params::gppAirT_externalNN, forcing, land, helpers)
    @unpack_nt (o_one, z_zero) ⇐ land.constants
    model           = nothing
    μ_x             = zeros(Float32, 7)
    σ_x             = ones(Float32, 7)
    μ_y             = z_zero
    σ_y             = o_one
    input_list      = Vector{String}(undef, 0)
    sensitivity     = o_one
    @pack_nt model      ⇒ land.gppAirT
    @pack_nt μ_x        ⇒ land.gppAirT
    @pack_nt σ_x        ⇒ land.gppAirT
    @pack_nt μ_y        ⇒ land.gppAirT
    @pack_nt σ_y        ⇒ land.gppAirT
    @pack_nt input_list ⇒ land.gppAirT
    @pack_nt sensitivity ⇒ land.diagnostics
    return land
end

# Extend precompute for gppAirT_externalNN
function precompute(params::gppAirT_externalNN, forcing, land, helpers)
    checkpoint  = load(joinpath(@__DIR__, "../../gpp_model.jld2"))
    input_list  = checkpoint["input_list"]
    F_loaded    = length(input_list)
    model       = Chain(Dense(F_loaded => 64, relu), Dense(64 => 32, relu), Dense(32 => 1))

    # here model weights are stored in checkpoint["model_state"]
    Flux.loadmodel!(model, checkpoint["model_state"])
    μ_x         = Float32.(checkpoint["μ_x"])
    σ_x         = Float32.(checkpoint["σ_x"])
    μ_y         = Float32(checkpoint["μ_y"])
    σ_y         = Float32(checkpoint["σ_y"])
    tair_name = "TA_DayTime_ERAIv2_gfld"
    abs_sens_lo = Float32(checkpoint["abs_sens_lo"])
    abs_sens_hi = Float32(checkpoint["abs_sens_hi"])

    @pack_nt tair_name    ⇒ land.gppAirT
    @pack_nt model        ⇒ land.gppAirT
    @pack_nt μ_x          ⇒ land.gppAirT
    @pack_nt σ_x          ⇒ land.gppAirT
    @pack_nt μ_y          ⇒ land.gppAirT
    @pack_nt σ_y          ⇒ land.gppAirT
    @pack_nt input_list   ⇒ land.gppAirT
    @pack_nt abs_sens_lo  ⇒ land.gppAirT
    @pack_nt abs_sens_hi  ⇒ land.gppAirT
    return land
end

# Extend compute for gppAirT_externalNN
function compute(params::gppAirT_externalNN, forcing, land, helpers)
    @unpack_nt model        ⇐ land.gppAirT
    @unpack_nt μ_x          ⇐ land.gppAirT
    @unpack_nt σ_x          ⇐ land.gppAirT
    @unpack_nt input_list   ⇐ land.gppAirT
    @unpack_nt abs_sens_lo  ⇐ land.gppAirT
    @unpack_nt abs_sens_hi  ⇐ land.gppAirT
    @unpack_nt (o_one, z_zero) ⇐ land.constants
    @unpack_nt tair_name     ⇐ land.gppAirT
    @unpack_nt f_airT_day    ⇐ forcing

    tair_idx  = findfirst(==(tair_name), input_list)
    x = zeros(Float32, length(input_list))
    for (i, name) in enumerate(input_list)
        if haskey(forcing, Symbol(name))
            x[i] = Float32(forcing[Symbol(name)])
        else
            x[i] = z_zero
        end
    end
    x_n = vec((x .- μ_x) ./ σ_x)
    # Use f_airT_day from forcing, normalize it
    tair_raw = f_airT_day
    tair_n = (tair_raw - μ_x[tair_idx]) / σ_x[tair_idx]
    function gpp_vs_tair(tair_n_val::T) where T<:Real
        x_t = Vector{T}(x_n)
        x_t[tair_idx] = tair_n_val
        return only(model(x_t))
    end
    sens = abs(ForwardDiff.derivative(gpp_vs_tair, tair_n))
    sens_norm = (sens - abs_sens_lo) / (abs_sens_hi - abs_sens_lo)
    gpp_f_airT = clamp(sens_norm, z_zero, o_one)
    @pack_nt gpp_f_airT ⇒ land.diagnostics
    return land
end

# --- Experiment setup ---
selected_site_indices = getSiteIndicesForHybrid()
do_random = 0
if do_random > 0
    Random.seed!(1234)
    selected_site_indices = first(shuffle(selected_site_indices), do_random)
end

path_experiment_json = joinpath(@__DIR__, "..", "setups", "WROASTED_HB", "experiment_hybrid.json")
path_setups = joinpath(@__DIR__, "..", "setups", "WROASTED_HB")

function build_replace_info(model_structure_file, optimization_file)
    return Dict(
        "forcing.subset.site" => selected_site_indices,
        "experiment.basics.config_files.model_structure" => joinpath(path_setups, model_structure_file),
        "experiment.basics.config_files.optimization" => joinpath(path_setups, optimization_file),
        "optimization.optimization_cost_threaded" => false,
        "optimization.optimization_parameter_scaling" => nothing,
        "hybrid.ml_training.fold_path" => nothing,
    )
end

function load_experiment(model_structure_file, optimization_file)
    replace_info = build_replace_info(model_structure_file, optimization_file)
    info = getExperimentInfo(path_experiment_json; replace_info=deepcopy(replace_info))
    forcing = getForcing(info)
    observations = getObservation(info, forcing.helpers)
    return info, forcing, observations
end

# --- Run the model for the site with the default parameters ---
function run_model_param_sensitivity(info, forcing, observations, site_index, loc_params)
    run_helpers = prepTEM(info.models.forward, forcing, observations, info)
    params = loc_params
    selected_models = info.models.forward
    parameter_scaling_type = info.optimization.run_options.parameter_scaling
    tbl_params = info.optimization.parameter_table
    param_to_index = getParameterIndices(selected_models, tbl_params)
    models = updateModels(params, param_to_index, parameter_scaling_type, selected_models)
    loc_forcing = run_helpers.space_forcing[site_index]
    loc_spinup_forcing = run_helpers.space_spinup_forcing[site_index]
    loc_forcing_t = run_helpers.loc_forcing_t
    loc_output = getCacheFromOutput(run_helpers.space_output[site_index], info.hybrid.ml_gradient.method)
    gradient_lib = info.hybrid.ml_gradient.method
    loc_output_from_cache = getOutputFromCache(loc_output, params, gradient_lib)
    land_init = deepcopy(run_helpers.loc_land)
    tem_info = run_helpers.tem_info
    loc_obs = run_helpers.space_observation[site_index]
    loc_cost_option = prepCostOptions(loc_obs, info.optimization.cost_options)
    constraint_method = info.optimization.run_options.multi_constraint_method
    coreTEM!(
        models,
        loc_forcing,
        loc_spinup_forcing,
        loc_forcing_t,
        loc_output_from_cache,
        land_init,
        tem_info)
    forward_output = (; Pair.(getUniqueVarNames(info.output.variables), loc_output_from_cache)...)
    loss_vector = SindbadTutorials.metricVector(loc_output_from_cache, loc_obs, loc_cost_option)
    t_loss = combineMetric(loss_vector, constraint_method)
    return forward_output, loss_vector, t_loss
end

function extract_site_series(output, var_name::Symbol)
    data = getproperty(output, var_name)
    if ndims(data) == 4
        return Array(data[:, 1, 1, 1])
    elseif ndims(data) == 3
        return Array(data[:, 1, 1])
    elseif ndims(data) == 2
        return Array(data[:, 1])
    end
    return Array(data)
end

function compare_model_structures(site_index=1)
    info_external, forcing_external, observations_external =
        load_experiment("model_structure_externalNN.json", "optimization_externalNN.json")
    info_standard, forcing_standard, observations_standard =
        load_experiment("model_structure.json", "optimization.json")

    output_external, _, _ = run_model_param_sensitivity(
        info_external,
        forcing_external,
        observations_external,
        site_index,
        info_external.optimization.parameter_table.default,
    )
    output_standard, _, _ = run_model_param_sensitivity(
        info_standard,
        forcing_standard,
        observations_standard,
        site_index,
        info_standard.optimization.parameter_table.default,
    )

    xdata = collect(info_external.helpers.dates.range)
    site_name = forcing_external.data[1].site[site_index]

    gpp_external = extract_site_series(output_external, :gpp)
    gpp_standard = extract_site_series(output_standard, :gpp)
    gpp_f_airT_external = extract_site_series(output_external, :gpp_f_airT)
    gpp_f_airT_standard = extract_site_series(output_standard, :gpp_f_airT)

    default(titlefont=(16, "times"), legendfontsize=11, tickfont=(10, :black))

    p_gpp = plot(
        xdata,
        gpp_external;
        label="externalNN",
        color=:firebrick,
        lw=2,
        xlabel="Date",
        ylabel="GPP",
        title="GPP comparison at $(site_name)",
        size=(1500, 500),
        left_margin=1Plots.cm,
    )
    plot!(p_gpp, xdata, gpp_standard; label="standard gppAirT", color=:steelblue, lw=2, ls=:dash)

    p_gpp_f_airT = plot(
        xdata,
        gpp_f_airT_external;
        label="externalNN",
        color=:firebrick,
        lw=2,
        xlabel="Date",
        ylabel="gpp_f_airT",
        title="gpp_f_airT comparison at $(site_name)",
        ylim=(0, 1.05),
        size=(1500, 500),
        left_margin=1Plots.cm,
    )
    plot!(p_gpp_f_airT, xdata, gpp_f_airT_standard; label="standard gppAirT", color=:steelblue, lw=2, ls=:dash)

    fig_dir = info_external.output.dirs.figure
    savefig(p_gpp, joinpath(fig_dir, "compare_gpp_externalNN_vs_standard_$(site_name).png"))
    savefig(p_gpp_f_airT, joinpath(fig_dir, "compare_gpp_f_airT_externalNN_vs_standard_$(site_name).png"))

    return (;
        site_name,
        info_external,
        info_standard,
        output_external,
        output_standard,
        p_gpp,
        p_gpp_f_airT,
    )
end

site_index = 1;
comparison = compare_model_structures(site_index);

