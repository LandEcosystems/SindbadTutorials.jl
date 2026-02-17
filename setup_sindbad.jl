#!/usr/bin/env julia
"""
Setup helper to configure the `SindbadTutorials` environment for development.

In dev mode:
- Clones all packages (Sindbad, ErrorMetrics, TimeSamplers, OmniTools) to dev/
- Develops them in the SindbadTutorials environment
- Also develops ErrorMetrics, TimeSamplers, OmniTools within Sindbad.jl's environment
- Any code changes in dev/* are immediately available via reload/reimport

Workflow:
  julia setup_sindbad.jl    # sets up dev mode for both environments
  julia
  julia> using Revise
  julia> using SindbadTutorials
  [edit code in dev/Sindbad.jl, dev/ErrorMetrics.jl, etc.]
  julia> includet("your_script.jl")  # auto-reload on changes

Configuration is read from `SindbadSetup.toml` if present.
"""

using Pkg
import TOML

const SETTINGS_FILE = "SindbadSetup.toml"

struct DevPkg
    name::String
    default_rel_path::String
    git_url::Union{String,Nothing}
end

const DEV_PACKAGES = [
    DevPkg("Sindbad",      "dev/Sindbad.jl",        "https://github.com/LandEcosystems/Sindbad.jl.git"),
    DevPkg("ErrorMetrics", "dev/ErrorMetrics.jl",   "https://github.com/LandEcosystems/ErrorMetrics.jl.git"),
    DevPkg("TimeSamplers", "dev/TimeSamplers.jl",   "https://github.com/LandEcosystems/TimeSamplers.jl.git"),
    DevPkg("OmniTools",    "dev/OmniTools.jl",      "https://github.com/LandEcosystems/OmniTools.jl.git"),
]

function load_settings()
    if isfile(SETTINGS_FILE)
        println("Loading settings from $(abspath(SETTINGS_FILE))")
        return TOML.parsefile(SETTINGS_FILE)
    else
        println("No settings file found ($SETTINGS_FILE); using defaults.")
        return Dict{String,Any}()
    end
end

function dev_path_for(pkg::DevPkg)
    return pkg.default_rel_path
end

function get_git_url(settings::Dict{String,Any}, pkg::DevPkg)
    pkg_settings = get(settings, pkg.name, Dict{String,Any}())
    git_url = get(pkg_settings, "git_url", pkg.git_url)
    return git_url === nothing ? nothing : String(git_url)
end

function enable_dev_mode(settings)
    println("\nEnabling DEV mode for Sindbad ecosystem packages...")
    
    # First: Clone all packages (so all paths exist)
    println("\n" * "=" ^ 60)
    println("STEP 1: Cloning packages...")
    println("=" ^ 60)
    
    for pkg in DEV_PACKAGES
        path = dev_path_for(pkg)
        git_url = get_git_url(settings, pkg)

        println("\n📦 $(pkg.name)...")

        if isdir(path)
            println("   ⚠️  Already cloned at `$path`")
            # Check for unstaged changes and prompt for update
            cd(path) do
                try
                    status = read(`git status --porcelain`, String)
                    if !isempty(status)
                        println("   ❌ Unstaged changes detected in $path. Please commit or stash them before updating.")
                        return
                    end
                catch e
                    println("   ❌ Error checking git status: $e")
                    return
                end
                # Prompt user to update
                print("   ❓ Do you want to update (git pull) $path? [y/N]: ")
                answer = readline()
                if lowercase(strip(answer)) in ["y", "yes"]
                    try
                        run(`git pull`)
                        println("   ✅ Updated $path from remote.")
                    catch e
                        println("   ❌ Error pulling updates: $e")
                    end
                else
                    println("   ℹ️  Skipping update for $path.")
                end
            end
        elseif git_url !== nothing
            # Need to clone the repo into the path
            println("   🔄 Cloning from $git_url...")
            # Create parent directories if needed
            parent_dir = dirname(path)
            if !isempty(parent_dir) && !isdir(parent_dir)
                mkpath(parent_dir)
            end
            try
                run(`git clone $git_url $path`)
                println("   ✅ Successfully cloned")
            catch e
                println("   ❌ Error cloning: $e")
            end
        else
            println("   ⚠️  No git URL configured, skipping clone")
        end
    end

    # Second: Develop all packages (now that all paths exist)
    println("\n" * "=" ^ 60)
    println("STEP 2: Developing packages...")
    println("=" ^ 60)
    
    for pkg in DEV_PACKAGES
        path = dev_path_for(pkg)
        
        if !isdir(path)
            println("\n⚠️  $(pkg.name): path does not exist at `$path`, skipping")
            continue
        end

        println("\n📦 $(pkg.name): developing from `$path`...")
        try
            Pkg.develop(path=path)
            println("   ✅ Successfully developed")
        catch e
            println("   ❌ Error: $e")
        end
    end

    # Third: Set up Sindbad.jl's dev dependencies
    setup_sindbad_dev_dependencies()
end

function setup_sindbad_dev_dependencies()
    println("\n" * "=" ^ 60)
    println("Setting up Sindbad.jl dev dependencies...")
    println("=" ^ 60)

    sindbad_path = "dev/Sindbad.jl"
    
    if !isdir(sindbad_path)
        println("⚠️  Sindbad.jl not found at $sindbad_path, skipping dev setup")
        return
    end

    sindbad_dev_deps = [
        ("ErrorMetrics", "dev/ErrorMetrics.jl"),
        ("TimeSamplers", "dev/TimeSamplers.jl"),
        ("OmniTools",    "dev/OmniTools.jl"),
        ("SindbadTEM",   "dev/Sindbad.jl/SindbadTEM"),
    ]

    # Temporarily activate Sindbad.jl environment
    original_env = Base.active_project()
    try
        Pkg.activate(sindbad_path)
        println("📦 Activated Sindbad.jl environment")

        for (dep_name, dep_path) in sindbad_dev_deps
            abs_dep_path = abspath(dep_path)
            
            if !isdir(abs_dep_path)
                println("   ⚠️  $dep_name not found at $abs_dep_path")
                continue
            end

            println("\n   📝 Developing $dep_name from $(dep_path)...")
            try
                Pkg.develop(path=abs_dep_path)
                println("      ✅ $dep_name developed successfully")
            catch e
                println("      ❌ Error developing $dep_name: $e")
            end
        end

        println("\n✨ Sindbad.jl dependencies configured for local development")
    finally
        # Restore original environment
        Pkg.activate(original_env)
    end
end

function enable_run_mode()
    println("\nEnabling RUN (registry) mode...")
    println("→ Removing local dev packages and installing from registry.")
    println("=" ^ 60)

    # Remove any dev packages that were set up
    dev_pkg_names = ["Sindbad", "ErrorMetrics", "TimeSamplers", "OmniTools"]
    
    println("\n📝 Removing dev packages...")
    for pkg_name in dev_pkg_names
        try
            Pkg.rm(pkg_name)
            println("   ✅ Removed $pkg_name")
        catch e
            # Likely not in dev mode, which is fine
            println("   ℹ️  $pkg_name not in dev mode (or already removed)")
        end
    end

    # Add Sindbad from registry (which will also add its dependencies)
    println("\n📝 Installing Sindbad from registry...")
    try
        Pkg.add("Sindbad")
        println("   ✅ Sindbad installed from registry (with dependencies)")
    catch e
        println("   ❌ Error installing Sindbad: $e")
    end

    println("\n" * "=" ^ 60)
    println("✨ Run mode enabled")
    println("   All packages are now using registry versions")
end

function test_dev_setup()
    println("\n" * "=" ^ 60)
    println("TESTING DEV MODE SETUP")
    println("=" ^ 60)

    println("\n✓ Checking package status...")
    
    all_ok = true
    
    # Check manifest for dev entries
    manifest_path = joinpath("Manifest.toml")
    if isfile(manifest_path)
        manifest_content = read(manifest_path, String)
        
        for pkg in DEV_PACKAGES
            if contains(manifest_content, "path = ") && contains(manifest_content, pkg.name)
                println("  ✅ $(pkg.name): in dev mode")
            else
                println("  ⚠️  $(pkg.name): not found as dev package")
                all_ok = false
            end
        end
    else
        println("  ⚠️  Manifest.toml not found")
        all_ok = false
    end

    println("\n✓ Checking dev folders exist...")
    for pkg in DEV_PACKAGES
        dev_path = pkg.default_rel_path
        
        if isdir(dev_path)
            println("  ✅ $dev_path exists")
        else
            println("  ⚠️  $dev_path not found")
            all_ok = false
        end
    end

    println("\n✓ Checking Sindbad.jl dev environment...")
    sindbad_project = joinpath("dev", "Sindbad.jl", "Project.toml")
    if isfile(sindbad_project)
        println("  ✅ Sindbad.jl/Project.toml exists")
    else
        println("  ⚠️  Sindbad.jl/Project.toml not found")
        all_ok = false
    end

    println("\n✓ Package paths...")
    for pkg in DEV_PACKAGES
        try
            # Try to load the package and get its path
            mod = eval(:(using $(Symbol(pkg.name)); $(Symbol(pkg.name))))
            path = pathof(mod)
            if contains(path, "dev")
                println("  ✅ $(pkg.name): $path")
            else
                println("  ⚠️  $(pkg.name): $path (not in dev/)")
                all_ok = false
            end
        catch e
            println("  ℹ️  $(pkg.name): not loaded (might need restart)")
        end
    end

    println("\n" * "=" ^ 60)
    if all_ok
        println("✨ DEV SETUP VERIFICATION PASSED")
    else
        println("⚠️  DEV SETUP: Some issues detected (see above)")
    end
    println("=" ^ 60)
end

function test_run_setup()
    println("\n" * "=" ^ 60)
    println("TESTING RUN MODE SETUP")
    println("=" ^ 60)

    println("\n✓ Checking package status...")
    
    all_ok = true
    
    # Check manifest - should NOT have dev entries for Sindbad
    manifest_path = joinpath("Manifest.toml")
    if isfile(manifest_path)
        manifest_content = read(manifest_path, String)
        
        # Look for dev entries - there should be none
        if contains(manifest_content, "Sindbad") && contains(manifest_content, "path = ")
            println("  ⚠️  Sindbad: still appears to be in dev mode")
            all_ok = false
        else
            println("  ✅ Sindbad: not in dev mode (using registry version)")
        end
    else
        println("  ⚠️  Manifest.toml not found")
        all_ok = false
    end

    println("\n✓ Package paths...")
    try
        # Check Sindbad path
        @eval using Sindbad
        sindbad_path = @eval pathof(Sindbad)
        if contains(sindbad_path, ".julia") || !contains(sindbad_path, "dev")
            println("  ✅ Sindbad: $sindbad_path")
        else
            println("  ⚠️  Sindbad: $sindbad_path (appears to be local dev)")
            all_ok = false
        end
    catch e
        println("  ℹ️  Sindbad: not loaded (might need restart)")
    end

    println("\n✓ Dependencies...")
    for pkg_name in ["ErrorMetrics", "TimeSamplers", "OmniTools"]
        println("  ✅ $pkg_name: included as dependency of Sindbad")
    end

    println("\n" * "=" ^ 60)
    if all_ok
        println("✨ RUN SETUP VERIFICATION PASSED")
    else
        println("⚠️  RUN SETUP: Some issues detected (see above)")
    end
    println("=" ^ 60)
end

function main()
    println("Activating SindbadTutorials environment at: $(pwd())")
    Pkg.activate(".")

    settings = load_settings()

    mode = get(get(settings, "mode", Dict{String,Any}()), "sindbad", "run")
    mode = String(mode)

    println("\nSindbad ecosystem mode: $mode")

    if mode == "dev"
        enable_dev_mode(settings)
    elseif mode == "run"
        enable_run_mode()
    else
        println("❌ Unknown mode: $mode")
        println("   Use \"run\" or \"dev\" in the [mode] section of $SETTINGS_FILE.")
        return
    end

    # Instantiate the environment (resolve and lock dependencies)
    println("\n" * "=" ^ 60)
    println("Instantiating SindbadTutorials environment...")
    println("=" ^ 60)
    try
        Pkg.instantiate()
        println("✅ Environment instantiated successfully")
    catch e
        println("❌ Error instantiating environment: $e")
    end

    # Run tests
    if mode == "dev"
        test_dev_setup()
    elseif mode == "run"
        test_run_setup()
    end
end

main()

