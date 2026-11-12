# syntax=docker/dockerfile:1.7
# Official Julia image
FROM julia:1.10-bookworm
LABEL org.opencontainers.image.source=https://github.com/pzaffino/Radiomics.jl
LABEL org.opencontainers.image.description="Radiomics.jl Docker Image"
LABEL maintainer="Paolo Zaffino <[p.zaffino@unicz.it]>, Aldo Giuliani <[aldo.giuliani@studenti.unicz.it]>, Ciro Benito Raggio <[ciro.raggio@kit.edu]>, Mohammadreza Javadi Namin <[mohammadreza.javadi@mail.polimi.it]>, Nastaran Ghaffari Elkhechi <[nastaran.ghaffari@mail.polimi.it]>, Jakub Mitura <[jakub.mitura14@gmail.com], Marco Tullio Giannotti <marcotullio.giannotti@studenti.unicz.it>"

# Build parameter: defaults to false (CPU-only)
ARG USE_CUDA=false
ENV USE_CUDA=${USE_CUDA}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Radiomics.jl from the build context: the image contains exactly the code of the commit being built
COPY Project.toml /opt/Radiomics.jl/Project.toml
COPY src /opt/Radiomics.jl/src
COPY ext /opt/Radiomics.jl/ext

# Conditional dependency installation
RUN julia -e 'using Pkg; \
    Pkg.develop(path="/opt/Radiomics.jl"); \
    pkgs = ["ArgParse", "CSV", "DataFrames", "PrecompileTools"]; \
    if get(ENV, "USE_CUDA", "false") == "true"; \
        push!(pkgs, "CUDA"); \
        Pkg.add(pkgs); \
        using CUDA; \
        CUDA.set_runtime_version!(v"12.2.0"; local_toolkit=false); \
    else; \
        Pkg.add(pkgs); \
    end; \
    Pkg.precompile()'

WORKDIR /app

# Command line entry point
COPY <<'EOF' /app/extract.jl
using Radiomics, ArgParse, CSV, DataFrames
if get(ENV, "USE_CUDA", "false") == "true"
    try
        using CUDA
    catch e
        @warn "CUDA not available, running on CPU"
    end
end

function parse_args_custom()
    s = ArgParseSettings(description="Radiomics.jl feature extraction")
    @add_arg_table! s begin
        "img"
            help = "Path to input image (NIfTI, NRRD or DICOM folder)"
            required = true
        "mask"
            help = "Path to mask (NIfTI, NRRD or DICOM folder)"
            required = true
        "--output", "-o"
            help = "Path to output CSV file"
            arg_type = String
            default = "features.csv"
        "--features"
            help = "Comma-separated feature families, e.g. first_order,glcm,shape3d (default: all)"
            arg_type = String
            default = nothing
        "--labels"
            help = "Comma-separated label values, e.g. 1 or 1,2,3 (default: 1)"
            arg_type = String
            default = nothing
        "--n_bins"
            help = "Number of bins for histogram"
            arg_type = Int
            default = nothing
        "--bin_width"
            help = "Bin width for histogram"
            arg_type = Float64
            default = nothing
        "--weighting_norm"
            help = "Weighting normalization method"
            arg_type = String
            default = nothing
        "--keep_largest_only"
            help = "Keep only the largest region in mask"
            arg_type = Bool
            default = true
        "--get_raw_matrices"
            help = "Return raw feature matrices"
            arg_type = Bool
            default = false
        "--features_std"
            help = "Also return std, min and max over directions for GLCM and GLRLM"
            arg_type = Bool
            default = false
        "--slices_2d"
            help = "Comma-separated plane:slice pairs (1-based, as in Julia), e.g. 3:10,3:11"
            arg_type = String
            default = nothing
        "--filter_type"
            help = "Image filter(s) applied before extraction: haar, log, gabor (comma-separated)"
            arg_type = String
            default = nothing
        "--filter_params"
            help = "Filter parameters as a Julia NamedTuple, e.g. '(level=2,)' or '(sigma=[1.0, 3.0],)'"
            arg_type = String
            default = nothing
        "--suv_ibsi_normalization"
            help = "Normalize a PET DICOM series to SUV (the image must be a DICOM PET series)"
            arg_type = Bool
            default = false
        "--use_gpu"
            help = "Use the GPU (only in the CUDA image)"
            arg_type = Bool
            default = get(ENV, "USE_CUDA", "false") == "true"
        "--verbose"
            help = "Enable verbose output"
            arg_type = Bool
            default = true
    end
    return parse_args(s)
end

splitlist(s) = [strip(x) for x in split(s, ",") if !isempty(strip(x))]

function parse_labels(s)
    isnothing(s) && return nothing
    v = parse.(Int, splitlist(s))
    return length(v) == 1 ? only(v) : v
end

function parse_slices(s)
    isnothing(s) && return nothing
    return [Tuple(parse.(Int, split(p, ":"))) for p in splitlist(s)]
end

function parse_filter_params(s)
    isnothing(s) && return (;)
    p = eval(Meta.parse(s))
    p isa NamedTuple || error("--filter_params must be a Julia NamedTuple, e.g. '(level=2,)', got: $s")
    return p
end

# One CSV row per feature dictionary. Nested results (several labels, slices or filter
# response maps) become one row each, identified by the columns filter_map, label and slice.
function flatten_rows(res, ctx=Dict{String,Any}())
    if res isa AbstractDict && !isempty(res) && all(v -> v isa AbstractDict, values(res))
        rows = Dict{String,Any}[]
        for (k, v) in res
            c = copy(ctx)
            if k isa Integer
                c["label"] = k
            elseif k isa Tuple
                c["slice"] = join(k, ":")
            else
                c["filter_map"] = string(k)
            end
            append!(rows, flatten_rows(v, c))
        end
        return rows
    end
    row = copy(ctx)
    for (k, v) in res
        row[string(k)] = v isa AbstractArray ? string(v) : v
    end
    return [row]
end

args = parse_args_custom()

features = isnothing(args["features"]) ? Symbol[] : Symbol.(splitlist(args["features"]))
filter_type = isnothing(args["filter_type"]) ? nothing :
              (ft = String.(splitlist(args["filter_type"])); length(ft) == 1 ? only(ft) : ft)

# Run radiomic feature extraction (image and mask are read from the given paths)
result = Radiomics.extract_radiomic_features(
    args["img"], args["mask"];
    features=features,
    labels=parse_labels(args["labels"]),
    n_bins=args["n_bins"],
    bin_width=args["bin_width"],
    weighting_norm=args["weighting_norm"],
    keep_largest_only=args["keep_largest_only"],
    get_raw_matrices=args["get_raw_matrices"],
    features_std=args["features_std"],
    slices_2d=parse_slices(args["slices_2d"]),
    use_gpu=args["use_gpu"],
    suv_ibsi_normalization=args["suv_ibsi_normalization"],
    filter_type=filter_type,
    filter_params=parse_filter_params(args["filter_params"]),
    verbose=args["verbose"]
)

# Convert results to DataFrame (identifier columns first) and save to CSV
df = DataFrame(flatten_rows(result))
id_cols = intersect(["filter_map", "label", "slice"], names(df))
select!(df, id_cols, Not(id_cols))
CSV.write(args["output"], df)
println("Extraction complete. ", nrow(df), " row(s) saved to: ", args["output"])
EOF

# Explicit installation of PackageCompiler and sysimage creation
RUN julia -e 'using Pkg; Pkg.add("PackageCompiler")'
RUN julia -e 'using Pkg, PackageCompiler; \
    pkgs = [:Radiomics, :ArgParse, :CSV, :DataFrames]; \
    ENV["USE_CUDA"] == "true" && push!(pkgs, :CUDA); \
    create_sysimage(pkgs; sysimage_path="/app/radiomics.so", cpu_target="generic")'

# Use the sysimage at each run
ENTRYPOINT ["julia", "--sysimage", "/app/radiomics.so", "/app/extract.jl"]