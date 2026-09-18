# Official Julia image
FROM julia:1.10-bookworm
LABEL org.opencontainers.image.source=https://github.com/pzaffino/Radiomics.jl
LABEL org.opencontainers.image.description="Radiomics.jl Docker Image"
LABEL maintainer="Paolo Zaffino <[p.zaffino@unicz.it]>, Aldo Giuliani <[aldo.giuliani@studenti.unicz.it]>, Ciro Benito Raggio <[ciro.raggio@kit.edu]>, Mohammadreza Javadi Namin <[mohammadreza.javadi@mail.polimi.it]>, Nastaran Ghaffari Elkhechi <[nastaran.ghaffari@mail.polimi.it]>, Jakub Mitura <[jakub.mitura14@gmail.com], Marco Tullio Giannotti <marcotullio.giannotti@studenti.unicz.it>">

# Build parameter: defaults to false (CPU-only)
ARG USE_CUDA=false
ENV USE_CUDA=${USE_CUDA}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Conditional dependency installation
RUN julia -e 'using Pkg; \
    Pkg.add(url="https://github.com/pzaffino/Radiomics.jl"); \
    pkgs = ["NIfTI", "ArgParse", "CSV", "DataFrames", "PrecompileTools"]; \
    if get(ENV, "USE_CUDA", "false") == "true"; \
        push!(pkgs, "CUDA"); \
        Pkg.add(pkgs); \
        using CUDA; \
        CUDA.set_runtime_version!(v"12.2.0"; local_toolkit=false); \
    else; \
        Pkg.add(pkgs); \
    end; \
    Pkg.precompile()'

# Set working directory and generate the embedded Julia script
WORKDIR /app

# Script generation with conditional CUDA import
RUN printf 'using Radiomics, NIfTI, ArgParse, CSV, DataFrames\n\
if get(ENV, "USE_CUDA", "false") == "true"\n\
    try\n\
        using CUDA\n\
    catch e\n\
        @warn "CUDA non disponibile, esecuzione su CPU"\n\
    end\n\
end\n\
\n\
function parse_args_custom()\n\
    s = ArgParseSettings()\n\
    @add_arg_table! s begin\n\
        "img"\n\
            help = "Path to input NIfTI image"\n\
            required = true\n\
        "mask"\n\
            help = "Path to NIfTI mask"\n\
            required = true\n\
        "--output", "-o"\n\
            help = "Path to output CSV file"\n\
            arg_type = String\n\
            default = "features.csv"\n\
        "--n_bins"\n\
            help = "Number of bins for histogram"\n\
            arg_type = Int\n\
            default = nothing\n\
        "--bin_width"\n\
            help = "Bin width for histogram"\n\
            arg_type = Float64\n\
            default = nothing\n\
        "--weighting_norm"\n\
            help = "Weighting normalization method"\n\
            arg_type = String\n\
            default = nothing\n\
        "--keep_largest_only"\n\
            help = "Keep only the largest region in mask"\n\
            arg_type = Bool\n\
            default = true\n\
        "--get_raw_matrices"\n\
            help = "Return raw feature matrices"\n\
            arg_type = Bool\n\
            default = false\n\
        "--verbose"\n\
            help = "Enable verbose output"\n\
            arg_type = Bool\n\
            default = true\n\
    end\n\
    return parse_args(s)\n\
end\n\
\n\
args = parse_args_custom()\n\
\n\
# Load NIfTI image and mask\n\
img  = niread(args["img"])\n\
mask = niread(args["mask"])\n\
\n\
# Extract voxel spacing from image header\n\
spacing = (img.header.pixdim[2], img.header.pixdim[3], img.header.pixdim[4])\n\
\n\
# Run radiomic feature extraction\n\
features = Radiomics.extract_radiomic_features(\n\
    img.raw, mask.raw, spacing;\n\
    features=Symbol[],\n\
    labels=nothing,\n\
    n_bins=args["n_bins"],\n\
    bin_width=args["bin_width"],\n\
    weighting_norm=args["weighting_norm"],\n\
    keep_largest_only=args["keep_largest_only"],\n\
    get_raw_matrices=args["get_raw_matrices"],\n\
    slices_2d=nothing,\n\
    verbose=args["verbose"]\n\
);\n\
\n\
# Convert results to DataFrame and save to CSV\n\
df = DataFrame([features])\n\
CSV.write(args["output"], df)\n\
println("Extraction complete. Results saved to: ", args["output"])\n\
' > /app/extract.jl

# Explicit installation of PackageCompiler and sysimage creation
RUN julia -e 'using Pkg; Pkg.add("PackageCompiler")'
RUN julia -e 'using Pkg, PackageCompiler; \
    pkgs = [:Radiomics, :NIfTI, :ArgParse, :CSV, :DataFrames]; \
    ENV["USE_CUDA"] == "true" && push!(pkgs, :CUDA); \
    create_sysimage(pkgs; sysimage_path="/app/radiomics.so", cpu_target="generic")'

# Use the sysimage at each run
ENTRYPOINT ["julia", "--sysimage", "/app/radiomics.so", "/app/extract.jl"]