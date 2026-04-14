# Official Julia image
FROM julia:1.10-bookworm

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Julia packages
RUN julia -e 'using Pkg; \
    Pkg.add(url="https://github.com/pzaffino/Radiomics.jl"); \
    Pkg.add(["NIfTI", "ArgParse"]); \
    Pkg.precompile()'

# Set working directory and generate the embedded Julia script
WORKDIR /app

RUN printf 'using Radiomics, NIfTI, ArgParse\n\
\n\
# Function to parse command line arguments\n\
function parse_args_custom()\n\
    s = ArgParseSettings()\n\
    @add_arg_table! s begin\n\
        "img"\n\
            help = "Path to input NIfTI image"\n\
            required = true\n\
        "mask"\n\
            help = "Path to NIfTI mask"\n\
            required = true\n\
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
        "--sample_rate"\n\
            help = "Sample rate for feature extraction"\n\
            arg_type = Float64\n\
            default = 0.03\n\
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
    sample_rate=args["sample_rate"],\n\
    slices_2d=nothing,\n\
    verbose=args["verbose"]\n\
);' > /app/extract.jl

# Set the entrypoint to Julia
ENTRYPOINT ["julia", "/app/extract.jl"]