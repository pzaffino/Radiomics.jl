#=
# First orde features
=#

function first_order_features(img, mask, verbose=false)
    println("Extracting first order features...")

    # Some data can be useful for features extraction
    voxel_volume = img.header.pixdim[2] * img.header.pixdim[3] * img.header.pixdim[4]
    roi_voxels = extract_roi_voxels(img, mask)

    # Energy
    energy = get_energy(roi_voxels)
    if verbose
        println("  Energy = $energy")
    end
    
    # Total energy
    total_energy = get_total_energy(voxel_volume, energy)
    if verbose
        println("  Total energy = $total_energy")
    end

end


function extract_roi_voxels(img, mask)
    roi_voxels = Vector{Float32}()
    for i in 1:size(img, 1)
        for j in 1:size(img, 2)
            for k in 1:size(img, 3)
                if mask[i,j,k] == 1
                    push!(roi_voxels, img[i,j,k])
                end
            end
        end
    end
    return roi_voxels
end


function get_energy(roi_voxels, c=0.0)
    energy = 0.0
    for i in 1:size(roi_voxels, 1)
        energy = energy + (roi_voxels[i] + c)^2
    end
    return energy
end


function get_total_energy(voxel_volume, energy)
    total_energy = voxel_volume * energy
    return total_energy 
end


