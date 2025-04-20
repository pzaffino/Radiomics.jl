function first_order_features(img, mask, verbose=false)
    println("Extracting first order features...")

    voxel_volume = img.header.pixdim[2] * img.header.pixdim[3] * img.header.pixdim[4]

    energy = get_energy(img, mask)
    if verbose
        println("Energy = $energy")
    end

    total_energy = get_total_energy(voxel_volume, energy)
    if verbose
        println("Total energy = $total_energy")
    end

end

function get_energy(img, mask, c=0)
    energy = 0.0
    for i in 1:size(img, 1)
        for j in 1:size(img, 2)
            for k in 1:size(img, 3)
                if mask[i,j,k] == 1
                    energy = energy + (img[i,j,k] + c)^2
                end
            end
        end
    end
    return energy
end

function get_total_energy(voxel_volume, energy)
    total_energy = voxel_volume * energy
    return total_energy 
end
