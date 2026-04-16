async function loadLatestVersion() {
    try {
        const res = await fetch(
            'https://api.github.com/repos/pzaffino/Radiomics.jl/releases/latest'
        );
        const release = await res.json();
        const el = document.getElementById('latest-version');
        if (el) el.textContent = release.tag_name;
    } catch (e) {
        const el = document.getElementById('latest-version');
        if (el) el.textContent = 'N/A';
    }
}

loadLatestVersion();