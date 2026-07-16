[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [string]$Seed = 'eviagri-mvp-20260712-v1',

    [ValidateRange(1, 100)]
    [int]$AgeCap = 3,

    [ValidateRange(1, 100)]
    [int]$IP102Cap = 5
)

$ErrorActionPreference = 'Stop'
$imagePattern = '^\.(jpg|jpeg|png|bmp|webp)$'

if (-not (Test-Path -LiteralPath $InputRoot -PathType Container)) {
    throw "Input root does not exist: $InputRoot"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Output root already exists: $OutputRoot"
}

function Get-RankHash {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes("$Seed|$RelativePath")
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLower()
    }
    finally {
        $sha.Dispose()
    }
}

$ip102ClassesPath = Join-Path $InputRoot 'IP102\classes.txt'
if (-not (Test-Path -LiteralPath $ip102ClassesPath -PathType Leaf)) {
    throw "IP102 class list does not exist: $ip102ClassesPath"
}

$ip102ClassMap = @{}
Get-Content -LiteralPath $ip102ClassesPath | ForEach-Object {
    if ($_ -match '^\s*(\d+)\s+(.*?)\s*$') {
        $ip102ClassMap[[int]$Matches[1] - 1] = $Matches[2].Trim()
    }
}

New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$rows = New-Object System.Collections.Generic.List[object]
$taxonomyRows = New-Object System.Collections.Generic.List[object]
$summaries = New-Object System.Collections.Generic.List[object]
$specs = @(
    [pscustomobject]@{ Dataset = 'ages'; Cap = $AgeCap },
    [pscustomobject]@{ Dataset = 'IP102'; Cap = $IP102Cap }
)

foreach ($spec in $specs) {
    $validationRoot = Join-Path $InputRoot ("{0}\val" -f $spec.Dataset)
    if (-not (Test-Path -LiteralPath $validationRoot -PathType Container)) {
        throw "Validation split does not exist: $validationRoot"
    }

    $classDirs = @(Get-ChildItem -LiteralPath $validationRoot -Directory)
    $nonEmptyClasses = 0
    $selectedForDataset = 0

    foreach ($classDir in $classDirs) {
        $classFiles = @(
            Get-ChildItem -LiteralPath $classDir.FullName -File |
                Where-Object { $_.Extension -match $imagePattern }
        )
        $classOriginal = $classDir.Name
        $classNormalized = $classOriginal
        $className = ''
        $species = ''
        $stageOriginal = ''
        $stageNormalized = ''

        if ($spec.Dataset -eq 'ages') {
            if ($classOriginal -match '^(.*)-([^-]+)$') {
                $species = $Matches[1]
                $stageOriginal = $Matches[2].ToLower()
                if ($stageOriginal -eq 'lavra') {
                    $stageNormalized = 'larva'
                }
                else {
                    $stageNormalized = $stageOriginal
                }
                $classNormalized = "$species-$stageNormalized"
            }
        }
        else {
            $zeroBasedClass = [int]$classOriginal
            if (-not $ip102ClassMap.ContainsKey($zeroBasedClass)) {
                throw "No IP102 class mapping for directory: $classOriginal"
            }
            $className = $ip102ClassMap[$zeroBasedClass]
        }

        $taxonomyRows.Add([pscustomobject]@{
            dataset = $spec.Dataset
            split = 'val'
            class_original = $classOriginal
            class_normalized = $classNormalized
            class_name = $className
            species = $species
            stage_original = $stageOriginal
            stage_normalized = $stageNormalized
            split_has_images = $classFiles.Count -gt 0
            split_image_count = $classFiles.Count
        })

        $candidates = @(
            $classFiles |
                ForEach-Object {
                    $relative = "{0}/val/{1}/{2}" -f $spec.Dataset, $classDir.Name, $_.Name
                    [pscustomobject]@{
                        File = $_
                        Relative = $relative
                        RankHash = Get-RankHash -RelativePath $relative
                    }
                } |
                Sort-Object RankHash |
                Select-Object -First $spec.Cap
        )

        if ($candidates.Count -gt 0) {
            $nonEmptyClasses++
        }

        $rank = 0
        foreach ($candidate in $candidates) {
            $rank++
            $selectedForDataset++
            $destinationRelative = $candidate.Relative
            $destination = Join-Path $OutputRoot ($destinationRelative -replace '/', '\')
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
            Copy-Item -LiteralPath $candidate.File.FullName -Destination $destination

            $fileHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLower()
            $rows.Add([pscustomobject]@{
                dataset = $spec.Dataset
                split = 'val'
                class_original = $classOriginal
                class_normalized = $classNormalized
                class_name = $className
                species = $species
                stage_original = $stageOriginal
                stage_normalized = $stageNormalized
                file_name = $candidate.File.Name
                source_relative = $candidate.Relative
                subset_relative = $destinationRelative
                selection_seed = $Seed
                selection_rank = $rank
                selection_hash = $candidate.RankHash
                bytes = $candidate.File.Length
                sha256 = $fileHash
            })
        }
    }

    $summaries.Add([pscustomobject]@{
        dataset = $spec.Dataset
        split = 'val'
        cap_per_class = $spec.Cap
        class_directories = $classDirs.Count
        non_empty_classes = $nonEmptyClasses
        empty_classes = $classDirs.Count - $nonEmptyClasses
        selected_images = $selectedForDataset
    })
}

$manifestPath = Join-Path $OutputRoot 'manifest.csv'
$rows |
    Sort-Object dataset, class_original, selection_rank |
    Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8

$taxonomyPath = Join-Path $OutputRoot 'taxonomy.csv'
$taxonomyRows |
    Sort-Object dataset, class_original |
    Export-Csv -LiteralPath $taxonomyPath -NoTypeInformation -Encoding UTF8

$summaryPath = Join-Path $OutputRoot 'summary.json'
[pscustomobject]@{
    protocol = 'deterministic_path_hash_stratified_by_class'
    selection_seed = $Seed
    datasets = $summaries
    total_selected_images = $rows.Count
    total_selected_bytes = ($rows | Measure-Object bytes -Sum).Sum
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

Write-Output ("MVP_SUBSET_OK images={0} bytes={1} manifest={2}" -f `
    $rows.Count, (($rows | Measure-Object bytes -Sum).Sum), $manifestPath)
