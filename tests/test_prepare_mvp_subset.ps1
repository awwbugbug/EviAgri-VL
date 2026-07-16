$ErrorActionPreference = 'Stop'

$workspace = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $workspace 'scripts\prepare_mvp_subset.ps1'
$tmp = Join-Path $PSScriptRoot ("tmp_mvp_" + [guid]::NewGuid().ToString('N'))

try {
    $inputRoot = Join-Path $tmp 'input'
    $outputRoot = Join-Path $tmp 'output'

    foreach ($className in @('Pest A-lavra', 'Pest B-adult')) {
        $dir = Join-Path $inputRoot ("ages\val\" + $className)
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        1..4 | ForEach-Object {
            Set-Content -LiteralPath (Join-Path $dir ("a$_.jpg")) -Value "age-$className-$_"
        }
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $inputRoot 'ages\val\Pest C-egg') | Out-Null

    $ipDir = Join-Path $inputRoot 'IP102\val\0'
    New-Item -ItemType Directory -Force -Path $ipDir | Out-Null
    1..7 | ForEach-Object {
        Set-Content -LiteralPath (Join-Path $ipDir ("i$_.jpg")) -Value "ip-$_"
    }
    Set-Content -LiteralPath (Join-Path $inputRoot 'IP102\classes.txt') -Value '1 rice leaf roller'

    & $scriptPath -InputRoot $inputRoot -OutputRoot $outputRoot -Seed 'fixture-seed'

    $rows = Import-Csv -LiteralPath (Join-Path $outputRoot 'manifest.csv')
    if ($rows.Count -ne 11) { throw "Expected 11 selected rows, got $($rows.Count)" }
    if (($rows | Where-Object { $_.dataset -eq 'ages' }).Count -ne 6) { throw 'Age cap must be 3 per class' }
    if (($rows | Where-Object { $_.dataset -eq 'IP102' }).Count -ne 5) { throw 'IP102 cap must be 5 per class' }

    $lavra = $rows | Where-Object { $_.class_original -eq 'Pest A-lavra' }
    if (($lavra | Select-Object -First 1).class_normalized -ne 'Pest A-larva') {
        throw 'lavra normalization was not recorded'
    }
    if (($rows | Where-Object { $_.dataset -eq 'IP102' } | Select-Object -First 1).class_name -ne 'rice leaf roller') {
        throw 'IP102 zero-based directory was not mapped to the one-based class list'
    }
    if ((Get-ChildItem -LiteralPath $outputRoot -File -Recurse -Filter '*.jpg').Count -ne 11) {
        throw 'Selected image copy count mismatch'
    }

    $taxonomy = Import-Csv -LiteralPath (Join-Path $outputRoot 'taxonomy.csv')
    if (($taxonomy | Where-Object { $_.dataset -eq 'ages' }).Count -ne 3) {
        throw 'Taxonomy must retain empty Age class directories'
    }
    $emptyClass = $taxonomy | Where-Object { $_.class_original -eq 'Pest C-egg' }
    if ($emptyClass.split_has_images -ne 'False') {
        throw 'Empty class must be explicitly marked in taxonomy'
    }

    Write-Output 'TEST_PREPARE_MVP_SUBSET_OK'
}
finally {
    $resolvedTmp = [IO.Path]::GetFullPath($tmp)
    $resolvedTests = [IO.Path]::GetFullPath($PSScriptRoot) + [IO.Path]::DirectorySeparatorChar
    if ($resolvedTmp.StartsWith($resolvedTests) -and (Test-Path -LiteralPath $resolvedTmp)) {
        Remove-Item -LiteralPath $resolvedTmp -Recurse -Force
    }
}
