<#
DelayCAST v2 - full protocol on Windows / PowerShell.

    .\scripts\run_delaycast.ps1            # 3 seeds, popmean control, cross-dataset transfer, negative control, figures, report
    .\scripts\run_delaycast.ps1 -Quick     # one seed, within-session only (sanity run)
    .\scripts\run_delaycast.ps1 -DataA D:\Rodent\Data -DataB D:\Rodent\Data2

Run it from the repository folder after `pip install -e .` (see README "Windows / PowerShell quick start").
Every step is a separate `python -m delaycast ...` call, so a failed step can be rerun on its own.
#>
param(
    [string]$DataA = "C:/PythonProject/Rodent/Data",
    [string]$DataB = "C:/PythonProject/Rodent/Data2",
    [string]$OutputDir = "outputs/delaycast",
    [string]$CacheDir = "cache",
    [string]$Seeds = "0,1,2",
    [switch]$Quick
)
$ErrorActionPreference = "Stop"
$common = @("--set", "data.data_a_root=$DataA", "--set", "data.data_b_root=$DataB",
            "--set", "output_dir=$OutputDir", "--set", "data.cache_dir=$CacheDir")

function Step($name, $args) {
    Write-Host "`n=== $name" -ForegroundColor Cyan
    & python -m delaycast @args
    if ($LASTEXITCODE -ne 0) { throw "step '$name' failed (exit $LASTEXITCODE)" }
}

Step "inspect" (@("inspect", "--npz-detail") + $common)
Step "cache"   (@("cache") + $common)
Step "select"  (@("select") + $common)
if ($Quick) {
    Step "train (quick)" (@("train", "--modes", "criteria,rate,random", "--seeds", "0") + $common)
} else {
    Step "train within-session" (@("train", "--modes", "criteria,rate,random", "--variants", "popmean", "--seeds", $Seeds) + $common)
    Step "train cross-dataset"  (@("train", "--modes", "criteria,random", "--seeds", "0", "--set", "train.eval_mode=cross_dataset") + $common)
    Step "negative control"     (@("train", "--modes", "criteria", "--seeds", "0", "--negative-control") + $common)
}
Step "figures" (@("figures", "--all-sessions") + $common)
Step "report"  (@("report") + $common)
Write-Host "`nDone. Figures: $OutputDir/figures ; verdicts: $OutputDir/REPORT.md" -ForegroundColor Green
