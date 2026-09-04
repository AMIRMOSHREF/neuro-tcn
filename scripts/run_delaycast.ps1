<#
DelayCAST v2 - full protocol on Windows / PowerShell.

    .\scripts\run_delaycast.ps1            # 3 seeds, popmean control, cross-dataset transfer, negative control, figures, report
    .\scripts\run_delaycast.ps1 -Quick     # one seed, within-session only (sanity run)
    .\scripts\run_delaycast.ps1 -DataA D:\Rodent\Data -DataB D:\Rodent\Data2
    .\scripts\run_delaycast.ps1 -Population   # Data + Data2 as identity-free population channels -> outputs/delaycast_pop

Run it from the repository folder after `pip install -e .` (see README "Windows / PowerShell quick start").
Every step is a separate `python -m delaycast ...` call, so a failed step can be rerun on its own.
#>
param(
    [string]$DataA = "C:/PythonProject/Rodent/Data",
    [string]$DataB = "C:/PythonProject/Rodent/Data2",
    [string]$OutputDir = "outputs/delaycast",
    [string]$CacheDir = "cache",
    [string]$Seeds = "0,1,2",
    [switch]$Quick,
    [switch]$Population       # identity-free population channels: Data + Data2 (11 sessions), outputs/delaycast_pop
)
$ErrorActionPreference = "Stop"
if ($Population) {
    if ($OutputDir -eq "outputs/delaycast") { $OutputDir = "outputs/delaycast_pop" }
    if ($CacheDir -eq "cache") { $CacheDir = "cache_pop" }
}
$common = @("--set", "data.data_a_root=$DataA", "--set", "data.data_b_root=$DataB",
            "--set", "output_dir=$OutputDir", "--set", "data.cache_dir=$CacheDir")
if ($Population) { $common += @("--set", "data.representation=population") }

# NB: the parameter must not be called $args - that is a PowerShell automatic variable and would be empty here.
function Step([string]$name, [string[]]$cmdArgs) {
    Write-Host "`n=== $name" -ForegroundColor Cyan
    Write-Host "python -m delaycast $($cmdArgs -join ' ')" -ForegroundColor DarkGray
    & python -m delaycast @cmdArgs
    if ($LASTEXITCODE -ne 0) { throw "step '$name' failed (exit $LASTEXITCODE)" }
}

Step "inspect" (@("inspect", "--npz-detail") + $common)
Step "cache"   (@("cache") + $common)
Step "select"  (@("select") + $common)
# Population representation: the channels are not neurons, so the rate/random arms and the linonly/noskip
# ablations would train on identical inputs - only the criteria arm and its spectral control are run.
$modes = if ($Population) { "criteria" } else { "criteria,rate,random" }
$variants = if ($Population) { "popmean" } else { "popmean,linonly,noskip" }
$xmodes = if ($Population) { "criteria" } else { "criteria,random" }
if ($Quick) {
    Step "train (quick)" (@("train", "--modes", $modes, "--seeds", "0") + $common)
} else {
    Step "train within-session" (@("train", "--modes", $modes, "--variants", $variants, "--seeds", $Seeds) + $common)
    Step "train cross-dataset"  (@("train", "--modes", $xmodes, "--seeds", "0", "--set", "train.eval_mode=cross_dataset") + $common)
    Step "negative control"     (@("train", "--modes", "criteria", "--seeds", "0", "--negative-control") + $common)
}
Step "figures" (@("figures", "--all-sessions") + $common)
Step "report"  (@("report") + $common)
Write-Host "`nDone. Figures: $OutputDir/figures ; verdicts: $OutputDir/REPORT.md" -ForegroundColor Green
