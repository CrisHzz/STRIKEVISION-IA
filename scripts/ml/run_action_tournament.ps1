<#!
.SYNOPSIS
Runs the reproducible development tournament for StrikeVision action recognition.

.DESCRIPTION
The Tournament stage prepares the derived corpus, then evaluates every configured
candidate on each of the three fight-level development folds. Existing completed
runs are preserved when -Resume is supplied, so it is safe to continue after a
terminal or power interruption. It never runs the held-out test set.

The Tune stage is deliberately separate: first inspect the tournament aggregate,
then tune only a chosen temporal candidate. This prevents choosing a model using
the test set or spending time tuning a model that loses the development tournament.
#>
[CmdletBinding()]
param(
    [ValidateSet('Tournament', 'Tune')]
    [string]$Stage = 'Tournament',

    [ValidateSet('tcn', 'hierarchical_tcn', 'stgcn_compact', 'x3d_linear_probe')]
    [string]$TuneModel = 'tcn',

    [ValidateRange(1, 200)]
    [int]$Trials = 20,

    [string]$Config = 'configs/models/action_training_v1.yaml',

    [string]$OutputRoot = 'outputs/action-recognition',

    [string]$RunSuffix = '',

    [ValidateRange(0, 2)]
    [int[]]$Folds = @(0, 1, 2),

    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $projectRoot
$resolvedOutputRoot = Join-Path $projectRoot $OutputRoot

# A child PowerShell can resolve `python` to Miniconda base even when this
# script was launched from an activated environment. Prefer the inherited
# Conda prefix so the project always uses strikevision_ML's dependencies.
$PythonExecutable = 'python'
if ($env:CONDA_PREFIX) {
    $condaPython = Join-Path $env:CONDA_PREFIX 'python.exe'
    if (Test-Path -LiteralPath $condaPython) {
        $PythonExecutable = $condaPython
    }
}
Write-Host "[environment] Python: $PythonExecutable"

function Invoke-ActionCommand {
    param([string[]]$Arguments)
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Action-recognition command failed with exit code ${LASTEXITCODE}: $PythonExecutable $($Arguments -join ' ')"
    }
}

if ($Stage -eq 'Tournament') {
    # This only regenerates derived manifests; it never changes source annotations.
    Invoke-ActionCommand @('-s', 'scripts/ml/run_action_recognition.py', 'prepare', '--config', $Config)

    $candidates = @(
        [pscustomobject]@{ Command = 'baseline'; Model = 'logistic_regression'; RunPrefix = 'baseline-logreg' },
        [pscustomobject]@{ Command = 'baseline'; Model = 'extra_trees'; RunPrefix = 'baseline-trees' },
        [pscustomobject]@{ Command = 'temporal'; Model = 'tcn'; RunPrefix = 'tcn' },
        [pscustomobject]@{ Command = 'temporal'; Model = 'stgcn_compact'; RunPrefix = 'stgcn' },
        [pscustomobject]@{ Command = 'rgb'; Model = 'x3d_linear_probe'; RunPrefix = 'x3d' }
    )
    $runs = [System.Collections.Generic.List[string]]::new()

    foreach ($fold in $Folds) {
        foreach ($candidate in $candidates) {
            $run = "$($candidate.RunPrefix)-fold$fold$RunSuffix"
            $runs.Add($run)
            $report = Join-Path $resolvedOutputRoot "$run\report.json"
            if ($Resume -and (Test-Path -LiteralPath $report)) {
                Write-Host "[resume] Already completed: $run"
                continue
            }
            Write-Host "[tournament] $run"
            Invoke-ActionCommand @('-s', 'scripts/ml/run_action_recognition.py', $candidate.Command, '--config', $Config, '--model', $candidate.Model, '--fold', "$fold", '--run', $run)
        }
    }

    $summaryPath = Join-Path $resolvedOutputRoot 'tournament.json'
    Invoke-ActionCommand (@('-s', 'scripts/ml/summarize_action_tournament.py') + $runs.ToArray() + @('--runs-root', $OutputRoot, '--output', $summaryPath))
    Write-Host "Tournament complete. Read $summaryPath."
    exit 0
}

# Optuna is required only for this stage. Install requirements-action-recognition.txt once if it is unavailable.
foreach ($fold in $Folds) {
    $run = "$TuneModel-optuna-fold$fold$RunSuffix"
    $report = Join-Path $resolvedOutputRoot "$run\tuning_report.json"
    if ($Resume -and (Test-Path -LiteralPath $report)) {
        Write-Host "[resume] Already completed: $run"
        continue
    }
    Write-Host "[tune] $run ($Trials trials)"
    Invoke-ActionCommand @('-s', 'scripts/ml/run_action_recognition.py', 'tune', '--config', $Config, '--model', $TuneModel, '--fold', "$fold", '--trials', "$Trials", '--run', $run)
}
