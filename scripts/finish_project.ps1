# Run in your own PowerShell terminal; requires Docker Desktop and GitHub browser sign-in.
param(
    [string]$RepoName = "machine-failure-mlops-demo",
    [string]$Owner = "sam-r-148"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
function Assert-Exit([string]$Message) {
    if ($LASTEXITCODE -ne 0) { throw $Message }
}
python scripts/validate_data.py
Assert-Exit "Dataset validation failed."
docker info --format '{{.ServerVersion}}'
Assert-Exit "Start Docker Desktop and run this script from a terminal with access to its engine."
gh auth status
Assert-Exit "Run gh auth login -h github.com, then retry."
$login = gh api user --jq .login
Assert-Exit "Could not verify the GitHub account."
if ($login.Trim() -ne $Owner) { throw "Expected GitHub account $Owner. Switch to that account first." }
# Reproduce the notebook in a real Jupyter kernel before deployment/submission.
python -m ipykernel install --user --name maintenance-demo --display-name "Maintenance demo"
Assert-Exit "Kernel registration failed."
$previousKernel = $env:NOTEBOOK_KERNEL
try {
    $env:NOTEBOOK_KERNEL = "maintenance-demo"
    python scripts/execute_notebook.py
    Assert-Exit "Jupyter training/evaluation failed."
} finally {
    if ($null -eq $previousKernel) { Remove-Item Env:NOTEBOOK_KERNEL -ErrorAction SilentlyContinue }
    else { $env:NOTEBOOK_KERNEL = $previousKernel }
}
python -m pytest -q --junitxml=artifacts/test-results.xml
Assert-Exit "Tests failed."
docker compose up --build -d --wait --wait-timeout 90
Assert-Exit "Container deployment failed."
python scripts/smoke_test.py
Assert-Exit "Docker HTTP smoke tests failed."
docker compose ps
& "$PSScriptRoot/publish.ps1" -RepoName $RepoName -Owner $Owner
Write-Host "Container remains running at http://localhost:8000/docs"
