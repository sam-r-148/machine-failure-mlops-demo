param(
    [string]$RepoName = "machine-failure-mlops-demo",
    [string]$Owner = "sam-r-148"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
function Assert-Exit([string]$Message) {
    if ($LASTEXITCODE -ne 0) { throw $Message }
}
gh auth status
Assert-Exit "Run gh auth login -h github.com first."
$login = gh api user --jq .login
Assert-Exit "Cannot verify GitHub account."
if ($login.Trim() -ne $Owner) { throw "Expected authenticated account $Owner." }
python scripts/validate_data.py
Assert-Exit "Kaggle data validation failed."
$meta = Get-Content artifacts/metadata.json -Raw | ConvertFrom-Json
if (-not $meta.dataset.assignment_dataset -or $meta.model_version -ne "maintenance-rf-v1") {
    throw "Train the Kaggle maintenance notebook before publishing."
}
python -m pytest -q
Assert-Exit "Tests failed."
git add README.md requirements.txt requirements-dev.txt .gitignore .dockerignore Dockerfile compose.yaml app scripts tests notebooks artifacts data .github
Assert-Exit "Could not stage project."
git diff --cached --check
Assert-Exit "Staged whitespace check failed."
git diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    git commit -m "Add Kaggle machine failure model, API, Docker deployment and CI"
    Assert-Exit "Commit failed. Configure your Git name/email if needed."
} elseif ($LASTEXITCODE -ne 0) { throw "Could not inspect staged changes." }
gh repo create "$Owner/$RepoName" --private --source . --remote origin --push
Assert-Exit "Repository creation/push failed; existing repos are not overwritten."
$commit = git rev-parse HEAD
$runId = $null
for ($attempt = 0; $attempt -lt 15; $attempt++) {
    $runs = gh run list --workflow ci.yml --commit $commit --json databaseId --limit 1 | ConvertFrom-Json
    Assert-Exit "Could not query workflow."
    if ($runs.Count -gt 0) { $runId = $runs[0].databaseId; break }
    Start-Sleep -Seconds 2
}
if (-not $runId) { throw "Run not listed yet. Check the Actions tab." }
gh run watch $runId --exit-status
Assert-Exit "CI failed. Inspect gh run view $runId --log-failed."
gh run view $runId --json url,conclusion
