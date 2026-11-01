$ErrorActionPreference='Stop'
$Repo=Split-Path $PSScriptRoot -Parent
& (Join-Path $Repo '.venv/Scripts/python.exe') -B (Join-Path $Repo 'tests/test_repository.py')
if ($LASTEXITCODE -ne 0) { throw 'Repository regression failed' }
& (Join-Path $Repo '.venv/Scripts/python.exe') -B (Join-Path $Repo 'scripts/rs05.py') verify
if ($LASTEXITCODE -ne 0) { throw 'Frozen release validation failed' }
