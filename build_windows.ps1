# Build LHK_Sculpt on Windows (GNU toolchain + portable MinGW in _tools).
Set-Location $PSScriptRoot
$env:PATH = "$env:USERPROFILE\.cargo\bin;$PSScriptRoot\_tools\mingw64\bin;$env:PATH"
cmd /c "cargo build --release 2>&1"
if ($LASTEXITCODE -ne 0) { Write-Host "Build fehlgeschlagen"; exit 1 }
New-Item -ItemType Directory -Force dist | Out-Null
Copy-Item target\release\LHK_Sculpt.exe, target\release\lhk_cli.exe dist\ -Force
Write-Host "Fertig: $PSScriptRoot\dist\LHK_Sculpt.exe"
