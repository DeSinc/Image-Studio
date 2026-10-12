# Image Studio Automated Deploy Script
# Target: desinc.my.to (/opt/projects/image-studio/app -> /var/www/desite/image-studio)

$ErrorActionPreference = "Stop"

Write-Host "=== Image Studio Deploy Pipeline ===" -ForegroundColor Cyan

# 1. Read version from Cargo.toml
$cargoToml = Get-Content Cargo.toml -Raw
if ($cargoToml -match 'version\s*=\s*"([^"]+)"') {
    $version = $matches[1]
} else {
    Write-Error "Could not detect version in Cargo.toml"
}

Write-Host "Target Version: v$version" -ForegroundColor Green

# 2. Build Release Binary
Write-Host "Building release binary..." -ForegroundColor Yellow
cargo build --release
if ($LASTEXITCODE -ne 0) {
    Write-Error "Cargo release build failed"
}

$exePath = "target/release/ImageStudio.exe"
if (-not (Test-Path $exePath)) {
    Write-Error "Release executable not found at $exePath"
}
$exeSize = (Get-Item $exePath).Length
Write-Host "Release binary built successfully ($([math]::Round($exeSize / 1MB, 2)) MB)" -ForegroundColor Green

# 3. Generate version.json
$versionJsonObj = @{
    version = $version
    download_url = "https://desinc.my.to/image-studio/ImageStudio.exe"
}
$versionJson = $versionJsonObj | ConvertTo-Json
$versionJsonPath = "target/version.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false); [System.IO.File]::WriteAllText((Join-Path (Get-Location) $versionJsonPath), $versionJson, $utf8NoBom)
Write-Host "Generated ${versionJsonPath}:" -ForegroundColor DarkGray
Write-Host $versionJson -ForegroundColor DarkGray

# 4. Upload to remote server (desinc.my.to)
$remoteHost = "ds@desinc.my.to"
$remoteDir = "/opt/projects/image-studio/app"

Write-Host "Uploading to ${remoteHost}:${remoteDir}..." -ForegroundColor Yellow
scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new $exePath "${remoteHost}:${remoteDir}/ImageStudio.exe"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upload ImageStudio.exe via SCP"
}

scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new $versionJsonPath "${remoteHost}:${remoteDir}/version.json"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upload version.json via SCP"
}

# 5. Verify live endpoint
Write-Host "Verifying live endpoint https://desinc.my.to/image-studio/version.json..." -ForegroundColor Yellow
Start-Sleep -Seconds 1
$liveResp = curl.exe -s https://desinc.my.to/image-studio/version.json
Write-Host "Live endpoint response: $liveResp" -ForegroundColor Green

Write-Host "=== Deployment Completed Successfully! ===" -ForegroundColor Cyan
