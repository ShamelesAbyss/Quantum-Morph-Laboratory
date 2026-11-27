param(
    [Parameter(Mandatory)][ValidatePattern('^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$')][string]$Tag,
    [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{40}$')][string]$Commit,
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')][string]$Repository
)
# CI verification only: no user-facing downloader, proxy, mirror or updater.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$headers = @{ Accept = 'application/vnd.github+json'; 'User-Agent' = 'QML-release-integrity-verification' }
$api = "https://api.github.com/repos/$Repository/releases"
$release = Invoke-RestMethod -Uri "$api/tags/$Tag" -Headers $headers
if ($release.draft -or $release.tag_name -cne $Tag) { throw 'Public release identity mismatch' }
$expectedPre = $env:ALLOW_PRERELEASE -eq 'true'
if ([bool]$release.prerelease -ne $expectedPre) { throw 'Public release maturity flag mismatch' }
if (-not $expectedPre) {
    $latest = Invoke-RestMethod -Uri "$api/latest" -Headers $headers
    if ($latest.tag_name -cne $Tag) { throw 'Release is not latest' }
}
$dist = Join-Path $PWD "target/public-verification-$Tag"
if (Test-Path -LiteralPath $dist) { throw 'Verification staging already exists' }
New-Item -ItemType Directory -Path $dist | Out-Null
$names = @(
    "Quantum-Morph-Laboratory-$Tag-Windows-x64.exe",
    "Quantum-Morph-Laboratory-$Tag-Windows-x64-Portable.zip",
    "Quantum-Morph-Laboratory-$Tag-Linux-x64.tar.gz",
    "Quantum-Morph-Laboratory-$Tag-Source.tar.gz",
    'RELEASE_MANIFEST.json', 'SHA256SUMS.txt'
)
if ($env:SIGN_RELEASES -eq 'true') { $names += 'RELEASE_MANIFEST.sigstore.json' }
if ($release.assets.Count -ne $names.Count) { throw 'Unexpected public asset coverage' }
foreach ($name in $names) {
    $assets = @($release.assets | Where-Object { $_.name -ceq $name })
    if ($assets.Count -ne 1 -or $assets[0].size -le 0 -or $assets[0].size -gt 1GB) { throw "Missing, duplicate or oversized public asset: $name" }
    # Resolve only native GitHub's asset API internally. Never print or publish asset URLs.
    $assetUri = "https://api.github.com/repos/$Repository/releases/assets/$($assets[0].id)?verification=$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
    Invoke-WebRequest -Uri $assetUri -Headers @{Accept='application/octet-stream'; 'User-Agent'='QML-release-integrity-verification'} -OutFile (Join-Path $dist $name)
    if ((Get-Item -LiteralPath (Join-Path $dist $name)).Length -ne $assets[0].size) { throw "Public asset byte count mismatch: $name" }
}
New-Item -ItemType Directory -Force -Path target | Out-Null
$contract = Join-Path $PWD 'target/public-release-contract'
rustc --edition=2024 -D warnings scripts/release_contract.rs -o $contract
if ($LASTEXITCODE -ne 0) { throw 'Cannot compile independent verifier' }
& $contract verify $dist $Tag $Commit
if ($LASTEXITCODE -ne 0) { throw 'Public provenance or checksum verification failed' }
$portable = Join-Path $dist 'windows-portable'
Expand-Archive -LiteralPath "$dist/Quantum-Morph-Laboratory-$Tag-Windows-x64-Portable.zip" -DestinationPath $portable
$standaloneHash = (Get-FileHash -LiteralPath "$dist/Quantum-Morph-Laboratory-$Tag-Windows-x64.exe" -Algorithm SHA256).Hash
$portableHash = (Get-FileHash -LiteralPath "$portable/qml-desktop.exe" -Algorithm SHA256).Hash
if ($standaloneHash -cne $portableHash) { throw 'Public standalone and portable desktop differ' }
& $contract package $portable
if ($LASTEXITCODE -ne 0) { throw 'Public Windows package documentation is incomplete' }
if (-not $IsWindows) {
    tar -xzf "$dist/Quantum-Morph-Laboratory-$Tag-Linux-x64.tar.gz" -C $dist
    if ($LASTEXITCODE -ne 0) { throw 'Public Linux archive cannot be extracted' }
    $linux = Join-Path $dist "Quantum-Morph-Laboratory-$Tag-Linux-x64"
    & $contract package $linux
    if ($LASTEXITCODE -ne 0) { throw 'Public Linux package documentation is incomplete' }
    $version = & "$linux/qml" --version
    if ($LASTEXITCODE -ne 0 -or $version.Trim() -cne "qml $($Tag.Substring(1))") { throw 'Public Linux CLI smoke test failed' }
} else {
    $version = & "$portable/qml.exe" --version
    if ($LASTEXITCODE -ne 0 -or $version.Trim() -cne "qml $($Tag.Substring(1))") { throw 'Public Windows CLI smoke test failed' }
}
if ($env:SIGN_RELEASES -eq 'true') {
    cosign verify-blob --bundle "$dist/RELEASE_MANIFEST.sigstore.json" --certificate-identity "https://github.com/$Repository/.github/workflows/release.yml@refs/tags/$Tag" --certificate-oidc-issuer https://token.actions.githubusercontent.com "$dist/RELEASE_MANIFEST.json"
    if ($LASTEXITCODE -ne 0) { throw 'Public signature identity validation failed' }
}
Write-Output "Verified native public release $Tag : $($names.Count) assets, exact source commit, complete SHA-256 coverage, latest and maturity flags"
