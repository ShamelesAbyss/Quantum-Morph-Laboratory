param([Parameter(Mandatory)][ValidatePattern('^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$')][string]$Tag)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$platform = if ($IsWindows) { 'Windows' } else { 'Linux' }
$extension = if ($IsWindows) { '.exe' } else { '' }
$contract = Join-Path $PWD 'target/release-contract.exe'
$base = "Quantum-Morph-Laboratory-$Tag-$platform-x64"
$package = Join-Path $PWD "target/package-$Tag-$platform"
if (Test-Path -LiteralPath $package) { throw 'Package staging already exists; refusing overwrite' }
if (Test-Path -LiteralPath 'dist') { throw 'Distribution staging already exists; refusing overwrite' }
New-Item -ItemType Directory -Path $package, 'dist' | Out-Null
Copy-Item -LiteralPath "target/release/qml-desktop$extension" -Destination $package
Copy-Item -LiteralPath "target/release/qml$extension" -Destination $package
# Copy tracked documentation and all referenced locally committed visual assets.
$tracked = @(git ls-files)
if ($LASTEXITCODE -ne 0) { throw 'Cannot inventory tracked documentation' }
foreach ($file in $tracked) {
    if ($file -match '^[^/]+\.md$|^LICENSE$|^(assets|docs|research|experiments|\.cursor/rules)/') {
        $target = Join-Path $package $file
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
        Copy-Item -LiteralPath $file -Destination $target
    }
}
& $contract package $package
if ($LASTEXITCODE -ne 0) { throw 'Packaged documentation validation failed' }
$version = & (Join-Path $package "qml$extension") --version
if ($LASTEXITCODE -ne 0 -or $version.Trim() -cne "qml $($Tag.Substring(1))") { throw 'Packaged CLI version smoke test failed' }
if ($IsWindows) {
    Copy-Item -LiteralPath "$package/qml-desktop.exe" -Destination "dist/$base.exe"
    Compress-Archive -Path "$package/*" -DestinationPath "dist/$base-Portable.zip"
} else {
    # Package names are stable and contain a single top-level directory.
    $renamed = Join-Path (Split-Path $package -Parent) $base
    if (Test-Path -LiteralPath $renamed) { throw 'Archive staging already exists' }
    Move-Item -LiteralPath $package -Destination $renamed
    tar -C (Split-Path $renamed -Parent) -czf "dist/$base.tar.gz" $base
    if ($LASTEXITCODE -ne 0) { throw 'Linux archive creation failed' }
}
Write-Output "Packaged $platform desktop, CLI, complete documentation and exact memorial"
