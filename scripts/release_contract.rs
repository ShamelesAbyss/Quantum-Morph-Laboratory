#![forbid(unsafe_code)]
//! CI-only local release contract. It does not implement a downloader or an updater.
#[path = "repository_policy.rs"]
pub mod policy;
#[path = "../crates/qml-core/src/sha256.rs"]
pub mod sha256;
use std::{fs, io::Write, path::Path, process::Command};
pub fn workspace_version(text: &str) -> Result<String, String> {
    let mut section = false;
    let mut version = None;
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') {
            section = line == "[workspace.package]";
            continue;
        }
        if section
            && let Some(value) = line
                .strip_prefix("version")
                .and_then(|s| s.trim_start().strip_prefix('='))
        {
            let value = value
                .trim()
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .ok_or("invalid workspace version")?;
            if version.replace(value.to_owned()).is_some() {
                return Err("duplicate workspace version".into());
            }
        }
    }
    version.ok_or_else(|| "missing workspace version".into())
}
pub fn validate_tag(version: &str, tag: &str, allow_prerelease: bool) -> Result<(), String> {
    if tag != format!("v{version}") {
        return Err("version and exact release tag disagree".into());
    }
    let base = version.split('-').next().unwrap_or("");
    if base.split('.').count() != 3
        || base
            .split('.')
            .any(|part| part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()))
    {
        return Err("expected semantic major.minor.patch".into());
    }
    if version.contains('-') && !allow_prerelease {
        return Err("prerelease requires explicit owner authorization".into());
    }
    if !version
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-')
    {
        return Err("unsafe version/tag".into());
    }
    Ok(())
}
fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
pub fn validate(
    root: &Path,
    tag: &str,
    commit: &str,
    allow_prerelease: bool,
) -> Result<(), String> {
    policy::validate_repository(root)?;
    let version = workspace_version(&policy::read_text_bounded(
        &root.join("Cargo.toml"),
        1024 * 1024,
    )?)?;
    validate_tag(&version, tag, allow_prerelease)?;
    if commit.len() != 40 || !commit.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("expected full source commit SHA".into());
    }
    if git(root, &["rev-parse", "HEAD"])? != commit {
        return Err("checkout does not match requested release commit".into());
    }
    if git(root, &["rev-parse", &format!("{tag}^{{commit}}")])? != commit {
        return Err("tag does not point at the checked-out source".into());
    }
    let notes = policy::read_text_bounded(&root.join("RELEASE_NOTES.md"), 16 * 1024 * 1024)?;
    let heading = notes
        .lines()
        .find(|line| line.starts_with("# "))
        .ok_or("missing release title")?;
    if !heading.starts_with(&format!("# {tag} | ")) {
        return Err("release notes do not identify the exact release tag".into());
    }
    if !notes.replace("\r\n", "\n").ends_with(policy::MEMORIAL) {
        return Err("release notes must preserve the exact final memorial".into());
    }
    Ok(())
}
pub fn artifact_names(tag: &str) -> [String; 4] {
    [
        "Windows-x64.exe",
        "Windows-x64-Portable.zip",
        "Linux-x64.tar.gz",
        "Source.tar.gz",
    ]
    .map(|suffix| format!("Quantum-Morph-Laboratory-{tag}-{suffix}"))
}
fn file_hash(path: &Path) -> Result<(u64, String), String> {
    let meta = fs::symlink_metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !meta.is_file()
        || meta.file_type().is_symlink()
        || meta.len() == 0
        || meta.len() > 1024 * 1024 * 1024
    {
        return Err("release asset must be a nonempty regular file within 1 GiB".into());
    }
    let hash = sha256::sha256_reader(fs::File::open(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if fs::metadata(path).map_err(|e| e.to_string())?.len() != meta.len() {
        return Err("release asset changed while hashing".into());
    }
    Ok((meta.len(), hash))
}
fn validate_identifiers(tag: &str, commit: &str) -> Result<(), String> {
    validate_tag(
        tag.strip_prefix('v').ok_or("missing version prefix")?,
        tag,
        true,
    )?;
    if commit.len() != 40 || !commit.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid source commit".into());
    }
    Ok(())
}
pub fn manifest(dist: &Path, tag: &str, commit: &str, signed: bool) -> Result<String, String> {
    validate_identifiers(tag, commit)?;
    let mut entries = Vec::new();
    for name in artifact_names(tag) {
        let (bytes, hash) = file_hash(&dist.join(&name))?;
        entries.push(format!(
            "    {{\"name\":\"{name}\",\"bytes\":{bytes},\"sha256\":\"{hash}\"}}"
        ));
    }
    let signing = if signed {
        "sigstore-keyless"
    } else {
        "not-configured"
    };
    Ok(format!(
        "{{\n  \"schema_version\":1,\n  \"tag\":\"{tag}\",\n  \"source_commit\":\"{commit}\",\n  \"signature_mode\":\"{signing}\",\n  \"distribution\":\"native GitHub Releases only\",\n  \"assets\":[\n{}\n  ]\n}}\n",
        entries.join(",\n")
    ))
}
fn create_file(path: &Path, text: &str) -> Result<(), String> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.write_all(text.as_bytes())
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())
}
pub fn prepare(dist: &Path, tag: &str, commit: &str, signed: bool) -> Result<(), String> {
    create_file(
        &dist.join("RELEASE_MANIFEST.json"),
        &manifest(dist, tag, commit, signed)?,
    )
}
pub fn verify_manifest(dist: &Path, tag: &str, commit: &str, signed: bool) -> Result<(), String> {
    let actual = policy::read_text_bounded(&dist.join("RELEASE_MANIFEST.json"), 1024 * 1024)?;
    if actual != manifest(dist, tag, commit, signed)? {
        return Err("release manifest provenance, sizes or integrity mismatch".into());
    }
    Ok(())
}
fn sums(dist: &Path, tag: &str, signed: bool) -> Result<String, String> {
    let mut names = artifact_names(tag).to_vec();
    names.push("RELEASE_MANIFEST.json".into());
    if signed {
        names.push("RELEASE_MANIFEST.sigstore.json".into());
    }
    let mut text = String::new();
    for name in names {
        let (_, hash) = file_hash(&dist.join(&name))?;
        text.push_str(&format!("{hash}  {name}\n"));
    }
    Ok(text)
}
pub fn checksums(dist: &Path, tag: &str, commit: &str, signed: bool) -> Result<(), String> {
    verify_manifest(dist, tag, commit, signed)?;
    create_file(&dist.join("SHA256SUMS.txt"), &sums(dist, tag, signed)?)
}
pub fn verify(dist: &Path, tag: &str, commit: &str, signed: bool) -> Result<(), String> {
    verify_manifest(dist, tag, commit, signed)?;
    let expected = sums(dist, tag, signed)?;
    let actual =
        policy::read_text_bounded(&dist.join("SHA256SUMS.txt"), 1024 * 1024)?.replace("\r\n", "\n");
    if actual != expected {
        return Err("release checksum coverage or integrity mismatch".into());
    }
    Ok(())
}
pub fn check_package(root: &Path) -> Result<(), String> {
    policy::validate_readme(
        root,
        &policy::read_text_bounded(&root.join("README.md"), 16 * 1024 * 1024)?,
    )?;
    if !policy::read_text_bounded(&root.join("RELEASE_NOTES.md"), 16 * 1024 * 1024)?
        .replace("\r\n", "\n")
        .ends_with(policy::MEMORIAL)
    {
        return Err("distributed release notes memorial mismatch".into());
    }
    if !policy::read_text_bounded(&root.join("LICENSE"), 1024 * 1024)?.starts_with("MIT License") {
        return Err("missing distributed MIT license".into());
    }
    Ok(())
}
#[cfg(not(test))]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = (|| -> Result<(), String> {
        match args.as_slice() {
            [mode,root,tag,commit] if mode=="validate"=>validate(Path::new(root),tag,commit,std::env::var("ALLOW_PRERELEASE").as_deref()==Ok("true")),
            [mode,root] if mode=="package"=>check_package(Path::new(root)),
            [mode,dist,tag,commit]=>{
                let signed=std::env::var("SIGN_RELEASES").as_deref()==Ok("true");
                match mode.as_str() {"prepare"=>prepare(Path::new(dist),tag,commit,signed),"manifest"=>verify_manifest(Path::new(dist),tag,commit,signed),"checksums"=>checksums(Path::new(dist),tag,commit,signed),"verify"=>verify(Path::new(dist),tag,commit,signed),_=>Err("unknown release-contract operation".into())}
            }
            _=>Err("expected validate ROOT TAG COMMIT, package ROOT, or prepare/manifest/checksums/verify DIST TAG COMMIT".into())
        }
    })();
    match result {
        Ok(()) => println!("Release contract verified"),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    const COMMIT: &str = "1111111111111111111111111111111111111111";
    #[test]
    fn exact_versions_and_prerelease_authorization_are_required() {
        assert_eq!(
            workspace_version("[workspace.package]\nversion = \"0.6.3\"\n[other]\nversion = \"9\"")
                .unwrap(),
            "0.6.3"
        );
        assert!(validate_tag("0.6.3", "v0.6.2", false).is_err());
        assert!(validate_tag("0.6.3-rc.1", "v0.6.3-rc.1", false).is_err());
        assert!(validate_tag("0.6.3-rc.1", "v0.6.3-rc.1", true).is_ok());
        assert!(validate_tag("../file", "v../file", true).is_err());
        assert!(workspace_version("[workspace.package]\nversion=\"1\"\nversion=\"2\"").is_err());
    }
    #[test]
    fn package_requires_license_memorial_and_resolvable_artwork() {
        let dir = std::env::temp_dir().join(format!(
            "qml-package-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&dir).unwrap();
        let readme = format!(
            "Compiled executables, portable packages, source archives, release notes and SHA-256 checksums\n\n{}",
            policy::MEMORIAL
        );
        fs::write(dir.join("README.md"), &readme).unwrap();
        fs::write(
            dir.join("RELEASE_NOTES.md"),
            policy::MEMORIAL.replace('\n', "\r\n"),
        )
        .unwrap();
        assert!(check_package(&dir).is_err());
        fs::write(dir.join("LICENSE"), "MIT License\n").unwrap();
        check_package(&dir).unwrap();
        fs::write(
            dir.join("README.md"),
            format!("![art](missing.svg)\n{readme}"),
        )
        .unwrap();
        assert!(check_package(&dir).is_err());
        fs::write(dir.join("missing.svg"), "<svg/>").unwrap();
        check_package(&dir).unwrap();
        fs::write(dir.join("RELEASE_NOTES.md"), "dedication removed").unwrap();
        assert!(check_package(&dir).is_err());
        for entry in fs::read_dir(&dir).unwrap() {
            fs::remove_file(entry.unwrap().path()).unwrap();
        }
        fs::remove_dir(dir).unwrap();
    }
    #[test]
    fn manifest_checksum_tampering_and_missing_assets_fail_closed() {
        let dir = std::env::temp_dir().join(format!(
            "qml-release-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&dir).unwrap();
        assert!(prepare(&dir, "v0.6.3", COMMIT, false).is_err());
        for name in artifact_names("v0.6.3") {
            fs::write(dir.join(name), b"synthetic package fixture").unwrap();
        }
        prepare(&dir, "v0.6.3", COMMIT, false).unwrap();
        assert!(prepare(&dir, "v0.6.3", COMMIT, false).is_err());
        checksums(&dir, "v0.6.3", COMMIT, false).unwrap();
        verify(&dir, "v0.6.3", COMMIT, false).unwrap();
        assert!(verify(&dir, "v0.6.3", COMMIT, true).is_err());
        assert!(
            verify(
                &dir,
                "v0.6.3",
                "2222222222222222222222222222222222222222",
                false
            )
            .is_err()
        );
        fs::write(dir.join(artifact_names("v0.6.3")[0].clone()), b"tampered").unwrap();
        assert!(verify(&dir, "v0.6.3", COMMIT, false).is_err());
        for entry in fs::read_dir(&dir).unwrap() {
            fs::remove_file(entry.unwrap().path()).unwrap();
        }
        fs::remove_dir(dir).unwrap();
    }
}
