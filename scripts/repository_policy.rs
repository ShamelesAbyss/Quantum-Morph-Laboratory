#![forbid(unsafe_code)]
//! Permanent documentation distribution and memorial validation.
use std::{
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
};
pub fn read_text_bounded(path: &Path, limit: u64) -> Result<String, String> {
    let mut text = String::new();
    fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take(limit.checked_add(1).ok_or("document length overflow")?)
        .read_to_string(&mut text)
        .map_err(|e| e.to_string())?;
    if text.len() as u64 > limit {
        return Err("document exceeds audit budget".into());
    }
    Ok(text)
}
pub const MEMORIAL: &str = "### In Loving Memory of Ashlan Haley Rapponotti\n\n*June 21, 2003 – November 13, 2026*\n\nFor the beautiful woman who listened to my passions, believed in me, and made my world a happier place simply by being part of it.\n\nEvery creation, every discovery, and every milestone reached carries a piece of my heart dedicated to you.\n\n**Forever loved. Forever missed. Forever remembered.** 💜\n";
pub const RELEASES: &str = "https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases";
fn normalized(text: &str) -> String {
    let mut result = text.to_ascii_lowercase().replace('\\', "/");
    for _ in 0..4 {
        let before = result.clone();
        let mut decoded = String::new();
        let bytes = result.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%'
                && i + 2 < bytes.len()
                && bytes[i + 1].is_ascii()
                && bytes[i + 2].is_ascii()
                && let Ok(value) = u8::from_str_radix(&result[i + 1..i + 3], 16)
                && value.is_ascii()
            {
                decoded.push(char::from(value));
                i += 3;
                continue;
            }
            let character = result[i..].chars().next().expect("character boundary");
            decoded.push(character);
            i += character.len_utf8();
        }
        result = decoded
            .replace("&#47;", "/")
            .replace("&#x2f;", "/")
            .replace("&sol;", "/")
            .replace("&#46;", ".")
            .replace("&#x2e;", ".")
            .replace("&period;", ".")
            .replace("&amp;", "&");
        if result == before {
            break;
        }
    }
    result
}
pub fn validate_distribution(text: &str) -> Result<(), String> {
    let decoded = normalized(text);
    for pattern in [
        "/releases/download/",
        "/releases/latest/download/",
        "browser_download_url",
    ] {
        if decoded.contains(pattern) {
            return Err(format!("prohibited distribution pattern: {pattern}"));
        }
    }
    for word in decoded.split(|c: char| {
        c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>' | '(' | ')' | '\u{60}' | '[' | ']')
    }) {
        if !word.starts_with("http://") && !word.starts_with("https://") {
            continue;
        }
        let host = word
            .split("://")
            .nth(1)
            .unwrap_or("")
            .split('/')
            .next()
            .unwrap_or("");
        if [
            "bit.ly",
            "tinyurl.com",
            "t.co",
            "tiny.cc",
            "is.gd",
            "goo.gl",
            "aka.ms",
        ]
        .contains(&host)
        {
            return Err(format!("redirect/shortener URL: {host}"));
        }
        let path = word
            .split(['?', '#'])
            .next()
            .unwrap_or(word)
            .trim_end_matches([',', ';', '!']);
        if [
            ".exe",
            ".apk",
            ".zip",
            ".tar.gz",
            ".tgz",
            ".tar.xz",
            ".7z",
            ".msi",
            ".dmg",
            ".deb",
            ".rpm",
            ".appimage",
            ".sha256",
        ]
        .iter()
        .any(|suffix| path.ends_with(suffix))
            || path.ends_with("sha256sums.txt")
        {
            return Err("direct executable/archive/checksum URL in documentation".into());
        }
    }
    Ok(())
}
fn markdown_targets(text: &str) -> Vec<(String, String, bool)> {
    let mut found = Vec::new();
    let mut offset = 0;
    while let Some(relative) = text[offset..].find("](") {
        let end = offset + relative;
        let start = text[..end].rfind('[').unwrap_or(end);
        let label = &text[start + 1..end];
        let after = end + 2;
        let Some(close) = text[after..].find(')') else {
            break;
        };
        let target = text[after..after + close]
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_owned();
        found.push((
            label.to_owned(),
            target,
            start > 0 && text.as_bytes()[start - 1] == b'!',
        ));
        offset = after + close + 1;
    }
    found
}
fn html_attributes(text: &str, attribute: &str) -> Vec<String> {
    let pattern = format!("{attribute}=\"");
    let mut out = Vec::new();
    let mut offset = 0;
    while let Some(at) = text[offset..].find(&pattern) {
        let start = offset + at + pattern.len();
        let Some(end) = text[start..].find('"') else {
            break;
        };
        out.push(text[start..start + end].to_owned());
        offset = start + end + 1;
    }
    out
}
pub fn exact_local_path(root: &Path, target: &str) -> Result<PathBuf, String> {
    let target = target.split(['#', '?']).next().unwrap_or(target);
    let mut path = root.to_owned();
    for component in Path::new(target).components() {
        match component {
            Component::CurDir => {}
            Component::Normal(name) => {
                let matched = fs::read_dir(&path)
                    .map_err(|e| e.to_string())?
                    .filter_map(Result::ok)
                    .any(|entry| entry.file_name() == name);
                if !matched {
                    return Err(format!("missing or incorrectly cased path: {target}"));
                }
                path.push(name);
            }
            _ => return Err(format!("absolute/escaping repository reference: {target}")),
        }
    }
    if !path.is_file() {
        return Err(format!("reference is not a file: {target}"));
    }
    Ok(path)
}
pub fn validate_readme(root: &Path, text: &str) -> Result<(), String> {
    validate_distribution(text)?;
    let text = text.replace("\r\n", "\n");
    if !text.ends_with(MEMORIAL) {
        return Err("README must end with the exact permanent memorial and one newline".into());
    }
    if !text.contains("Compiled executables, portable packages, source archives, release notes and SHA-256 checksums") {return Err("missing approved neutral distribution wording".into());}
    let mut targets = markdown_targets(&text);
    targets.extend(
        html_attributes(&text, "src")
            .into_iter()
            .map(|t| (String::new(), t, true)),
    );
    targets.extend(
        html_attributes(&text, "href")
            .into_iter()
            .map(|t| (String::new(), t, false)),
    );
    for (label, target, image) in targets {
        if normalized(&label).starts_with("download") {
            return Err("download action labels/buttons are prohibited".into());
        }
        if target.starts_with('#') {
            continue;
        }
        if target.starts_with("https://") {
            if image {
                if !target.starts_with(
                    "https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/workflows/",
                ) || !target.contains("/badge.svg")
                {
                    return Err(format!(
                        "artwork must be committed locally; only native CI badges may be remote: {target}"
                    ));
                }
            } else if !target
                .starts_with("https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory")
                && !target.starts_with("https://www.rust-lang.org/")
                && !target.starts_with("https://doc.rust-lang.org/")
            {
                return Err(format!("noncanonical README navigation target: {target}"));
            }
        } else if target.contains("://") {
            return Err(format!("unsupported README link: {target}"));
        } else {
            exact_local_path(root, &target)?;
        }
    }
    if text.to_ascii_lowercase().contains("<button") {
        return Err("README distribution buttons are prohibited".into());
    }
    Ok(())
}
fn documents(dir: &Path, output: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if entry.file_type().map_err(|e| e.to_string())?.is_symlink() {
            continue;
        }
        if path.is_dir() {
            if ![".git", "target", "node_modules"]
                .iter()
                .any(|name| path.file_name().is_some_and(|n| n == *name))
            {
                documents(&path, output)?;
            }
        } else if path
            .extension()
            .is_some_and(|e| e == "md" || e == "mdc" || e == "txt")
        {
            output.push(path);
        }
    }
    Ok(())
}
pub fn validate_repository(root: &Path) -> Result<usize, String> {
    let readme = read_text_bounded(&root.join("README.md"), 16 * 1024 * 1024)?;
    validate_readme(root, &readme)?;
    for path in [
        ".cursor/rules/github-distribution-compliance.mdc",
        "AGENTS.md",
    ] {
        let rules = read_text_bounded(&root.join(path), 16 * 1024 * 1024)?;
        if !rules.contains("explicit owner authorization AND clarification from GitHub Support") {
            return Err("permanent owner/Support enforcement rule is missing".into());
        }
    }
    let mut files = Vec::new();
    documents(root, &mut files)?;
    for file in &files {
        if fs::metadata(file).map_err(|e| e.to_string())?.len() > 16 * 1024 * 1024 {
            return Err(format!("document exceeds audit budget: {}", file.display()));
        }
        validate_distribution(&read_text_bounded(file, 16 * 1024 * 1024)?)
            .map_err(|e| format!("{}: {e}", file.display()))?;
    }
    Ok(files.len())
}
#[cfg(not(test))]
fn main() {
    let root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    match validate_repository(&root) {
        Ok(count) => println!(
            "Repository compliance verified: {count} documents, artwork paths and permanent memorial"
        ),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_direct_encoded_and_alternate_distribution() {
        for target in [
            "https://github.com/example/repo/releases/download/v1/app.exe",
            "https://github.com/example/repo/releases/latest/download/app",
            "https://github.com/example/repo/releases%2Fdownload%2Fv1%2Fapp",
            "https://github.com/example/repo/releases&#47;download&#47;v1",
            "https://mirror.example/app.zip?raw=1",
            "https://example.com/SHA256SUMS.txt",
            "https://example.com/app%2EEXE",
            "https://bit.ly/example",
            "browser_download_url",
        ] {
            assert!(validate_distribution(target).is_err(), "{target}");
        }
        assert!(validate_distribution(RELEASES).is_ok());
    }
    #[test]
    fn requires_exact_final_memorial() {
        let valid = format!(
            "Compiled executables, portable packages, source archives, release notes and SHA-256 checksums are available through the repository's GitHub Releases section.\n\n{MEMORIAL}"
        );
        assert!(validate_readme(Path::new("."), &valid).is_ok());
        assert!(validate_readme(Path::new("."), &format!("{valid}extra")).is_err());
        assert!(
            validate_readme(
                Path::new("."),
                &valid.replace("**Forever loved.", "Forever loved.")
            )
            .is_err()
        );
    }
    #[test]
    fn artwork_case_and_missing_files_fail_on_every_os() {
        let root = std::env::temp_dir().join(format!(
            "qml-policy-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        fs::write(root.join("hero.svg"), "<svg/>").unwrap();
        assert!(exact_local_path(&root, "hero.svg").is_ok());
        assert!(exact_local_path(&root, "Hero.svg").is_err());
        assert!(exact_local_path(&root, "missing.svg").is_err());
        assert!(exact_local_path(&root, "../hero.svg").is_err());
        fs::remove_file(root.join("hero.svg")).unwrap();
        fs::remove_dir(root).unwrap();
    }
    #[test]
    fn repository_documentation_is_compliant() {
        let root = option_env!("CARGO_MANIFEST_DIR")
            .map(|s| Path::new(s).join("../.."))
            .unwrap_or_else(|| PathBuf::from("."));
        validate_repository(&root).unwrap();
    }
}
