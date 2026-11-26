//! Safe sibling staging with explicit overwrite authorization.
use std::{fs, io, path::Path};

pub fn validate_destination(path: &Path, source: Option<&Path>) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(meta) => {
            if !meta.is_file() || meta.file_type().is_symlink() {
                return Err(io::Error::other("destination must be a regular file"));
            }
            if let Some(src) = source
                && same_file::is_same_file(src, path)?
            {
                return Err(io::Error::other("destination aliases immutable source"));
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    if let Some(src) = source {
        fs::canonicalize(src)?;
    }
    Ok(())
}

pub fn stage(path: &Path) -> io::Result<tempfile::NamedTempFile> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    tempfile::Builder::new()
        .prefix(".qml-stage-")
        .tempfile_in(parent)
}

pub fn publish(
    temp: tempfile::NamedTempFile,
    path: &Path,
    source: Option<&Path>,
    overwrite: bool,
) -> io::Result<()> {
    temp.as_file().sync_all()?;
    validate_destination(path, source)?;
    if overwrite {
        temp.persist(path).map_err(|e| e.error)?;
    } else {
        temp.persist_noclobber(path).map_err(|e| e.error)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[test]
    fn distinct_staging_and_authorized_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("artifact");
        fs::write(&path, b"old").unwrap();
        let mut first = stage(&path).unwrap();
        let second = stage(&path).unwrap();
        assert_ne!(first.path(), second.path());
        first.write_all(b"new").unwrap();
        publish(first, &path, None, true).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new");
    }
    #[cfg(unix)]
    #[test]
    fn rejects_symbolic_and_dangling_links() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let link = dir.path().join("link");
        fs::write(&source, b"source").unwrap();
        std::os::unix::fs::symlink(&source, &link).unwrap();
        assert!(validate_destination(&link, Some(&source)).is_err());
        fs::remove_file(source).unwrap();
        assert!(validate_destination(&link, None).is_err());
    }
    #[test]
    fn failed_publication_preserves_existing_and_source() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("source");
        let dst = dir.path().join("destination");
        fs::write(&src, b"source").unwrap();
        fs::write(&dst, b"old").unwrap();
        let mut temp = stage(&dst).unwrap();
        temp.write_all(b"new").unwrap();
        assert!(publish(temp, &dst, Some(&src), false).is_err());
        assert_eq!(fs::read(&dst).unwrap(), b"old");
        fs::remove_file(&dst).unwrap();
        fs::hard_link(&src, &dst).unwrap();
        assert!(validate_destination(&dst, Some(&src)).is_err());
        assert_eq!(fs::read(&src).unwrap(), b"source");
        let temp = stage(&dst).unwrap();
        assert!(publish(temp, dir.path(), None, true).is_err());
        assert_eq!(fs::read(&src).unwrap(), b"source");
    }
}
