//! Versioned, bounded cache envelopes; legacy caches require regeneration.
use crate::{Sha256, sha256_reader};
use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
};
pub const MAX_CACHE_BYTES: u64 = 512 * 1024 * 1024;
pub fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
pub fn write(
    path: &Path,
    identity: &str,
    encode: impl FnOnce(&mut File) -> io::Result<()>,
) -> io::Result<()> {
    if identity.len() != 64 || !identity.bytes().all(|v| v.is_ascii_hexdigit()) {
        return Err(invalid("cache identity must be SHA-256"));
    }
    let mut temp = crate::publication::stage(path)?;
    let file = temp.as_file_mut();
    file.write_all(b"QMLCACHE2\0")?;
    file.write_all(identity.as_bytes())?;
    encode(file)?;
    if file.stream_position()? > MAX_CACHE_BYTES - 64 {
        return Err(invalid("cache exceeds budget"));
    }
    file.seek(SeekFrom::Start(0))?;
    let digest = sha256_reader(&mut *file)?;
    file.seek(SeekFrom::End(0))?;
    file.write_all(digest.as_bytes())?;
    crate::publication::publish(temp, path, None, false)
}
pub fn read(path: &Path, identity: &str) -> io::Result<Reader> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    if !(138..=MAX_CACHE_BYTES).contains(&size) {
        return Err(invalid(
            "unverified or oversized cache; regenerate in a new directory",
        ));
    }
    let mut bytes = Vec::new();
    file.take(MAX_CACHE_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != size {
        return Err(invalid("cache changed while reading"));
    }
    let end = bytes.len() - 64;
    let mut sha = Sha256::new();
    sha.update(&bytes[..end]);
    if sha.finalize_hex().as_bytes() != &bytes[end..] {
        return Err(invalid("cache checksum mismatch"));
    }
    if &bytes[..10] != b"QMLCACHE2\0" || &bytes[10..74] != identity.as_bytes() {
        return Err(invalid(
            "unverified cache provenance; regenerate in a new directory",
        ));
    }
    bytes.truncate(end);
    Ok(Reader { bytes, offset: 74 })
}
pub struct Reader {
    bytes: Vec<u8>,
    offset: usize,
}
impl Reader {
    pub fn bytes(&mut self, n: usize) -> io::Result<&[u8]> {
        let end = self
            .offset
            .checked_add(n)
            .ok_or_else(|| invalid("length overflow"))?;
        let v = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| invalid("truncated cache"))?;
        self.offset = end;
        Ok(v)
    }
    pub fn len(&mut self, max: usize) -> io::Result<usize> {
        let n = u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()) as usize;
        if n > max {
            return Err(invalid("cache length exceeds budget"));
        }
        Ok(n)
    }
    pub fn count(&mut self) -> io::Result<usize> {
        usize::try_from(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
            .map_err(|_| invalid("count overflow"))
    }
    pub fn string(&mut self) -> io::Result<String> {
        let n = self.len(4096)?;
        String::from_utf8(self.bytes(n)?.to_vec()).map_err(|_| invalid("invalid UTF-8"))
    }
    pub fn values(&mut self) -> io::Result<Vec<f64>> {
        let n = self.len(16_777_216)?;
        let bytes = self.bytes(n.checked_mul(4).ok_or_else(|| invalid("length overflow"))?)?;
        let mut v = Vec::with_capacity(n);
        for c in bytes.as_chunks::<4>().0 {
            let x = f32::from_le_bytes(*c);
            if !x.is_finite() {
                return Err(invalid("non-finite cache value"));
            }
            v.push(f64::from(x));
        }
        Ok(v)
    }
    pub fn finish(self) -> io::Result<()> {
        if self.offset != self.bytes.len() {
            return Err(invalid("trailing cache data"));
        }
        Ok(())
    }
}
pub fn length(w: &mut impl Write, n: usize) -> io::Result<()> {
    w.write_all(
        &u32::try_from(n)
            .map_err(|_| invalid("length overflow"))?
            .to_le_bytes(),
    )
}
pub fn string(w: &mut impl Write, s: &str) -> io::Result<()> {
    if s.len() > 4096 {
        return Err(invalid("string too long"));
    }
    length(w, s.len())?;
    w.write_all(s.as_bytes())
}
pub fn values(w: &mut impl Write, v: &[f64]) -> io::Result<()> {
    length(w, v.len())?;
    for &x in v {
        let x = x as f32;
        if !x.is_finite() {
            return Err(invalid("non-finite cache value"));
        }
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod aegis_cache_tests {
    use super::*;
    #[test]
    fn envelope_rejects_corruption_wrong_identity_lengths_and_legacy() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache");
        let identity = crate::sha256_hex(b"model+config+tokenizer+datasets");
        write(&path, &identity, |w| values(w, &[1.0, 2.0])).unwrap();
        let original = std::fs::read(&path).unwrap();
        assert!(write(&path, &identity, |w| values(w, &[3.0])).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), original);
        assert!(read(&path, &crate::sha256_hex(b"changed weights")).is_err());
        let mut r = read(&path, &identity).unwrap();
        assert_eq!(r.values().unwrap(), vec![1.0, 2.0]);
        r.finish().unwrap();
        let mut corrupt = original.clone();
        corrupt[80] ^= 1;
        std::fs::write(&path, corrupt).unwrap();
        assert!(read(&path, &identity).is_err());
        std::fs::write(&path, b"QCAL1\0").unwrap();
        assert!(read(&path, &identity).is_err());
        std::fs::remove_file(&path).unwrap();
        write(&path, &identity, |w| w.write_all(&u32::MAX.to_le_bytes())).unwrap();
        assert!(read(&path, &identity).unwrap().values().is_err());
        assert!(values(&mut Vec::new(), &[f64::NAN]).is_err());
        assert!(values(&mut Vec::new(), &[f64::MAX]).is_err());
    }
}
