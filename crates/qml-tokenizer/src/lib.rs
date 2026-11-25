//! Pure safe Rust Hugging Face `tokenizer.json` Byte-Level BPE tokenizer.
//!
//! Supports GPT-2 / SmolLM / SmolLM2 / Llama-3 / Qwen2-style `tokenizer.json`
//! files without any Python or C++ runtime dependencies.

use serde::Deserialize;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Maximum permitted `tokenizer.json` file size (64 MiB) to bound memory use.
pub const MAX_TOKENIZER_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// Errors returned while loading or using a tokenizer.
#[derive(Debug)]
pub enum TokenizerError {
    Io(std::io::Error),
    FileTooLarge { size: u64, max: u64 },
    Json(serde_json::Error),
    UnsupportedModelType(String),
    InvalidMergeEntry(String),
    MissingVocab,
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "tokenizer I/O error: {err}"),
            Self::FileTooLarge { size, max } => {
                write!(f, "tokenizer.json size {size} exceeds limit {max} bytes")
            }
            Self::Json(err) => write!(f, "tokenizer.json parse error: {err}"),
            Self::UnsupportedModelType(kind) => {
                write!(
                    f,
                    "unsupported tokenizer model type '{kind}' (expected BPE)"
                )
            }
            Self::InvalidMergeEntry(entry) => {
                write!(f, "invalid BPE merge entry: '{entry}'")
            }
            Self::MissingVocab => write!(f, "tokenizer.json contains an empty vocabulary"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<std::io::Error> for TokenizerError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

/// Raw JSON schema for Hugging Face `tokenizer.json`.
#[derive(Debug, Deserialize)]
struct HfTokenizerJson {
    #[serde(default)]
    added_tokens: Vec<HfAddedToken>,
    #[serde(default)]
    pre_tokenizer: Option<serde_json::Value>,
    model: HfModel,
}

#[derive(Debug, Deserialize)]
struct HfAddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Debug, Deserialize)]
struct HfModel {
    #[serde(rename = "type")]
    model_type: Option<String>,
    vocab: HashMap<String, u32>,
    #[serde(default)]
    merges: Vec<MergeEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MergeEntry {
    SpaceSeparated(String),
    Pair([String; 2]),
}

/// Pure-Rust Byte-Level BPE tokenizer loaded from a Hugging Face `tokenizer.json`.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    merges: HashMap<(String, String), usize>,
    /// Added/special tokens sorted by descending byte length for greedy matching.
    added_tokens: Vec<(String, u32, bool)>,
    special_ids: HashMap<u32, String>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    split_individual_digits: bool,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

impl BpeTokenizer {
    /// Loads a Hugging Face `tokenizer.json` from disk with bounded file size.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let mut file = File::open(path)?;
        let size = file.metadata()?.len();
        if size > MAX_TOKENIZER_FILE_BYTES {
            return Err(TokenizerError::FileTooLarge {
                size,
                max: MAX_TOKENIZER_FILE_BYTES,
            });
        }
        let mut buf = Vec::with_capacity(size as usize);
        file.read_to_end(&mut buf)?;
        Self::from_json_bytes(&buf)
    }

    /// Parses a Hugging Face `tokenizer.json` from UTF-8 bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let parsed: HfTokenizerJson = serde_json::from_slice(bytes)?;
        if let Some(kind) = &parsed.model.model_type
            && kind != "BPE"
        {
            return Err(TokenizerError::UnsupportedModelType(kind.clone()));
        }
        if parsed.model.vocab.is_empty() {
            return Err(TokenizerError::MissingVocab);
        }

        let mut vocab = parsed.model.vocab;
        let mut special_ids = HashMap::new();
        let mut added_tokens = Vec::with_capacity(parsed.added_tokens.len());

        for tok in parsed.added_tokens {
            vocab.entry(tok.content.clone()).or_insert(tok.id);
            if tok.special {
                special_ids.insert(tok.id, tok.content.clone());
            }
            added_tokens.push((tok.content, tok.id, tok.special));
        }
        // Longest token first for greedy special-token splitting.
        added_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.1.cmp(&b.1)));

        let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
        let mut id_to_token = vec![String::new(); max_id + 1];
        for (tok, &id) in &vocab {
            if let Some(slot) = id_to_token.get_mut(id as usize) {
                *slot = tok.clone();
            }
        }

        let mut merges = HashMap::with_capacity(parsed.model.merges.len());
        for (rank, entry) in parsed.model.merges.into_iter().enumerate() {
            let (left, right) = match entry {
                MergeEntry::SpaceSeparated(s) => {
                    let mut parts = s.splitn(2, ' ');
                    let a = parts
                        .next()
                        .ok_or_else(|| TokenizerError::InvalidMergeEntry(s.clone()))?;
                    let b = parts
                        .next()
                        .ok_or_else(|| TokenizerError::InvalidMergeEntry(s.clone()))?;
                    (a.to_string(), b.to_string())
                }
                MergeEntry::Pair([a, b]) => (a, b),
            };
            merges.insert((left, right), rank);
        }

        let split_individual_digits = detect_individual_digits(parsed.pre_tokenizer.as_ref());
        let (byte_to_char, char_to_byte) = gpt2_byte_unicode_tables();

        let bos_token_id = vocab
            .get("<|im_start|>")
            .or_else(|| vocab.get("<|begin_of_text|>"))
            .or_else(|| vocab.get("<s>"))
            .or_else(|| vocab.get("<|endoftext|>"))
            .copied();
        let eos_token_id = vocab
            .get("<|im_end|>")
            .or_else(|| vocab.get("<|end_of_text|>"))
            .or_else(|| vocab.get("</s>"))
            .or_else(|| vocab.get("<|endoftext|>"))
            .copied();

        Ok(Self {
            vocab,
            id_to_token,
            merges,
            added_tokens,
            special_ids,
            byte_to_char,
            char_to_byte,
            split_individual_digits,
            bos_token_id,
            eos_token_id,
        })
    }

    /// Returns the total vocabulary size (max token id + 1).
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Returns the detected BOS token ID, if present.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Returns the detected EOS token ID, if present.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Looks up a token ID by exact token string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Returns true if `id` is registered as a special token.
    pub fn is_special_token(&self, id: u32) -> bool {
        self.special_ids.contains_key(&id)
    }

    /// Returns true if this tokenizer supports ChatML `<|im_start|>` / `<|im_end|>`.
    pub fn supports_chatml(&self) -> bool {
        self.vocab.contains_key("<|im_start|>") && self.vocab.contains_key("<|im_end|>")
    }

    /// Formats a system + user prompt into ChatML when supported, else plain text.
    pub fn format_chat_prompt(&self, system: Option<&str>, user_prompt: &str) -> String {
        if self.supports_chatml() {
            let sys = system.unwrap_or("You are a helpful AI assistant.");
            format!(
                "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            )
        } else {
            user_prompt.to_string()
        }
    }

    /// Encodes a UTF-8 string into token IDs, honoring added/special tokens.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut cursor = 0usize;

        while cursor < text.len() {
            // Find the earliest added token match starting at or after `cursor`.
            let mut next_match: Option<(usize, usize, u32)> = None;
            for (tok, id, _) in &self.added_tokens {
                if tok.is_empty() {
                    continue;
                }
                if let Some(rel) = text[cursor..].find(tok.as_str()) {
                    let abs_start = cursor + rel;
                    let replace = match next_match {
                        None => true,
                        Some((best_start, best_len, _)) => {
                            abs_start < best_start
                                || (abs_start == best_start && tok.len() > best_len)
                        }
                    };
                    if replace {
                        next_match = Some((abs_start, tok.len(), *id));
                    }
                }
            }

            match next_match {
                Some((start, len, id)) => {
                    if start > cursor {
                        self.encode_plain_segment(&text[cursor..start], &mut ids);
                    }
                    ids.push(id);
                    cursor = start + len;
                }
                None => {
                    self.encode_plain_segment(&text[cursor..], &mut ids);
                    break;
                }
            }
        }

        ids
    }

    fn encode_plain_segment(&self, segment: &str, out: &mut Vec<u32>) {
        for piece in pretokenize_gpt2(segment, self.split_individual_digits) {
            let encoded_piece: String = piece
                .as_bytes()
                .iter()
                .map(|&b| self.byte_to_char[b as usize])
                .collect();
            for bpe_tok in self.bpe_merge(&encoded_piece) {
                if let Some(&id) = self.vocab.get(&bpe_tok) {
                    out.push(id);
                } else {
                    // Fallback to individual byte-level chars if unseen.
                    for ch in bpe_tok.chars() {
                        let s = ch.to_string();
                        if let Some(&id) = self.vocab.get(&s) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    fn bpe_merge(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if word.len() <= 1 {
            return word;
        }

        loop {
            let mut best_rank = usize::MAX;
            let mut best_idx = None;
            for i in 0..word.len() - 1 {
                if let Some(&rank) = self.merges.get(&(word[i].clone(), word[i + 1].clone()))
                    && rank < best_rank
                {
                    best_rank = rank;
                    best_idx = Some(i);
                }
            }
            let Some(idx) = best_idx else {
                break;
            };
            let merged = format!("{}{}", word[idx], word[idx + 1]);
            // Merge all non-overlapping occurrences of this pair left-to-right.
            let left = word[idx].clone();
            let right = word[idx + 1].clone();
            let mut next = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == left && word[i + 1] == right {
                    next.push(merged.clone());
                    i += 2;
                } else {
                    next.push(word[i].clone());
                    i += 1;
                }
            }
            word = next;
            if word.len() <= 1 {
                break;
            }
        }

        word
    }

    /// Decodes a sequence of token IDs back into a UTF-8 string.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        let mut out = String::new();
        let mut pending_bytes = Vec::new();

        for &id in ids {
            if let Some(special) = self.special_ids.get(&id) {
                if !pending_bytes.is_empty() {
                    out.push_str(&String::from_utf8_lossy(&pending_bytes));
                    pending_bytes.clear();
                }
                if !skip_special_tokens {
                    out.push_str(special);
                }
                continue;
            }

            if let Some(tok) = self.id_to_token.get(id as usize) {
                for ch in tok.chars() {
                    if let Some(&b) = self.char_to_byte.get(&ch) {
                        pending_bytes.push(b);
                    } else {
                        if !pending_bytes.is_empty() {
                            out.push_str(&String::from_utf8_lossy(&pending_bytes));
                            pending_bytes.clear();
                        }
                        out.push(ch);
                    }
                }
            }
        }

        if !pending_bytes.is_empty() {
            out.push_str(&String::from_utf8_lossy(&pending_bytes));
        }
        out
    }

    /// Decodes a single token ID for streaming display.
    pub fn decode_token(&self, id: u32, skip_special_tokens: bool) -> String {
        self.decode(&[id], skip_special_tokens)
    }
}

fn detect_individual_digits(pre_tokenizer: Option<&serde_json::Value>) -> bool {
    let Some(val) = pre_tokenizer else {
        return false;
    };
    if val.get("type").and_then(|v| v.as_str()) == Some("Digits") {
        return val
            .get("individual_digits")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
    }
    if let Some(list) = val.get("pretokenizers").and_then(|v| v.as_array()) {
        return list.iter().any(|item| detect_individual_digits(Some(item)));
    }
    false
}

/// Standard GPT-2 bijective mapping between raw bytes `0..=255` and printable
/// Unicode code points.
fn gpt2_byte_unicode_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u8> = Vec::with_capacity(256);
    bs.extend(b'!'..=b'~');
    bs.extend(0xA1u8..=0xACu8);
    bs.extend(0xAEu8..=0xFFu8);

    let mut cs: Vec<u32> = bs.iter().map(|&b| u32::from(b)).collect();
    let mut n = 0u32;
    for b in 0u8..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);
    for (b, c) in bs.into_iter().zip(cs) {
        let ch = char::from_u32(c).expect("GPT-2 byte unicode scalar is valid");
        byte_to_char[b as usize] = ch;
        char_to_byte.insert(ch, b);
    }
    (byte_to_char, char_to_byte)
}

/// Pure-Rust implementation of the GPT-2 byte-level pre-tokenizer regex:
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
/// with optional `Digits { individual_digits: true }` pre-splitting.
fn pretokenize_gpt2(input: &str, split_individual_digits: bool) -> Vec<&str> {
    let mut out = Vec::new();
    let mut idx = 0usize;
    let bytes = input.as_bytes();

    while idx < input.len() {
        let rest = &input[idx..];

        // 1. English contractions: 's, 't, 're, 've, 'm, 'll, 'd
        if let Some(len) = match_contraction(rest) {
            out.push(&input[idx..idx + len]);
            idx += len;
            continue;
        }

        // 2. Optional single ASCII space followed by letters, digits, or punctuation.
        let (has_lead_space, after_space) = if bytes[idx] == b' ' {
            (true, idx + 1)
        } else {
            (false, idx)
        };

        if after_space < input.len() {
            let next_ch = input[after_space..].chars().next().unwrap();
            if next_ch.is_alphabetic() {
                let mut end = after_space;
                for (off, c) in input[after_space..].char_indices() {
                    if c.is_alphabetic() {
                        end = after_space + off + c.len_utf8();
                    } else {
                        break;
                    }
                }
                out.push(&input[idx..end]);
                idx = end;
                continue;
            }
            if next_ch.is_numeric() {
                if split_individual_digits {
                    if has_lead_space {
                        out.push(&input[idx..after_space]);
                        idx = after_space;
                    }
                    let d_len = next_ch.len_utf8();
                    out.push(&input[idx..idx + d_len]);
                    idx += d_len;
                } else {
                    let mut end = after_space;
                    for (off, c) in input[after_space..].char_indices() {
                        if c.is_numeric() {
                            end = after_space + off + c.len_utf8();
                        } else {
                            break;
                        }
                    }
                    out.push(&input[idx..end]);
                    idx = end;
                }
                continue;
            }
            if !next_ch.is_whitespace() {
                let mut end = after_space;
                for (off, c) in input[after_space..].char_indices() {
                    if !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric() {
                        end = after_space + off + c.len_utf8();
                    } else {
                        break;
                    }
                }
                out.push(&input[idx..end]);
                idx = end;
                continue;
            }
        }

        // 3. Whitespace runs: `\s+(?!\S)|\s+`
        let mut ws_end = idx;
        let mut last_ws_start = idx;
        for (off, c) in rest.char_indices() {
            if c.is_whitespace() {
                last_ws_start = idx + off;
                ws_end = idx + off + c.len_utf8();
            } else {
                break;
            }
        }
        if ws_end < input.len() && last_ws_start > idx {
            // Leave the final whitespace character attached to the following token.
            out.push(&input[idx..last_ws_start]);
            idx = last_ws_start;
        } else {
            out.push(&input[idx..ws_end]);
            idx = ws_end;
        }
    }

    out
}

fn match_contraction(s: &str) -> Option<usize> {
    for pat in ["'re", "'ve", "'ll", "'s", "'t", "'m", "'d"] {
        if s.starts_with(pat) {
            return Some(pat.len());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tokenizer_json() -> &'static str {
        r#"{
            "version": "1.0",
            "added_tokens": [
                {"id": 0, "content": "<|endoftext|>", "special": true},
                {"id": 1, "content": "<|im_start|>", "special": true},
                {"id": 2, "content": "<|im_end|>", "special": true}
            ],
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Digits", "individual_digits": true},
                    {"type": "ByteLevel", "add_prefix_space": false}
                ]
            },
            "model": {
                "type": "BPE",
                "vocab": {
                    "<|endoftext|>": 0,
                    "<|im_start|>": 1,
                    "<|im_end|>": 2,
                    "h": 3,
                    "e": 4,
                    "l": 5,
                    "o": 6,
                    "he": 7,
                    "ll": 8,
                    "hello": 9,
                    "Ġ": 10,
                    "w": 11,
                    "r": 12,
                    "d": 13,
                    "Ġw": 14,
                    "or": 15,
                    "ld": 16,
                    "Ġworld": 17,
                    "!": 18,
                    "1": 19,
                    "2": 20,
                    "Ċ": 21
                },
                "merges": [
                    "h e",
                    "l l",
                    "he ll",
                    "hell o",
                    "Ġ w",
                    "o r",
                    "l d",
                    "Ġw or",
                    "Ġwor ld"
                ]
            }
        }"#
    }

    #[test]
    fn bpe_encode_decode_round_trip_and_special_tokens() {
        let tok = BpeTokenizer::from_json_bytes(sample_tokenizer_json().as_bytes())
            .expect("valid sample tokenizer");
        assert_eq!(tok.bos_token_id(), Some(1));
        assert_eq!(tok.eos_token_id(), Some(2));
        assert!(tok.supports_chatml());

        let ids = tok.encode("<|im_start|>hello world! 12<|im_end|>");
        assert_eq!(ids.first().copied(), Some(1));
        assert_eq!(ids.last().copied(), Some(2));
        // Digits are split individually: '1' (19) and '2' (20).
        assert!(ids.contains(&19));
        assert!(ids.contains(&20));

        let decoded_keep = tok.decode(&ids, false);
        assert_eq!(decoded_keep, "<|im_start|>hello world! 12<|im_end|>");
        let decoded_skip = tok.decode(&ids, true);
        assert_eq!(decoded_skip, "hello world! 12");
    }

    #[test]
    fn gpt2_byte_table_is_bijective() {
        let (b2c, c2b) = gpt2_byte_unicode_tables();
        assert_eq!(c2b.len(), 256);
        for b in 0u8..=255u8 {
            assert_eq!(c2b.get(&b2c[b as usize]), Some(&b));
        }
    }
}
