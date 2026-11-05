//! Proof of equivalence for the existing sklearn L2 model, not a new
//! detector and not yet a production model loader.
//! Runtime dependencies: Rust std + PyO3; no numpy or sklearn.
mod format;
use pyo3::exceptions::{PyValueError,PyIOError};
use memmap2::MmapOptions;
use std::fs::File;
use pyo3::prelude::*;
use std::collections::HashMap;

pub(crate) fn pack_ngram(chars: &[char]) -> Option<u64> {
    if chars.is_empty() || chars.len() > 3 { return None; }
    let mut key = (chars.len() as u64) << 48;
    for c in chars {
        if (*c as u32) > u16::MAX as u32 { return None; }
    }
    for (i, c) in chars.iter().enumerate() {
        key |= (*c as u64) << (16 * (2 - i));
    }
    Some(key)
}

/// Compact open-addressing map: one contiguous buffer, no HashMap allocations.
fn build_vocab_table(entries:Vec<(u64,u32)>)->Vec<(u64,u32)>{
    let cap=(entries.len().saturating_mul(3)/2+1).next_power_of_two().max(8);
    let mut table=vec![(0,u32::MAX);cap];
    for (key,index) in entries{
        let mut i=(hash_ngram(key) as usize)&(cap-1);
        loop{
            if table[i].1==u32::MAX{table[i]=(key,index);break;}
            i=(i+1)&(cap-1);
        }
    }
    table
}
#[inline]
fn hash_ngram(mut key:u64)->u64{
    key^=key>>33;
    key=key.wrapping_mul(0xff51afd7ed558ccd);
    key^=key>>33;
    key
}

impl NativeL2 {
    #[inline]
    fn lookup_index(&self,key:u64)->Option<usize>{
        let mask=self.vocabulary.len()-1;
        let mut i=(hash_ngram(key) as usize)&mask;
        loop{
            let (candidate,index)=self.vocabulary[i];
            if index==u32::MAX{return None;}
            if candidate==key{return Some(index as usize);}
            i=(i+1)&mask;
        }
    }
}

#[pyclass]
pub struct NativeL2 {
    vocabulary: Vec<(u64,u32)>,
    idf: Vec<f64>,
    weights: format::Weights,
    intercept: Vec<f64>,
    min_ngram: usize,
    max_ngram: usize,
    binary: bool,
    sublinear_tf: bool,
    class_names: Option<Vec<String>>,
}

#[pymethods]
impl NativeL2 {
    #[new]
    #[pyo3(signature = (vocabulary, idf, weights, intercept, min_ngram, max_ngram, binary, sublinear_tf))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        vocabulary: HashMap<String, usize>,
        idf: Vec<f64>,
        weights: Vec<Vec<f64>>,
        intercept: Vec<f64>,
        min_ngram: usize,
        max_ngram: usize,
        binary: bool,
        sublinear_tf: bool,
    ) -> PyResult<Self> {
        if !(1..=3).contains(&min_ngram) || min_ngram > max_ngram || max_ngram > 3 {
            return Err(PyValueError::new_err("unsupported ngram range"));
        }
        let width = idf.len();
        if width == 0 || vocabulary.len() != width || intercept.len() != weights.len()
            || weights.is_empty() || weights.iter().any(|v| v.len() != width)
            || idf.iter().any(|x| !x.is_finite())
            || intercept.iter().any(|x| !x.is_finite())
            || weights.iter().flatten().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err("invalid finite model tensor shapes"));
        }
        let mut indexed = Vec::with_capacity(width);
        let mut used = vec![false;width];
        for (gram, index) in vocabulary {
            if index >= width || used[index] { return Err(PyValueError::new_err("vocabulary index out of bounds")); }
            used[index]=true;
            let chars: Vec<char> = gram.chars().collect();
            if chars.len() < min_ngram || chars.len() > max_ngram {
                return Err(PyValueError::new_err("vocabulary ngram length disagrees with schema"));
            }
            let Some(key) = pack_ngram(&chars) else {
                return Err(PyValueError::new_err("non-BMP vocabulary ngram unsupported"));
            };
            indexed.push((key,index as u32));
        }
        indexed.sort_unstable_by_key(|v|v.0);
        if indexed.windows(2).any(|w|w[0].0==w[1].0){return Err(PyValueError::new_err("duplicate ngram"));}
        Ok(Self { vocabulary: build_vocab_table(indexed), idf, weights:format::Weights::F64(weights.into_iter().flatten().collect()), intercept,
                  min_ngram, max_ngram, binary, sublinear_tf, class_names: None })
    }

    /// Load an authenticated, bounded numeric model artifact without NumPy.
    #[staticmethod]
    fn from_model_bytes(blob: &[u8]) -> PyResult<Self> {
        let parsed = format::read(blob)?;
        Ok(Self {
            vocabulary:build_vocab_table(parsed.vocab),
            idf:parsed.idf,weights:parsed.weights,intercept:parsed.intercept,
            min_ngram:parsed.ngram_min,max_ngram:parsed.ngram_max,
            binary:parsed.binary,sublinear_tf:parsed.sublinear,
            class_names:Some(parsed.classes),
        })
    }

    /// Load the model through a read-only OS file mapping instead of
    /// allocating a second whole-file bytes buffer in the Python process.
    #[staticmethod]
    fn from_model_file(path:&str)->PyResult<Self>{
        let file=File::open(path).map_err(|e|PyIOError::new_err(e.to_string()))?;
        let map=unsafe {MmapOptions::new().map(&file)}
            .map_err(|e|PyIOError::new_err(e.to_string()))?;
        Self::from_model_bytes(&map)
    }

    fn numeric_format(&self)->&'static str{
        match &self.weights {
            format::Weights::F64(_)=>"f64",
            format::Weights::F32(_)=>"f32",
        }
    }

    /// Class label order bound to the exported numeric weights.
    fn class_names(&self) -> Option<Vec<String>> {
        self.class_names.clone()
    }

    /// Return raw one-vs-rest class scores in classifier.classes_ order.
    /// Matches the existing first-4096-byte Latin-1 TF-IDF L2 calculation.
    fn scores(&self, raw: &[u8]) -> Vec<f64> {
        let mut chars = Vec::<char>::with_capacity(raw.len().min(4096));
        for &b in raw.iter().take(4096) {
            chars.extend((b as char).to_lowercase());
        }
        // sklearn's char analyzer first collapses runs of TWO OR MORE
        // Unicode whitespace characters to a single ASCII space.  Python
        // regards ASCII 0x1c..0x1f as whitespace as well.
        let mut collapsed = Vec::with_capacity(chars.len());
        let mut i = 0;
        while i < chars.len() {
            let is_space = |c: char| c.is_whitespace() || matches!(c, '\u{001c}'..='\u{001f}');
            if is_space(chars[i]) {
                let mut end = i + 1;
                while end < chars.len() && is_space(chars[end]) { end += 1; }
                if end - i > 1 { collapsed.push(' '); }
                else { collapsed.push(chars[i]); }
                i = end;
            } else {
                collapsed.push(chars[i]);
                i += 1;
            }
        }
        let chars = collapsed;
        let mut counts: HashMap<usize, u32> = HashMap::new();
        for n in self.min_ngram..=self.max_ngram {
            if chars.len() < n { continue; }
            for gram in chars.windows(n) {
                if let Some(key) = pack_ngram(gram) {
                    if let Some(index) = self.lookup_index(key) {
                        *counts.entry(index).or_default() += 1;
                    }
                }
            }
        }
        let mut sparse: Vec<(usize, f64)> = counts.into_iter().map(|(i, count)| {
            let tf = if self.binary { 1.0 } else if self.sublinear_tf {
                1.0 + (count as f64).ln()
            } else { count as f64 };
            (i, tf * self.idf[i])
        }).collect();
        sparse.sort_unstable_by_key(|x| x.0);
        let norm = sparse.iter().map(|(_, x)| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for (_, value) in &mut sparse { *value /= norm; }
        }
        let mut output = self.intercept.clone();
        let width=self.idf.len();
        for (class,out) in output.iter_mut().enumerate() {
            for &(index,value) in &sparse {
                let pos=class*width+index;
                let weight=match &self.weights {
                    format::Weights::F64(v)=>v[pos],
                    format::Weights::F32(v)=>v[pos] as f64,
                };
                *out+=weight*value;
            }
        }
        if output.len() == 1 {
            vec![-output[0], output[0]]
        } else { output }
    }
}

#[pymodule]
fn _codeccat_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeL2>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ngram_packing_is_unique_for_short_byte_sequences() {
        assert_ne!(pack_ngram(&['a']), pack_ngram(&['a', '\0']));
        assert_ne!(pack_ngram(&['a', 'b']), pack_ngram(&['b', 'a']));
        assert!(pack_ngram(&['a', 'b', 'c', 'd']).is_none());
    }
}
