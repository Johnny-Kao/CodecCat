//! Compact, versioned, numeric-only CodecCat model format for Native Runtime.
//! This is a prototype schema (CCM1), not a public stable serialization ABI.
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use sha2::{Digest,Sha256};
use std::collections::HashMap;

const MAX_MODEL_BYTES: usize = 256 * 1024 * 1024;
const MAX_FEATURES: usize = 200_000;
const MAX_CLASSES: usize = 256;

pub struct ParsedModel {
    pub vocab: HashMap<String,usize>,
    pub idf: Vec<f64>,
    pub weights: Vec<Vec<f64>>,
    pub intercept: Vec<f64>,
    pub classes: Vec<String>,
    pub ngram_min: usize,
    pub ngram_max: usize,
    pub binary: bool,
    pub sublinear: bool,
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> PyResult<&'a [u8]> {
        let end = self.pos.checked_add(len)
            .ok_or_else(|| PyValueError::new_err("model length overflow"))?;
        let bytes = self.data.get(self.pos..end)
            .ok_or_else(|| PyValueError::new_err("truncated native model"))?;
        self.pos = end;
        Ok(bytes)
    }
    fn u8(&mut self) -> PyResult<u8> { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> PyResult<usize> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as usize)
    }
    fn u32(&mut self) -> PyResult<usize> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }
    fn float(&mut self) -> PyResult<f64> {
        let f=f64::from_le_bytes(self.take(8)?.try_into().unwrap());
        if !f.is_finite() { return Err(PyValueError::new_err("non-finite model weight")); }
        Ok(f)
    }
    fn string(&mut self) -> PyResult<String> {
        let n = self.u16()?;
        if n == 0 || n > 256 { return Err(PyValueError::new_err("invalid model string")); }
        let value = std::str::from_utf8(self.take(n)?)
            .map_err(|_| PyValueError::new_err("invalid model UTF-8"))?;
        Ok(value.to_owned())
    }
}

pub fn read(blob: &[u8]) -> PyResult<ParsedModel> {
    if blob.len() < 4+4+4+4+32 || blob.len() > MAX_MODEL_BYTES {
        return Err(PyValueError::new_err("model size exceeds limit"));
    }
    let (body, digest) = blob.split_at(blob.len()-32);
    if Sha256::digest(body).as_slice() != digest {
        return Err(PyValueError::new_err("native model SHA256 mismatch"));
    }
    let mut r = Reader { data: body, pos:0 };
    if r.take(4)? != b"CCM1" { return Err(PyValueError::new_err("unsupported model schema")); }
    let width=r.u32()?;
    let classes_n=r.u32()?;
    let min=r.u8()? as usize;
    let max=r.u8()? as usize;
    let flags=r.u8()?;
    let reserved=r.u8()?;
    if width==0 || width>MAX_FEATURES || classes_n<2 || classes_n>MAX_CLASSES
        || min==0 || max>3 || min>max || flags & !3 != 0 || reserved!=0 {
        return Err(PyValueError::new_err("invalid model metadata"));
    }
    // Preflight the matrix allocation against encoded size, not only row count.
    let rows = if classes_n == 2 { 1 } else { classes_n };
    let floats=(width.checked_mul(rows)
        .and_then(|x| x.checked_add(width+rows)))
        .and_then(|x| x.checked_mul(8))
        .ok_or_else(|| PyValueError::new_err("model tensor dimensions overflow"))?;
    if floats>body.len() { return Err(PyValueError::new_err("model weights truncated")); }

    let mut classes=Vec::with_capacity(classes_n);
    for _ in 0..classes_n { classes.push(r.string()?); }
    let mut vocab=HashMap::with_capacity(width);
    let mut used=vec![false;width];
    for _ in 0..width {
        let idx=r.u32()?;
        let gram=r.string()?;
        if idx>=width || used[idx] || vocab.insert(gram,idx).is_some() {
            return Err(PyValueError::new_err("invalid or duplicate vocabulary index"));
        }
        used[idx]=true;
    }
    let mut idf=Vec::with_capacity(width);
    for _ in 0..width { idf.push(r.float()?); }
    let mut intercept=Vec::with_capacity(rows);
    for _ in 0..rows { intercept.push(r.float()?); }
    let mut weights=Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut row=Vec::with_capacity(width);
        for _ in 0..width { row.push(r.float()?); }
        weights.push(row);
    }
    if r.pos != body.len() { return Err(PyValueError::new_err("unexpected trailing model payload")); }
    Ok(ParsedModel {
        vocab,idf,weights,intercept,classes,
        ngram_min:min,ngram_max:max,
        binary:flags & 1 != 0,
        sublinear:flags & 2 != 0,
    })
}
