//! Bounded numeric CCM1 / experimental CCM2 model parsing.
//! CCM1: f64 IDF, bias and dense weights; CCM2: f64 IDF/bias and f32 weights.
//! Neither file permits executable deserialization. Indices are verified.
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use sha2::{Digest,Sha256};
use crate::pack_ngram;

const MAX_MODEL_BYTES: usize = 256 * 1024 * 1024;
const MAX_FEATURES: usize = 200_000;
const MAX_CLASSES: usize = 256;

pub enum Weights {
    F64(Vec<f64>),
    F32(Vec<f32>),
}

pub struct ParsedModel {
    pub vocab: Vec<(u64,u32)>,
    pub idf: Vec<f64>,
    pub weights: Weights,
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
        let end=self.pos.checked_add(len)
            .ok_or_else(|| PyValueError::new_err("model length overflow"))?;
        let bytes=self.data.get(self.pos..end)
            .ok_or_else(|| PyValueError::new_err("truncated native model"))?;
        self.pos=end;
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
    fn f32(&mut self) -> PyResult<f32> {
        let f=f32::from_le_bytes(self.take(4)?.try_into().unwrap());
        if !f.is_finite() { return Err(PyValueError::new_err("non-finite f32 model weight")); }
        Ok(f)
    }
    fn string(&mut self) -> PyResult<String> {
        let n=self.u16()?;
        if n==0 || n>256 { return Err(PyValueError::new_err("invalid model string")); }
        let str=std::str::from_utf8(self.take(n)?)
            .map_err(|_| PyValueError::new_err("invalid model UTF-8"))?;
        Ok(str.to_owned())
    }
}

pub fn read(blob: &[u8]) -> PyResult<ParsedModel> {
    if blob.len()<4+4+4+4+32 || blob.len()>MAX_MODEL_BYTES {
        return Err(PyValueError::new_err("model size exceeds limit"));
    }
    let (body,digest)=blob.split_at(blob.len()-32);
    if Sha256::digest(body).as_slice()!=digest {
        return Err(PyValueError::new_err("native model SHA256 mismatch"));
    }
    let mut r=Reader { data:body,pos:0 };
    let header=r.take(4)?;
    let weight32=match header {
        b"CCM1"=>false,
        b"CCM2"=>true,
        _=>return Err(PyValueError::new_err("unsupported model schema")),
    };
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
    let rows=if classes_n==2 { 1 } else { classes_n };
    let length=width.checked_mul(rows)
        .and_then(|w|w.checked_mul(if weight32 {4} else {8}))
        .and_then(|w|w.checked_add(width.checked_mul(8)?))
        .and_then(|w|w.checked_add(rows.checked_mul(8)?))
        .ok_or_else(|| PyValueError::new_err("model tensor dimensions overflow"))?;
    if length>body.len() {
        return Err(PyValueError::new_err("model weights truncated"));
    }
    let mut classes=Vec::with_capacity(classes_n);
    for _ in 0..classes_n { classes.push(r.string()?); }
    let mut vocab=Vec::with_capacity(width);
    let mut used=vec![false;width];
    for _ in 0..width {
        let index=r.u32()?;
        let gram=r.string()?;
        let chars:Vec<char>=gram.chars().collect();
        if chars.len()<min || chars.len()>max {
            return Err(PyValueError::new_err("invalid vocabulary ngram length"));
        }
        let packed=pack_ngram(&chars)
            .ok_or_else(|| PyValueError::new_err("vocabulary ngram cannot be packed"))?;
        if index>=width || used[index] {
            return Err(PyValueError::new_err("duplicate vocabulary index"));
        }
        used[index]=true;
        vocab.push((packed,index as u32));
    }
    vocab.sort_unstable_by_key(|x|x.0);
    if vocab.windows(2).any(|x|x[0].0==x[1].0) {
        return Err(PyValueError::new_err("duplicate vocabulary key"));
    }
    let mut idf=Vec::with_capacity(width);
    for _ in 0..width { idf.push(r.float()?); }
    let mut intercept=Vec::with_capacity(rows);
    for _ in 0..rows { intercept.push(r.float()?); }
    let weight_count=width*rows;
    let weights=if weight32 {
        let mut values=Vec::with_capacity(weight_count);
        for _ in 0..weight_count {values.push(r.f32()?);}
        Weights::F32(values)
    }else{
        let mut values=Vec::with_capacity(weight_count);
        for _ in 0..weight_count {values.push(r.float()?);}
        Weights::F64(values)
    };
    if r.pos!=body.len(){
        return Err(PyValueError::new_err("unexpected trailing model payload"));
    }
    Ok(ParsedModel {
        vocab,idf,weights,intercept,classes,
        ngram_min:min,ngram_max:max,
        binary:flags & 1 != 0,sublinear:flags & 2 != 0,
    })
}
