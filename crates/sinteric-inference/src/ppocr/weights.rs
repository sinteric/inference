//! Strict, mmap-backed FP32 assets. Paths are explicit; graph JSON is an
//! authenticated description of the compiled model contract, never executable.
use crate::assets::PinnedFile;
use crate::ppocr::{Error, Result, ASSET_MANIFEST};
use crate::tensor_file::{Dtype, Index, TensorFile};
use crate::{checked_size, try_vec};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Det,
    Rec,
    Uvdoc,
}
impl ModelKind {
    #[cfg(test)]
    pub fn name(self) -> &'static str {
        match self {
            Self::Det => "det",
            Self::Rec => "rec",
            Self::Uvdoc => "uvdoc",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AssetConfig {
    pub model_dir: PathBuf,
    pub kind: ModelKind,
    /// Optional explicit manifest. Graph/dictionary identity stays compiled in.
    #[serde(default)]
    pub manifest: Option<PathBuf>,
    #[serde(default)]
    pub verify_weights: bool,
}
impl AssetConfig {
    pub fn new(model_dir: impl Into<PathBuf>, kind: ModelKind) -> Self {
        Self {
            model_dir: model_dir.into(),
            kind,
            manifest: None,
            verify_weights: false,
        }
    }
}
#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    det: Pin,
    rec: Pin,
    uvdoc: Pin,
}
impl Manifest {
    fn pin(self, kind: ModelKind) -> Pin {
        match kind {
            ModelKind::Det => self.det,
            ModelKind::Rec => self.rec,
            ModelKind::Uvdoc => self.uvdoc,
        }
    }
}
#[derive(Deserialize)]
struct Pin {
    onnx_sha256: String,
    files: Vec<PinnedFile>,
}
impl Pin {
    fn file(&self, name: &str) -> Result<&PinnedFile> {
        self.files
            .iter()
            .find(|f| f.path == name)
            .ok_or_else(|| Error::model_files(format!("missing OCR file pin {name}")))
    }
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphSpec {
    pub schema_version: u32,
    pub importer_version: u32,
    pub model: ModelKind,
    pub source_sha256: String,
    pub inputs: Vec<InputSpec>,
    pub outputs: Vec<String>,
    pub nodes: Vec<NodeSpec>,
    pub integer_constants: BTreeMap<String, IntegerConstant>,
    pub tensors: BTreeMap<String, Vec<usize>>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputSpec {
    pub name: String,
    pub shape: Vec<Option<usize>>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeSpec {
    pub name: String,
    /// Declared so `deny_unknown_fields` accepts the graph file; the
    /// embedded pin hashes the whole file, and the build reads only the
    /// fields below.
    #[expect(dead_code, reason = "parsed for the schema, never read")]
    pub source_name: String,
    pub op: String,
    pub inputs: Vec<String>,
    pub output: String,
    #[expect(dead_code, reason = "parsed for the schema, never read")]
    pub attrs: BTreeMap<String, serde_json::Value>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegerConstant {
    pub shape: Vec<usize>,
    pub values: Vec<i64>,
}

fn digest(bytes: &[u8], expected: &str) -> Result<()> {
    let got = hex::encode(Sha256::digest(bytes));
    if got != expected {
        return Err(Error::model_files(format!(
            "OCR SHA256 mismatch: got {got}, expected {expected}"
        )));
    }
    Ok(())
}
fn mapped(path: &Path, pin: &PinnedFile, verify: bool) -> Result<Mmap> {
    let result = (|| {
        let f = File::open(path)?;
        let len = f.metadata()?.len();
        if len != pin.size || len == 0 {
            return Err(Error::model_files(format!(
                "OCR file size {len} != {}",
                pin.size
            )));
        }
        // Model files must remain immutable for the lifetime of this mapping.
        let map = unsafe { Mmap::map(&f) }?;
        if verify {
            digest(&map, &pin.sha256)?;
        }
        Ok(map)
    })();
    result.map_err(|e: Error| e.at(path))
}
fn checked_elements(shape: &[usize]) -> Result<usize> {
    if shape.len() > 12 {
        return Err(Error::model_files(format!(
            "invalid OCR tensor shape {shape:?}"
        )));
    }
    let len = checked_size("OCR tensor", shape)?;
    if len > isize::MAX as usize / 4 {
        return Err(Error::size_overflow("OCR tensor bytes"));
    }
    Ok(len)
}
/// The file holds exactly the graph's tensors, each FP32 of its shape.
fn check_tensors(index: &Index, spec: &GraphSpec) -> Result<()> {
    if index.len() != spec.tensors.len() {
        return Err(Error::model_files("OCR tensor inventory mismatch"));
    }
    for (name, shape) in &spec.tensors {
        checked_elements(shape)?;
        if index.dtype(name)? != Dtype::F32 || index.shape(name)? != shape {
            return Err(Error::model_files(format!(
                "OCR tensor dtype/shape mismatch: {name}"
            )));
        }
    }
    Ok(())
}
impl GraphSpec {
    fn validate(&self, kind: ModelKind, source_sha256: &str) -> Result<()> {
        if self.schema_version != 1
            || self.importer_version != 1
            || self.model != kind
            || self.source_sha256 != source_sha256
        {
            return Err(Error::model_files("OCR graph schema/model/source mismatch"));
        }
        if self.inputs.len() != 1
            || self.inputs[0].shape.len() != 4
            || self.inputs[0].shape[1] != Some(3)
            || (kind == ModelKind::Rec && self.inputs[0].shape[2] != Some(48))
            || self.nodes.is_empty()
            || self.outputs.len() != 1
        {
            return Err(Error::model_files(
                "OCR graph input/output contract mismatch",
            ));
        }
        let mut available = BTreeSet::new();
        for i in &self.inputs {
            available.insert(i.name.as_str());
        }
        for (name, shape) in &self.tensors {
            checked_elements(shape)?;
            available.insert(name.as_str());
        }
        for (name, c) in &self.integer_constants {
            if checked_elements(&c.shape)? != c.values.len() {
                return Err(Error::model_files(format!(
                    "OCR shape constant {name} extent"
                )));
            }
            available.insert(name.as_str());
        }
        let mut names = BTreeSet::new();
        for node in &self.nodes {
            if node.name.is_empty()
                || !names.insert(&node.name)
                || node.inputs.is_empty()
                || node
                    .inputs
                    .iter()
                    .filter(|s| !s.is_empty())
                    .any(|s| !available.contains(s.as_str()))
                || !available.insert(node.output.as_str())
            {
                return Err(Error::model_files(format!(
                    "OCR graph topology at {}",
                    node.name
                )));
            }
            if !matches!(
                node.op.as_str(),
                "Conv"
                    | "ConvTranspose"
                    | "Relu"
                    | "HardSigmoid"
                    | "HardSwish"
                    | "Swish"
                    | "GeluErf"
                    | "Sigmoid"
                    | "GlobalAveragePool"
                    | "ReduceMean"
                    | "Resize"
                    | "MaxPool"
                    | "AveragePool"
                    | "Concat"
                    | "Add"
                    | "Mul"
                    | "MatMul"
                    | "Softmax"
                    | "LayerNorm"
                    | "Shape"
                    | "Slice"
                    | "Squeeze"
                    | "Unsqueeze"
                    | "Reshape"
                    | "Transpose"
            ) && !(kind == ModelKind::Uvdoc
                && matches!(node.op.as_str(), "PRelu" | "Pad" | "GridSample"))
            {
                return Err(Error::model_files(format!("unmatched OCR op {}", node.op)));
            }
        }
        if self.outputs.iter().any(|s| !available.contains(s.as_str())) {
            return Err(Error::model_files("OCR graph output missing"));
        }
        Ok(())
    }
}

/// Validated source tensors and graph identity. The exact graph and dictionary
/// are always hashed against embedded pins; hashing tensor payloads is optional.
pub struct Assets {
    weights: TensorFile,
    graph: GraphSpec,
    dictionary: Vec<String>,
}
impl Assets {
    pub fn load(config: &AssetConfig) -> Result<Self> {
        let compiled: Manifest = toml::from_str(ASSET_MANIFEST)
            .map_err(|e| Error::model_files(format!("OCR embedded manifest: {e}")))?;
        let manifest: Manifest = if let Some(path) = &config.manifest {
            let file = File::open(path).map_err(|e| Error::from(e).at(path))?;
            if file.metadata().map_err(|e| Error::from(e).at(path))?.len() > 128 * 1024 * 1024 {
                return Err(Error::model_files("OCR manifest exceeds 128 MiB").at(path));
            }
            let mut text = String::new();
            file.take(128 * 1024 * 1024 + 1)
                .read_to_string(&mut text)
                .map_err(|e| Error::from(e).at(path))?;
            if text.len() > 128 * 1024 * 1024 {
                return Err(Error::model_files("OCR manifest exceeds 128 MiB").at(path));
            }
            toml::from_str(&text)
                .map_err(|e| Error::model_files(format!("OCR manifest: {e}")).at(path))?
        } else {
            compiled
        };
        if manifest.schema_version != 1 {
            return Err(Error::model_files("OCR manifest schema"));
        }
        let pin = manifest.pin(config.kind);
        let pin = &pin;
        // Explicit manifests may pin payloads but cannot select another graph
        // or dictionary for these hand-written networks.
        let expected: Manifest = toml::from_str(ASSET_MANIFEST)
            .map_err(|e| Error::model_files(format!("OCR embedded manifest: {e}")))?;
        let expected = expected.pin(config.kind);
        for name in if config.kind == ModelKind::Rec {
            &["graph.json", "dict.txt"][..]
        } else {
            &["graph.json"][..]
        } {
            let a = pin.file(name)?;
            let b = expected.file(name)?;
            if a.sha256 != b.sha256 || a.size != b.size || pin.onnx_sha256 != expected.onnx_sha256 {
                return Err(Error::model_files(format!(
                    "manifest changes compiled OCR identity: {name}"
                )));
            }
        }
        let path = config.model_dir.join("graph.json");
        let graph_bytes = mapped(&path, pin.file("graph.json")?, true)?;
        let graph: GraphSpec =
            serde_json::from_slice(&graph_bytes).map_err(|e| Error::from(e).at(&path))?;
        graph
            .validate(config.kind, &pin.onnx_sha256)
            .map_err(|e| e.at(&path))?;
        let path = config.model_dir.join("model.safetensors");
        let weights = mapped(&path, pin.file("model.safetensors")?, config.verify_weights)?;
        let weights = TensorFile::from_map(weights).map_err(|e| e.at(&path))?;
        check_tensors(weights.index(), &graph).map_err(|e| e.at(&path))?;
        let dictionary = if config.kind == ModelKind::Rec {
            let path = config.model_dir.join("dict.txt");
            let bytes = mapped(&path, pin.file("dict.txt")?, true)?;
            let text = std::str::from_utf8(&bytes)
                .map_err(|e| Error::model_files(format!("OCR dictionary UTF-8: {e}")).at(&path))?;
            let entries = text.lines().map(str::to_owned).collect::<Vec<_>>();
            if entries.len() != 11945 || entries.iter().any(|s| s.is_empty()) {
                return Err(
                    Error::model_files("OCR dictionary must have 11945 nonempty entries").at(&path),
                );
            }
            entries
        } else {
            Vec::new()
        };
        Ok(Self {
            weights,
            graph,
            dictionary,
        })
    }
    pub fn graph(&self) -> &GraphSpec {
        &self.graph
    }
    /// Dictionary excludes the CTC blank (class 0) and appended space.
    pub fn dictionary(&self) -> &[String] {
        &self.dictionary
    }
    pub fn tensor(&self, name: &str) -> Result<Vec<f32>> {
        let shape = self.weights.index().shape(name)?;
        let values = self.weights.view::<f32>(name, shape)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::model_files(format!("nonfinite OCR weights: {name}")));
        }
        let mut result = try_vec(values.len())?;
        result.copy_from_slice(values);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_headers_and_shapes() {
        let mut spec = GraphSpec {
            schema_version: 1,
            importer_version: 1,
            model: ModelKind::Det,
            source_sha256: String::new(),
            inputs: vec![],
            outputs: vec![],
            nodes: vec![],
            integer_constants: BTreeMap::new(),
            tensors: BTreeMap::new(),
        };
        let tensors = |bytes: &[u8], spec: &GraphSpec| {
            Index::parse(bytes).and_then(|index| check_tensors(&index, spec))
        };
        assert_eq!(checked_elements(&[0, 1]).unwrap(), 0);
        assert!(checked_elements(&[usize::MAX, 2]).is_err());
        assert!(spec.validate(ModelKind::Det, "").is_err());
        assert!(Assets::load(&AssetConfig::new("/explicit/missing/ocr", ModelKind::Rec)).is_err());
        spec.tensors.insert("w".into(), vec![2]);
        let encoded = |dtype: &str, shape: Vec<usize>| {
            let value = serde_json::json!({"w":{"dtype":dtype,"shape":shape,"data_offsets":[0,8]}});
            crate::tensor_file::tests::encode(value, &[0; 8])
        };
        assert!(tensors(&encoded("F32", vec![2]), &spec).is_ok());
        for bytes in [encoded("F16", vec![4]), encoded("F32", vec![1, 2])] {
            assert!(tensors(&bytes, &spec).is_err_and(|e| e.kind() == crate::ErrorKind::ModelFiles));
        }
        spec.inputs.push(InputSpec {
            name: "x".into(),
            shape: vec![None, Some(3), None, None],
        });
        spec.nodes.push(NodeSpec {
            name: "det.relu000".into(),
            source_name: "relu".into(),
            op: "Relu".into(),
            inputs: vec!["x".into()],
            output: "y".into(),
            attrs: BTreeMap::new(),
        });
        spec.outputs.push("y".into());
        assert!(spec.validate(ModelKind::Det, "").is_ok());
        spec.nodes[0].inputs[0] = "missing".into();
        assert!(spec
            .validate(ModelKind::Det, "")
            .is_err_and(|e| e.kind() == crate::ErrorKind::ModelFiles));
        spec.nodes[0].inputs[0] = "x".into();
        spec.nodes[0].op = "Sin".into();
        assert!(spec
            .validate(ModelKind::Det, "")
            .is_err_and(|e| e.kind() == crate::ErrorKind::ModelFiles));
    }
}
