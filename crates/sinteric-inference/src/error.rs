use std::path::PathBuf;

/// Errors shared by the runtime and its library boundaries.
///
/// Match variants rather than display strings. Context preserves the original
/// error in [`std::error::Error::source`]. New variants may be added.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("{0}")]
    Message(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("{path}: {source}")]
    File {
        path: PathBuf,
        #[source]
        source: Box<Error>,
    },
    #[error("{0}")]
    Shape(String),
    #[error("{0}")]
    Asset(String),
    #[error("{0}")]
    Backend(String),
    #[error("{operation}: oneDNN status {status}")]
    BackendStatus {
        operation: &'static str,
        status: i32,
    },
    #[error("{0}")]
    Config(String),
    #[error("missing or invalid {0}")]
    Missing(String),
    #[error("unsupported {what}: {reason}")]
    Unsupported { what: &'static str, reason: String },
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Layout(#[from] std::alloc::LayoutError),
    #[error("{0}")]
    Slice(#[from] std::array::TryFromSliceError),
    #[error("size overflow computing {0}")]
    SizeOverflow(&'static str),
    #[error("{operation} needs {requested} bytes, workspace limit is {limit} bytes")]
    WorkspaceLimit {
        operation: &'static str,
        requested: usize,
        limit: usize,
    },
    #[error("allocation failed for {bytes} bytes")]
    Allocation { bytes: usize },
    #[error("{0} lock poisoned")]
    Poisoned(&'static str),
    #[error("tokenizer: {0}")]
    Tokenizer(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error("{context}: {source}")]
    Context {
        context: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl Error {
    /// Attach the actual source path to a file, JSON, tensor or cache error.
    pub fn at(self, path: impl Into<PathBuf>) -> Self {
        Self::File {
            path: path.into(),
            source: Box::new(self),
        }
    }
}
impl From<String> for Error {
    fn from(s: String) -> Self {
        Self::Message(s)
    }
}
impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Self::Message(s.to_owned())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Add diagnostic context while retaining an error's source.
pub trait ResultContext<T> {
    fn context(self, context: impl Into<String>) -> Result<T>;
    fn with_context(self, context: impl FnOnce() -> String) -> Result<T>;
}
impl<T, E: Into<Box<dyn std::error::Error + Send + Sync>>> ResultContext<T>
    for std::result::Result<T, E>
{
    fn context(self, context: impl Into<String>) -> Result<T> {
        self.map_err(|source| Error::Context {
            context: context.into(),
            source: source.into(),
        })
    }
    fn with_context(self, context: impl FnOnce() -> String) -> Result<T> {
        self.map_err(|source| Error::Context {
            context: context(),
            source: source.into(),
        })
    }
}
impl<T> ResultContext<T> for Option<T> {
    fn context(self, context: impl Into<String>) -> Result<T> {
        self.ok_or_else(|| Error::Missing(context.into()))
    }
    fn with_context(self, context: impl FnOnce() -> String) -> Result<T> {
        self.ok_or_else(|| Error::Missing(context()))
    }
}

/// Checked product for dimensions and byte counts derived from external data.
pub fn checked_size(what: &'static str, dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d).ok_or(Error::SizeOverflow(what))
    })
}

/// Allocate a zeroed vector without an allocation panic or capacity overflow.
pub fn try_vec<T: Default + Clone>(len: usize) -> Result<Vec<T>> {
    let bytes = checked_size("vector bytes", &[len, std::mem::size_of::<T>()])?;
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| Error::Allocation { bytes })?;
    out.resize(len, T::default());
    Ok(out)
}

/// Validate a shape before entering a numerical kernel.
macro_rules! check_shape {
    ($cond:expr, $($arg:tt)*) => {{
        let holds: bool = $cond;
        if !holds {
            return Err($crate::Error::Shape(format!($($arg)*)));
        }
    }};
}
pub(crate) use check_shape;

/// Return a shape error from the enclosing function.
macro_rules! invalid_shape {
    ($($arg:tt)*) => { return Err($crate::Error::Shape(format!($($arg)*))) };
}
pub(crate) use invalid_shape;

/// Read bounded JSON input, preserving its source path in every failure.
/// Model config/tokenizer/header JSON is limited to 128 MiB.
pub fn read_json(path: impl AsRef<std::path::Path>) -> Result<serde_json::Value> {
    let path = path.as_ref();
    let result = (|| {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| Error::SizeOverflow("JSON file"))?;
        if len > 128 * 1024 * 1024 {
            return Err(Error::Config(format!(
                "JSON file is too large: {len} bytes"
            )));
        }
        let mut bytes = try_vec::<u8>(len)?;
        file.read_exact(&mut bytes)?;
        Ok(serde_json::from_slice(&bytes)?)
    })();
    result.map_err(|e: Error| e.at(path))
}

/// Checked extent of a strided, batched matrix; zero dimensions are invalid.
pub fn matrix_span(
    g: usize,
    rows: usize,
    cols: usize,
    batch_stride: usize,
    row_stride: usize,
) -> Result<usize> {
    if g == 0 || rows == 0 || cols == 0 {
        return Err(Error::Shape(format!("empty matrix [{g}, {rows}, {cols}]")));
    }
    (g - 1)
        .checked_mul(batch_stride)
        .and_then(|n| {
            (rows - 1)
                .checked_mul(row_stride)
                .and_then(|r| n.checked_add(r))
        })
        .and_then(|n| n.checked_add(cols))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or(Error::SizeOverflow("strided matrix extent"))
}
