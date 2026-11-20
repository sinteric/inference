//! The one error type every call returns, and what kind of failure it is.

use std::fmt;
use std::path::Path;

/// What kind of failure an [`Error`] is, so a host can act on it without
/// reading the message. Match it with a wildcard arm: new kinds may be added.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// The input is malformed: an empty batch, a pixel buffer shorter than
    /// its stated size, a box outside the image, text the tokenizer rejects.
    InvalidInput,
    /// The input is well formed but past a limit: longer than the model's
    /// window, a batch past `max_batch`, or more working memory than the
    /// workspace limit allows.
    TooLarge,
    /// This model, precision or option does not run here: a precision this
    /// CPU has no kernels for, a tower the model was loaded without, a CPU
    /// below the library's AVX2 floor.
    Unsupported,
    /// A model file is missing, unreadable, malformed, or not the pinned one.
    ModelFiles,
    /// A config or runtime option is invalid, or conflicts with the runtime.
    Config,
    /// Memory for a buffer could not be allocated.
    OutOfMemory,
    /// A broken internal invariant or a oneDNN failure: a bug, not the
    /// caller's input.
    Internal,
}

/// An error from the library: a [`ErrorKind`], a message, and the error it
/// came from when there is one ([`std::error::Error::source`]).
///
/// The message is for people and may change; branch on [`Error::kind`].
pub struct Error(Box<Inner>);

struct Inner {
    kind: ErrorKind,
    message: String,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// An error of `kind`. Code built on the library (a fine-tune's own heads
    /// on [`crate::guard`]) uses it to fail in the library's terms.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self(Box::new(Inner {
            kind,
            message: message.into(),
            source: None,
        }))
    }

    /// What kind of failure this is.
    pub fn kind(&self) -> ErrorKind {
        self.0.kind
    }

    /// Prefix the message with the file it concerns.
    pub fn at(self, path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().display().to_string();
        self.context(path)
    }

    pub(crate) fn with_source(
        kind: ErrorKind,
        message: impl Into<String>,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self(Box::new(Inner {
            kind,
            message: message.into(),
            source: Some(source.into()),
        }))
    }

    /// Prefix the message with what was being done; the kind stays.
    pub(crate) fn context(mut self, context: impl fmt::Display) -> Self {
        self.0.message = format!("{context}: {}", self.0.message);
        self
    }

    pub(crate) fn invalid_input(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidInput, message)
    }
    pub(crate) fn too_large(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::TooLarge, message)
    }
    pub(crate) fn unsupported(what: &str, reason: impl fmt::Display) -> Self {
        Self::new(
            ErrorKind::Unsupported,
            format!("unsupported {what}: {reason}"),
        )
    }
    pub(crate) fn model_files(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::ModelFiles, message)
    }
    /// A tensor, key or file a model needs and its files do not have.
    pub(crate) fn missing(what: impl fmt::Display) -> Self {
        Self::new(ErrorKind::ModelFiles, format!("missing or invalid {what}"))
    }
    pub(crate) fn config(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Config, message)
    }
    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Internal, message)
    }
    pub(crate) fn out_of_memory(bytes: usize) -> Self {
        Self::new(
            ErrorKind::OutOfMemory,
            format!("allocation failed for {bytes} bytes"),
        )
    }
    /// A size computed from the input or the model files overflows `usize`.
    pub(crate) fn size_overflow(what: &str) -> Self {
        Self::new(
            ErrorKind::TooLarge,
            format!("size overflow computing {what}"),
        )
    }
    pub(crate) fn workspace_limit(operation: &str, requested: usize, limit: usize) -> Self {
        Self::new(
            ErrorKind::TooLarge,
            format!("{operation} needs {requested} bytes, workspace limit is {limit} bytes"),
        )
    }
    pub(crate) fn backend_status(operation: &str, status: i32) -> Self {
        Self::new(
            ErrorKind::Internal,
            format!("{operation}: oneDNN status {status}"),
        )
    }
    pub(crate) fn poisoned(what: &str) -> Self {
        Self::new(ErrorKind::Internal, format!("{what} lock poisoned"))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.message)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("Error");
        d.field("kind", &self.0.kind)
            .field("message", &self.0.message);
        if let Some(source) = &self.0.source {
            d.field("source", source);
        }
        d.finish()
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0
            .source
            .as_deref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
}

/// File I/O in the library reads and writes model files and their caches.
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::with_source(ErrorKind::ModelFiles, format!("io: {e}"), e)
    }
}

/// JSON in the library is a model's config or tokenizer file.
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::with_source(ErrorKind::ModelFiles, e.to_string(), e)
    }
}

impl From<std::alloc::LayoutError> for Error {
    fn from(e: std::alloc::LayoutError) -> Self {
        Self::with_source(ErrorKind::Internal, e.to_string(), e)
    }
}

impl From<std::array::TryFromSliceError> for Error {
    fn from(e: std::array::TryFromSliceError) -> Self {
        Self::with_source(ErrorKind::Internal, e.to_string(), e)
    }
}

/// Add what was being done to an error; its kind stays.
pub(crate) trait ResultContext<T> {
    fn context(self, context: impl fmt::Display) -> Result<T>;
    fn with_context<C: fmt::Display>(self, context: impl FnOnce() -> C) -> Result<T>;
}

impl<T> ResultContext<T> for Result<T> {
    fn context(self, context: impl fmt::Display) -> Result<T> {
        self.map_err(|e| e.context(context))
    }
    fn with_context<C: fmt::Display>(self, context: impl FnOnce() -> C) -> Result<T> {
        self.map_err(|e| e.context(context()))
    }
}

/// `None` is something the model files should have held.
impl<T> ResultContext<T> for Option<T> {
    fn context(self, context: impl fmt::Display) -> Result<T> {
        self.ok_or_else(|| Error::missing(context))
    }
    fn with_context<C: fmt::Display>(self, context: impl FnOnce() -> C) -> Result<T> {
        self.ok_or_else(|| Error::missing(context()))
    }
}

/// Checked product for dimensions and byte counts derived from external data.
pub fn checked_size(what: &'static str, dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d).ok_or_else(|| Error::size_overflow(what))
    })
}

/// Allocate a zeroed vector without an allocation panic or capacity overflow.
pub fn try_vec<T: Default + Clone>(len: usize) -> Result<Vec<T>> {
    let bytes = checked_size("vector bytes", &[len, std::mem::size_of::<T>()])?;
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| Error::out_of_memory(bytes))?;
    out.resize(len, T::default());
    Ok(out)
}

/// Check an internal invariant (a shape the library itself computed) before
/// a numerical kernel; failing is [`ErrorKind::Internal`]. Input from the
/// caller is checked with [`check_input!`], model files with
/// [`check_files!`].
macro_rules! check_shape {
    ($cond:expr, $($arg:tt)*) => {{
        let holds: bool = $cond;
        if !holds {
            return Err($crate::Error::internal(format!($($arg)*)));
        }
    }};
}
pub(crate) use check_shape;

/// Check the caller's input; failing is [`ErrorKind::InvalidInput`].
macro_rules! check_input {
    ($cond:expr, $($arg:tt)*) => {{
        let holds: bool = $cond;
        if !holds {
            return Err($crate::Error::invalid_input(format!($($arg)*)));
        }
    }};
}
pub(crate) use check_input;

/// Check what a model file holds; failing is [`ErrorKind::ModelFiles`].
macro_rules! check_files {
    ($cond:expr, $($arg:tt)*) => {{
        let holds: bool = $cond;
        if !holds {
            return Err($crate::Error::model_files(format!($($arg)*)));
        }
    }};
}
pub(crate) use check_files;

/// Return an internal-invariant error from the enclosing function.
macro_rules! invalid_shape {
    ($($arg:tt)*) => { return Err($crate::Error::internal(format!($($arg)*))) };
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
            .map_err(|_| Error::size_overflow("JSON file"))?;
        if len > 128 * 1024 * 1024 {
            return Err(Error::model_files(format!(
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
        return Err(Error::internal(format!(
            "empty matrix [{g}, {rows}, {cols}]"
        )));
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
        .ok_or_else(|| Error::size_overflow("strided matrix extent"))
}
