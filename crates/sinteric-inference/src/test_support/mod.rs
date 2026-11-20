//! Helpers for the crate's tests only; never part of the library.

pub(crate) mod amx;
pub(crate) mod assets;
pub(crate) mod env;
pub(crate) mod media;
pub(crate) mod qual;

/// The variable that tells a child test process which test it runs.
const OWN_PROCESS: &str = "SINTERIC_TEST_OWN_PROCESS";

/// Whether the calling test should run its body here: in a child process
/// of this test binary started for that test alone. In the parent it starts
/// that child, waits for it, fails if the child failed, and returns `false`.
///
/// A test needs its own process when it sets process-global state: the
/// oneDNN ISA cap, the thread pool or [`crate::Runtime::new`]. Before the
/// crates were folded into one, each such test was an integration test
/// binary of its own; this keeps that isolation. `path` is the test's
/// `module_path!()` and name, as [`own_process!`] passes it.
pub(crate) fn in_own_process(path: &str) -> bool {
    let name = test_name(path);
    if std::env::var_os(OWN_PROCESS).is_some_and(|v| v == name) {
        return true;
    }
    let exe = std::env::current_exe().expect("the test binary's path");
    let status = std::process::Command::new(exe)
        .args([name, "--exact", "--include-ignored", "--nocapture"])
        .env(OWN_PROCESS, name)
        .status()
        .expect("start the test in its own process");
    assert!(status.success(), "{name} failed in its own process");
    false
}

/// Runs the test at `path` (its `module_path!()` and name) in a child
/// process of this test binary with `envs` set, asserts that it passed, and
/// returns its output. The child runs the test's body even when the test
/// starts with [`own_process!`]. A test that compares several thread counts
/// starts one child per count, because the pool width is fixed per process.
pub(crate) fn run_child(path: &str, envs: &[(&str, String)]) -> std::process::Output {
    let name = test_name(path);
    let exe = std::env::current_exe().expect("the test binary's path");
    let output = std::process::Command::new(exe)
        .args([name, "--exact", "--include-ignored", "--nocapture"])
        .env(OWN_PROCESS, name)
        .envs(envs.iter().map(|(k, v)| (k, v)))
        .output()
        .expect("start the child test");
    assert!(
        output.status.success(),
        "{name} failed in a child process ({envs:?}):\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

/// The libtest name of the test at `path`: the path without the crate.
fn test_name(path: &str) -> &str {
    path.split_once("::").map_or(path, |(_, rest)| rest)
}

/// Runs the rest of the calling test in its own process
/// ([`in_own_process`]). Put it first in the test, with the test's name.
macro_rules! own_process {
    ($test:ident) => {
        if !$crate::test_support::in_own_process(concat!(module_path!(), "::", stringify!($test))) {
            return;
        }
    };
}
pub(crate) use own_process;
