//! Statically link pinned oneDNN 3.13.4 (`c531fe1a`).
//!
//! Build from source with cmake (`DNNL_LIBRARY_TYPE=STATIC`, CPU + THREADPOOL)
//! unless `ONEDNN_STATIC_ROOT` (or a cached default prefix) already contains
//! `libdnnl.a` for this release with `DNNL_CPU_RUNTIME=THREADPOOL`. Bindings
//! are the committed `src/bindings.rs`; regenerate with
//! `ONEDNN_BINDGEN=1 cargo build -p onednn-sys --features bindgen`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

const PINNED_COMMIT: &str = "c531fe1a5cd631b7f60277cdf6675c588cadbc28";
const PINNED_MAJOR: i32 = 3;
const PINNED_MINOR: i32 = 13;
const PINNED_PATCH: i32 = 4;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=vendor");
    println!("cargo:rerun-if-changed=snt_threadpool.cpp");
    println!("cargo:rerun-if-changed=snt_amx_probe.cpp");
    println!("cargo:rerun-if-env-changed=ONEDNN_BINDGEN");
    println!("cargo:rerun-if-env-changed=ONEDNN_STATIC_ROOT");
    println!("cargo:rerun-if-env-changed=ONEDNN_REBUILD_NARROW_PREFIX");
    println!("cargo:rerun-if-env-changed=ONEDNN_ROOT");
    compile_threadpool_shim();

    #[cfg(feature = "bindgen")]
    if env::var("ONEDNN_BINDGEN").ok().as_deref() == Some("1") {
        generate();
    }

    let prefix = resolve_prefix();
    let lib = ensure_static_lib(&prefix);
    let lib_dir = lib.parent().expect("libdnnl.a directory");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=dnnl");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:root={}", prefix.display());
}

fn compile_threadpool_shim() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src = manifest.join("snt_threadpool.cpp");
    let obj = out.join("snt_threadpool.o");
    let lib = out.join("libsnt_threadpool.a");
    let cxx = env::var("CXX").unwrap_or_else(|_| "g++".into());
    let vendor = manifest.join("vendor");
    run(
        Command::new(&cxx)
            .args(["-c", "-fPIC", "-O2", "-std=c++14", "-I"])
            .arg(&vendor)
            .arg("-o")
            .arg(&obj)
            .arg(&src),
        "compile snt_threadpool.cpp",
    );
    let probe_src = manifest.join("snt_amx_probe.cpp");
    let probe_obj = out.join("snt_amx_probe.o");
    run(
        Command::new(&cxx)
            .args(["-c", "-fPIC", "-O2", "-std=c++14"])
            .arg("-o")
            .arg(&probe_obj)
            .arg(&probe_src),
        "compile snt_amx_probe.cpp",
    );
    let ar = env::var("AR").unwrap_or_else(|_| "ar".into());
    run(
        Command::new(&ar)
            .args(["crs"])
            .arg(&lib)
            .arg(&obj)
            .arg(&probe_obj),
        "ar libsnt_threadpool.a",
    );
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=snt_threadpool");
}

fn default_tp_prefix() -> PathBuf {
    let home =
        env::var_os("HOME").expect("set HOME or ONEDNN_STATIC_ROOT for the oneDNN build prefix");
    PathBuf::from(home).join(".cache/sinteric-sdk/onednn-static-tp")
}

fn resolve_prefix() -> PathBuf {
    for name in ["ONEDNN_STATIC_ROOT", "ONEDNN_ROOT"] {
        if let Some(p) = env::var_os(name).filter(|p| !p.is_empty()) {
            return PathBuf::from(p);
        }
    }
    default_tp_prefix()
}

fn ensure_static_lib(prefix: &Path) -> PathBuf {
    if let Some(lib) = find_archive(prefix) {
        assert!(version_ok(prefix), "existing oneDNN archive at {} is not pinned 3.13.4; supply a matching ONEDNN_STATIC_ROOT", prefix.display());
        assert!(runtime_is_threadpool(prefix), "existing oneDNN archive at {} is not THREADPOOL; supply a matching ONEDNN_STATIC_ROOT (the prefix was not modified)", prefix.display());
        let missing = missing_ocr_primitives(prefix);
        if missing.is_empty() {
            return lib;
        }
        if env::var("ONEDNN_REBUILD_NARROW_PREFIX").ok().as_deref() != Some("1") {
            panic!("oneDNN prefix {} is missing {}. No implicit rebuild was performed. Refresh explicitly: CC=gcc CXX=g++ ONEDNN_STATIC_ROOT='{}' ONEDNN_REBUILD_NARROW_PREFIX=1 cargo build -j6 -p onednn-sys --release",prefix.display(),missing.join(", "),prefix.display());
        }
    }
    build_from_source(prefix);
    find_archive(prefix).unwrap_or_else(|| {
        panic!(
            "static oneDNN 3.13.4 THREADPOOL build did not produce libdnnl.a under {}",
            prefix.display()
        )
    })
}

fn find_archive(prefix: &Path) -> Option<PathBuf> {
    for rel in ["lib/libdnnl.a", "lib64/libdnnl.a", "libdnnl.a"] {
        let p = prefix.join(rel);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn version_ok(prefix: &Path) -> bool {
    let header = prefix.join("include/oneapi/dnnl/dnnl_version.h");
    let Ok(text) = fs::read_to_string(&header) else {
        return false;
    };
    text.contains(&format!("#define DNNL_VERSION_MAJOR {PINNED_MAJOR}"))
        && text.contains(&format!("#define DNNL_VERSION_MINOR {PINNED_MINOR}"))
        && text.contains(&format!("#define DNNL_VERSION_PATCH {PINNED_PATCH}"))
}

fn missing_ocr_primitives(prefix: &Path) -> Vec<String> {
    let text =
        fs::read_to_string(prefix.join("include/oneapi/dnnl/dnnl_config.h")).unwrap_or_default();
    if text.contains("#define BUILD_PRIMITIVE_ALL 1") {
        return Vec::new();
    }
    [
        "CONVOLUTION",
        "DECONVOLUTION",
        "POOLING",
        "RESAMPLING",
        "REDUCTION",
        "CONCAT",
        "PRELU",
    ]
    .iter()
    .filter(|op| !text.contains(&format!("#define BUILD_{op} 1")))
    .map(|op| format!("BUILD_{op}"))
    .collect()
}

fn runtime_is_threadpool(prefix: &Path) -> bool {
    let header = prefix.join("include/oneapi/dnnl/dnnl_config.h");
    let Ok(text) = fs::read_to_string(&header) else {
        return false;
    };
    text.contains("DNNL_CPU_RUNTIME DNNL_RUNTIME_THREADPOOL")
}

fn build_from_source(prefix: &Path) {
    let cmake = which("cmake").expect(
        "cmake is required to build pinned oneDNN 3.13.4 (or set ONEDNN_STATIC_ROOT to a prefix with libdnnl.a)",
    );
    let src = source_dir();
    if !src.join("CMakeLists.txt").is_file() {
        fetch_source(&src);
    }
    let build = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR")).join("onednn-build");
    fs::create_dir_all(prefix).expect("create ONEDNN_STATIC_ROOT");
    fs::create_dir_all(&build).expect("create oneDNN build dir");

    let mut conf = Command::new(&cmake);
    conf.arg("-S")
        .arg(&src)
        .arg("-B")
        .arg(&build)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!("-DCMAKE_INSTALL_PREFIX={}", prefix.display()))
        .arg("-DDNNL_LIBRARY_TYPE=STATIC")
        .arg("-DONEDNN_CPU_RUNTIME=THREADPOOL")
        .arg("-DONEDNN_GPU_RUNTIME=NONE")
        .arg("-DONEDNN_BUILD_GRAPH=OFF")
        .arg("-DONEDNN_BUILD_TESTS=OFF")
        .arg("-DONEDNN_BUILD_EXAMPLES=OFF")
        .arg("-DONEDNN_ENABLE_WORKLOAD=INFERENCE")
        .arg("-DONEDNN_ENABLE_PRIMITIVE=MATMUL;REORDER;LAYER_NORMALIZATION;SOFTMAX;ELTWISE;BINARY;REDUCTION;CONVOLUTION;DECONVOLUTION;POOLING;RESAMPLING;CONCAT;PRELU")
        .arg("-DONEDNN_ENABLE_CONCURRENT_EXEC=ON")
        .arg("-DONEDNN_ENABLE_PRIMITIVE_CACHE=OFF");
    if which("ninja").is_some() {
        conf.arg("-G").arg("Ninja");
    }
    run(&mut conf, "cmake configure oneDNN");

    let jobs = env::var("NUM_JOBS")
        .ok()
        .or_else(|| env::var("CARGO_NUM_JOBS").ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "4".into())
        });
    let mut build_cmd = Command::new(&cmake);
    build_cmd
        .arg("--build")
        .arg(&build)
        .arg("--parallel")
        .arg(&jobs);
    run(&mut build_cmd, "cmake build oneDNN");
    run(
        Command::new(&cmake).arg("--install").arg(&build),
        "cmake install oneDNN",
    );
    let _ = fs::write(prefix.join("PINNED_COMMIT"), PINNED_COMMIT);
    let _ = fs::write(prefix.join("CPU_RUNTIME"), "THREADPOOL");
    if !version_ok(prefix) || !runtime_is_threadpool(prefix) {
        panic!(
            "installed oneDNN at {} is not 3.13.4 THREADPOOL (want {}.{}.{} / {PINNED_COMMIT})",
            prefix.display(),
            PINNED_MAJOR,
            PINNED_MINOR,
            PINNED_PATCH
        );
    }
}

fn source_dir() -> PathBuf {
    let cache = PathBuf::from(env::var_os("HOME").expect("set HOME to cache oneDNN build sources"))
        .join(".cache/sinteric-sdk/.sdk-src");
    cache.join(format!("onednn-{PINNED_COMMIT}"))
}

/// Where the pinned commit is fetched from.
const REPOSITORY: &str = "https://github.com/uxlfoundation/oneDNN";

/// Fetch the pinned source into `src`: GitHub's archive of the commit, or,
/// when that is refused, a shallow git fetch of the same commit. A sandbox can
/// let git read a public repository and still refuse its archive URL: measured
/// 2026-11-04 behind a sandbox proxy that answered the archive with 403 and
/// served `git fetch` of the commit.
fn fetch_source(src: &Path) {
    let Err(archive) = fetch_archive(src) else {
        return;
    };
    // Whatever the archive left half-written would pass for a source tree.
    let _ = fs::remove_dir_all(src);
    if let Err(git) = fetch_git(src) {
        panic!("could not fetch oneDNN {PINNED_COMMIT}: {archive}; then {git}");
    }
}

fn fetch_archive(src: &Path) -> Result<(), String> {
    fs::create_dir_all(src).map_err(|e| format!("create {}: {e}", src.display()))?;
    let url = format!("{REPOSITORY}/archive/{PINNED_COMMIT}.tar.gz");
    let tar = src
        .parent()
        .unwrap_or(src)
        .join(format!("onednn-{PINNED_COMMIT}.tar.gz"));
    let status = Command::new("curl")
        .args(["-fsSL", "-o"])
        .arg(&tar)
        .arg(&url)
        .status()
        .map_err(|e| format!("curl {url}: {e}"))?;
    if !status.success() {
        let _ = fs::remove_file(&tar);
        return Err(format!("curl {url} failed: {status}"));
    }
    let status = Command::new("tar")
        .args(["-xz", "--strip-components=1", "-C"])
        .arg(src)
        .arg("-f")
        .arg(&tar)
        .status()
        .map_err(|e| format!("tar xf {}: {e}", tar.display()))?;
    if !status.success() {
        return Err(format!("tar xf {} failed: {status}", tar.display()));
    }
    Ok(())
}

fn fetch_git(src: &Path) -> Result<(), String> {
    fs::create_dir_all(src).map_err(|e| format!("create {}: {e}", src.display()))?;
    let git = |args: &[&str]| -> Result<String, String> {
        let out = Command::new("git")
            .arg("-C")
            .arg(src)
            .args(args)
            .output()
            .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
        if !out.status.success() {
            return Err(format!(
                "git {} failed: {}: {}",
                args.join(" "),
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
    };
    git(&["init", "-q"])?;
    git(&["fetch", "-q", "--depth", "1", REPOSITORY, PINNED_COMMIT])?;
    git(&["checkout", "-q", "FETCH_HEAD"])?;
    let head = git(&["rev-parse", "HEAD"])?;
    if head != PINNED_COMMIT {
        return Err(format!("git fetched {head}, not {PINNED_COMMIT}"));
    }
    Ok(())
}

fn which(name: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|paths| {
        env::split_paths(&paths).find_map(|p| {
            let cand = p.join(name);
            cand.is_file().then_some(cand)
        })
    })
}

fn run(cmd: &mut Command, what: &str) {
    let status = cmd.status().unwrap_or_else(|e| panic!("{what}: {e}"));
    if !status.success() {
        panic!("{what} failed: {status}");
    }
}

#[cfg(feature = "bindgen")]
fn generate() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let bindings = bindgen::Builder::default()
        .header(manifest.join("wrapper.h").to_string_lossy())
        .clang_arg(format!("-I{}", manifest.join("vendor").display()))
        .allowlist_function("dnnl_.*")
        .allowlist_type("dnnl_.*")
        .allowlist_var("DNNL_.*")
        .allowlist_var("dnnl_.*")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .layout_tests(false)
        .generate_comments(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen of oneDNN C API");
    let out = manifest.join("src/bindings.rs");
    bindings
        .write_to_file(&out)
        .unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
    println!("cargo:warning=regenerated {}", out.display());
}
