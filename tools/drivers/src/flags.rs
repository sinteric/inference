//! `--name value` flags and bare `--switch`es, each given at most once.

use std::collections::HashMap;
use std::str::FromStr;

use anyhow::{bail, Context, Result};

/// The flags of one command line, checked against what the driver accepts.
#[derive(Debug)]
pub struct Flags {
    values: HashMap<String, String>,
    switches: Vec<String>,
}

impl Flags {
    /// Parse `args` (without the program name). `valued` take a value;
    /// `switches` do not. Anything else is an error.
    pub fn parse(
        args: impl IntoIterator<Item = String>,
        valued: &[&str],
        switches: &[&str],
    ) -> Result<Self> {
        let mut values = HashMap::new();
        let mut on = Vec::new();
        let mut args = args.into_iter();
        while let Some(key) = args.next() {
            if switches.contains(&key.as_str()) {
                if on.contains(&key) {
                    bail!("duplicate argument {key}");
                }
                on.push(key);
            } else if valued.contains(&key.as_str()) {
                let value = args
                    .next()
                    .with_context(|| format!("missing value for {key}"))?;
                if values.insert(key.clone(), value).is_some() {
                    bail!("duplicate argument {key}");
                }
            } else {
                bail!("unknown argument {key}");
            }
        }
        Ok(Self {
            values,
            switches: on,
        })
    }

    /// Whether `--name` was given.
    pub fn switch(&self, name: &str) -> bool {
        self.switches.iter().any(|s| s == name)
    }

    /// The value of `--name`, parsed, if given.
    pub fn get<T: FromStr>(&self, name: &str) -> Result<Option<T>>
    where
        T::Err: std::fmt::Display,
    {
        self.values
            .get(name)
            .map(|v| v.parse().map_err(|e| anyhow::anyhow!("{name} {v}: {e}")))
            .transpose()
    }

    /// The value of `--name`, parsed; an error when it is missing.
    pub fn required<T: FromStr>(&self, name: &str) -> Result<T>
    where
        T::Err: std::fmt::Display,
    {
        self.get(name)?
            .with_context(|| format!("required argument {name}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn values_switches_and_errors() {
        let f = Flags::parse(args("--threads 4 --warmup"), &["--threads"], &["--warmup"]).unwrap();
        assert_eq!(f.required::<usize>("--threads").unwrap(), 4);
        assert!(f.switch("--warmup"));
        assert!(f.get::<usize>("--tile").unwrap().is_none());
        assert!(f.required::<usize>("--tile").is_err());
        assert!(Flags::parse(args("--threads"), &["--threads"], &[]).is_err());
        assert!(Flags::parse(args("--threads 1 --threads 2"), &["--threads"], &[]).is_err());
        assert!(Flags::parse(args("--other 1"), &["--threads"], &[]).is_err());
        let f = Flags::parse(args("--threads x"), &["--threads"], &[]).unwrap();
        assert!(f.get::<usize>("--threads").is_err());
    }
}
