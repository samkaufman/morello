use anyhow::{bail, Result};

pub fn clang_path() -> Result<String> {
    match std::env::var("CLANG") {
        Ok(v) => Ok(v),
        Err(_) => bail!("Environment variable CLANG is not set"),
    }
}
