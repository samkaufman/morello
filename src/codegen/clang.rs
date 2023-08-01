use anyhow::{bail, Result};

pub fn get_path() -> Result<String> {
    match std::env::var("CLANG") {
        Ok(v) => Ok(v),
        Err(e) => bail!("Environment variable CLANG is not set"),
    }
}
