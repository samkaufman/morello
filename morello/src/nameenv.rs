use crate::{opaque_symbol::OpaqueSymbol, target::Target, utils::ASCII_PAIRS, views::View};
use std::collections::HashMap;

#[derive(Default)]
pub struct NameEnv {
    names: HashMap<OpaqueSymbol, String>,
}

impl NameEnv {
    pub fn new() -> Self {
        NameEnv::default()
    }
}

impl NameEnv {
    pub fn name<K: View + ?Sized>(&mut self, view: &K) -> &str {
        let cnt = self.names.len();
        let name = self
            .names
            .entry(view.identifier())
            .or_insert_with(|| String::from_iter(ASCII_PAIRS[cnt]));
        name
    }

    pub fn get_name<K: View + ?Sized>(&self, view: &K) -> Option<&str> {
        self.names.get(&view.identifier()).map(|s| s.as_str())
    }

    pub fn get_name_or_display<Tgt: Target>(&self, view: &dyn View<Tgt = Tgt>) -> String {
        if let Some(present_name) = self.get_name(view) {
            present_name.to_owned()
        } else if let Some(param) = view.to_param() {
            param.to_string()
        } else {
            panic!("No name for non-Param view");
        }
    }
}
