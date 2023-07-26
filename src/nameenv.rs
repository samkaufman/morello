use by_address::ByThinAddress;
use std::collections::HashMap;

use crate::{target::Target, utils::ASCII_PAIRS, views::View};

pub struct NameEnv<'t, K: ?Sized> {
    names: HashMap<ByThinAddress<&'t K>, String>,
}

impl<'t, K: ?Sized> NameEnv<'t, K> {
    pub fn new() -> Self {
        NameEnv {
            names: HashMap::new(),
        }
    }

    pub fn name(&mut self, view: &'t K) -> &str {
        let view_by_address = ByThinAddress(view);
        let cnt = self.names.len();
        let name = self
            .names
            .entry(view_by_address)
            .or_insert_with(|| String::from_iter(ASCII_PAIRS[cnt]));
        name
    }

    pub fn get_name(&self, view: &'t K) -> Option<&str> {
        let view_by_address = ByThinAddress(view);
        self.names.get(&view_by_address).map(|s| s.as_str())
    }
}

impl<'t, Tgt: Target> NameEnv<'t, dyn View<Tgt = Tgt>> {
    pub fn get_name_or_display(&self, view: &'t dyn View<Tgt = Tgt>) -> String {
        if let Some(present_name) = self.get_name(view) {
            present_name.to_owned()
        } else if let Some(param) = view.to_param() {
            param.to_string()
        } else {
            panic!("No name for non-Param view");
        }
    }
}
