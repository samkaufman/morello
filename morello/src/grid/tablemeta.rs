use super::general::{BiMapExt, SurMap};

pub trait TableMeta: SurMap {
    fn dimension_types(&self, input: &Self::Domain) -> Vec<DimensionType>;
}

pub trait TableBiMap: BiMapExt + TableMeta {}

pub trait TableSurMap: SurMap + TableMeta {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimensionType {
    Other,
    Accum,
    Shape,
    Dtype,
    Contig,
    Aligned,
    Level,
    Layout,
    VectorSize,
    SerialOnly,
    MemoryLimits,
}

impl<T> TableBiMap for T where T: BiMapExt + TableMeta {}
impl<T> TableSurMap for T where T: SurMap + TableMeta {}
