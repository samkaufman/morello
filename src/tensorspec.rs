use itertools::Itertools;
use serde::{Deserialize, Serialize};

use std::collections::HashSet;
use std::fmt::Display;

use crate::common::{Contig, DimSize, Dtype, Shape};
use crate::grid::canon::CanonicalBimap;
use crate::grid::general::Bimap;
use crate::grid::linear::BimapInt;
use crate::layout::{row_major, Layout};
use crate::target::{MemoryLevel, Target};
use crate::utils::join_into_string;

#[derive(Clone, PartialEq, Eq, Debug, Hash, Deserialize, Serialize)]
#[serde(bound = "")]
pub struct TensorSpec<Tgt: Target> {
    pub shape: Shape, // TODO: Rename to shape
    pub dtype: Dtype,
    pub aux: TensorSpecAux<Tgt>,
}

// TODO: This probably shouldn't be public.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct TensorSpecAux<Tgt: Target> {
    pub contig: Contig,
    pub aligned: bool,
    pub level: Tgt::Level,
    pub layout: Layout,
    pub vector_size: Option<DimSize>,
}

#[derive(Default)]
pub struct TensorSpecAuxBimap<Tgt: Target> {
    phantom: std::marker::PhantomData<Tgt>,
}

#[derive(Default)]
pub struct TensorSpecAuxNonDepBimap<Tgt: Target> {
    phantom: std::marker::PhantomData<Tgt>,
}

impl<Tgt: Target> TensorSpec<Tgt> {
    pub fn new_canon(
        shape: Shape,
        dtype: Dtype,
        contiguous_abs: Contig,
        aligned: bool,
        level: Tgt::Level,
        layout: Layout,
        vector_size: Option<DimSize>,
    ) -> Self {
        let mut r = Self::new_noncanon(
            shape,
            dtype,
            contiguous_abs,
            aligned,
            level,
            layout,
            vector_size,
        );
        r.canonicalize();
        r
    }

    pub fn new_noncanon(
        shape: Shape,
        dtype: Dtype,
        contiguous_abs: Contig,
        aligned: bool,
        level: Tgt::Level,
        layout: Layout,
        vector_size: Option<DimSize>,
    ) -> Self {
        Self::new_noncanon_with_aux(
            shape,
            dtype,
            TensorSpecAux {
                contig: contiguous_abs,
                aligned,
                level,
                layout,
                vector_size,
            },
        )
    }

    pub fn new_noncanon_with_aux(shape: Shape, dtype: Dtype, aux: TensorSpecAux<Tgt>) -> Self {
        if shape.is_empty() || shape.iter().any(|&d| d < 1) {
            panic!("Invalid shape: {:?}", shape);
        }

        if aux.vector_size.is_some() != aux.level.vector_rf() {
            panic!(
                "vector_size must be specified if and only if the bank ({:?}) is a vector register file", aux.level
            )
        }

        TensorSpec { shape, dtype, aux }
    }

    pub fn layout(&self) -> Layout {
        self.aux.layout.clone()
    }

    pub fn set_layout(&mut self, new_layout: Layout) {
        self.aux.layout = new_layout;
    }

    pub fn set_contiguous_abs(&mut self, contiguous_abs: Contig) {
        self.aux.contig = contiguous_abs;
    }

    pub fn is_contiguous(&self) -> bool {
        self.aux.contig == self.aux.layout.contiguous_full()
    }

    /// Returns true if this TensorSpec can be tiled to the given shape.
    pub fn is_valid_tile_shape(&self, shape: &[DimSize]) -> bool {
        if shape.len() != self.shape.len() {
            return false;
        }

        if !shape.iter().zip(self.shape.iter()).all(|(i, o)| i <= o) {
            return false;
        }

        let all_ones = shape.iter().all(|d| *d == 1);
        if !all_ones && !self.aux.layout.applies_to_shape(shape) {
            return false;
        }

        true
    }

    pub fn bytes_used(&self) -> u64 {
        u64::from(self.dtype.size()) * self.shape.iter().copied().map(u64::from).product::<u64>()
    }

    pub fn shape(&self) -> &[DimSize] {
        &self.shape
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn contiguous_abs(&self) -> Contig {
        self.aux.contig
    }

    pub fn aligned(&self) -> bool {
        self.aux.aligned
    }

    pub fn level(&self) -> <Tgt as Target>::Level {
        self.aux.level
    }

    pub fn vector_size(&self) -> Option<DimSize> {
        self.aux.vector_size
    }

    pub fn set_level(&mut self, level: Tgt::Level, vector_size: Option<DimSize>) {
        assert_eq!(
            level.vector_rf(),
            vector_size.is_some(),
            "Cannot set level to {:?} with vector shape {:?}",
            level,
            vector_size
        );
        self.aux.level = level;
        self.aux.vector_size = vector_size;
    }

    /// Returns a new TensorSpec with the given shape and alignment.
    ///
    /// The result's layout and contiguousness abstraction will have been
    /// canoncialized for the given shape.
    pub fn shrink(&mut self, shape: &[DimSize], aligned: bool) {
        // This implementation is similar to `canonicalize`, but the tile
        // contiguousness is computed from both old and new shapes.
        self.aux.contig = self
            .layout()
            .tile_contiguity(shape, &self.shape, self.aux.contig);
        self.shape = Shape::from(shape);
        self.aux.layout = self.aux.layout.canonicalize_for_shape(&self.shape);
        self.aux.aligned = aligned;
    }

    pub fn canonicalize(&mut self) {
        self.aux.canonicalize(&self.shape, self.aux.aligned);
    }

    /// Returns a TensorSpec with given size-one dimensions dropped.
    ///
    /// The given dimension indices must be sorted in ascending order.
    ///
    /// The result will be canoncialized. If any given dimension index is not
    /// size one, this method panics.
    pub fn squeeze_dims(&self, dropped_dims: &[u8]) -> TensorSpec<Tgt> {
        // Make sure dropped_dims is sorted.
        debug_assert!(dropped_dims.windows(2).all(|w| w[0] < w[1]));

        let mut new_shape = Shape::from(self.shape());
        for &dim in dropped_dims.iter().rev() {
            assert_eq!(
                new_shape[usize::from(dim)],
                1,
                "Cannot drop non-degenerate dimension {} of shape {:?}",
                dim,
                new_shape
            );
            new_shape.remove(dim.into());
        }

        let (new_layout, new_contig) = if new_shape.iter().all(|&d| d == 1) {
            let new_layout = row_major(new_shape.len().try_into().unwrap());
            (new_layout.clone(), new_layout.contiguous_full())
        } else {
            let dropped_dims_set = dropped_dims.iter().copied().collect::<HashSet<_>>();
            self.layout()
                .dim_drop(&dropped_dims_set, self.contiguous_abs())
        };

        TensorSpec::new_canon(
            new_shape,
            self.dtype(),
            new_contig,
            self.aligned(),
            self.level(),
            new_layout,
            self.vector_size(),
        )
    }

    // TODO: Shouldn't need this method. Should be implicit in Spec validity.
    pub fn can_move_to(&self, dest_layout: &Layout, dest_level: &Tgt::Level) -> bool {
        // If the destination is in VRF, then the operand volume must be a multiple of at least one
        // of the vector sizes.
        let vector_bytes = dest_level.vector_bytes();
        if !vector_bytes.is_empty() {
            let vol: DimSize = self.shape().iter().product();
            let bytes = vol * DimSize::from(self.dtype.size());
            if vector_bytes.iter().all(|&vb| bytes % vb != 0) {
                return false;
            }
        }
        true
    }
}

impl<Tgt: Target> Display for TensorSpec<Tgt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dims_part = self
            .shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("×");
        let aux_part = tensorspec_aux_str(&self.aux);

        write!(f, "({}, {}", dims_part, self.dtype)?;
        if aux_part.is_empty() {
            write!(f, ")")
        } else {
            write!(f, ", {})", aux_part)
        }
    }
}

impl<Tgt: Target> TensorSpecAux<Tgt> {
    pub fn canonicalize(&mut self, shape: &Shape, aligned: bool) {
        self.contig = self.layout.tile_contiguity(shape, shape, self.contig);
        self.layout = self.layout.canonicalize_for_shape(shape);
        self.aligned = aligned;
    }
}

impl<Tgt> CanonicalBimap for TensorSpecAux<Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: Bimap<Domain = Tgt::Level, Codomain = BimapInt>,
{
    type Bimap = TensorSpecAuxBimap<Tgt>;

    fn bimap() -> Self::Bimap {
        TensorSpecAuxBimap::default()
    }
}

impl<Tgt: Target> Display for TensorSpecAux<Tgt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let aux_part = tensorspec_aux_str(self);
        if aux_part.is_empty() {
            write!(f, "(_)")
        } else {
            write!(f, "({})", aux_part)
        }
    }
}

impl<Tgt> Bimap for TensorSpecAuxBimap<Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: Bimap<Domain = Tgt::Level, Codomain = BimapInt>,
{
    type Domain = TensorSpecAux<Tgt>;
    type Codomain = ((Contig, bool, Layout, Option<DimSize>), [BimapInt; 1]);
    type DomainIter = std::iter::Once<TensorSpecAux<Tgt>>;

    fn apply(&self, t: &Self::Domain) -> Self::Codomain {
        (
            (t.contig, t.aligned, t.layout.clone(), t.vector_size),
            [Tgt::Level::bimap().apply(&t.level)],
        )
    }

    fn apply_inverse(&self, i: &Self::Codomain) -> Self::DomainIter {
        let ((contig, aligned, layout, vector_size), [level_val]) = i.clone();
        std::iter::once(TensorSpecAux {
            contig,
            aligned,
            layout,
            vector_size,
            level: Tgt::Level::bimap()
                .apply_inverse(&level_val)
                .exactly_one()
                .unwrap_or_else(|_| panic!()),
        })
    }
}

impl<Tgt> Bimap for TensorSpecAuxNonDepBimap<Tgt>
where
    Tgt: Target,
    Tgt::Level: CanonicalBimap,
    <Tgt::Level as CanonicalBimap>::Bimap: Bimap<Domain = Tgt::Level, Codomain = BimapInt>,
{
    type Domain = TensorSpecAux<Tgt>;
    type Codomain = (Layout, [BimapInt; 4]);
    type DomainIter = std::iter::Once<Self::Domain>;

    fn apply(&self, aux: &TensorSpecAux<Tgt>) -> Self::Codomain {
        let inverted_contig = aux.layout.contiguous_full() - aux.contig;
        (
            aux.layout.clone(),
            [
                inverted_contig.into(),
                aux.aligned as _,
                Tgt::Level::bimap().apply(&aux.level),
                aux.vector_size.unwrap_or(0),
            ],
        )
    }

    fn apply_inverse(&self, v: &Self::Codomain) -> Self::DomainIter {
        let (layout, pt) = v;
        let [inverted_contig, aligned_val, level_val, vector_size_val] = pt;
        // `unwrap_or_else` rather than `unwrap` to avoid needing a Debug bound
        let level = Tgt::Level::bimap()
            .apply_inverse(level_val)
            .exactly_one()
            .unwrap_or_else(|_| panic!());
        std::iter::once(TensorSpecAux {
            layout: layout.clone(),
            contig: layout.contiguous_full() - Contig::try_from(*inverted_contig).unwrap(),
            aligned: *aligned_val != 0,
            level,
            vector_size: if *vector_size_val == 0 {
                None
            } else {
                Some(*vector_size_val)
            },
        })
    }
}

fn tensorspec_aux_str<Tgt: Target>(aux: &TensorSpecAux<Tgt>) -> String {
    let mut parts = Vec::with_capacity(5);

    if aux.level != Tgt::default_level() {
        parts.push(aux.level.to_string());
    }

    if !aux.layout.is_row_major() {
        parts.push(aux.layout.to_string());
    }

    if aux.contig != aux.layout.contiguous_full() {
        parts.push(format!("c{}", aux.contig));
    }

    if !aux.aligned {
        parts.push(String::from("ua"));
    }

    if let Some(vector_size) = aux.vector_size {
        parts.push(vector_size.to_string());
    }

    join_into_string(&parts, ", ")
}
