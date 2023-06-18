use crate::{
    alignment::aligned_approx, common::DimSize, target::Target, tensorspec::TensorSpec,
    tiling::Tiling,
};

use auto_impl::auto_impl;
use smallvec::{smallvec, SmallVec};
use std::{
    borrow::Borrow,
    fmt::{Debug, Display, Formatter},
};

#[auto_impl(Box, Rc)]
pub trait View: Debug {
    type Tgt: Target;

    fn spec(&self) -> &TensorSpec<Self::Tgt>;

    fn shape(&self) -> &[DimSize] {
        self.spec().dim_sizes()
    }

    fn to_param(&self) -> Option<&Param<Self::Tgt>> {
        None
    }
}

// TODO: Rename to ViewTransforms
pub trait ViewExt: View {
    fn squeeze_dims<I: IntoIterator<Item = u8>>(self, dims: I) -> SqueezeDimsView<Self>
    where
        Self: Sized,
    {
        let dims_smallvec = dims.into_iter().collect::<SmallVec<_>>();
        let spec = self.spec().squeeze_dims(&dims_smallvec);
        SqueezeDimsView {
            inner: self,
            dims: dims_smallvec,
            spec,
        }
    }

    fn transpose(self) -> TransposeView<Self>
    where
        Self: Sized,
    {
        let [h, w] = self.shape() else {
            panic!("Cannot transpose a tensor with shape {:?}", self.shape());
        };
        let dim_sizes = smallvec![*w, *h];
        let mut vector_shape = None;
        if let Some(original_vector_shape) = self.spec().vector_shape() {
            debug_assert_eq!(original_vector_shape.len(), 2);
            vector_shape = Some(smallvec![
                original_vector_shape[1],
                original_vector_shape[0]
            ]);
        }

        let (transposed_layout, new_contig) = self
            .spec()
            .layout()
            .transpose((0, 1), self.spec().contiguous_abs());
        let spec = TensorSpec::new_canon(
            dim_sizes,
            self.spec().dtype(),
            new_contig,
            self.spec().aligned(),
            self.spec().level(),
            transposed_layout,
            vector_shape,
        );
        TransposeView { inner: self, spec }
    }
}

impl<V: View> ViewExt for V {}

/// A reference to an Impl node parameter.
///
/// Remember that some scheduling actions result in multi-node Impls, such as for movement which
/// may produce a MoveLet binding and a nested Block. In this case, parameters---and, therefore,
/// the referants of a Param---differ between the MoveLet and Block.
#[derive(Debug, Clone)]
pub struct Param<Tgt: Target>(pub u8, pub TensorSpec<Tgt>);

#[derive(Debug, Clone)]
pub struct Tensor<Tgt: Target>(pub TensorSpec<Tgt>);

#[derive(Debug, Clone)]
pub struct Tile<V: View> {
    pub tiling: Tiling,
    pub view: V,
    spec: TensorSpec<V::Tgt>,
}

#[derive(Debug, Clone)]
pub struct SqueezeDimsView<V: View> {
    pub inner: V,
    pub dims: SmallVec<[u8; 4]>,
    spec: TensorSpec<V::Tgt>,
}

#[derive(Debug, Clone)]
pub struct TransposeView<V: View> {
    pub inner: V,
    spec: TensorSpec<V::Tgt>,
}

impl<Tgt: Target> View for Param<Tgt> {
    type Tgt = Tgt;

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.1
    }

    fn to_param(&self) -> Option<&Param<Self::Tgt>> {
        Some(self)
    }
}

impl<Tgt: Target> Display for Param<Tgt> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

impl<Tgt: Target> View for Tensor<Tgt> {
    type Tgt = Tgt;

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.0
    }
}

impl<V: View> Tile<V> {
    // TODO: Drop this. Callers can build.
    pub fn new(tiling: Tiling, view: V) -> Self {
        let mut spec = view.spec().clone();
        spec.shrink(tiling.shape(), aligned_approx(&tiling, view.spec()));
        Self { tiling, view, spec }
    }

    pub fn steps_dim(&self, dim: u8) -> u32 {
        let origin_size = self.view.borrow().shape()[usize::from(dim)];
        self.tiling.steps_dim(dim, origin_size)
    }

    pub fn boundary_size(&self, dim: u8) -> u32 {
        let origin_size = self.view.borrow().shape()[usize::from(dim)];
        self.tiling.boundary_size(dim, origin_size)
    }
}

impl<V: View> View for Tile<V> {
    type Tgt = V::Tgt;

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.spec
    }
}

impl<V: View> View for SqueezeDimsView<V> {
    type Tgt = V::Tgt;

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.spec
    }
}

impl<V: View> View for TransposeView<V> {
    type Tgt = V::Tgt;

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.spec
    }
}
