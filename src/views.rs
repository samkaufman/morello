use crate::{
    alignment::aligned_approx,
    common::{DimSize, Shape},
    expr::{AffineExpr, Term},
    layout::BufferExprTerm,
    opaque_symbol::OpaqueSymbol,
    target::Target,
    tensorspec::TensorSpec,
};

use auto_impl::auto_impl;
use smallvec::{smallvec, SmallVec};
use std::{
    collections::HashMap,
    fmt::{Debug, Display, Formatter},
};

#[auto_impl(&, Box, Rc)]
pub trait View: Debug {
    type Tgt: Target;

    fn identifier(&self) -> OpaqueSymbol {
        todo!();
    }

    fn backing_tensor<'a>(
        &'a self,
        env: &'a HashMap<Param<Self::Tgt>, &'a dyn View<Tgt = Self::Tgt>>,
    ) -> Option<&'a Tensor<Self::Tgt>>;

    fn spec(&self) -> &TensorSpec<Self::Tgt>;

    fn shape(&self) -> &[DimSize] {
        self.spec().dim_sizes()
    }

    fn make_buffer_indexing_expr(
        &self,
        env: &HashMap<Param<Self::Tgt>, &dyn View<Tgt = Self::Tgt>>,
    ) -> AffineExpr<BufferExprTerm>;

    /// Update environment to map all nested [Param]s to Views, given `args`.
    fn bind<'i>(
        &self,
        args: &[&'i dyn View<Tgt = Self::Tgt>],
        env: &mut HashMap<Param<Self::Tgt>, &'i dyn View<Tgt = Self::Tgt>>,
    );

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
            self.spec().vector_size(),
        );
        TransposeView { inner: self, spec }
    }
}

impl<V: View> ViewExt for V {}

/// A reference to an Impl node parameter.
///
/// Remember that some scheduling actions result in multi-node Impls, such as for movement which
/// may produce a MoveLet binding and a nested Block. In this case, parameters---and, therefore,
/// the referents of a Param---differ between the MoveLet and Block.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Param<Tgt: Target>(pub u8, pub TensorSpec<Tgt>, OpaqueSymbol);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tensor<Tgt: Target>(pub TensorSpec<Tgt>, OpaqueSymbol);

#[derive(Debug, Clone)]
pub struct CacheView<V: View> {
    pub source: V,
    spec: TensorSpec<V::Tgt>,
}

impl<V: View> CacheView<V> {
    pub fn new(source: V, spec: TensorSpec<V::Tgt>) -> CacheView<V> {
        CacheView { source, spec }
    }
}

/// A tile with a fixed shape, resulting from applying a [Tiling] to a [View].
#[derive(Debug, Clone)]
pub struct Tile<V: View> {
    // TODO: Since Tiles don't have boundaries, we shouldn't store Tiling.
    shape: Shape,
    step_sizes: SmallVec<[DimSize; 5]>,
    pub view: V,
    expr_term_id: OpaqueSymbol,
    spec: TensorSpec<V::Tgt>,
}

#[derive(Debug)]
pub struct SqueezeDimsView<V: View> {
    pub inner: V,
    pub dims: SmallVec<[u8; 4]>,
    spec: TensorSpec<V::Tgt>,
}

#[derive(Debug)]
pub struct TransposeView<V: View> {
    pub inner: V,
    spec: TensorSpec<V::Tgt>,
}

impl<Tgt: Target> Param<Tgt> {
    pub fn new(dim_idx: u8, spec: TensorSpec<Tgt>) -> Param<Tgt> {
        Param(dim_idx, spec, OpaqueSymbol::new())
    }
}

impl<Tgt: Target> View for Param<Tgt> {
    type Tgt = Tgt;

    fn backing_tensor<'a>(
        &'a self,
        env: &'a HashMap<Param<Self::Tgt>, &'a dyn View<Tgt = Self::Tgt>>,
    ) -> Option<&'a Tensor<Self::Tgt>> {
        env.get(self).and_then(|v| v.backing_tensor(env))
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.1
    }

    fn make_buffer_indexing_expr(
        &self,
        env: &HashMap<Param<Self::Tgt>, &dyn View<Tgt = Self::Tgt>>,
    ) -> AffineExpr<BufferExprTerm> {
        let resolved = env.get(self).expect("Param should be in environment");
        resolved.make_buffer_indexing_expr(env)
    }

    fn bind<'i>(
        &self,
        args: &[&'i dyn View<Tgt = Self::Tgt>],
        env: &mut HashMap<Param<Self::Tgt>, &'i dyn View<Tgt = Self::Tgt>>,
    ) {
        if env
            .insert(self.clone(), args[usize::from(self.0)])
            .is_some()
        {
            panic!("Identifier was already in environment")
        }
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

impl<Tgt: Target> Tensor<Tgt> {
    pub fn new(spec: TensorSpec<Tgt>) -> Self {
        Tensor(spec, OpaqueSymbol::new())
    }
}

impl<Tgt: Target> View for Tensor<Tgt> {
    type Tgt = Tgt;

    fn backing_tensor<'a>(
        &'a self,
        _env: &'a HashMap<Param<Self::Tgt>, &'a dyn View<Tgt = Self::Tgt>>,
    ) -> Option<&'a Tensor<Self::Tgt>> {
        Some(self)
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.0
    }

    fn make_buffer_indexing_expr(
        &self,
        _env: &HashMap<Param<Self::Tgt>, &dyn View<Tgt = Self::Tgt>>,
    ) -> AffineExpr<BufferExprTerm> {
        self.spec()
            .layout()
            .buffer_indexing_expr(&self.1, self.shape())
    }

    fn bind<'i>(
        &self,
        _args: &[&'i dyn View<Tgt = Self::Tgt>],
        _env: &mut HashMap<Param<Self::Tgt>, &'i dyn View<Tgt = Self::Tgt>>,
    ) {
    }
}

impl<V: View> View for CacheView<V> {
    type Tgt = V::Tgt;

    fn backing_tensor<'a>(
        &'a self,
        env: &'a HashMap<Param<Self::Tgt>, &'a dyn View<Tgt = Self::Tgt>>,
    ) -> Option<&'a Tensor<Self::Tgt>> {
        self.source.backing_tensor(env)
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.spec
    }

    fn make_buffer_indexing_expr(
        &self,
        env: &HashMap<Param<Self::Tgt>, &dyn View<Tgt = Self::Tgt>>,
    ) -> AffineExpr<BufferExprTerm> {
        self.source.make_buffer_indexing_expr(env)
    }

    fn bind<'i>(
        &self,
        args: &[&'i dyn View<Tgt = Self::Tgt>],
        env: &mut HashMap<Param<Self::Tgt>, &'i dyn View<Tgt = Self::Tgt>>,
    ) {
        self.source.bind(args, env)
    }
}

impl<V: View> Tile<V> {
    // TODO: Drop this. Callers can build.
    pub fn new(shape: Shape, step_sizes: Shape, view: V) -> Self {
        let expr_term_id = OpaqueSymbol::new();
        let mut spec = view.spec().clone();
        spec.shrink(&shape, aligned_approx(&shape, &step_sizes, view.spec()));
        Self {
            shape,
            step_sizes,
            view,
            expr_term_id,
            spec,
        }
    }

    pub fn step_sizes(&self) -> &[DimSize] {
        &self.step_sizes
    }

    /// Yields [`BufferExprTerm::TileIdx`] terms for each non-degenerate tiling dimension.
    pub fn tile_dim_terms(&self) -> impl Iterator<Item = BufferExprTerm> + '_ {
        (0..self.shape.len()).filter_map(|dim| {
            let steps = self.steps_dim(dim.try_into().unwrap());
            debug_assert_ne!(steps, 0);
            if steps != 1 {
                Some(BufferExprTerm::TileIdx(
                    dim.try_into().unwrap(),
                    self.expr_term_id.clone(),
                ))
            } else {
                None
            }
        })
    }

    pub fn steps_dim(&self, dim: u8) -> u32 {
        let origin_size = self.view.shape()[usize::from(dim)];
        divrem::DivCeil::div_ceil(origin_size, self.step_sizes[usize::from(dim)])
    }
}

impl<T: View> View for Tile<T> {
    type Tgt = T::Tgt;

    fn backing_tensor<'a>(
        &'a self,
        env: &'a HashMap<Param<Self::Tgt>, &'a dyn View<Tgt = Self::Tgt>>,
    ) -> Option<&'a Tensor<Self::Tgt>> {
        self.view.backing_tensor(env)
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.spec
    }

    fn make_buffer_indexing_expr(
        &self,
        env: &HashMap<Param<Self::Tgt>, &dyn View<Tgt = Self::Tgt>>,
    ) -> AffineExpr<BufferExprTerm> {
        let expr = self.view.make_buffer_indexing_expr(env);
        if self
            .shape()
            .iter()
            .zip(&self.step_sizes)
            .any(|(a, b)| *a != *b)
        {
            todo!("Implement support for sliding tilings.");
        }

        let mut new_expr = expr.clone();
        for t in &expr.0 {
            let BufferExprTerm::Pt(dim, _) = &t.1 else {
                continue;
            };

            let size_in_dim = self.shape()[usize::from(*dim)];

            let logical_substitution = {
                let mut terms = vec![Term(1, BufferExprTerm::Pt(*dim, self.expr_term_id.clone()))];
                if size_in_dim != self.view.shape()[usize::from(*dim)] {
                    terms.push(Term(
                        size_in_dim.try_into().unwrap(),
                        BufferExprTerm::TileIdx(*dim, self.expr_term_id.clone()),
                    ));
                }
                AffineExpr(terms, 0)
            };
            new_expr = new_expr.subs(&t.1, logical_substitution);
        }
        new_expr
    }

    fn bind<'i>(
        &self,
        args: &[&'i dyn View<Tgt = Self::Tgt>],
        env: &mut HashMap<Param<Self::Tgt>, &'i dyn View<Tgt = Self::Tgt>>,
    ) {
        self.view.bind(args, env)
    }
}

impl<T: View> View for SqueezeDimsView<T> {
    type Tgt = T::Tgt;

    fn backing_tensor<'a>(
        &'a self,
        env: &'a HashMap<Param<Self::Tgt>, &'a dyn View<Tgt = Self::Tgt>>,
    ) -> Option<&'a Tensor<Self::Tgt>> {
        self.inner.backing_tensor(env)
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.spec
    }

    fn make_buffer_indexing_expr(
        &self,
        _env: &HashMap<Param<Self::Tgt>, &dyn View<Tgt = Self::Tgt>>,
    ) -> AffineExpr<BufferExprTerm> {
        todo!()
    }

    fn bind<'i>(
        &self,
        args: &[&'i dyn View<Tgt = Self::Tgt>],
        env: &mut HashMap<Param<Self::Tgt>, &'i dyn View<Tgt = Self::Tgt>>,
    ) {
        self.inner.bind(args, env)
    }
}

impl<T: View> View for TransposeView<T> {
    type Tgt = T::Tgt;

    fn backing_tensor<'a>(
        &'a self,
        env: &'a HashMap<Param<Self::Tgt>, &'a dyn View<Tgt = Self::Tgt>>,
    ) -> Option<&'a Tensor<Self::Tgt>> {
        self.inner.backing_tensor(env)
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.spec
    }

    fn make_buffer_indexing_expr(
        &self,
        _env: &HashMap<Param<Self::Tgt>, &dyn View<Tgt = Self::Tgt>>,
    ) -> AffineExpr<BufferExprTerm> {
        todo!()
    }

    fn bind<'i>(
        &self,
        args: &[&'i dyn View<Tgt = Self::Tgt>],
        env: &mut HashMap<Param<Self::Tgt>, &'i dyn View<Tgt = Self::Tgt>>,
    ) {
        self.inner.bind(args, env)
    }
}
