use crate::{
    common::{DimSize, Shape},
    expr::{AffineForm, NonAffine, NonAffineExpr, Substitute, Term},
    layout::{BufferVar, Layout, LayoutError},
    opaque_symbol::OpaqueSymbol,
    target::Target,
    tensorspec::TensorSpec,
};
use smallvec::smallvec;
use std::{
    borrow::Borrow,
    fmt::{Debug, Display, Formatter},
    rc::Rc,
};

pub trait View: Clone {
    type Tgt: Target;

    /// Return an identifier which can be used to test reference equality.
    fn identifier(&self) -> OpaqueSymbol;

    fn backing_tensor(&self) -> Option<&Tensor<Self::Tgt>>;

    fn spec(&self) -> &TensorSpec<Self::Tgt>;

    fn shape(&self) -> &[DimSize] {
        self.spec().shape()
    }

    fn make_buffer_indexing_expr(&self) -> NonAffineExpr<BufferVar> {
        let backing_tensor = self.backing_tensor().unwrap();
        let backing_layout = backing_tensor.spec().layout();
        self.make_buffer_indexing_expr_with_layout(backing_layout)
    }

    // TODO: Rename
    fn make_buffer_indexing_expr_with_layout(&self, layout: &Layout) -> NonAffineExpr<BufferVar>;

    /// Replace any nested [Param]s with the corresponding [View]s from the given
    /// function.
    #[must_use]
    fn bind(self, get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Self::Tgt>>)
        -> ViewE<Self::Tgt>;

    fn to_param(&self) -> Option<&Param<Self::Tgt>> {
        None
    }

    // TODO: Drop this. Temporary hack for pprint'ing BoundaryTiles, which don't have syntax.
    fn is_boundary_tile(&self) -> bool {
        false
    }

    /// Visit parameters from this View using the provided visitor function.
    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Self::Tgt>);
}

// TODO: Replace with an impl on all smart pointers? Maybe a macro?
impl<V> View for &V
where
    V: View,
{
    type Tgt = V::Tgt;

    fn identifier(&self) -> OpaqueSymbol {
        (**self).identifier()
    }

    fn backing_tensor(&self) -> Option<&Tensor<Self::Tgt>> {
        (**self).backing_tensor()
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        (**self).spec()
    }

    fn make_buffer_indexing_expr_with_layout(&self, layout: &Layout) -> NonAffineExpr<BufferVar> {
        (**self).make_buffer_indexing_expr_with_layout(layout)
    }

    fn bind(
        self,
        get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Self::Tgt>>,
    ) -> ViewE<Self::Tgt> {
        (*self).clone().bind(get_argument)
    }

    fn to_param(&self) -> Option<&Param<Self::Tgt>> {
        (**self).to_param()
    }

    fn shape(&self) -> &[DimSize] {
        (**self).shape()
    }

    fn make_buffer_indexing_expr(&self) -> NonAffineExpr<BufferVar> {
        (**self).make_buffer_indexing_expr()
    }

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Self::Tgt>),
    {
        (**self).visit_params(visitor)
    }
}

impl<V> View for Box<V>
where
    V: View,
{
    type Tgt = V::Tgt;

    fn identifier(&self) -> OpaqueSymbol {
        (**self).identifier()
    }

    fn backing_tensor(&self) -> Option<&Tensor<Self::Tgt>> {
        (**self).backing_tensor()
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        (**self).spec()
    }

    fn make_buffer_indexing_expr_with_layout(&self, layout: &Layout) -> NonAffineExpr<BufferVar> {
        (**self).make_buffer_indexing_expr_with_layout(layout)
    }

    fn bind(
        self,
        get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Self::Tgt>>,
    ) -> ViewE<Self::Tgt> {
        (*self).clone().bind(get_argument)
    }

    fn to_param(&self) -> Option<&Param<Self::Tgt>> {
        (**self).to_param()
    }

    fn is_boundary_tile(&self) -> bool {
        (**self).is_boundary_tile()
    }

    fn shape(&self) -> &[DimSize] {
        (**self).shape()
    }

    fn make_buffer_indexing_expr(&self) -> NonAffineExpr<BufferVar> {
        (**self).make_buffer_indexing_expr()
    }

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Self::Tgt>),
    {
        (**self).visit_params(visitor)
    }
}

impl<V> View for Rc<V>
where
    V: View,
{
    type Tgt = V::Tgt;

    fn identifier(&self) -> OpaqueSymbol {
        (**self).identifier()
    }

    fn backing_tensor(&self) -> Option<&Tensor<Self::Tgt>> {
        (**self).backing_tensor()
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        (**self).spec()
    }

    fn make_buffer_indexing_expr_with_layout(&self, layout: &Layout) -> NonAffineExpr<BufferVar> {
        (**self).make_buffer_indexing_expr_with_layout(layout)
    }

    fn bind(
        self,
        get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Self::Tgt>>,
    ) -> ViewE<Self::Tgt> {
        (*self).clone().bind(get_argument)
    }

    fn to_param(&self) -> Option<&Param<Self::Tgt>> {
        (**self).to_param()
    }

    fn is_boundary_tile(&self) -> bool {
        (**self).is_boundary_tile()
    }

    fn shape(&self) -> &[DimSize] {
        (**self).shape()
    }

    fn make_buffer_indexing_expr(&self) -> NonAffineExpr<BufferVar> {
        (**self).make_buffer_indexing_expr()
    }

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Self::Tgt>),
    {
        (**self).visit_params(visitor)
    }
}

pub trait ViewExt: View {
    fn squeeze_dims<I: IntoIterator<Item = u8>>(self, dims: I) -> SqueezeDimsView<Self>
    where
        Self: Sized,
    {
        let dims_vec = dims.into_iter().collect::<Vec<_>>();
        let spec = self.spec().squeeze_dims(&dims_vec);
        SqueezeDimsView {
            inner: self,
            dims: dims_vec,
            spec,
            unique_id: OpaqueSymbol::new(),
        }
    }

    fn one_prefix(self) -> OnePrefixView<Self>
    where
        Self: Sized,
    {
        let spec = self.spec().one_prefix();
        OnePrefixView {
            inner: self,
            spec,
            unique_id: OpaqueSymbol::new(),
        }
    }

    /// Yields a view of the matrix with its two logical dimensions swapped.
    ///
    /// The underlying data is not modified. Instead, both the dimension sizes and the
    /// logical dimensions of the layout are swapped to be consistent with the new
    /// dimension ordering.
    fn transpose(self) -> TransposeView<Self>
    where
        Self: Sized,
    {
        let [h, w] = self.shape() else {
            panic!("Cannot transpose a tensor with shape {:?}", self.shape());
        };
        let shape = smallvec![*w, *h];

        let transposed_layout = self.spec().layout().swap_dims((0, 1));
        let spec = TensorSpec::new_canon(
            shape,
            self.spec().dtype(),
            self.spec().level(),
            transposed_layout,
            self.spec().vector_size(),
        );
        TransposeView {
            inner: self,
            spec,
            unique_id: OpaqueSymbol::new(),
        }
    }
}

impl<V: View> ViewExt for V {}

// TODO: Rename. `ViewE` is a confusing name.
#[derive(Debug, Clone)]
pub enum ViewE<Tgt: Target> {
    Tensor(Tensor<Tgt>),
    Param(Param<Tgt>),
    CacheView(CacheView<Box<ViewE<Tgt>>>),
    Tile(Tile<Box<ViewE<Tgt>>>),
    BoundaryTile(BoundaryTile<Box<ViewE<Tgt>>>),
    SqueezeDimsView(SqueezeDimsView<Box<ViewE<Tgt>>>),
    OnePrefixView(OnePrefixView<Box<ViewE<Tgt>>>),
    TransposeView(TransposeView<Box<ViewE<Tgt>>>),
}

impl<Tgt: Target> View for ViewE<Tgt> {
    type Tgt = Tgt;

    fn identifier(&self) -> OpaqueSymbol {
        match self {
            ViewE::Tensor(tensor) => tensor.identifier(),
            ViewE::Param(param) => param.identifier(),
            ViewE::CacheView(cache_view) => cache_view.identifier(),
            ViewE::Tile(tile) => tile.identifier(),
            ViewE::BoundaryTile(boundary_tile) => boundary_tile.identifier(),
            ViewE::SqueezeDimsView(squeeze_dims_view) => squeeze_dims_view.identifier(),
            ViewE::OnePrefixView(one_prefix_view) => one_prefix_view.identifier(),
            ViewE::TransposeView(transpose_view) => transpose_view.identifier(),
        }
    }

    fn backing_tensor(&self) -> Option<&Tensor<Self::Tgt>> {
        match self {
            ViewE::Tensor(tensor) => tensor.backing_tensor(),
            ViewE::Param(param) => param.backing_tensor(),
            ViewE::CacheView(cache_view) => cache_view.backing_tensor(),
            ViewE::Tile(tile) => tile.backing_tensor(),
            ViewE::BoundaryTile(boundary_tile) => boundary_tile.backing_tensor(),
            ViewE::SqueezeDimsView(squeeze_dims_view) => squeeze_dims_view.backing_tensor(),
            ViewE::OnePrefixView(one_prefix_view) => one_prefix_view.backing_tensor(),
            ViewE::TransposeView(transpose_view) => transpose_view.backing_tensor(),
        }
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        match self {
            ViewE::Tensor(tensor) => tensor.spec(),
            ViewE::Param(param) => param.spec(),
            ViewE::CacheView(cache_view) => cache_view.spec(),
            ViewE::Tile(tile) => tile.spec(),
            ViewE::BoundaryTile(boundary_tile) => boundary_tile.spec(),
            ViewE::SqueezeDimsView(squeeze_dims_view) => squeeze_dims_view.spec(),
            ViewE::OnePrefixView(one_prefix_view) => one_prefix_view.spec(),
            ViewE::TransposeView(transpose_view) => transpose_view.spec(),
        }
    }

    fn make_buffer_indexing_expr_with_layout(&self, layout: &Layout) -> NonAffineExpr<BufferVar> {
        match self {
            ViewE::Tensor(tensor) => tensor.make_buffer_indexing_expr_with_layout(layout),
            ViewE::Param(param) => param.make_buffer_indexing_expr_with_layout(layout),
            ViewE::CacheView(cache_view) => {
                cache_view.make_buffer_indexing_expr_with_layout(layout)
            }
            ViewE::Tile(tile) => tile.make_buffer_indexing_expr_with_layout(layout),
            ViewE::BoundaryTile(boundary_tile) => {
                boundary_tile.make_buffer_indexing_expr_with_layout(layout)
            }
            ViewE::SqueezeDimsView(squeeze_dims_view) => {
                squeeze_dims_view.make_buffer_indexing_expr_with_layout(layout)
            }
            ViewE::OnePrefixView(one_prefix_view) => {
                one_prefix_view.make_buffer_indexing_expr_with_layout(layout)
            }
            ViewE::TransposeView(transpose_view) => {
                transpose_view.make_buffer_indexing_expr_with_layout(layout)
            }
        }
    }

    fn bind(
        self,
        get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Self::Tgt>>,
    ) -> ViewE<Self::Tgt> {
        match self {
            ViewE::Tensor(tensor) => tensor.bind(get_argument),
            ViewE::Param(param) => param.bind(get_argument),
            ViewE::CacheView(cache_view) => cache_view.bind(get_argument),
            ViewE::Tile(tile) => tile.bind(get_argument),
            ViewE::BoundaryTile(boundary_tile) => boundary_tile.bind(get_argument),
            ViewE::SqueezeDimsView(squeeze_dims_view) => squeeze_dims_view.bind(get_argument),
            ViewE::OnePrefixView(one_prefix_view) => one_prefix_view.bind(get_argument),
            ViewE::TransposeView(transpose_view) => transpose_view.bind(get_argument),
        }
    }

    fn shape(&self) -> &[DimSize] {
        match self {
            ViewE::Tensor(tensor) => tensor.shape(),
            ViewE::Param(param) => param.shape(),
            ViewE::CacheView(cache_view) => cache_view.shape(),
            ViewE::Tile(tile) => tile.shape(),
            ViewE::BoundaryTile(boundary_tile) => boundary_tile.shape(),
            ViewE::SqueezeDimsView(squeeze_dims_view) => squeeze_dims_view.shape(),
            ViewE::OnePrefixView(one_prefix_view) => one_prefix_view.shape(),
            ViewE::TransposeView(transpose_view) => transpose_view.shape(),
        }
    }

    fn make_buffer_indexing_expr(&self) -> NonAffineExpr<BufferVar> {
        match self {
            ViewE::Tensor(tensor) => tensor.make_buffer_indexing_expr(),
            ViewE::Param(param) => param.make_buffer_indexing_expr(),
            ViewE::CacheView(cache_view) => cache_view.make_buffer_indexing_expr(),
            ViewE::Tile(tile) => tile.make_buffer_indexing_expr(),
            ViewE::BoundaryTile(boundary_tile) => boundary_tile.make_buffer_indexing_expr(),
            ViewE::SqueezeDimsView(squeeze_dims_view) => {
                squeeze_dims_view.make_buffer_indexing_expr()
            }
            ViewE::OnePrefixView(one_prefix_view) => one_prefix_view.make_buffer_indexing_expr(),
            ViewE::TransposeView(transpose_view) => transpose_view.make_buffer_indexing_expr(),
        }
    }

    fn to_param(&self) -> Option<&Param<Self::Tgt>> {
        match self {
            ViewE::Tensor(tensor) => tensor.to_param(),
            ViewE::Param(param) => param.to_param(),
            ViewE::CacheView(cache_view) => cache_view.to_param(),
            ViewE::Tile(tile) => tile.to_param(),
            ViewE::BoundaryTile(boundary_tile) => boundary_tile.to_param(),
            ViewE::SqueezeDimsView(squeeze_dims_view) => squeeze_dims_view.to_param(),
            ViewE::OnePrefixView(one_prefix_view) => one_prefix_view.to_param(),
            ViewE::TransposeView(transpose_view) => transpose_view.to_param(),
        }
    }

    fn is_boundary_tile(&self) -> bool {
        matches!(self, ViewE::BoundaryTile(_))
    }

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Self::Tgt>),
    {
        match self {
            ViewE::Tensor(tensor) => tensor.visit_params(visitor),
            ViewE::Param(param) => param.visit_params(visitor),
            ViewE::CacheView(cache_view) => cache_view.visit_params(visitor),
            ViewE::Tile(tile) => tile.visit_params(visitor),
            ViewE::BoundaryTile(boundary_tile) => boundary_tile.visit_params(visitor),
            ViewE::SqueezeDimsView(squeeze_dims_view) => squeeze_dims_view.visit_params(visitor),
            ViewE::OnePrefixView(one_prefix_view) => one_prefix_view.visit_params(visitor),
            ViewE::TransposeView(transpose_view) => transpose_view.visit_params(visitor),
        }
    }
}

impl<Tgt: Target> From<Tensor<Tgt>> for ViewE<Tgt> {
    fn from(tensor: Tensor<Tgt>) -> Self {
        ViewE::Tensor(tensor)
    }
}

impl<Tgt: Target> From<Param<Tgt>> for ViewE<Tgt> {
    fn from(param: Param<Tgt>) -> Self {
        ViewE::Param(param)
    }
}

impl<V> From<CacheView<V>> for ViewE<V::Tgt>
where
    V: View + Into<ViewE<V::Tgt>>,
{
    fn from(cache_view: CacheView<V>) -> Self {
        ViewE::CacheView(CacheView {
            source: Box::new(cache_view.source.into()),
            spec: cache_view.spec,
            unique_id: cache_view.unique_id,
        })
    }
}

impl<V> From<Tile<V>> for ViewE<V::Tgt>
where
    V: View + Into<ViewE<V::Tgt>>,
{
    fn from(tile: Tile<V>) -> Self {
        ViewE::Tile(Tile {
            shape: tile.shape,
            step_sizes: tile.step_sizes,
            view: Box::new(tile.view.into()),
            expr_term_id: tile.expr_term_id,
            spec: tile.spec,
            unique_id: tile.unique_id,
        })
    }
}

impl<V> From<SqueezeDimsView<V>> for ViewE<V::Tgt>
where
    V: View + Into<ViewE<V::Tgt>>,
{
    fn from(view: SqueezeDimsView<V>) -> Self {
        ViewE::SqueezeDimsView(SqueezeDimsView {
            inner: Box::new(view.inner.into()),
            dims: view.dims,
            spec: view.spec,
            unique_id: view.unique_id,
        })
    }
}

impl<V> From<OnePrefixView<V>> for ViewE<V::Tgt>
where
    V: View + Into<ViewE<V::Tgt>>,
{
    fn from(view: OnePrefixView<V>) -> Self {
        ViewE::OnePrefixView(OnePrefixView {
            inner: Box::new(view.inner.into()),
            spec: view.spec,
            unique_id: view.unique_id,
        })
    }
}

impl<V> From<TransposeView<V>> for ViewE<V::Tgt>
where
    V: View + Into<ViewE<V::Tgt>>,
{
    fn from(view: TransposeView<V>) -> Self {
        ViewE::TransposeView(TransposeView {
            inner: Box::new(view.inner.into()),
            spec: view.spec,
            unique_id: view.unique_id,
        })
    }
}

impl<V> From<BoundaryTile<V>> for ViewE<V::Tgt>
where
    V: View + Into<ViewE<V::Tgt>>,
{
    fn from(boundary_tile: BoundaryTile<V>) -> Self {
        ViewE::BoundaryTile(BoundaryTile {
            shape: boundary_tile.shape,
            offsets: boundary_tile.offsets,
            view: Box::new(boundary_tile.view.into()),
            expr_term_id: boundary_tile.expr_term_id,
            spec: boundary_tile.spec,
            unique_id: boundary_tile.unique_id,
        })
    }
}

#[derive(thiserror::Error, Debug)]
pub enum TileError {
    #[error("Layout does not apply to tile size")]
    LayoutIncompatible(#[from] LayoutError),
}

#[derive(Debug, Clone)]
pub enum ParamOr<V, P> {
    Param(P),
    Other(V),
}

impl<V, P> View for ParamOr<V, P>
where
    V: View,
    P: Borrow<Param<V::Tgt>> + Clone + Debug, // TODO: Remove Debug bound one removed from View
{
    type Tgt = V::Tgt;

    fn identifier(&self) -> OpaqueSymbol {
        match self {
            ParamOr::Param(p) => p.borrow().identifier(),
            ParamOr::Other(v) => v.identifier(),
        }
    }

    fn backing_tensor(&self) -> Option<&Tensor<Self::Tgt>> {
        match self {
            ParamOr::Param(p) => p.borrow().backing_tensor(),
            ParamOr::Other(v) => v.backing_tensor(),
        }
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        match self {
            ParamOr::Param(p) => p.borrow().spec(),
            ParamOr::Other(v) => v.spec(),
        }
    }

    fn make_buffer_indexing_expr_with_layout(&self, layout: &Layout) -> NonAffineExpr<BufferVar> {
        match self {
            ParamOr::Param(p) => p.borrow().make_buffer_indexing_expr_with_layout(layout),
            ParamOr::Other(v) => v.make_buffer_indexing_expr_with_layout(layout),
        }
    }

    fn bind(
        self,
        get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Self::Tgt>>,
    ) -> ViewE<Self::Tgt> {
        match self {
            ParamOr::Param(p) => p.borrow().clone().bind(get_argument),
            ParamOr::Other(v) => v.bind(get_argument),
        }
    }

    fn to_param(&self) -> Option<&Param<Self::Tgt>> {
        match self {
            ParamOr::Param(p) => Some(p.borrow()),
            ParamOr::Other(v) => v.to_param(),
        }
    }

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Self::Tgt>),
    {
        match self {
            ParamOr::Param(p) => p.borrow().visit_params(visitor),
            ParamOr::Other(v) => v.visit_params(visitor),
        }
    }
}

/// A reference to an Impl node parameter.
///
/// Remember that some scheduling actions result in multi-node Impls, such as for movement which
/// may produce a Alloc binding and a nested Block. In this case, parameters---and, therefore,
/// the referents of a Param---differ between the Alloc and Block.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Param<Tgt: Target>(pub u8, pub TensorSpec<Tgt>, pub(crate) OpaqueSymbol);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tensor<Tgt: Target>(pub TensorSpec<Tgt>, OpaqueSymbol);

#[derive(Debug, Clone)]
pub struct CacheView<V: View> {
    pub source: V,
    spec: TensorSpec<V::Tgt>,
    unique_id: OpaqueSymbol,
}

impl<V: View> CacheView<V> {
    pub fn new(source: V, spec: TensorSpec<V::Tgt>) -> Self {
        CacheView {
            source,
            spec,
            unique_id: OpaqueSymbol::new(),
        }
    }
}

/// A tile with a fixed shape, resulting from applying a [Tiling] to a [View].
#[derive(Debug, Clone)]
pub struct Tile<V: View> {
    shape: Shape,
    step_sizes: Shape,
    pub view: V,
    expr_term_id: OpaqueSymbol,
    spec: TensorSpec<V::Tgt>,
    unique_id: OpaqueSymbol,
}

/// A boundary tile that represents a fixed region of a tensor without loop iteration.
/// Unlike regular tiles, boundary tiles generate buffer indexing expressions that
/// use constant offsets instead of loop iterator variables.
#[derive(Debug, Clone)]
pub struct BoundaryTile<V: View> {
    shape: Shape,
    offsets: Vec<u32>,
    pub view: V,
    expr_term_id: OpaqueSymbol,
    spec: TensorSpec<V::Tgt>,
    unique_id: OpaqueSymbol,
}

#[derive(Debug, Clone)]
pub struct SqueezeDimsView<V: View> {
    pub inner: V,
    pub dims: Vec<u8>,
    spec: TensorSpec<V::Tgt>,
    unique_id: OpaqueSymbol,
}

#[derive(Debug, Clone)]
pub struct OnePrefixView<V: View> {
    pub inner: V,
    spec: TensorSpec<V::Tgt>,
    unique_id: OpaqueSymbol,
}

#[derive(Debug, Clone)]
pub struct TransposeView<V: View> {
    pub inner: V,
    spec: TensorSpec<V::Tgt>,
    unique_id: OpaqueSymbol,
}

impl<Tgt: Target> Param<Tgt> {
    pub fn new(dim_idx: u8, spec: TensorSpec<Tgt>) -> Param<Tgt> {
        Param(dim_idx, spec, OpaqueSymbol::new())
    }
}

impl<Tgt: Target> View for Param<Tgt> {
    type Tgt = Tgt;

    fn identifier(&self) -> OpaqueSymbol {
        self.2
    }

    fn backing_tensor(&self) -> Option<&Tensor<Tgt>> {
        None
    }

    fn spec(&self) -> &TensorSpec<Tgt> {
        &self.1
    }

    fn make_buffer_indexing_expr_with_layout(&self, _layout: &Layout) -> NonAffineExpr<BufferVar> {
        todo!()
    }

    fn bind(
        self,
        get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Self::Tgt>>,
    ) -> ViewE<Self::Tgt> {
        get_argument(self.0).unwrap_or_else(|| self.into())
    }

    fn to_param(&self) -> Option<&Param<Tgt>> {
        Some(self)
    }

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Tgt>),
    {
        visitor(self.0, &self.1);
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

    fn identifier(&self) -> OpaqueSymbol {
        self.1
    }

    fn backing_tensor(&self) -> Option<&Tensor<Self::Tgt>> {
        Some(self)
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.0
    }

    fn make_buffer_indexing_expr_with_layout(&self, layout: &Layout) -> NonAffineExpr<BufferVar> {
        layout.buffer_indexing_expr(self.1, self.shape())
    }

    fn bind(
        self,
        _get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Self::Tgt>>,
    ) -> ViewE<Self::Tgt> {
        self.into()
    }

    fn visit_params<F>(&self, _visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Self::Tgt>),
    {
        // Tensors have no parameters
    }
}

impl<V: View> View for CacheView<V> {
    type Tgt = V::Tgt;

    fn identifier(&self) -> OpaqueSymbol {
        self.unique_id
    }

    fn backing_tensor(&self) -> Option<&Tensor<Self::Tgt>> {
        self.source.backing_tensor()
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.spec
    }

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Self::Tgt>),
    {
        self.source.visit_params(visitor);
    }

    fn make_buffer_indexing_expr_with_layout(&self, layout: &Layout) -> NonAffineExpr<BufferVar> {
        self.source.make_buffer_indexing_expr_with_layout(layout)
    }

    fn bind(
        self,
        get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Self::Tgt>>,
    ) -> ViewE<Self::Tgt> {
        ViewE::from(CacheView {
            source: self.source.bind(get_argument),
            spec: self.spec.clone(),
            unique_id: self.unique_id,
        })
    }
}

impl<V: View> Tile<V> {
    pub fn new(shape: Shape, step_sizes: Shape, view: V) -> Result<Self, TileError> {
        let expr_term_id = OpaqueSymbol::new();
        let unique_id = OpaqueSymbol::new();
        let mut spec = view.spec().clone();
        spec.shrink(&shape)?;
        Ok(Self {
            shape,
            step_sizes,
            view,
            expr_term_id,
            spec,
            unique_id,
        })
    }

    pub fn shape(&self) -> &[DimSize] {
        &self.shape
    }

    pub fn step_sizes(&self) -> &[DimSize] {
        &self.step_sizes
    }

    /// Yields [`BufferVar::TileIdx`] terms for all tiling dimensions, including size=1.
    pub fn tile_dim_terms(&self) -> impl Iterator<Item = BufferVar> + '_ {
        (0..self.shape.len())
            .map(|dim| BufferVar::TileIdx(dim.try_into().unwrap(), self.expr_term_id))
    }

    pub fn steps_dim(&self, dim: u8) -> u32 {
        let origin_size = self.view.shape()[usize::from(dim)];
        divrem::DivCeil::div_ceil(origin_size.get(), self.step_sizes[usize::from(dim)].get())
    }

    pub fn full_steps_dim(&self, dim: u8) -> u32 {
        let origin_size = self.view.shape()[usize::from(dim)];
        origin_size.get() / self.step_sizes[usize::from(dim)].get()
    }

    /// Replace points in the given indexing expression with tile coordinate-adjusted points.
    pub fn compose_buffer_indexing_expr(
        &self,
        inner_expr: NonAffineExpr<BufferVar>,
    ) -> NonAffineExpr<BufferVar> {
        if self
            .shape()
            .iter()
            .zip(&self.step_sizes)
            .any(|(a, b)| *a != *b)
        {
            todo!(
                "Implement support for sliding tilings. (Shape was {:?} and step sizes were {:?}.)",
                self.shape(),
                self.step_sizes
            );
        }
        inner_expr.map_vars(&mut |term_var| match term_var {
            BufferVar::Pt(dim, _) => {
                let e = self.expr_term_id;
                let size_in_dim = self.shape()[usize::from(dim)];
                let mut terms = vec![Term(1, NonAffine::Leaf(BufferVar::Pt(dim, e)))]; // pt_{dim}
                if size_in_dim != self.view.shape()[usize::from(dim)] {
                    // pt_{dim} + tile_idx_{dim} * size_in_dim
                    terms.push(Term(
                        size_in_dim.get().try_into().unwrap(),
                        NonAffine::Leaf(BufferVar::TileIdx(dim, e)),
                    ));
                }
                AffineForm(terms, 0)
            }
            BufferVar::TileIdx(_, _) => NonAffine::Leaf(term_var).into(),
        })
    }

    pub(crate) fn boxed_viewe(self) -> Tile<Box<ViewE<V::Tgt>>>
    where
        V: Into<ViewE<V::Tgt>>,
    {
        Tile {
            view: Box::new(self.view.into()),
            shape: self.shape,
            step_sizes: self.step_sizes,
            expr_term_id: self.expr_term_id,
            spec: self.spec,
            unique_id: self.unique_id,
        }
    }
}

impl<T: View> View for Tile<T> {
    type Tgt = T::Tgt;

    fn identifier(&self) -> OpaqueSymbol {
        self.unique_id
    }

    fn backing_tensor(&self) -> Option<&Tensor<Self::Tgt>> {
        self.view.backing_tensor()
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.spec
    }

    fn make_buffer_indexing_expr_with_layout(&self, layout: &Layout) -> NonAffineExpr<BufferVar> {
        self.compose_buffer_indexing_expr(self.view.make_buffer_indexing_expr_with_layout(layout))
    }

    fn bind(
        self,
        get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Self::Tgt>>,
    ) -> ViewE<Self::Tgt> {
        ViewE::from(Tile {
            view: self.view.bind(get_argument),
            unique_id: self.unique_id,
            shape: self.shape,
            step_sizes: self.step_sizes,
            expr_term_id: self.expr_term_id,
            spec: self.spec,
        })
    }

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Self::Tgt>),
    {
        self.view.visit_params(visitor);
    }
}

impl<T: View> View for SqueezeDimsView<T> {
    type Tgt = T::Tgt;

    fn identifier(&self) -> OpaqueSymbol {
        self.unique_id
    }

    fn backing_tensor(&self) -> Option<&Tensor<Self::Tgt>> {
        self.inner.backing_tensor()
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.spec
    }

    fn make_buffer_indexing_expr_with_layout(&self, _layout: &Layout) -> NonAffineExpr<BufferVar> {
        todo!()
    }

    fn bind(
        self,
        get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Self::Tgt>>,
    ) -> ViewE<Self::Tgt> {
        ViewE::from(SqueezeDimsView {
            inner: self.inner.bind(get_argument),
            dims: self.dims,
            spec: self.spec,
            unique_id: self.unique_id,
        })
    }

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Self::Tgt>),
    {
        self.inner.visit_params(visitor);
    }
}

impl<T: View> View for OnePrefixView<T> {
    type Tgt = T::Tgt;

    fn identifier(&self) -> OpaqueSymbol {
        self.unique_id
    }

    fn backing_tensor(&self) -> Option<&Tensor<Self::Tgt>> {
        self.inner.backing_tensor()
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.spec
    }

    fn make_buffer_indexing_expr_with_layout(&self, _layout: &Layout) -> NonAffineExpr<BufferVar> {
        todo!()
    }

    fn bind(
        self,
        get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Self::Tgt>>,
    ) -> ViewE<Self::Tgt> {
        ViewE::from(OnePrefixView {
            inner: self.inner.bind(get_argument),
            spec: self.spec,
            unique_id: self.unique_id,
        })
    }

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Self::Tgt>),
    {
        self.inner.visit_params(visitor);
    }
}

impl<T: View> View for TransposeView<T> {
    type Tgt = T::Tgt;

    fn identifier(&self) -> OpaqueSymbol {
        self.unique_id
    }

    fn backing_tensor(&self) -> Option<&Tensor<Self::Tgt>> {
        self.inner.backing_tensor()
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.spec
    }

    fn make_buffer_indexing_expr_with_layout(&self, _layout: &Layout) -> NonAffineExpr<BufferVar> {
        todo!()
    }

    fn bind(
        self,
        get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Self::Tgt>>,
    ) -> ViewE<Self::Tgt> {
        ViewE::from(TransposeView {
            inner: self.inner.bind(get_argument),
            spec: self.spec,
            unique_id: self.unique_id,
        })
    }

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Self::Tgt>),
    {
        self.inner.visit_params(visitor);
    }
}

impl<V: View> BoundaryTile<V> {
    pub fn new(shape: Shape, offsets: Vec<u32>, view: V) -> Result<Self, TileError> {
        let expr_term_id = OpaqueSymbol::new();
        let unique_id = OpaqueSymbol::new();
        let mut spec = view.spec().clone();
        spec.shrink(&shape)?;
        Ok(BoundaryTile {
            shape,
            offsets,
            view,
            expr_term_id,
            spec,
            unique_id,
        })
    }

    pub fn shape(&self) -> &[DimSize] {
        &self.shape
    }

    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Replace points in the given indexing expression with boundary-specific constant offsets.
    /// Unlike regular tiles, boundary tiles use constant offsets instead of tile coordinate variables.
    pub fn compose_buffer_indexing_expr(
        &self,
        inner_expr: NonAffineExpr<BufferVar>,
    ) -> NonAffineExpr<BufferVar> {
        inner_expr.map_vars(&mut |term_var| match term_var {
            BufferVar::Pt(dim, _) => {
                let offset_in_dim = self.offsets[usize::from(dim)];
                let e = self.expr_term_id;
                // pt_{dim} + offset_{dim}
                AffineForm(
                    vec![Term(1, NonAffine::Leaf(BufferVar::Pt(dim, e)))],
                    offset_in_dim.try_into().unwrap(),
                )
            }
            BufferVar::TileIdx(_, _) => NonAffine::Leaf(term_var).into(),
        })
    }
}

impl<T: View> View for BoundaryTile<T> {
    type Tgt = T::Tgt;

    fn identifier(&self) -> OpaqueSymbol {
        self.unique_id
    }

    fn backing_tensor(&self) -> Option<&Tensor<Self::Tgt>> {
        self.view.backing_tensor()
    }

    fn spec(&self) -> &TensorSpec<Self::Tgt> {
        &self.spec
    }

    fn make_buffer_indexing_expr_with_layout(&self, layout: &Layout) -> NonAffineExpr<BufferVar> {
        self.compose_buffer_indexing_expr(self.view.make_buffer_indexing_expr_with_layout(layout))
    }

    fn bind(
        self,
        get_argument: &mut dyn FnMut(u8) -> Option<ViewE<Self::Tgt>>,
    ) -> ViewE<Self::Tgt> {
        ViewE::from(BoundaryTile {
            view: self.view.bind(get_argument),
            unique_id: self.unique_id,
            shape: self.shape,
            offsets: self.offsets,
            expr_term_id: self.expr_term_id,
            spec: self.spec,
        })
    }

    fn is_boundary_tile(&self) -> bool {
        true
    }

    fn visit_params<F>(&self, visitor: &mut F)
    where
        F: FnMut(u8, &TensorSpec<Self::Tgt>),
    {
        self.view.visit_params(visitor);
    }
}
