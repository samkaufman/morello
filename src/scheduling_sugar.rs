use crate::common::DimSize;
use crate::imp::kernels::KernelType;
use crate::imp::{Impl, ImplNode};
use crate::layout::Layout;
use crate::scheduling::Action;
use crate::spec::Spec;
use crate::target::Target;
use std::iter;

/// A trait extending [ImplNode]s and [Spec]s with methods for more conveniently applying [Action]s.
///
/// These methods are intended to be used for manual scheduling by developers. Other clients should
/// probably apply [Action]s directly.
pub trait SchedulingSugar<Tgt: Target> {
    fn tile_out(&self, output_shape: &[DimSize], parallel: bool) -> ImplNode<Tgt, ()>;
    fn split(&self, k: DimSize) -> ImplNode<Tgt, ()>;
    fn move_param(
        &self,
        source_idx: u8,
        destination_level: Tgt::Level,
        destination_layout: Layout,
        destination_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt, ()>;
    fn to_accum(&self) -> ImplNode<Tgt, ()>;
    fn peel(
        &self,
        layout: Layout,
        level: Tgt::Level,
        vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt, ()>;
    fn spatial_split(&self) -> ImplNode<Tgt, ()>;
    fn place(&self, kernel_type: KernelType) -> ImplNode<Tgt, ()>;
}

pub trait Subschedule<Tgt: Target> {
    /// Apply a function to schedule a nested sub-Spec.
    fn subschedule(
        &self,
        path: &[usize],
        f: &impl Fn(&Spec<Tgt>) -> ImplNode<Tgt, ()>,
    ) -> ImplNode<Tgt, ()>;
}

impl<Tgt: Target> SchedulingSugar<Tgt> for Spec<Tgt> {
    fn tile_out(&self, output_shape: &[DimSize], parallel: bool) -> ImplNode<Tgt, ()> {
        Action::TileOut {
            output_shape: output_shape.into(),
            parallel,
        }
        .apply(self)
        .unwrap()
    }

    fn split(&self, k: DimSize) -> ImplNode<Tgt, ()> {
        Action::Split { k }.apply(self).unwrap()
    }

    fn move_param(
        &self,
        source_idx: u8,
        destination_level: Tgt::Level,
        destination_layout: Layout,
        destination_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt, ()> {
        Action::Move {
            source_idx,
            destination_level,
            destination_layout,
            destination_vector_size,
        }
        .apply(self)
        .unwrap()
    }

    fn to_accum(&self) -> ImplNode<Tgt, ()> {
        Action::ToAccum.apply(self).unwrap()
    }

    fn peel(
        &self,
        layout: Layout,
        level: Tgt::Level,
        vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt, ()> {
        Action::Peel {
            layout,
            level,
            vector_size,
        }
        .apply(self)
        .unwrap()
    }

    fn spatial_split(&self) -> ImplNode<Tgt, ()> {
        Action::SpatialSplit.apply(self).unwrap()
    }

    fn place(&self, kernel_type: KernelType) -> ImplNode<Tgt, ()> {
        Action::Place(kernel_type).apply(self).unwrap()
    }
}

impl<Tgt: Target> SchedulingSugar<Tgt> for ImplNode<Tgt, ()> {
    fn tile_out(&self, output_shape: &[DimSize], parallel: bool) -> ImplNode<Tgt, ()> {
        apply_to_leaf_spec(self, |spec| spec.tile_out(output_shape, parallel))
    }

    fn split(&self, k: DimSize) -> ImplNode<Tgt, ()> {
        apply_to_leaf_spec(self, |spec| spec.split(k))
    }

    fn move_param(
        &self,
        source_idx: u8,
        destination_level: Tgt::Level,
        destination_layout: Layout,
        destination_vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt, ()> {
        apply_to_leaf_spec(self, |spec| {
            spec.move_param(
                source_idx,
                destination_level,
                destination_layout,
                destination_vector_size,
            )
        })
    }

    fn to_accum(&self) -> ImplNode<Tgt, ()> {
        apply_to_leaf_spec(self, |spec| spec.to_accum())
    }

    fn peel(
        &self,
        layout: Layout,
        level: Tgt::Level,
        vector_size: Option<DimSize>,
    ) -> ImplNode<Tgt, ()> {
        apply_to_leaf_spec(self, |spec| spec.peel(layout, level, vector_size))
    }

    fn spatial_split(&self) -> ImplNode<Tgt, ()> {
        apply_to_leaf_spec(self, |spec| spec.spatial_split())
    }

    fn place(&self, kernel_type: KernelType) -> ImplNode<Tgt, ()> {
        apply_to_leaf_spec(self, |spec| spec.place(kernel_type))
    }
}

impl<Tgt: Target> Subschedule<Tgt> for ImplNode<Tgt, ()> {
    fn subschedule(
        &self,
        path: &[usize],
        f: &impl Fn(&Spec<Tgt>) -> ImplNode<Tgt, ()>,
    ) -> ImplNode<Tgt, ()> {
        let children = self.children();
        if children.is_empty() {
            match self {
                ImplNode::SpecApp(spec_app) => f(&spec_app.0),
                _ => panic!("subschedule path chose non-Spec leaf: {:?}", self),
            }
        } else if children.len() == 1 {
            self.replace_children(iter::once(children[0].subschedule(path, f)))
        } else {
            self.replace_children(children.iter().enumerate().map(|(i, child)| {
                if i == path[0] {
                    child.subschedule(&path[1..], f)
                } else {
                    child.clone()
                }
            }))
        }
    }
}

fn apply_to_leaf_spec<Tgt, F>(node: &ImplNode<Tgt, ()>, f: F) -> ImplNode<Tgt, ()>
where
    Tgt: Target,
    F: FnOnce(&Spec<Tgt>) -> ImplNode<Tgt, ()>,
{
    match node {
        ImplNode::SpecApp(app) => f(&app.0),
        _ => match &node.children() {
            [] => panic!("Not a Spec application and no children."),
            [child] => node.replace_children(iter::once(apply_to_leaf_spec(child, f))),
            _ => panic!("Ambiguous choice of child. Use `subschedule`."),
        },
    }
}
