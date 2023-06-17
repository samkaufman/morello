use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::iter;

use smallvec::SmallVec;

use crate::alignment::aligned_approx;
use crate::common::{DimSize, Shape};
use crate::layout::Layout;
use crate::memorylimits::MemVec;
use crate::reinterpret::squeeze_tile;
use crate::spec::{PrimitiveAux, PrimitiveBasics, PrimitiveSpecType, Spec};
use crate::target::{MemoryLevel, Target, X86MemoryLevel, X86Target};
use crate::tensorspec::{TensorSpec, TensorSpecAux};
use crate::tiling::{OperandTile, Tiling};

/// A single node in an Impl.
///
/// This does not retain its own Spec or references to its children. It contains the
/// minimal amount of information needed to distinguish a scheduling decision, which
/// makes it appropriate for storing in a database because, in a database, the Spec is
/// stored separately and the children can be looked up, given their Specs, which are
/// computable from this node and its Spec.
///
/// ImplNodes, given their outer Spec, can compute the Specs of their children.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum ImplNode<Tgt: Target> {
    Loop {
        subscripts: SmallVec<[u8; 2]>,
        tiles: Vec<OperandTile>,
        parallel: bool,
    },
    MoveLet {
        source_idx: u8,
        destination_level: Tgt::Level,
        destination_layout: Layout,
        destination_vector_shape: Option<Shape>,
        prefetch: bool,
    },
    AccumBlock,
    Pipeline {
        layout: Layout,
        level: Tgt::Level,
        vector_shape: Option<Shape>,
    },
    SpatialSplit,
    /* Mult Impls */
    Mult,
    BroadcastVecMult,
    /* Load & Store Impls */
    ValueAssign,
    VectorAssign,
    /* Zero Impls */
    MemsetZero,
    VectorZero,
}

/// A description of how much memory is allocated by an ImplNode.
///
/// Memory consumed during the entire lifetime of an Impl is separated
/// from the memory consumed during the lifetime of each child. This makes
/// it possible to encode memory consumption by a leaf node with no children.
pub struct MemoryAllocation {
    pub base: MemVec,
    pub during_children: Vec<MemVec>,
}

impl<Tgt: Target> ImplNode<Tgt> {
    pub fn child_count(&self, node_spec: &Spec<Tgt>) -> usize {
        match self {
            ImplNode::Loop { .. } => 1,
            ImplNode::AccumBlock => 2,
            ImplNode::SpatialSplit => 1,
            ImplNode::Mult
            | ImplNode::BroadcastVecMult
            | ImplNode::ValueAssign
            | ImplNode::VectorAssign
            | ImplNode::MemsetZero
            | ImplNode::VectorZero => 0,
            _ => {
                // This is slow. Ideally, all cases are implemented.
                self.child_specs(node_spec).len()
            }
        }
    }

    pub fn child_specs(&self, node_spec: &Spec<Tgt>) -> Vec<Spec<Tgt>> {
        match self {
            ImplNode::Loop { tiles, .. } => {
                let mut new_operands = node_spec.operands();
                for OperandTile {
                    tiling,
                    source_idx,
                    aligned,
                } in tiles
                {
                    let ref_op = &mut new_operands[usize::from(*source_idx)];
                    ref_op.shrink(tiling.shape(), *aligned);
                }
                let mut inner_spec = node_spec.clone();
                inner_spec.replace_io(&new_operands);
                vec![inner_spec]
            }
            ImplNode::Pipeline {
                layout,
                level,
                vector_shape,
            } => {
                let Spec::Compose { components, operand_auxes, serial_only } = node_spec else {
                    panic!();
                };
                debug_assert!(components.len() >= 2);

                // Figure out the output shape of the next-to-outermost component Spec.
                let next_to_outer_basics = &components[1];
                let intermediate_tensor = TensorSpec::<Tgt>::new_canon(
                    next_to_outer_basics
                        .operand_shapes()
                        .swap_remove(next_to_outer_basics.typ.output_idx()),
                    next_to_outer_basics.dtype,
                    layout.contiguous_full(),
                    true,
                    *level,
                    layout.clone(),
                    vector_shape.clone(),
                );

                // The head of a Compose is the final function evaluated. Build
                // a full Spec so that it can be further scheduled independently.
                let external_head_input_cnt = components[0].typ.input_count() - 1;
                let head_spec = {
                    // TODO: Simplfy this block.
                    let head_operand_auxes = iter::once(&intermediate_tensor.aux)
                        .chain(&operand_auxes[..external_head_input_cnt])
                        .chain(iter::once(&operand_auxes[node_spec.output_idx()]));
                    let head_basics = &components[0];
                    Spec::Primitive(
                        head_basics.clone(),
                        head_basics.aux_from_operand_auxes(head_operand_auxes),
                        *serial_only,
                    )
                };

                let next_to_head_input_auxes = &operand_auxes[external_head_input_cnt
                    ..external_head_input_cnt + next_to_outer_basics.typ.input_count()];
                let remainder: Spec<Tgt> = if components.len() == 2 {
                    Spec::Primitive(
                        next_to_outer_basics.clone(),
                        next_to_outer_basics.aux_from_operand_auxes(
                            next_to_head_input_auxes
                                .iter()
                                .chain(iter::once(&intermediate_tensor.aux)),
                        ),
                        *serial_only,
                    )
                } else {
                    let remainder_inputs = &node_spec.operands()
                        [external_head_input_cnt..next_to_head_input_auxes.len() - 1];
                    let remainder_operand_auxes = remainder_inputs
                        .iter()
                        .map(|t| t.aux.clone())
                        .chain(iter::once(intermediate_tensor.aux))
                        .collect::<Vec<_>>();
                    Spec::Compose {
                        components: components[1..].to_vec(),
                        operand_auxes: remainder_operand_auxes,
                        serial_only: *serial_only,
                    }
                };
                vec![remainder, head_spec]
            }
            ImplNode::SpatialSplit => {
                let Spec::Primitive(PrimitiveBasics { typ: PrimitiveSpecType::Conv { .. }, spec_shape: _, dtype }, _, serial_only) = node_spec else {
                    panic!();
                };
                let operands = node_spec.operands();

                // We're going to introduce a Loop which traverses all the
                // pixels over all spatial dimensions, vectorizing across
                // the batch, channels, and filters dimmensions. This means
                // the loop body/the sub-Spec will have its spatial dims.
                // dropped, even though they are non-degenerate.

                let dropped_dims: Vec<u8> =
                    (2..u8::try_from(operands[0].dim_sizes().len()).unwrap()).collect();
                let spatial_ones = Shape::from_elem(1, dropped_dims.len());

                let tiled_image_spec_shape = operands[0].dim_sizes()[..2]
                    .iter()
                    .chain(spatial_ones.iter())
                    .copied()
                    .collect::<Shape>();
                // TODO: The below `shrink` is messy because we, the caller, need to
                // compute alignment. Fix that.
                let mut tiled_image_spec = operands[0].clone();
                tiled_image_spec.shrink(
                    &tiled_image_spec_shape,
                    aligned_approx(
                        &Tiling::new_simple(tiled_image_spec_shape.clone()),
                        &tiled_image_spec_shape,
                        &operands[0],
                    ),
                );

                let tiled_filters_spec_shape = operands[1].dim_sizes()[..2]
                    .iter()
                    .chain(spatial_ones.iter())
                    .copied()
                    .collect::<Shape>();
                let mut tiled_filters_spec = operands[1].clone();
                tiled_filters_spec.shrink(
                    &tiled_filters_spec_shape,
                    aligned_approx(
                        &Tiling::new_simple(tiled_filters_spec_shape.clone()),
                        &tiled_filters_spec_shape,
                        &operands[1],
                    ),
                );

                let squeezed_image_spec = squeeze_tile(&tiled_image_spec, &dropped_dims);
                let squeezed_filters_spec = squeeze_tile(&tiled_filters_spec, &dropped_dims);
                let squeezed_output_spec = squeeze_tile(&operands[2], &dropped_dims);
                vec![Spec::Primitive(
                    PrimitiveBasics {
                        typ: PrimitiveSpecType::Matmul { accum: true },
                        spec_shape: operands[0].dim_sizes()[..2]
                            .iter()
                            .copied()
                            .chain(iter::once(operands[1].dim_sizes()[0]))
                            .collect(),
                        dtype: *dtype,
                    },
                    PrimitiveAux::Standard(vec![
                        TensorSpecAux {
                            contig: squeezed_image_spec.contiguous_abs(),
                            aligned: squeezed_image_spec.aligned(),
                            level: squeezed_image_spec.level(),
                            layout: squeezed_image_spec.layout(),
                            vector_shape: squeezed_image_spec.vector_shape().cloned(),
                        },
                        TensorSpecAux {
                            contig: squeezed_filters_spec.contiguous_abs(),
                            aligned: squeezed_filters_spec.aligned(),
                            level: squeezed_filters_spec.level(),
                            layout: squeezed_filters_spec.layout(),
                            vector_shape: squeezed_filters_spec.vector_shape().cloned(),
                        },
                        TensorSpecAux {
                            contig: squeezed_output_spec.contiguous_abs(),
                            aligned: squeezed_output_spec.aligned(),
                            level: squeezed_output_spec.level(),
                            layout: squeezed_output_spec.layout(),
                            vector_shape: squeezed_output_spec.vector_shape().cloned(),
                        },
                    ]),
                    *serial_only,
                )]
            }
            ImplNode::MoveLet {
                source_idx,
                destination_level,
                destination_layout,
                destination_vector_shape,
                prefetch,
            } => {
                if *prefetch {
                    unimplemented!()
                }

                let outer_operands = node_spec.operands();
                let operand: &TensorSpec<Tgt> = &outer_operands[usize::from(*source_idx)];
                let new_mat_spec = movelet_inner_tensorspec(
                    operand,
                    destination_level,
                    &destination_layout.canonicalize_for_shape(operand.dim_sizes()),
                    destination_vector_shape.as_ref().map(|v| v.as_slice()),
                );

                let make_logue = |typ, f: &dyn Fn(_, _, _) -> bool| {
                    if f(destination_level, *source_idx, node_spec) {
                        Some(Spec::Primitive(
                            PrimitiveBasics {
                                typ,
                                spec_shape: operand.dim_sizes().clone(),
                                dtype: operand.dtype(),
                            },
                            PrimitiveAux::Move {
                                outer_aux: operand.aux.clone(),
                                inner_level: new_mat_spec.level(),
                                inner_layout: new_mat_spec.layout(),
                                inner_vector_shape: new_mat_spec.vector_shape().cloned(),
                            },
                            node_spec.serial_only(),
                        ))
                    } else {
                        None
                    }
                };
                let prologue = make_logue(PrimitiveSpecType::Load, &movelet_gens_prologue);
                let epilogue = make_logue(PrimitiveSpecType::Store, &movelet_gens_epilogue);

                let mut result = vec![];
                let mut new_operands = outer_operands.clone();
                new_operands[usize::from(*source_idx)] = new_mat_spec;
                let mut new_inner_spec = node_spec.clone();
                new_inner_spec.replace_io(&new_operands);
                result.extend(prologue);
                result.push(new_inner_spec);
                result.extend(epilogue);

                result
            }
            ImplNode::AccumBlock => {
                let Spec::Primitive(PrimitiveBasics { typ, spec_shape: _, dtype: _ }, _, _) = node_spec else {
                    panic!();
                };
                let (PrimitiveSpecType::Matmul { accum } | PrimitiveSpecType::Conv { accum }) = typ else {
                    panic!();
                };
                if *accum {
                    panic!("Spec is already accumulating");
                }
                let TensorSpec {
                    dim_sizes: output_dim_sizes,
                    dtype: output_dtype,
                    aux: output_aux,
                } = node_spec.output();
                let mut result = vec![
                    Spec::Primitive(
                        PrimitiveBasics {
                            typ: PrimitiveSpecType::Zero,
                            spec_shape: output_dim_sizes,
                            dtype: output_dtype,
                        },
                        PrimitiveAux::Standard(vec![output_aux]),
                        node_spec.serial_only(),
                    ),
                    node_spec.clone_as_accum(),
                ];
                result.iter_mut().for_each(|s| s.canonicalize());
                result
            }
            ImplNode::Mult
            | ImplNode::BroadcastVecMult
            | ImplNode::ValueAssign
            | ImplNode::VectorAssign
            | ImplNode::MemsetZero
            | ImplNode::VectorZero => vec![],
        }
    }

    pub fn peak_memory_from_child_peaks(&self, spec: &Spec<Tgt>, child_peaks: &[MemVec]) -> MemVec {
        // Take the per-level peak from all children, adding in the additional
        // memory that `self` allocates per-child, and finally put `self`'s
        // base memory allocation on top.
        let own_allocation = self.memory_allocated(spec);
        debug_assert_eq!(child_peaks.len(), own_allocation.during_children.len());
        let mut peak = MemVec::zero::<Tgt>();
        for (child_peak, own_child_alloc) in child_peaks
            .iter()
            .zip(own_allocation.during_children.iter())
        {
            for i in 0..peak.len() {
                peak[i] = peak[i].max(own_allocation.base[i] + child_peak[i] + own_child_alloc[i]);
            }
        }
        peak
    }

    pub fn steps(&self, spec: &Spec<Tgt>) -> u32 {
        match &self {
            ImplNode::Loop { subscripts, .. } => {
                let mut val = 1;
                for &s in subscripts {
                    val *= self.steps_subscript(spec, s);
                }
                val
            }
            // Type specialization would avoid needing cases like this.
            _ => panic!("steps() called on non-loop node"),
        }
    }

    pub fn full_steps(&self, spec: &Spec<Tgt>) -> u32 {
        match &self {
            ImplNode::Loop { subscripts, .. } => {
                let mut val = 1;
                for &s in subscripts {
                    let all_steps = self.steps_subscript(spec, s);
                    if self.boundary_size(spec, s) != 0 {
                        val *= all_steps - 1;
                    } else {
                        val *= all_steps;
                    }
                }
                val
            }
            // Type specialization would avoid needing cases like this.
            _ => panic!("full_steps() called on non-loop node"),
        }
    }

    fn steps_subscript(&self, spec: &Spec<Tgt>, subscript: u8) -> u32 {
        match &self {
            ImplNode::Loop { .. } => {
                self.apply_to_subscript(spec, subscript, |t, dim, origin_size| {
                    t.tiling.steps_dim(dim, origin_size)
                })
            }
            // Type specialization would avoid needing cases like this.
            _ => panic!("steps_subscript() called on non-loop node"),
        }
    }

    fn boundary_size(&self, spec: &Spec<Tgt>, subscript: u8) -> u32 {
        match &self {
            ImplNode::Loop { .. } => {
                self.apply_to_subscript(spec, subscript, |t, dim, origin_size| {
                    t.tiling.boundary_size(dim, origin_size)
                })
            }
            // Type specialization would avoid needing cases like this.
            _ => panic!("boundary_size() called on non-loop node"),
        }
    }

    fn apply_to_subscript<F>(&self, spec: &Spec<Tgt>, subscript: u8, mut fn_: F) -> u32
    where
        F: FnMut(&OperandTile, u8, DimSize) -> u32,
    {
        match &self {
            ImplNode::Loop { tiles, .. } => {
                let operands = spec.operands();
                for tile in tiles {
                    let source_usize = usize::from(tile.source_idx);
                    let subs: &SmallVec<[u8; 4]> = &spec.operands_dim_subscripts()[source_usize];
                    for (dim, &sub) in subs.iter().enumerate() {
                        if sub == subscript {
                            let osize = operands[source_usize].dim_sizes()[dim];
                            return fn_(tile, dim.try_into().unwrap(), osize);
                        }
                    }
                }
                panic!("No subscript {} found among tiles", subscript);
            }
            _ => unimplemented!(),
        }
    }

    pub fn memory_allocated(&self, spec: &Spec<Tgt>) -> MemoryAllocation {
        match &self {
            ImplNode::MoveLet {
                source_idx,
                destination_level,
                destination_layout: _,
                destination_vector_shape: _,
                prefetch,
            } => {
                // We assume bytes_used will be the same for source and destination
                // tensors.
                let mut additional = spec.operands()[usize::from(*source_idx)].bytes_used();
                if *prefetch {
                    additional *= 2;
                }

                let mem: MemVec = Tgt::levels()
                    .iter()
                    .map(|level| {
                        if destination_level == level {
                            u64::from(additional)
                        } else {
                            0u64
                        }
                    })
                    .collect();

                let has_pro = movelet_gens_prologue(destination_level, *source_idx, spec);
                let has_epi = movelet_gens_epilogue(destination_level, *source_idx, spec);

                let to_return_len = 1 + has_pro as usize + has_epi as usize;
                let mut to_return = Vec::with_capacity(to_return_len);
                if has_pro {
                    to_return.push(MemVec::zero::<Tgt>());
                }
                to_return.push(mem);
                if has_epi {
                    to_return.push(MemVec::zero::<Tgt>());
                }
                MemoryAllocation {
                    base: MemVec::zero::<Tgt>(),
                    during_children: to_return,
                }
            }
            ImplNode::Pipeline {
                layout: _,
                level,
                vector_shape: _,
            } => {
                // We only introduce Pipelines by peeling, so we know that the
                // intermediate tensor is the first input to to final/head
                // component.
                let Spec::Compose { components, .. } = spec else {
                    panic!();
                };
                let next_to_outer_basics = &components[1];
                let output_shape =
                    &next_to_outer_basics.operand_shapes()[next_to_outer_basics.typ.output_idx()];
                let base = Tgt::levels()
                    .iter()
                    .map(|l| {
                        if level == l {
                            u64::from(next_to_outer_basics.dtype.size())
                                * u64::from(output_shape.into_iter().product::<DimSize>())
                        } else {
                            0u64
                        }
                    })
                    .collect();
                let during_children = (0..self.child_count(spec))
                    .map(|_| MemVec::zero::<Tgt>())
                    .collect();
                MemoryAllocation {
                    base,
                    during_children,
                }
            }
            ImplNode::Loop { .. }
            | ImplNode::SpatialSplit
            | ImplNode::AccumBlock
            | ImplNode::Mult
            | ImplNode::BroadcastVecMult
            | ImplNode::ValueAssign
            | ImplNode::VectorAssign
            | ImplNode::MemsetZero
            | ImplNode::VectorZero => MemoryAllocation {
                base: MemVec::zero::<Tgt>(),
                during_children: (0..self.child_count(spec))
                    .map(|_| MemVec::zero::<Tgt>())
                    .collect(),
            },
        }
    }
}

impl<Tgt: Target> Display for ImplNode<Tgt> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            ImplNode::Loop {
                subscripts,
                tiles,
                parallel: _,
            } => {
                write!(
                    f,
                    "Loop([{}] {})",
                    subscripts
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    tiles
                        .iter()
                        .map(|t| format!("{:?}", t.tiling))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            ImplNode::MoveLet {
                source_idx,
                destination_level,
                destination_layout,
                destination_vector_shape,
                prefetch,
            } => write!(
                f,
                "MoveLet({}, {}, {}, {:?}, {:?})",
                source_idx,
                destination_level,
                destination_layout,
                destination_vector_shape,
                prefetch
            ),
            ImplNode::Mult => write!(f, "Mult"),
            ImplNode::BroadcastVecMult => write!(f, "BroadcastVecMult"),
            ImplNode::ValueAssign => write!(f, "ValueAssign"),
            ImplNode::VectorAssign => write!(f, "VectorAssign"),
            ImplNode::MemsetZero => write!(f, "MemsetZero"),
            ImplNode::VectorZero => write!(f, "VectorZero"),
            _ => write!(f, "{:?}", self),
        }
    }
}

fn movelet_gens_prologue<Tgt: Target>(
    destination_level: &Tgt::Level,
    source_idx: u8,
    node_spec: &Spec<Tgt>,
) -> bool {
    let operand_count = node_spec.operand_count();
    let is_output = usize::from(source_idx) == operand_count - 1;
    destination_level.is_addressed() && (!is_output || node_spec.output_is_read())
}

fn movelet_gens_epilogue<Tgt: Target>(
    destination_level: &Tgt::Level,
    source_idx: u8,
    node_spec: &Spec<Tgt>,
) -> bool {
    let operand_count = node_spec.operand_count();
    let is_output = usize::from(source_idx) == operand_count - 1;
    destination_level.is_addressed() && is_output
}

pub fn mult_applies_to_operands(operands: &[TensorSpec<X86Target>]) -> bool {
    operands
        .iter()
        .all(|o| o.level() == X86MemoryLevel::RF && o.dim_sizes().iter().all(|&d| d == 1))
}

pub(crate) fn movelet_inner_tensorspec<Tgt: Target>(
    operand: &TensorSpec<Tgt>,
    destination_level: &Tgt::Level,
    destination_layout: &Layout,
    destination_vector_shape: Option<&[DimSize]>,
) -> TensorSpec<Tgt> {
    // When moving into an addressed bank, we'll generate an aligned destination.
    // If it's into a cache level, alignment won't change.
    let aligned = if destination_level.is_addressed() {
        true
    } else {
        operand.aligned()
    };

    // Will the result be contiguous? If the move is into a cache, it might be.
    // If it's into memory bank with its own address space, then yes.
    let contiguous_abs = if destination_level.is_addressed() {
        destination_layout.contiguous_full()
    } else {
        operand.contiguous_abs()
    };

    TensorSpec::<Tgt>::new_canon(
        operand.dim_sizes().clone(),
        operand.dtype(),
        contiguous_abs,
        aligned,
        *destination_level,
        destination_layout.clone(),
        destination_vector_shape.map(SmallVec::from),
    )
}

pub fn broadcastvecmult_applies_to_operands(operands: &[TensorSpec<X86Target>]) -> bool {
    if operands[0].level() != X86MemoryLevel::RF {
        return false;
    }
    for i in 1..3 {
        if operands[i].level() != X86MemoryLevel::VRF {
            return false;
        }
        if &operands[i].dim_sizes() != operands[i].vector_shape().as_ref().unwrap() {
            return false;
        }
        if !operands[i].aligned() || !operands[i].is_contiguous() {
            return false;
        }
        if operands[0].dtype() != operands[i].dtype() {
            return false;
        }
    }
    if operands[0].dim_sizes().iter().any(|d| *d != 1) {
        return false;
    }
    if operands[1].dim_sizes().len() != 2 || operands[1].dim_sizes()[0] != 1 {
        return false;
    }
    if operands[2].dim_sizes().to_vec() != vec![1, operands[1].dim_sizes()[1]] {
        return false;
    }
    true
}

pub fn valueassign_applies_to_operands<Tgt: Target>(operands: &[TensorSpec<Tgt>]) -> bool {
    if operands[0].level() == operands[1].level() {
        return false;
    }
    for o in operands {
        for &d in o.dim_sizes() {
            if d != 1 {
                return false;
            }
        }
    }
    true
}

pub fn vectorassign_applies_to_operands<Tgt: Target>(operands: &[TensorSpec<Tgt>]) -> bool {
    if operands.iter().any(|o| !o.is_contiguous()) {
        return false;
    }
    if operands[0].dtype() != operands[1].dtype() {
        return false;
    }
    if operands[0].dim_sizes() != operands[1].dim_sizes() {
        return false;
    }
    if operands[0].layout() != operands[1].layout() {
        return false;
    }

    let mut has_vrf = false;
    for o in operands {
        if o.level().vector_rf() {
            has_vrf = true;
            match &o.vector_shape() {
                Some(vshape) => {
                    if vshape != &o.dim_sizes() {
                        return false;
                    }
                }
                None => {
                    panic!("No vector_shape on operand in level {:?}", o.level());
                }
            }
        }
    }
    if !has_vrf {
        // Neither operand is in a vector RF.
        return false;
    }

    true
}

pub fn memsetzero_applies_to_operands(operands: &[TensorSpec<X86Target>]) -> bool {
    if !operands[0].is_contiguous() {
        return false;
    }
    if operands[0].level() != X86MemoryLevel::RF {
        return false;
    }
    true
}

pub fn vectorzero_applies_to_operands(operands: &[TensorSpec<X86Target>]) -> bool {
    if !operands[0].is_contiguous() {
        return false;
    }
    if operands[0].level() != X86MemoryLevel::VRF {
        return false;
    }
    match operands[0].vector_shape() {
        Some(vshape) if vshape != operands[0].dim_sizes() => {
            return false;
        }
        None => return false,
        _ => (),
    };
    true
}
