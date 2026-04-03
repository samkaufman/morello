use super::cpu::HEAP_BUFFER_ALIGNMENT_BYTES;
use crate::imp::{Impl, ImplNode};
use crate::opaque_symbol::OpaqueSymbol;
use crate::target::Target;
use crate::tensorspec::TensorSpec;
use crate::views::View;
use std::cmp::max;
use std::collections::HashMap;

/// Identifies a single emitted workspace buffer.
pub type WorkspaceId = usize;

/// Supplies target-specific rules for which tensors need workspace-backed storage.
pub trait PlanMemoryTarget: Target {
    /// Returns the byte size to reserve for `spec`, or `None` if no workspace bytes are needed.
    fn heap_allocated_tensor_bytes(spec: &TensorSpec<Self>) -> Option<u64>;
}

/// Records the workspace scopes and tensor placements needed for static allocation.
#[derive(Debug, Clone, Default)]
pub struct StaticAllocationPlan<Tgt: Target> {
    scope_workspaces: HashMap<*const ImplNode<Tgt>, WorkspaceId>,
    workspaces: Vec<u64>,
    tensor_placements: HashMap<OpaqueSymbol, (WorkspaceId, u64)>,
}

/// Tracks the current bump-allocation frontier within one workspace scope.
#[derive(Debug, Clone, Copy)]
struct WorkspaceCursor {
    workspace_id: WorkspaceId,
    /// The next free byte offset within the workspace.
    frontier: u64,
}

impl<Tgt: Target> StaticAllocationPlan<Tgt> {
    /// Returns the planned byte size for a given workspace id.
    pub fn workspace_size(&self, workspace_id: WorkspaceId) -> u64 {
        self.workspaces[workspace_id]
    }

    /// Returns the static placement assigned to a tensor identifier, if any.
    pub fn placement(&self, symbol: OpaqueSymbol) -> Option<(WorkspaceId, u64)> {
        self.tensor_placements.get(&symbol).copied()
    }

    /// Returns the workspace to allocate while emitting this exact Impl node, if any.
    pub fn workspace_for(&self, node: &ImplNode<Tgt>) -> Option<WorkspaceId> {
        self.scope_workspaces.get(&(node as *const _)).copied()
    }
}

impl<Tgt: PlanMemoryTarget> StaticAllocationPlan<Tgt> {
    /// Plan workspace-backed heap buffers for a bound Impl tree in a single pass.
    pub fn plan(root: &ImplNode<Tgt>) -> Self {
        let mut plan = Self::default();
        let root_workspace = plan.make_workspace();
        let root_workspace_id = root_workspace.workspace_id;
        plan.plan_node(root, root_workspace);

        if plan.workspaces[root_workspace_id] > 0 {
            plan.scope_workspaces
                .insert(root as *const _, root_workspace_id);
        }
        plan
    }

    /// Create a fresh workspace and return a cursor at its origin.
    fn make_workspace(&mut self) -> WorkspaceCursor {
        let workspace_id = self.workspaces.len();
        self.workspaces.push(0);
        WorkspaceCursor {
            workspace_id,
            frontier: 0,
        }
    }

    /// Traverse one Impl node's children within one workspace scope.
    fn plan_children(&mut self, node: &ImplNode<Tgt>, cursor: WorkspaceCursor) {
        for node in node.children() {
            self.plan_node(node, cursor);
        }
    }

    /// Traverse one Impl node and assign workspace offsets to heap-backed buffers.
    fn plan_node(&mut self, node: &ImplNode<Tgt>, cursor: WorkspaceCursor) {
        match node {
            ImplNode::Loop(l) if l.parallel => {
                let loop_workspace = self.make_workspace();
                let workspace_id = loop_workspace.workspace_id;
                self.plan_children(node, loop_workspace);
                if self.workspaces[workspace_id] > 0 {
                    for child in node.children() {
                        self.scope_workspaces
                            .insert(child as *const _, workspace_id);
                    }
                }
            }
            ImplNode::Alloc(alloc) => {
                let mut alloc_cursor = cursor;
                if let crate::views::ViewE::Tensor(tensor) = &alloc.introduced {
                    let (next_cursor, offset_bytes) =
                        self.allocate_tensor(alloc_cursor, tensor.spec());
                    alloc_cursor = next_cursor;
                    if let Some(offset_bytes) = offset_bytes {
                        self.tensor_placements.insert(
                            tensor.identifier(),
                            (alloc_cursor.workspace_id, offset_bytes),
                        );
                    }
                }
                self.plan_children(node, alloc_cursor);
            }
            ImplNode::Pipeline(pipeline) => {
                let pipeline_cursor = self.allocate_pipeline_region(pipeline, cursor);
                self.plan_children(node, pipeline_cursor);
            }
            _ => self.plan_children(node, cursor),
        }
    }

    /// Reserve a raw byte range from the current workspace.
    ///
    /// Returns the updated cursor and the range's starting offset.
    fn allocate_bytes(&mut self, cursor: WorkspaceCursor, bytes: u64) -> (WorkspaceCursor, u64) {
        let offset_bytes = align_up::<Tgt>(cursor.frontier);
        let cursor = WorkspaceCursor {
            frontier: offset_bytes + bytes,
            ..cursor
        };
        self.workspaces[cursor.workspace_id] =
            max(self.workspaces[cursor.workspace_id], cursor.frontier);
        (cursor, offset_bytes)
    }

    /// Reserve space for one heap-backed tensor within the active workspace.
    ///
    /// Returns the updated workspace cursor and the tensor's starting byte offset
    /// within the workspace, or `None` if the tensor would not be heap-allocated
    /// in generated C.
    fn allocate_tensor(
        &mut self,
        cursor: WorkspaceCursor,
        spec: &TensorSpec<Tgt>,
    ) -> (WorkspaceCursor, Option<u64>) {
        match Tgt::heap_allocated_tensor_bytes(spec) {
            Some(bytes) => {
                let (cursor, offset_bytes) = self.allocate_bytes(cursor, bytes);
                (cursor, Some(offset_bytes))
            }
            None => (cursor, None),
        }
    }

    /// Reserve a pipeline sub-region whose wiring blocks alternate between the front and back.
    ///
    /// Each wiring in `pipeline.wirings` represents the intermediate tensors flowing
    /// from one pipeline stage to the next.
    fn allocate_pipeline_region(
        &mut self,
        pipeline: &crate::imp::pipeline::Pipeline<Tgt>,
        cursor: WorkspaceCursor,
    ) -> WorkspaceCursor {
        // For each wiring, compute the aligned total bytes of its heap-backed intermediates.
        let wiring_heap_bytes = pipeline
            .wirings
            .iter()
            .map(|wiring| {
                let mut block_size = 0;
                for bytes in wiring
                    .intermediate_tensors
                    .iter()
                    .filter_map(|tensor| Tgt::heap_allocated_tensor_bytes(tensor.spec()))
                {
                    block_size = align_up::<Tgt>(block_size) + bytes;
                }
                align_up::<Tgt>(block_size)
            })
            .collect::<Vec<_>>();

        let mut region_size = wiring_heap_bytes.iter().copied().max().unwrap_or(0);
        for pair in wiring_heap_bytes.windows(2) {
            region_size = max(region_size, pair[0] + pair[1]);
        }
        if region_size == 0 {
            return cursor;
        }

        let (cursor, region_offset) = self.allocate_bytes(cursor, region_size);
        for (stage_idx, (wiring, &block_size)) in
            pipeline.wirings.iter().zip(&wiring_heap_bytes).enumerate()
        {
            let stage_base = if stage_idx % 2 == 0 {
                region_offset
            } else {
                region_offset + region_size - block_size
            };
            let mut local_offset = 0;
            for (symbol, bytes) in wiring.intermediate_tensors.iter().filter_map(|tensor| {
                Tgt::heap_allocated_tensor_bytes(tensor.spec())
                    .map(|bytes| (tensor.identifier(), bytes))
            }) {
                local_offset = align_up::<Tgt>(local_offset);
                self.tensor_placements
                    .insert(symbol, (cursor.workspace_id, stage_base + local_offset));
                local_offset += bytes;
            }
            debug_assert_eq!(align_up::<Tgt>(local_offset), block_size);
        }
        cursor
    }
}

/// Round a byte count up to the workspace sub-allocation alignment.
fn align_up<Tgt: Target>(value: u64) -> u64 {
    let workspace_alignment = max(HEAP_BUFFER_ALIGNMENT_BYTES, u64::from(Tgt::line_size()));
    if workspace_alignment == 0 {
        return value;
    }
    value.div_ceil(workspace_alignment) * workspace_alignment
}
