use std::fmt;

use crate::common::Dtype;
use crate::utils::indent;

// TODO: Remove Debug.
#[derive(Debug, Clone)]
pub enum CBuffer {
    Ptr {
        name: String,
        backing_buffer: Box<CBuffer>,
    },
    UnsizedHeapArray {
        // TODO: Merge UnsizedHeapArray with HeapArray.
        name: String,
        dtype: Dtype,
    },
    HeapArray {
        name: String,
        size: u32,
        dtype: Dtype,
    },
    StackArray {
        name: String,
        size: u32,
        dtype: Dtype,
    },
    ValueVar {
        name: String,
        dtype: Dtype,
    },
}

impl CBuffer {
    /// Returns the C identifier name if the receiver has just one (i.e., is not distributed).
    pub fn name(&self) -> Option<&str> {
        match self {
            CBuffer::Ptr { name, .. }
            | CBuffer::UnsizedHeapArray { name, .. }
            | CBuffer::HeapArray { name, .. }
            | CBuffer::StackArray { name, .. }
            | CBuffer::ValueVar { name, .. } => Some(name),
        }
    }

    pub fn dtype(&self) -> Dtype {
        match self {
            CBuffer::Ptr {
                backing_buffer: backing_tensor,
                ..
            } => backing_tensor.dtype(),
            CBuffer::UnsizedHeapArray { dtype, .. }
            | CBuffer::HeapArray { dtype, .. }
            | CBuffer::StackArray { dtype, .. }
            | CBuffer::ValueVar { dtype, .. } => *dtype,
        }
    }

    pub fn emit<W: fmt::Write>(&self, w: &mut W, zero_init: bool, depth: usize) -> fmt::Result {
        match self {
            CBuffer::Ptr { .. } => unimplemented!(),
            CBuffer::UnsizedHeapArray { .. } => unimplemented!(),
            CBuffer::HeapArray { name, size, dtype } => {
                writeln!(w, "{}{} *restrict {};", indent(depth), c_type(*dtype), name)?;
                writeln!(
                    w,
                    "{}posix_memalign((void **)&{}, 128, {}*sizeof({}));",
                    indent(depth),
                    name,
                    size,
                    c_type(*dtype)
                )?;

                if zero_init {
                    writeln!(
                        w,
                        "{}memset({}, 0, {}*sizeof({}));",
                        indent(depth),
                        name,
                        size,
                        c_type(*dtype)
                    )?;
                }

                Ok(())
            }
            CBuffer::StackArray { name, size, dtype } => {
                let epi = if zero_init { " = {0}" } else { "" };
                writeln!(
                    w,
                    "{}{} {}[{}] __attribute__((aligned (128))){};",
                    indent(depth),
                    c_type(*dtype),
                    name,
                    size,
                    epi
                )?;
                Ok(())
            }
            CBuffer::ValueVar { name, dtype } => {
                let epi = if zero_init { " = {0}" } else { "" };
                writeln!(w, "{}{} {}{};", indent(depth), c_type(*dtype), name, epi)?;
                Ok(())
            }
        }
    }

    pub fn emit_free<W: fmt::Write>(&self, w: &mut W, depth: usize) -> fmt::Result {
        match self {
            CBuffer::Ptr { .. } => unimplemented!(),
            CBuffer::UnsizedHeapArray { name, .. } | CBuffer::HeapArray { name, .. } => {
                writeln!(w, "{}free({name});", indent(depth))?;
                Ok(())
            }
            CBuffer::StackArray { .. } | CBuffer::ValueVar { .. } => Ok(()),
        }
    }

    pub fn declared_type(&self) -> &'static str {
        match self {
            CBuffer::Ptr {
                backing_buffer: backing_tensor,
                ..
            } => backing_tensor.declared_type(),
            CBuffer::UnsizedHeapArray { dtype, .. } | CBuffer::HeapArray { dtype, .. } => {
                c_type(*dtype)
            }
            CBuffer::StackArray { .. } => todo!(),
            CBuffer::ValueVar { .. } => todo!(),
        }
    }

    pub fn should_unroll(&self) -> bool {
        false
    }
}

pub fn c_type(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::Uint8 => "uint8_t",
        Dtype::Uint32 => "uint32_t",
    }
}
