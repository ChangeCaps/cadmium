use std::collections::HashMap;

use crate::{Block, Type};

#[derive(Clone, Debug)]
pub enum FnType {
    External(u64),
    Internal(Block),
}

impl From<u64> for FnType {
    #[inline]
    fn from(id: u64) -> Self {
        Self::External(id)
    }
}

impl From<Block> for FnType {
    #[inline]
    fn from(block: Block) -> Self {
        Self::Internal(block)
    }
}

#[derive(Clone, Debug)]
pub struct ModFn {
    pub ty: FnType,
    pub args: Vec<Type>,
    pub return_type: Type,
}

impl ModFn {
    #[inline]
    pub fn new(ty: impl Into<FnType>, args: Vec<Type>, return_type: Type) -> Self {
        Self {
            ty: ty.into(),
            args,
            return_type,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Module {
    pub modules: HashMap<String, Module>,
    pub functions: HashMap<String, ModFn>,
}

impl Module {
    #[inline]
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    #[inline]
    pub fn insert_module(&mut self, ident: impl Into<String>, module: Module) {
        self.modules.insert(ident.into(), module);
    }

    #[inline]
    pub fn insert_function(&mut self, ident: impl Into<String>, mod_fn: ModFn) {
        self.functions.insert(ident.into(), mod_fn);
    }
}
