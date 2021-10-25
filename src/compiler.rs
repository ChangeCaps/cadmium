use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    panic,
};

use lasagna::{Span, Spanned};

use crate::{
    AssignStmt, BinaryExpr, BinaryOp, BlockStmt, BoolExpr, CallExpr, Expr, FnDecl, FnType, IfStmt,
    Instruction, LetStmt, ModFn, Module, Register, ReturnStmt, Stmt, TermExpr, Type, UnaryExpr,
    UnaryOp, Word,
};

#[derive(Clone, Debug)]
pub struct CompilerError {
    pub span: Span,
    pub message: String,
}

impl CompilerError {
    #[inline]
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for CompilerError {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "compiler error: '{}' at {}", self.message, self.span)
    }
}

impl std::error::Error for CompilerError {}

/// Register allocator.
#[derive(Default)]
pub struct Registers {
    allocated: BTreeSet<Register>,
    unallocated: BTreeSet<Register>,
}

impl Registers {
    #[inline]
    pub fn new(max_registers: u32) -> Self {
        let mut unallocated = BTreeSet::new();

        for i in 0..max_registers {
            unallocated.insert(Register::from_u8(i as u8));
        }

        Self {
            allocated: BTreeSet::new(),
            unallocated,
        }
    }

    #[inline]
    pub fn alloc(&mut self) -> Option<Register> {
        if let Some(reg) = self.unallocated.iter().next().cloned() {
            self.allocated.insert(reg);
            self.unallocated.remove(&reg);

            Some(reg)
        } else {
            None
        }
    }

    #[inline]
    pub fn free(&mut self, reg: Register) {
        self.allocated.remove(&reg);
        self.unallocated.insert(reg);
    }

    #[inline]
    pub fn allocated(&self) -> impl DoubleEndedIterator<Item = Register> {
        self.allocated.clone().into_iter()
    }
}

#[derive(Debug)]
pub struct Value {
    pub ty: Type,
    pub deref: bool,
    pub reg: Register,
}

#[derive(Clone, Debug)]
pub struct Variable {
    pub ty: Type,
    pub addr: u64,
}

#[derive(Clone, Debug)]
pub struct Scope {
    /// Number of pushes made in this scope.
    pub local_size: u64,

    /// Number of pushes made in this function.
    pub stack_size: u64,

    pub variables: HashMap<String, Variable>,

    pub return_type: Type,
    pub returned: bool,
}

impl Scope {
    #[inline]
    pub fn new() -> Self {
        Self {
            local_size: 0,
            stack_size: 0,
            variables: HashMap::new(),
            return_type: Type::Void,
            returned: false,
        }
    }

    #[inline]
    pub fn sub(&self) -> Self {
        Self {
            local_size: 0,
            ..self.clone()
        }
    }

    #[inline]
    pub fn push(&mut self) {
        self.local_size += 1;
        self.stack_size += 1;
    }

    #[inline]
    pub fn push_variable(&mut self, ident: String, var: Variable) {
        self.variables.insert(ident, var);

        self.local_size += 1;
        self.stack_size += 1;
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Block(u64);

impl Block {
    #[inline]
    pub const fn from_u64(val: u64) -> Self {
        Self(val)
    }

    #[inline]
    pub const fn to_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for Block {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%b{}", self.0)
    }
}

#[derive(Clone, Debug)]
pub struct CompiledBlock {
    pub label: String,
    pub instructions: Vec<Instruction>,
}

impl CompiledBlock {
    #[inline]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            instructions: Vec::new(),
        }
    }
}

pub struct Compiler {
    pub next_block: Block,
    pub blocks: BTreeMap<Block, CompiledBlock>,
    pub finalized_blocks: Vec<Block>,
    pub current_block: Option<Block>,
    pub externals: BTreeMap<u64, Block>,
    pub max_registers: u32,
    pub module: Module,
}

impl Compiler {
    #[inline]
    pub fn new() -> Self {
        Self {
            next_block: Block(0),
            blocks: BTreeMap::new(),
            finalized_blocks: Vec::new(),
            current_block: None,
            externals: BTreeMap::new(),
            max_registers: 256,
            module: Module::new(),
        }
    }

    #[inline]
    pub fn dump_asm(&self) -> String {
        let mut dump = String::new();

        for block in &self.finalized_blocks {
            let block = &self.blocks[block];

            dump += &format!("{}\n", block.label);

            for ins in &block.instructions {
                dump += &format!("\t{}\n", ins);
            }

            dump += "\n";
        }

        dump
    }

    #[inline]
    pub fn registers(&self) -> Registers {
        Registers::new(self.max_registers)
    }

    #[inline]
    pub fn create_block(&mut self, label: impl Into<String>) -> Block {
        let block = self.next_block;
        self.next_block.0 += 1;

        self.blocks.insert(block, CompiledBlock::new(label));

        block
    }

    #[inline]
    pub fn set_block(&mut self, block: Block) {
        if self.finalized_blocks.contains(&block) {
            panic!("cannot use finalized blocks");
        }

        self.current_block = Some(block);
    }

    #[inline]
    pub fn current_block(&self) -> Block {
        self.current_block.expect("current block not set")
    }

    #[inline]
    pub fn finalize(&mut self) {
        let block = self.current_block.take().expect("current block not set");

        self.finalized_blocks.push(block);
    }

    #[inline]
    pub fn label(&self) -> &String {
        let block = self.current_block();

        &self.blocks.get(&block).unwrap().label
    }

    /// Push instruction to the current function
    #[inline]
    pub fn ins(&mut self, ins: Instruction) {
        let block = self.current_block();

        self.blocks.get_mut(&block).unwrap().instructions.push(ins);
    }

    /// Crates an internal function that wraps the external function.
    #[inline]
    pub fn compile_external(&mut self, func: &ModFn) {
        if let FnType::External(id) = func.ty {
            let mut registers = self.registers();

            // crate block
            let external = self.create_block(format!("<external {}>", id));
            self.set_block(external);

            let mut args = Vec::new();

            // push args
            for i in 0..func.args.len() {
                let reg = registers.alloc().unwrap();

                self.ins(Instruction::StackAddr(reg, i as i64 * -8 - 8));
                self.ins(Instruction::Read(reg, reg));

                args.push(reg);
            }

            for arg in &args {
                registers.free(*arg);
            }

            let reg = registers.alloc().unwrap();

            // call the external function
            self.ins(Instruction::CallExternal(id, reg, args));

            // pop stack
            for _ in 0..func.args.len() {
                let reg = registers.alloc().unwrap();
                self.ins(Instruction::Pop(reg));
            }

            // return
            self.ins(Instruction::Return(reg));

            self.finalize();

            // insert external
            self.externals.insert(id, external);
        }
    }

    #[inline]
    pub fn compile_deref(&mut self, value: Value) -> Value {
        if value.deref {
            self.ins(Instruction::Read(value.reg, value.reg));

            Value {
                reg: value.reg,
                deref: false,
                ty: value.ty,
            }
        } else {
            value
        }
    }

    #[inline]
    pub fn compile_term_expr(
        &mut self,
        term: &TermExpr,
        scope: &mut Scope,
        registers: &mut Registers,
    ) -> Result<Value, CompilerError> {
        match term {
            TermExpr::Paren(paren) => self.compile_expr(&paren.expr, scope, registers),
            TermExpr::Variable(ident) => {
                if let Some(var) = scope.variables.get(&ident.value) {
                    let reg = registers.alloc().expect("ran out of registers");

                    let offset = var.addr as i64 - scope.stack_size as i64 * 8;

                    self.ins(Instruction::StackAddr(reg, offset));

                    Ok(Value {
                        reg,
                        deref: true,
                        ty: var.ty.clone(),
                    })
                } else if let Some(func) = self.module.functions.get(&ident.value).cloned() {
                    match func.ty {
                        FnType::External(id) => {
                            if let Some(block) = self.externals.get(&id).cloned() {
                                let reg = registers.alloc().expect("ran out of registers");

                                self.ins(Instruction::FnAddr(reg, block));

                                Ok(Value {
                                    reg,
                                    deref: false,
                                    ty: Type::Fn(
                                        func.args.clone(),
                                        Box::new(func.return_type.clone()),
                                    ),
                                })
                            } else {
                                let current = self.current_block();
                                self.compile_external(&func);
                                self.set_block(current);

                                let block = self.externals[&id];

                                let reg = registers.alloc().expect("ran out of registers");

                                self.ins(Instruction::FnAddr(reg, block));

                                Ok(Value {
                                    reg,
                                    deref: false,
                                    ty: Type::Fn(
                                        func.args.clone(),
                                        Box::new(func.return_type.clone()),
                                    ),
                                })
                            }
                        }
                        FnType::Internal(block) => {
                            let reg = registers.alloc().unwrap();

                            self.ins(Instruction::FnAddr(reg, block));

                            Ok(Value {
                                reg,
                                deref: false,
                                ty: Type::Fn(func.args.clone(), Box::new(func.return_type.clone())),
                            })
                        }
                    }
                } else {
                    Err(CompilerError::new(
                        ident.span(),
                        format!("'{}' is not defined", ident.value),
                    ))
                }
            }
            TermExpr::Integer(int) => {
                let reg = registers.alloc().expect("ran out of registers");

                self.ins(Instruction::Init(reg, Word::from_i64(int.value)));

                Ok(Value {
                    reg,
                    deref: false,
                    ty: Type::I32,
                })
            }
            TermExpr::Bool(b) => {
                let reg = registers.alloc().expect("ran out of registers");

                let val = match b {
                    BoolExpr::True(_) => true,
                    BoolExpr::False(_) => false,
                };

                self.ins(Instruction::Init(reg, Word::from_bool(val)));

                Ok(Value {
                    reg,
                    deref: false,
                    ty: Type::Bool,
                })
            }
        }
    }

    #[inline]
    pub fn compile_call_expr(
        &mut self,
        call: &CallExpr,
        scope: &mut Scope,
        registers: &mut Registers,
    ) -> Result<Value, CompilerError> {
        match call {
            CallExpr::Term(term) => self.compile_term_expr(term, scope, registers),
            CallExpr::Call(call, call_args) => {
                let mut callee = self.compile_call_expr(call, scope, registers)?;
                callee = self.compile_deref(callee);

                if let Type::Fn(ty_args, return_type) = callee.ty {
                    let mut args = Vec::new();

                    // compile arguments
                    for (i, arg) in call_args.args.iter().enumerate() {
                        // compile arg
                        let mut val = self.compile_expr(arg, scope, registers)?;
                        val = self.compile_deref(val);

                        // check arg type
                        if ty_args.get(i) != Some(&val.ty) {
                            return Err(CompilerError::new(
                                arg.span(),
                                format!("expected '{}' found '{}'", ty_args[i], val.ty),
                            ));
                        }

                        // push arg
                        args.push(val.reg);
                    }

                    for arg in &args {
                        registers.free(*arg);
                    }

                    // save allocated registers
                    for reg in registers.allocated() {
                        self.ins(Instruction::Push(reg));
                    }

                    // allocate return value
                    let reg = registers.alloc().unwrap();

                    // call function
                    self.ins(Instruction::Call(callee.reg, reg, args));

                    // restore allocated registers
                    for a_reg in registers.allocated().rev() {
                        if a_reg != reg {
                            self.ins(Instruction::Pop(a_reg));
                        }
                    }

                    Ok(Value {
                        ty: *return_type,
                        deref: false,
                        reg,
                    })
                } else {
                    Err(CompilerError::new(
                        call.span(),
                        format!("cannot call '{}'", callee.ty),
                    ))
                }
            }
        }
    }

    #[inline]
    pub fn compile_unary_expr(
        &mut self,
        unary: &UnaryExpr,
        scope: &mut Scope,
        registers: &mut Registers,
    ) -> Result<Value, CompilerError> {
        match unary {
            UnaryExpr::Unary(op, unary) => match op {
                UnaryOp::Ref(_) => {
                    let val = self.compile_unary_expr(unary, scope, registers)?;

                    if val.deref {
                        Ok(Value {
                            ty: Type::Ptr(Box::new(val.ty)),
                            deref: false,
                            reg: val.reg,
                        })
                    } else {
                        scope.push();

                        self.ins(Instruction::Push(val.reg));
                        self.ins(Instruction::StackAddr(val.reg, -8));

                        Ok(Value {
                            ty: Type::Ptr(Box::new(val.ty)),
                            deref: false,
                            reg: val.reg,
                        })
                    }
                }
                UnaryOp::Deref(_) => {
                    let mut val = self.compile_unary_expr(unary, scope, registers)?;
                    val = self.compile_deref(val);

                    if let Type::Ptr(inner) = val.ty {
                        Ok(Value {
                            ty: *inner,
                            deref: true,
                            reg: val.reg,
                        })
                    } else {
                        Err(CompilerError::new(
                            unary.span(),
                            format!("cannot deref '{}'", val.ty),
                        ))
                    }
                }
            },
            UnaryExpr::Call(call) => self.compile_call_expr(call, scope, registers),
        }
    }

    #[inline]
    pub fn compile_binary_expr(
        &mut self,
        binary: &BinaryExpr,
        scope: &mut Scope,
        registers: &mut Registers,
    ) -> Result<Value, CompilerError> {
        let mut lhs = self.compile_expr(&binary.lhs, scope, registers)?;
        let mut rhs = self.compile_expr(&binary.rhs, scope, registers)?;
        lhs = self.compile_deref(lhs);
        rhs = self.compile_deref(rhs);

        match lhs.ty {
            Type::I32 if rhs.ty == Type::I32 && binary.op.is_numeric() => {
                let ins = match binary.op {
                    BinaryOp::Add(_) => Instruction::IAdd(lhs.reg, lhs.reg, rhs.reg),
                    BinaryOp::Sub(_) => Instruction::ISub(lhs.reg, lhs.reg, rhs.reg),
                    BinaryOp::Mul(_) => Instruction::IMul(lhs.reg, lhs.reg, rhs.reg),
                    BinaryOp::Div(_) => Instruction::IDiv(lhs.reg, lhs.reg, rhs.reg),
                    _ => unreachable!(),
                };

                self.ins(ins);

                registers.free(rhs.reg);

                return Ok(lhs);
            }
            _ => {}
        }

        Err(CompilerError::new(
            binary.span(),
            format!(
                "binary operator '{}' not implemented for '{}' '{}'",
                binary.op, lhs.ty, rhs.ty,
            ),
        ))
    }

    #[inline]
    pub fn compile_expr(
        &mut self,
        expr: &Expr,
        scope: &mut Scope,
        registers: &mut Registers,
    ) -> Result<Value, CompilerError> {
        match expr {
            Expr::Unary(unary) => self.compile_unary_expr(unary, scope, registers),
            Expr::Binary(binary) => self.compile_binary_expr(binary, scope, registers),
        }
    }

    #[inline]
    pub fn compile_let_stmt(
        &mut self,
        let_stmt: &LetStmt,
        scope: &mut Scope,
    ) -> Result<(), CompilerError> {
        let mut registers = self.registers();
        let mut value = self.compile_expr(&let_stmt.expr, scope, &mut registers)?;
        value = self.compile_deref(value);

        let addr = scope.stack_size * 8;

        scope.push_variable(
            let_stmt.ident.value.clone(),
            Variable { addr, ty: value.ty },
        );

        self.ins(Instruction::Comment(format!(
            "push {}",
            let_stmt.ident.value
        )));

        self.ins(Instruction::Push(value.reg));

        Ok(())
    }

    #[inline]
    pub fn compile_assign_stmt(
        &mut self,
        assign_stmt: &AssignStmt,
        scope: &mut Scope,
    ) -> Result<(), CompilerError> {
        let mut registers = self.registers();

        let lhs = self.compile_expr(&assign_stmt.lhs, scope, &mut registers)?;
        let mut rhs = self.compile_expr(&assign_stmt.rhs, scope, &mut registers)?;
        rhs = self.compile_deref(rhs);

        if lhs.deref {
            if lhs.ty == rhs.ty {
                self.ins(Instruction::Write(lhs.reg, rhs.reg));

                Ok(())
            } else {
                Err(CompilerError::new(
                    assign_stmt.span(),
                    format!("cannot assign '{}' to '{}'", lhs.ty, rhs.ty),
                ))
            }
        } else {
            Err(CompilerError::new(
                assign_stmt.lhs.span(),
                "cannot assign to value",
            ))
        }
    }

    #[inline]
    pub fn compile_block_stmt(
        &mut self,
        block_stmt: &BlockStmt, 
        scope: &Scope,
    ) -> Result<bool, CompilerError> {
        let mut scope = scope.sub();

        for stmt in block_stmt.stmts.iter() {
            self.compile_stmt(stmt, &mut scope)?;
        }

        let mut registers = self.registers();
        self.pop_scope(&mut scope, &mut registers); 

        Ok(scope.returned)
    }

    #[inline]
    pub fn compile_if_stmt(
        &mut self,
        if_stmt: &IfStmt,
        scope: &mut Scope,
    ) -> Result<(), CompilerError> {
        let mut registers = self.registers();
        let reg = registers.alloc().expect("ran out of registers");

        let mut val = self.compile_expr(&if_stmt.expr, scope, &mut registers)?;
        val = self.compile_deref(val);

        if let Type::Bool = val.ty {
            let block = self.create_block(self.label().clone() + "::true_block");
            let end = self.create_block(self.label().clone() + "::end");

            self.ins(Instruction::Branch(block, end, reg));

            self.finalize();

            self.set_block(block);
            self.compile_block_stmt(&if_stmt.block, scope)?;
            self.ins(Instruction::Comment(String::from("jump to end")));
            self.ins(Instruction::Jump(end));

            self.finalize();

            self.set_block(end);

            Ok(())
        } else {
            Err(CompilerError::new(
                if_stmt.expr.span(),
                format!("expected '{}' found '{}'", Type::Bool, val.ty),
            ))
        }
    }

    #[inline]
    pub fn compile_return_stmt(
        &mut self,
        return_stmt: &ReturnStmt,
        scope: &mut Scope,
    ) -> Result<(), CompilerError> {
        let mut registers = self.registers();

        let mut val = self.compile_expr(&return_stmt.expr, scope, &mut registers)?;
        val = self.compile_deref(val);

        if val.ty != scope.return_type {
            return Err(CompilerError::new(
                return_stmt.span(),
                format!(
                    "expected return type '{}' found '{}",
                    scope.return_type, val.ty
                ),
            ));
        }

        self.pop_stack(scope, &mut registers);

        self.ins(Instruction::Return(val.reg));

        scope.returned = true;

        Ok(())
    }

    #[inline]
    pub fn compile_stmt(&mut self, stmt: &Stmt, scope: &mut Scope) -> Result<(), CompilerError> {
        match stmt {
            Stmt::Let(let_stmt) => self.compile_let_stmt(let_stmt, scope),
            Stmt::Assign(assign_stmt) => self.compile_assign_stmt(assign_stmt, scope),
            Stmt::If(if_stmt) => self.compile_if_stmt(if_stmt, scope),
            Stmt::Expr(expr, _) => {
                let mut registers = self.registers();

                self.compile_expr(expr, scope, &mut registers)?;

                Ok(())
            }
            Stmt::Fn(fn_decl) => self.compile_function(fn_decl),
            Stmt::Return(return_stmt) => self.compile_return_stmt(return_stmt, scope),
        }
    }

    #[inline]
    pub fn compile_function(&mut self, func: &FnDecl) -> Result<(), CompilerError> {
        let mut scope = Scope::new();

        // register variables in scope
        for (i, arg) in func.args.iter().rev().enumerate() {
            scope.push_variable(
                arg.ident.value.clone(),
                Variable {
                    ty: arg.ty.as_type(),
                    addr: i as u64 * 8,
                },
            );
        }

        let current = self.current_block();

        let mod_fn = self.module.functions.get(&func.ident.value).unwrap();
        scope.return_type = mod_fn.return_type.clone();

        let block = if let FnType::Internal(block) = mod_fn.ty {
            block
        } else {
            unreachable!()
        };

        // create function block
        self.set_block(block);

        // compile function block
        scope.returned = self.compile_block_stmt(&func.block, &scope)?;

        if !scope.returned {
            // compile stack
            if scope.return_type != Type::Void {
                return Err(CompilerError::new(
                    func.block.span(),
                    format!(
                        "expected return type '{}' found '{}",
                        scope.return_type,
                        Type::Void
                    ),
                ));
            }

            let mut registers = self.registers();
            self.pop_stack(&mut scope, &mut registers);

            // return from function
            let mut registers = self.registers();
            let reg = registers.alloc().unwrap();
            self.ins(Instruction::Return(reg));
        } 

        self.finalize();
        self.set_block(current);

        Ok(())
    }

    #[inline]
    pub fn prepare_fn(&mut self, fn_decl: &FnDecl) {
        let args = fn_decl.args.iter().map(|arg| arg.ty.as_type()).collect();

        let return_type = fn_decl
            .return_type
            .as_ref()
            .map(|rt| rt.ty.as_type())
            .unwrap_or(Type::Void);

        let block = self.create_block(&fn_decl.ident.value);
        self.module
            .insert_function(&fn_decl.ident.value, ModFn::new(block, args, return_type));
    }

    #[inline]
    pub fn prepare_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Fn(fn_decl) => self.prepare_fn(fn_decl),
            _ => {}
        }
    }

    #[inline]
    pub fn pop_scope(&mut self, scope: &mut Scope, registers: &mut Registers) {
        let reg = registers.alloc().expect("ran out of registers");

        if scope.local_size > 0 {
            self.ins(Instruction::Comment(String::from("pop scope stack")));
        }

        for _ in 0..scope.local_size {
            self.ins(Instruction::Pop(reg));
        }
    }

    #[inline]
    pub fn pop_stack(&mut self, scope: &mut Scope, registers: &mut Registers) {
        let reg = registers.alloc().expect("ran out of registers");

        if scope.stack_size > 0 {
            self.ins(Instruction::Comment(String::from("pop stack")));
        }

        for _ in 0..scope.stack_size {
            self.ins(Instruction::Pop(reg));
        }
    }
}
