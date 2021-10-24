use std::collections::{BTreeMap, BTreeSet, HashMap};

use lasagna::{Span, Spanned};

use crate::{
    AssignStmt, BinaryExpr, BinaryOp, BlockStmt, Bool, Expr, IfStmt, Instruction, LetStmt,
    Register, Stmt, TermExpr, Type, UnaryExpr, Word,
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
    pub fn allocated(&self) -> impl Iterator<Item = Register> {
        self.allocated.clone().into_iter()
    }
}

pub struct Value {
    pub ty: Type,
    pub deref: bool,
    pub reg: Register,
}

pub struct Variable {
    pub ty: Type,
    pub addr: u64,
}

pub struct Scope {
    /// Number of pushes made in this scope.
    pub local_size: u64,

    /// Number of pushes made in this function.
    pub stack_size: u64,

    pub variables: HashMap<String, Variable>,
}

impl Scope {
    #[inline]
    pub fn new() -> Self {
        Self {
            local_size: 0,
            stack_size: 0,
            variables: HashMap::new(),
        }
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
    pub current_block: Option<Block>,
    pub max_registers: u32,
}

impl Compiler {
    #[inline]
    pub fn new() -> Self {
        Self {
            next_block: Block(0),
            blocks: BTreeMap::new(),
            current_block: None,
            max_registers: 256,
        }
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
        self.current_block = Some(block);
    }

    #[inline]
    pub fn label(&self) -> &String {
        let block = self.current_block.expect("current block not set");

        &self.blocks.get(&block).unwrap().label
    }

    #[inline]
    pub fn ins(&mut self, ins: Instruction) {
        let block = self.current_block.expect("current block not set");

        self.blocks.get_mut(&block).unwrap().instructions.push(ins);
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
                    Bool::True(_) => true,
                    Bool::False(_) => false,
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
    pub fn compile_unary_expr(
        &mut self,
        unary: &UnaryExpr,
        scope: &mut Scope,
        registers: &mut Registers,
    ) -> Result<Value, CompilerError> {
        match unary {
            UnaryExpr::Term(term) => self.compile_term_expr(term, scope, registers),
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
        let addr = scope.stack_size * 8;

        let mut registers = self.registers();
        let value = self.compile_expr(&let_stmt.expr, scope, &mut registers)?;

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
    pub fn compile_block_stmt(&mut self, block_stmt: &BlockStmt) -> Result<(), CompilerError> {
        let mut scope = Scope::new();

        for stmt in block_stmt.stmts.iter() {
            self.compile_stmt(stmt, &mut scope)?;
        }

        self.pop_scope(&mut scope);

        Ok(())
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

            self.set_block(block);
            self.compile_block_stmt(&if_stmt.block)?;
            self.ins(Instruction::Comment(String::from("jump to end")));
            self.ins(Instruction::Jump(end));

            self.set_block(end);

            Ok(())
        } else {
            Err(CompilerError::new(
                if_stmt.expr.span(),
                format!("expected '{}' found '{}'", Type::Bool, val.ty,),
            ))
        }
    }

    #[inline]
    pub fn compile_stmt(&mut self, stmt: &Stmt, scope: &mut Scope) -> Result<(), CompilerError> {
        match stmt {
            Stmt::Let(let_stmt) => self.compile_let_stmt(let_stmt, scope),
            Stmt::Assign(assign_stmt) => self.compile_assign_stmt(assign_stmt, scope),
            Stmt::If(if_stmt) => self.compile_if_stmt(if_stmt, scope),
        }
    }

    #[inline]
    pub fn pop_scope(&mut self, scope: &mut Scope) {
        let mut registers = self.registers();
        let reg = registers.alloc().expect("ran out of registers");

        self.ins(Instruction::Comment(String::from("pop scope stack")));

        for _ in 0..scope.local_size {
            self.ins(Instruction::Pop(reg));
        }
    }

    #[inline]
    pub fn pop_stack(&mut self, scope: &mut Scope) {
        let mut registers = self.registers();
        let reg = registers.alloc().expect("ran out of registers");

        self.ins(Instruction::Comment(String::from("pop stack")));

        for _ in 0..scope.stack_size {
            self.ins(Instruction::Pop(reg));
        }
    }
}
