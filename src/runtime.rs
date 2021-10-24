use crate::{AstProgram, Block, CompiledBlock, Compiler, Scope};
use std::collections::BTreeMap;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Register(u8);

impl Register {
    #[inline]
    pub fn from_u8(byte: u8) -> Self {
        Self(byte)
    }

    #[inline]
    pub fn inner(self) -> u8 {
        self.0
    }
}

impl std::fmt::Display for Register {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%r{}", self.0)
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Word([u8; 8]);

impl Word {
    pub const ZERO: Self = Self([0; 8]);

    /// Creates a [`Word`] from a slice.
    ///
    /// # Panic
    /// Panics if ``bytes.len() != 8``.
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.len() != 8 {
            panic!("word must contain 8 bytes")
        }

        // SAFETY: we know that bytes has the correct length, and directly copying bytes is safe
        // therefore ptr::read is safe.
        Self(unsafe { std::ptr::read(bytes as *const _ as *const _) })
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }

    #[inline]
    pub fn from_f32(val: f32) -> Self {
        let bytes: [u8; 4] = unsafe { std::mem::transmute(val) };

        Self([0, 0, 0, 0, bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    #[inline]
    pub fn from_f64(val: f64) -> Self {
        let bytes: [u8; 8] = unsafe { std::mem::transmute(val) };

        Self(bytes)
    }

    #[inline]
    pub fn from_i32(val: i32) -> Self {
        let bytes: [u8; 4] = val.to_be_bytes();

        Self([0, 0, 0, 0, bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    #[inline]
    pub fn from_i64(val: i64) -> Self {
        Self(val.to_be_bytes())
    }

    #[inline]
    pub fn from_u64(val: u64) -> Self {
        Self(val.to_be_bytes())
    }

    #[inline]
    pub fn from_bool(val: bool) -> Self {
        if val {
            Self([0, 0, 0, 0, 0, 0, 0, 1])
        } else {
            Self([0, 0, 0, 0, 0, 0, 0, 0])
        }
    }

    #[inline]
    pub fn as_f32(self) -> f32 {
        unsafe { std::mem::transmute([self.0[4], self.0[5], self.0[6], self.0[7]]) }
    }

    #[inline]
    pub fn as_f64(self) -> f64 {
        unsafe { std::mem::transmute(self.0) }
    }

    #[inline]
    pub fn as_i32(self) -> i32 {
        i32::from_be_bytes([self.0[4], self.0[5], self.0[6], self.0[7]])
    }

    #[inline]
    pub fn as_i64(self) -> i64 {
        i64::from_be_bytes(self.0)
    }

    #[inline]
    pub fn as_u64(self) -> u64 {
        u64::from_be_bytes(self.0)
    }

    #[inline]
    pub fn as_bool(self) -> bool {
        self.0[7] != 0
    }
}

impl std::fmt::Display for Word {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:02X}", self.0[0])?;
        write!(f, "{:02X}", self.0[1])?;
        write!(f, "{:02X}", self.0[2])?;
        write!(f, "{:02X}", self.0[3])?;
        write!(f, "{:02X}", self.0[4])?;
        write!(f, "{:02X}", self.0[5])?;
        write!(f, "{:02X}", self.0[6])?;
        write!(f, "{:02X}", self.0[7])
    }
}

#[derive(Clone, Debug)]
pub enum Instruction {
    // stack
    Push(Register),
    Pop(Register),
    StackAddr(Register, i64),
    Read(Register, Register),
    Write(Register, Register),

    // register
    Mov(Register, Register),
    Init(Register, Word),

    // integer
    IAdd(Register, Register, Register),
    ISub(Register, Register, Register),
    IMul(Register, Register, Register),
    IDiv(Register, Register, Register),

    // control flow
    Jump(Block),
    JumpNZ(Block, Register),
    Branch(Block, Block, Register),
    Exit(Register),

    // debug
    Comment(String),
}

impl std::fmt::Display for Instruction {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Push(src) => write!(f, "push {}", src),
            Self::Pop(dst) => write!(f, "pop {}", dst),
            Self::StackAddr(dst, offset) => write!(f, "saddr {} {}", dst, offset),
            Self::Read(dst, src) => write!(f, "read {} {}", dst, src),
            Self::Write(dst, src) => write!(f, "write {} {}", dst, src),
            Self::Mov(dst, src) => write!(f, "mov {} {}", dst, src),
            Self::Init(dst, src) => write!(f, "init {} {}", dst, src),
            Self::IAdd(dst, lhs, rhs) => write!(f, "iadd {} {} {}", dst, lhs, rhs),
            Self::ISub(dst, lhs, rhs) => write!(f, "isub {} {} {}", dst, lhs, rhs),
            Self::IMul(dst, lhs, rhs) => write!(f, "imul {} {} {}", dst, lhs, rhs),
            Self::IDiv(dst, lhs, rhs) => write!(f, "idiv {} {} {}", dst, lhs, rhs),
            Self::Jump(block) => write!(f, "jmp {}", block),
            Self::JumpNZ(block, src) => write!(f, "jmpnz {} {}", block, src),
            Self::Branch(block_true, block_false, src) => {
                write!(f, "br {} {} {}", block_true, block_false, src)
            }
            Self::Exit(src) => write!(f, "exit {}", src),
            Self::Comment(comment) => write!(f, "// {}", comment),
        }
    }
}

#[derive(Clone)]
pub struct Runtime {
    pub block: Block,
    pub inst: usize,
    pub registers: [Word; 256],
    pub stack_pointer: usize,
    pub memory: Vec<u8>,
}

impl Runtime {
    #[inline]
    pub fn new() -> Self {
        Self {
            block: Block::from_u64(0),
            inst: 0,
            registers: [Word::ZERO; 256],
            stack_pointer: 0,
            memory: Vec::new(),
        }
    }

    #[inline]
    pub fn read_reg(&self, register: Register) -> Word {
        self.registers[register.0 as usize]
    }

    #[inline]
    pub fn write_reg(&mut self, register: Register, word: Word) {
        self.registers[register.0 as usize] = word;
    }

    #[inline]
    pub fn read_mem(&mut self, address: usize) -> Word {
        Word::from_bytes(&self.memory[address..address + 8])
    }

    #[inline]
    pub fn write_mem(&mut self, address: usize, word: Word) {
        let upper = address + 8;

        if self.memory.len() < upper {
            self.memory.resize(upper, 0);
        }

        self.memory[address..address + 8].copy_from_slice(word.as_bytes())
    }

    #[inline]
    pub fn push(&mut self, word: Word) {
        self.write_mem(self.stack_pointer, word);

        self.stack_pointer += 8;
    }

    #[inline]
    pub fn eval(&mut self, inst: &Instruction) -> Option<i32> {
        match *inst {
            Instruction::Push(src) => {
                self.push(self.read_reg(src));
            }
            Instruction::Pop(dst) => {
                self.stack_pointer -= 8;

                let val = self.read_mem(self.stack_pointer);

                self.write_reg(dst, val);
            }
            Instruction::StackAddr(dst, offset) => {
                let addr = self.stack_pointer as i64 + offset;

                self.write_reg(dst, Word::from_u64(addr as u64));
            }
            Instruction::Read(dst, src) => {
                let ptr = self.read_reg(src);
                let word = self.read_mem(ptr.as_u64() as usize);
                self.write_reg(dst, word);
            }
            Instruction::Write(dst, src) => {
                let ptr = self.read_reg(dst);
                let word = self.read_reg(src);
                self.write_mem(ptr.as_u64() as usize, word);
            }
            Instruction::Mov(dst, src) => {
                let word = self.read_reg(src);

                self.write_reg(dst, word);
            }
            Instruction::Init(dst, val) => {
                self.write_reg(dst, val);
            }
            Instruction::IAdd(dst, lhs, rhs) => {
                let res = self.read_reg(lhs).as_i64() + self.read_reg(rhs).as_i64();

                self.write_reg(dst, Word::from_i64(res));
            }
            Instruction::ISub(dst, lhs, rhs) => {
                let res = self.read_reg(lhs).as_i64() - self.read_reg(rhs).as_i64();

                self.write_reg(dst, Word::from_i64(res));
            }
            Instruction::IMul(dst, lhs, rhs) => {
                let res = self.read_reg(lhs).as_i64() * self.read_reg(rhs).as_i64();

                self.write_reg(dst, Word::from_i64(res));
            }
            Instruction::IDiv(dst, lhs, rhs) => {
                let res = self.read_reg(lhs).as_i64() / self.read_reg(rhs).as_i64();

                self.write_reg(dst, Word::from_i64(res));
            }
            Instruction::Jump(block) => {
                self.block = block;
                self.inst = 0;
            }
            Instruction::JumpNZ(block, src) => {
                if self.read_reg(src).as_i64() != 0 {
                    self.block = block;
                    self.inst = 0;
                }
            }
            Instruction::Branch(block_true, block_false, src) => {
                if self.read_reg(src).as_bool() {
                    self.block = block_true;
                    self.inst = 0;
                } else {
                    self.block = block_false;
                    self.inst = 0;
                }
            }
            Instruction::Exit(src) => {
                let exit_code = self.read_reg(src);

                return Some(exit_code.as_i32());
            }
            Instruction::Comment(_) => {}
        }

        None
    }

    #[inline]
    pub fn run(&mut self, program: &JitProgram) -> i32 {
        self.block = program.main_block;
        self.inst = 0;

        loop {
            let inst = &program.blocks[&self.block].instructions[self.inst];
            self.inst += 1;

            let exit_code = self.eval(inst);

            if let Some(code) = exit_code {
                break code;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct JitProgram {
    pub blocks: BTreeMap<Block, CompiledBlock>,
    pub main_block: Block,
}

impl JitProgram {
    #[inline]
    pub fn compile(source: &str) -> Result<Self, anyhow::Error> {
        let program = AstProgram::parse(&source)?;

        let mut compiler = Compiler::new();

        let main = compiler.create_block("main");
        compiler.set_block(main);

        let mut scope = Scope::new();

        for stmt in program.stmts.iter() {
            compiler.compile_stmt(stmt, &mut scope)?;
        }

        compiler.pop_stack(&mut scope);

        let mut registers = compiler.registers();
        let reg = registers.alloc().expect("ran out of registers");

        // exit program
        compiler.ins(Instruction::Comment(String::from("exit program")));
        compiler.ins(Instruction::Init(reg, Word::from_i64(0)));
        compiler.ins(Instruction::Exit(reg));

        Ok(Self {
            blocks: compiler.blocks,
            main_block: main,
        })
    }

    #[inline]
    pub fn dump(&self) {
        for block in self.blocks.values() {
            println!("{}:", block.label);

            for instruction in &block.instructions {
                println!("\t{}", instruction);
            }
        }
    }
}
