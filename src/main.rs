use std::fs::read_to_string;

use cadmium::*;

fn main() {
    let source = read_to_string("test.cm").unwrap();

    let program = JitProgram::compile(&source).unwrap();

    program.dump();

    let mut runtime = Runtime::new();

    let code = runtime.run(&program);

    std::process::exit(code as i32);
}
