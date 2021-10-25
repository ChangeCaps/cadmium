use std::{fs::read_to_string, sync::Arc};

use cadmium::*;

fn main() {
    let source = read_to_string("test.cm").unwrap();

    let mut module = Module::new();
    module.insert_function("println", ModFn::new(0, vec![Type::I32], Type::Void));
    let program = JitProgram::compile(&source, module).unwrap();

    program.dump();

    let mut runtime = Runtime::new();
    runtime.externals.insert(
        0,
        Arc::new(|_, args| -> Word {
            println!("ext: {}", args[0]);

            Word::ZERO
        }),
    );

    let code = runtime.run(&program);

    std::process::exit(code as i32);
}
