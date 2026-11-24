use std::env;
use std::process::ExitCode;

const HELP: &str = "Quantum Morph Laboratory (qml)\n\nUSAGE:\n    qml <COMMAND>\n\nCOMMANDS:\n    doctor      Show basic host platform diagnostics\n    help        Show this help text\n    version     Show the CLI version\n\nModel inspection, transformation, benchmarking, and inference commands are not implemented yet.";

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("doctor") if args.next().is_none() => doctor(),
        Some("help" | "--help" | "-h") if args.next().is_none() => {
            println!("{HELP}");
            ExitCode::SUCCESS
        }
        Some("version" | "--version" | "-V") if args.next().is_none() => {
            println!("qml {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some(command) => {
            eprintln!("unknown command or unexpected argument: {command}\n\n{HELP}");
            ExitCode::from(2)
        }
        None => {
            println!("{HELP}");
            ExitCode::SUCCESS
        }
    }
}

fn doctor() -> ExitCode {
    println!("QML host diagnostics (basic)");
    println!("  OS: {}", env::consts::OS);
    println!("  Architecture: {}", env::consts::ARCH);
    match std::thread::available_parallelism() {
        Ok(count) => println!("  Available logical parallelism: {}", count.get()),
        Err(error) => println!("  Available logical parallelism: unavailable ({error})"),
    }
    println!("  GPU, memory, disk, and backend discovery: not implemented");
    ExitCode::SUCCESS
}
