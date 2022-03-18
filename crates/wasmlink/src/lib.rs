//! WebAssembly linker API.

#![deny(missing_docs)]

mod adapter;
mod linker;
mod module;
mod profile;
mod resources;
mod dy;

pub use self::adapter::ModuleAdapter;
pub use self::linker::Linker;
pub use self::module::Module;
pub use self::profile::Profile;
