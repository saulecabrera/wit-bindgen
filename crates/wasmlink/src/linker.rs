use crate::{
    adapter::{
        ModuleAdapter, FUNCTION_TABLE_NAME, PARENT_MODULE_NAME, REALLOC_EXPORT_NAME,
        REALLOC_FUNC_TYPE, RUNTIME_MODULE_NAME,
    },
    Module, Profile,
};
use anyhow::{anyhow, bail, Result};
use petgraph::{algo::toposort, graph::NodeIndex, Graph};
use std::{collections::{hash_map::Entry, HashMap}, hash::Hasher};
use wasmparser::{ExternalKind, FuncType, ImportSectionEntryType, Type, TypeDef};

pub const CANONICAL_ABI_MODULE_NAME: &str = "canonical_abi";

pub fn to_val_type(ty: &Type) -> wasm_encoder::ValType {
    match ty {
        Type::I32 => wasm_encoder::ValType::I32,
        Type::I64 => wasm_encoder::ValType::I64,
        Type::F32 => wasm_encoder::ValType::F32,
        Type::F64 => wasm_encoder::ValType::F64,
        Type::V128 => wasm_encoder::ValType::V128,
        Type::FuncRef => wasm_encoder::ValType::FuncRef,
        Type::ExternRef => wasm_encoder::ValType::ExternRef,
        Type::ExnRef | Type::Func | Type::EmptyBlockType => {
            unimplemented!("unsupported value type")
        }
    }
}

/// Represents a linked module built from a dependency graph.
#[derive(Default)]
struct LinkedModule<'a> {
    types: Vec<&'a FuncType>,
    imports: Vec<(&'a str, Option<&'a str>, wasm_encoder::EntityType)>,
    implicit_instances: HashMap<&'a str, u32>,
    modules: Vec<wasm_encoder::Module>,
    module_map: HashMap<&'a ModuleAdapter<'a>, (u32, Option<u32>)>,
    instances: Vec<(u32, Vec<(&'a str, u32)>)>,
    func_aliases: Vec<(u32, &'a str)>,
    memory_aliases: Vec<(u32, &'a str)>,
    table_aliases: Vec<(u32, &'a str)>,
    segments: Vec<(u32, Vec<wasm_encoder::Element>)>,
    exports: Vec<(&'a str, wasm_encoder::Export)>,
}

impl<'a> LinkedModule<'a> {
    /// Creates a new linked module from the given dependency graph.
    fn new(
        graph: &'a Graph<ModuleAdapter<'a>, ()>,
        needs_runtime: bool,
        profile: &Profile,
    ) -> Result<Self> {
        let mut linked = Self::default();

        let mut types = HashMap::new();
        let mut profile_imports = HashMap::new();
        for f in graph.node_indices() {
            let adapter = &graph[f];

            // Add all profile imports to the base set of types and imports
            for import in &adapter.module.imports {
                let ty = adapter
                    .module
                    .import_func_type(import)
                    .expect("expected import to be a function");

                if !profile.provides(import.module, import.field, ty) {
                    continue;
                }

                let type_index = *types.entry(ty).or_insert_with(|| {
                    let index = linked.types.len() as u32;
                    linked.types.push(ty);
                    index
                });

                match profile_imports.insert((import.module, import.field), type_index) {
                    Some(previous) => {
                        if previous != type_index {
                            bail!(
                                "profile import `{}` from module `{}` has a conflicting type between different importing modules",
                                import.field.unwrap_or(""),
                                import.module
                            );
                        }
                    }
                    None => {
                        linked.imports.push((
                            import.module,
                            import.field,
                            wasm_encoder::EntityType::Function(type_index),
                        ));

                        let len = linked.implicit_instances.len();
                        linked
                            .implicit_instances
                            .entry(import.module)
                            .or_insert(len as u32);
                    }
                }
            }

            let module_index = linked.modules.len() as u32;
            linked.modules.push(adapter.adapt()?);

            let shim_index = adapter.encode_shim().map(|m| {
                let index = linked.modules.len() as u32;
                linked.modules.push(m);
                index
            });

            linked
                .module_map
                .insert(adapter, (module_index, shim_index));
        }

        if needs_runtime {
            let bytes = include_bytes!(env!("RUNTIME_WASM_PATH"));
            let module = Module::new(RUNTIME_MODULE_NAME, bytes, [])?;
            let index = linked.modules.len() as u32;
            linked.modules.push(module.encode());
            let mut args = Vec::new();
            for (name, index) in &linked.implicit_instances {
                args.push((*name, *index));
            }
            assert!(linked.instances.is_empty());
            linked.instances.push((index, args));
        }

        // Instantiate the root module
        let (root_index, _) = linked.instantiate(graph, NodeIndex::new(0), None)?;

        let root = &graph[NodeIndex::new(0)];

        // Re-export all supported exports from the root module
        for export in &root.module.exports {
            match export.kind {
                ExternalKind::Function => {
                    let func_index = linked.func_aliases.len() as u32;
                    linked.func_aliases.push((root_index, export.field));
                    linked.exports.push((
                        export.field,
                        wasm_encoder::Export::Function(linked.imports.len() as u32 + func_index),
                    ));
                }
                ExternalKind::Memory => {
                    let memory_index = linked.memory_aliases.len() as u32;
                    linked.memory_aliases.push((root_index, export.field));
                    linked
                        .exports
                        .push((export.field, wasm_encoder::Export::Memory(memory_index)));
                }
                _ => {}
            }
        }

        Ok(linked)
    }

    fn instantiate(
        &mut self,
        graph: &'a Graph<ModuleAdapter<'a>, ()>,
        current: NodeIndex,
        parent: Option<u32>,
    ) -> Result<(u32, bool)> {
        // TODO: make this iterative instead of recursive?

        // If a parent module was specified and this is a shim module, just instantiate it
        let (module_index, shim_index) = self.module_map[&graph[current]];
        if parent.is_none() {
            // Instantiate shims for adapted modules
            if let Some(shim_index) = shim_index {
                let index = (self.instances.len() + self.implicit_instances.len()) as u32;
                self.instances.push((shim_index, Vec::new()));
                return Ok((index, true));
            }
        }

        // Add the implicit instances to the instantiation args
        let mut args = Vec::new();
        for (name, index) in &self.implicit_instances {
            args.push((*name, *index));
        }

        // Add the parent instance
        if let Some(parent) = parent {
            args.push((PARENT_MODULE_NAME, parent));
        }

        // If the module has resources, import the runtime module
        if graph[current].module.has_resources {
            args.push((RUNTIME_MODULE_NAME, self.implicit_instances.len() as u32));
        }

        // Recurse on each direct dependency in the graph
        let mut shims = Vec::new();
        let mut neighbors = graph.neighbors(current).detach();
        while let Some(neighbor) = neighbors.next_node(graph) {
            let (index, is_shim) = self.instantiate(graph, neighbor, None)?;
            let neighbor_adapter = &graph[neighbor];

            args.push((neighbor_adapter.module.name, index));

            // If the adapted module has resources, import it as the canonical ABI module too
            if neighbor_adapter.module.has_resources {
                args.push((CANONICAL_ABI_MODULE_NAME, index));
            }

            if is_shim {
                shims.push((neighbor, index));
            }
        }

        // Instantiate the current module
        let parent_index = (self.instances.len() + self.implicit_instances.len()) as u32;
        self.instances.push((module_index, args));

        // If there are shims, ensure the parent exports the required realloc function
        if !shims.is_empty() {
            let adapter = &graph[current];

            let export = adapter
                .module
                .exports
                .iter()
                .find(|e| {
                    e.field == REALLOC_EXPORT_NAME && matches!(e.kind, ExternalKind::Function)
                })
                .ok_or_else(|| {
                    anyhow!(
                        "module `{}` does not export the required function `{}`",
                        adapter.module.name,
                        REALLOC_EXPORT_NAME
                    )
                })?;

            if adapter
                .module
                .func_type(export.index)
                .expect("function index must be in range")
                != &REALLOC_FUNC_TYPE as &FuncType
            {
                bail!(
                    "module `{}` exports function `{}` but it is not the expected type",
                    adapter.module.name,
                    REALLOC_EXPORT_NAME
                );
            }
        }

        // For each shim that was instantiated, instantiate the real module passing in the parent
        for (shim, shim_index) in shims {
            let (child_index, _) = self.instantiate(graph, shim, Some(parent_index))?;

            // Emit the shim function table
            let adapter = &graph[shim];
            let table_index = self.table_aliases.len() as u32;
            self.table_aliases.push((shim_index, FUNCTION_TABLE_NAME));

            // Emit the segments populating the function table
            let mut segments = Vec::new();
            for name in adapter.aliases() {
                let func_index = self.imports.len() as u32 + self.func_aliases.len() as u32;
                self.func_aliases.push((child_index, name));
                segments.push(wasm_encoder::Element::Func(func_index));
            }

            self.segments.push((table_index, segments));
        }

        Ok((parent_index, false))
    }

    fn encode(&self) -> wasm_encoder::Module {
        let mut module = wasm_encoder::Module::new();

        self.write_type_section(&mut module);
        self.write_import_section(&mut module);
        self.write_module_section(&mut module);
        self.write_instance_section(&mut module);
        self.write_alias_section(&mut module);
        self.write_export_section(&mut module);
        self.write_element_section(&mut module);

        module
    }

    fn write_type_section(&self, module: &mut wasm_encoder::Module) {
        let mut section = wasm_encoder::TypeSection::new();

        for ty in &self.types {
            section.function(
                ty.params.iter().map(to_val_type),
                ty.returns.iter().map(to_val_type),
            );
        }

        module.section(&section);
    }

    fn write_import_section(&self, module: &mut wasm_encoder::Module) {
        let mut section = wasm_encoder::ImportSection::new();

        for (module, field, ty) in &self.imports {
            section.import(module, *field, *ty);
        }

        module.section(&section);
    }

    fn write_module_section(&self, module: &mut wasm_encoder::Module) {
        let mut section = wasm_encoder::ModuleSection::new();

        for module in &self.modules {
            section.module(module);
        }

        module.section(&section);
    }

    fn write_instance_section(&self, module: &mut wasm_encoder::Module) {
        let mut section = wasm_encoder::InstanceSection::new();

        for (module, args) in &self.instances {
            section.instantiate(
                *module,
                args.iter()
                    .map(|(name, index)| (*name, wasm_encoder::Export::Instance(*index))),
            );
        }

        module.section(&section);
    }

    fn write_alias_section(&self, module: &mut wasm_encoder::Module) {
        let mut section = wasm_encoder::AliasSection::new();

        for (index, name) in &self.func_aliases {
            section.instance_export(*index, wasm_encoder::ItemKind::Function, name);
        }

        for (index, name) in &self.table_aliases {
            section.instance_export(*index, wasm_encoder::ItemKind::Table, name);
        }

        for (index, name) in &self.memory_aliases {
            section.instance_export(*index, wasm_encoder::ItemKind::Memory, name);
        }

        module.section(&section);
    }

    fn write_export_section(&self, module: &mut wasm_encoder::Module) {
        let mut section = wasm_encoder::ExportSection::new();

        for (name, export) in &self.exports {
            section.export(name, *export);
        }

        module.section(&section);
    }

    fn write_element_section(&self, module: &mut wasm_encoder::Module) {
        let mut section = wasm_encoder::ElementSection::new();

        for (table_index, elements) in &self.segments {
            section.active(
                Some(*table_index),
                wasm_encoder::Instruction::I32Const(0),
                wasm_encoder::ValType::FuncRef,
                wasm_encoder::Elements::Expressions(elements),
            );
        }

        module.section(&section);
    }
}

/// Implements a WebAssembly module linker.
#[derive(Debug)]
pub struct Linker {
    profile: Profile,
}

impl Linker {
    /// Constructs a new WebAssembly module linker with the given profile.
    pub fn new(profile: Profile) -> Self {
        Self { profile }
    }

    /// Links the given module with the given set of imported modules.
    ///
    /// On success, returns a vector of bytes representing the linked module.
    pub fn link(&self, module: &Module, imports: &HashMap<&str, Module>) -> Result<Vec<u8>> {
        let (graph, needs_runtime) = self.build_graph(module, imports)?;

        let module = LinkedModule::new(&graph, needs_runtime, &self.profile)?;

        Ok(module.encode().finish())
    }

    /// Dynamically links modules
    pub fn dylink(&self, module: &Module) -> Result<Vec<u8>> {
        let dynamic_adapter = crate::dy::DyAdapter::new(&module);
        let module = dynamic_adapter.adapt()?;
        let bytes = module.finish();
        let pretty = wasmprinter::print_bytes(&bytes)?;
        std::fs::write("dyn.wat", pretty)?;
        std::fs::write("dyn.wasm", &bytes)?;
        Ok(bytes)
    }

    fn build_graph<'a>(
        &self,
        module: &'a Module,
        imports: &'a HashMap<&str, Module>,
    ) -> Result<(Graph<ModuleAdapter<'a>, ()>, bool)> {
        let mut queue: Vec<(Option<petgraph::graph::NodeIndex>, &Module)> = Vec::new();
        let mut seen = HashMap::new();
        let mut graph: Graph<ModuleAdapter, ()> = Graph::new();

        let mut needs_runtime = module.has_resources;

        queue.push((None, module));

        let mut next_resource_id = 0;

        while let Some((predecessor, module)) = queue.pop() {
            let index = match seen.entry(module as *const _) {
                Entry::Occupied(e) => *e.get(),
                Entry::Vacant(e) => {
                    needs_runtime |= module.has_resources;
                    let index = graph.add_node(ModuleAdapter::new(module, &mut next_resource_id));

                    for import in &module.imports {
                        let imported_module = imports.get(import.module);

                        // Check for profile provided function imports before resolving exports on the imported module
                        if let ImportSectionEntryType::Function(i) = &import.ty {
                            match module
                                .types
                                .get(*i as usize)
                                .expect("function index must be in range")
                            {
                                TypeDef::Func(ft) => {
                                    if import.module == CANONICAL_ABI_MODULE_NAME
                                        || self.profile.provides(import.module, import.field, ft)
                                    {
                                        continue;
                                    }
                                }
                                _ => unreachable!("import must be a function"),
                            }
                        }

                        let imported_module = imported_module.ok_or_else(|| {
                            anyhow!(
                                "module `{}` imports from unknown module `{}`",
                                module.name,
                                import.module
                            )
                        })?;

                        imported_module.resolve_import(import, module)?;

                        queue.push((Some(index), imported_module));
                    }

                    *e.insert(index)
                }
            };

            if let Some(predecessor) = predecessor {
                if !graph.contains_edge(predecessor, index) {
                    graph.add_edge(predecessor, index, ());
                }
            };
        }

        // Ensure the graph is acyclic by performing a topographical sort.
        // This algorithm requires more space than `is_cyclic_directed`, but
        // performs the check iteratively rather than recursively.
        toposort(&graph, None).map_err(|e| {
            anyhow!(
                "module `{}` and its imports form a cycle in the import graph",
                graph[e.node_id()].module.name
            )
        })?;

        Ok((graph, needs_runtime))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn it_errors_on_missing_import() -> Result<()> {
        let bytes = wat::parse_str(
            r#"(module (import "unknown" "import" (func)) (func (export "_start")))"#,
        )?;
        let main = Module::new("main", &bytes, [])?;

        let linker = Linker::new(Profile::new());

        assert_eq!(
            linker.link(&main, &HashMap::new()).unwrap_err().to_string(),
            "module `main` imports from unknown module `unknown`"
        );

        Ok(())
    }

    #[test]
    fn it_errors_on_an_import_with_missing_export() -> Result<()> {
        let bytes = wat::parse_str(r#"(module (import "a" "a" (func)) (func (export "_start")))"#)?;
        let a = wat::parse_str(r#"(module (import "b" "b" (func)))"#)?;

        let main = Module::new("main", &bytes, [])?;

        let mut imports = HashMap::new();
        imports.insert("a", Module::new("a", &a, [])?);

        let linker = Linker::new(Profile::new());

        assert_eq!(
            linker.link(&main, &imports).unwrap_err().to_string(),
            "module `a` does not export a function named `a`"
        );

        Ok(())
    }

    #[test]
    fn it_errors_on_an_import_with_export_mismatch() -> Result<()> {
        let bytes = wat::parse_str(r#"(module (import "a" "a" (func)) (func (export "_start")))"#)?;
        let a = wat::parse_str(r#"(module (import "b" "b" (func)) (memory (export "a") 0))"#)?;

        let main = Module::new("main", &bytes, [])?;

        let mut imports = HashMap::new();
        imports.insert("a", Module::new("a", &a, [])?);

        let linker = Linker::new(Profile::new());

        assert_eq!(
            linker.link(&main, &imports).unwrap_err().to_string(),
            "expected a function for export `a` from module `a` but found a memory"
        );

        Ok(())
    }

    #[test]
    fn it_errors_on_an_import_with_export_signature_mismatch() -> Result<()> {
        let bytes = wat::parse_str(r#"(module (import "a" "a" (func)) (func (export "_start")))"#)?;
        let a =
            wat::parse_str(r#"(module (import "b" "b" (func)) (func (export "a") (param i32)))"#)?;

        let main = Module::new("main", &bytes, [])?;

        let mut imports = HashMap::new();
        imports.insert("a", Module::new("a", &a, [])?);

        let linker = Linker::new(Profile::new());

        assert_eq!(
            linker.link(&main, &imports).unwrap_err().to_string(),
            "module `main` imports function `a` from module `a` but the types are incompatible"
        );

        Ok(())
    }

    #[test]
    fn it_errors_on_an_import_cycle() -> Result<()> {
        let bytes = wat::parse_str(r#"(module (import "a" "a" (func)) (func (export "_start")))"#)?;
        let a = wat::parse_str(r#"(module (import "b" "b" (func)) (func (export "a")))"#)?;
        let b = wat::parse_str(r#"(module (import "c" "c" (func)) (func (export "b")))"#)?;
        let c = wat::parse_str(r#"(module (import "a" "a" (func)) (func (export "c")))"#)?;

        let main = Module::new("main", &bytes, [])?;

        let mut imports = HashMap::new();
        imports.insert("a", Module::new("a", &a, [])?);
        imports.insert("b", Module::new("b", &b, [])?);
        imports.insert("c", Module::new("c", &c, [])?);

        let linker = Linker::new(Profile::new());

        assert_eq!(
            linker.link(&main, &imports).unwrap_err().to_string(),
            "module `c` and its imports form a cycle in the import graph"
        );

        Ok(())
    }

    #[test]
    fn it_errors_on_incompatible_profile_imports() -> Result<()> {
        let bytes = wat::parse_str(
            r#"(module (import "a" "a" (func)) (import "b" "b" (func)) (func (export "_start")))"#,
        )?;
        let a = wat::parse_str(
            r#"(module (import "wasi_snapshot_preview1" "c" (func)) (func (export "a")))"#,
        )?;
        let b = wat::parse_str(
            r#"(module (import "wasi_snapshot_preview1" "c" (func (param i32))) (func (export "b")))"#,
        )?;

        let main = Module::new("main", &bytes, [])?;

        let mut imports = HashMap::new();
        imports.insert("a", Module::new("a", &a, [])?);
        imports.insert("b", Module::new("b", &b, [])?);

        let linker = Linker::new(Profile::new());

        assert_eq!(
            linker.link(&main, &imports).unwrap_err().to_string(),
            "profile import `c` from module `wasi_snapshot_preview1` has a conflicting type between different importing modules"
        );

        Ok(())
    }

    #[test]
    fn it_links() -> Result<()> {
        let bytes = wat::parse_str(r#"(module (import "a" "a" (func)) (func (export "_start")))"#)?;
        let a = wat::parse_str(
            r#"(module (import "wasi_snapshot_preview1" "a" (func)) (func (export "a")))"#,
        )?;

        let main = Module::new("main", &bytes, [])?;

        let mut imports = HashMap::new();
        imports.insert("a", Module::new("a", &a, [])?);

        let linker = Linker::new(Profile::new());

        let bytes = linker.link(&main, &imports)?;

        assert_eq!(
            wasmprinter::print_bytes(&bytes)?,
            "\
(module
  (type (;0;) (func))
  (import \"wasi_snapshot_preview1\" \"a\" (func (;0;) (type 0)))
  (module (;0;)
    (type (;0;) (func))
    (import \"a\" \"a\" (func (;0;) (type 0)))
    (func (;1;) (type 0))
    (export \"_start\" (func 1)))
  (module (;1;)
    (type (;0;) (func))
    (import \"wasi_snapshot_preview1\" \"a\" (func (;0;) (type 0)))
    (func (;1;) (type 0))
    (export \"a\" (func 1)))
  (instance (;1;)
    (instantiate 1
      (import \"wasi_snapshot_preview1\" (instance 0))))
  (instance (;2;)
    (instantiate 0
      (import \"wasi_snapshot_preview1\" (instance 0))
      (import \"a\" (instance 1))))
  (alias 2 \"_start\" (func (;1;)))
  (export \"_start\" (func 1)))"
        );

        Ok(())
    }

    #[test]
    fn it_errors_with_missing_parent_realloc() -> Result<()> {
        let bytes = wat::parse_str(
            r#"(module (import "a" "a" (func (param i32 i32))) (func (export "_start")))"#,
        )?;
        let a = wat::parse_str(
            r#"(module (import "wasi_snapshot_preview1" "a" (func)) (func (export "a") (param i32 i32)) (memory (export "memory") 0) (func (export "canonical_abi_realloc") (param i32 i32 i32 i32) (result i32) unreachable) (func (export "canonical_abi_free") (param i32 i32 i32)))"#,
        )?;

        let main = Module::new("main", &bytes, [])?;
        let a = Module::new(
            "a",
            &a,
            [wit_parser::Interface::parse("a", "a: function(p: string)")?],
        )?;

        let mut imports = HashMap::new();
        imports.insert("a", a);

        let linker = Linker::new(Profile::new());

        assert_eq!(
            linker.link(&main, &imports).unwrap_err().to_string(),
            "module `main` does not export the required function `canonical_abi_realloc`"
        );

        Ok(())
    }

    #[test]
    fn it_errors_with_incorrect_parent_realloc() -> Result<()> {
        let bytes = wat::parse_str(
            r#"(module (import "a" "a" (func (param i32 i32))) (func (export "_start")) (func (export "canonical_abi_realloc") (param i32 i32 i32 i32)))"#,
        )?;
        let a = wat::parse_str(
            r#"(module (import "wasi_snapshot_preview1" "a" (func)) (func (export "a") (param i32 i32)) (memory (export "memory") 0) (func (export "canonical_abi_realloc") (param i32 i32 i32 i32) (result i32) unreachable) (func (export "canonical_abi_free") (param i32 i32 i32)))"#,
        )?;

        let main = Module::new("main", &bytes, [])?;
        let a = Module::new(
            "a",
            &a,
            [wit_parser::Interface::parse("a", "a: function(p: string)")?],
        )?;

        let mut imports = HashMap::new();
        imports.insert("a", a);

        let linker = Linker::new(Profile::new());

        assert_eq!(
            linker.link(&main, &imports).unwrap_err().to_string(),
            "module `main` exports function `canonical_abi_realloc` but it is not the expected type"
        );

        Ok(())
    }

    #[test]
    fn it_links_with_interface() -> Result<()> {
        let bytes = wat::parse_str(
            r#"(module (import "a" "a" (func (param i32 i32))) (func (export "_start")) (func (export "canonical_abi_realloc") (param i32 i32 i32 i32) (result i32) unreachable) (memory (export "memory") 0))"#,
        )?;
        let a = wat::parse_str(
            r#"(module (import "wasi_snapshot_preview1" "a" (func)) (func (export "a") (param i32 i32)) (memory (export "memory") 0) (func (export "canonical_abi_realloc") (param i32 i32 i32 i32) (result i32) unreachable) (func (export "canonical_abi_free") (param i32 i32 i32)))"#,
        )?;

        let main = Module::new("main", &bytes, [])?;
        let a = Module::new(
            "a",
            &a,
            [wit_parser::Interface::parse("a", "a: function(p: string)")?],
        )?;

        let mut imports = HashMap::new();
        imports.insert("a", a);

        let linker = Linker::new(Profile::new());

        let bytes = linker.link(&main, &imports)?;

        assert_eq!(
            wasmprinter::print_bytes(&bytes)?,
            "\
(module
  (type (;0;) (func))
  (import \"wasi_snapshot_preview1\" \"a\" (func (;0;) (type 0)))
  (module (;0;)
    (type (;0;) (func (param i32 i32)))
    (type (;1;) (func))
    (type (;2;) (func (param i32 i32 i32 i32) (result i32)))
    (import \"a\" \"a\" (func (;0;) (type 0)))
    (func (;1;) (type 1))
    (func (;2;) (type 2) (param i32 i32 i32 i32) (result i32)
      unreachable)
    (memory (;0;) 0)
    (export \"_start\" (func 1))
    (export \"canonical_abi_realloc\" (func 2))
    (export \"memory\" (memory 0)))
  (module (;1;)
    (type (;0;) (func))
    (type (;1;) (func (param i32 i32)))
    (type (;2;) (func (param i32 i32 i32 i32) (result i32)))
    (import \"wasi_snapshot_preview1\" \"a\" (func (;0;) (type 0)))
    (import \"$parent\" \"memory\" (memory (;0;) 0))
    (import \"$parent\" \"canonical_abi_realloc\" (func (;1;) (type 2)))
    (module (;0;)
      (type (;0;) (func))
      (type (;1;) (func (param i32 i32)))
      (type (;2;) (func (param i32 i32 i32 i32) (result i32)))
      (type (;3;) (func (param i32 i32 i32)))
      (import \"wasi_snapshot_preview1\" \"a\" (func (;0;) (type 0)))
      (func (;1;) (type 1) (param i32 i32))
      (func (;2;) (type 2) (param i32 i32 i32 i32) (result i32)
        unreachable)
      (func (;3;) (type 3) (param i32 i32 i32))
      (memory (;0;) 0)
      (export \"a\" (func 1))
      (export \"memory\" (memory 0))
      (export \"canonical_abi_realloc\" (func 2))
      (export \"canonical_abi_free\" (func 3)))
    (instance (;2;)
      (instantiate 0
        (import \"wasi_snapshot_preview1\" (instance 0))))
    (alias 2 \"memory\" (memory (;1;)))
    (alias 2 \"canonical_abi_realloc\" (func (;2;)))
    (alias 2 \"canonical_abi_free\" (func (;3;)))
    (alias 2 \"a\" (func (;4;)))
    (func (;5;) (type 1) (param i32 i32)
      (local i32)
      block  ;; label = @1
        i32.const 0
        i32.const 0
        i32.const 1
        local.get 1
        call 2
        local.tee 2
        br_if 0 (;@1;)
        unreachable
      end
      local.get 2
      local.get 0
      local.get 1
      memory.copy 1 0
      local.get 2
      local.get 1
      call 4)
    (export \"memory\" (memory 1))
    (export \"canonical_abi_realloc\" (func 2))
    (export \"canonical_abi_free\" (func 3))
    (export \"a\" (func 5)))
  (module (;2;)
    (type (;0;) (func (param i32 i32)))
    (func (;0;) (type 0) (param i32 i32)
      local.get 0
      local.get 1
      i32.const 0
      call_indirect (type 0))
    (table (;0;) 1 1 funcref)
    (export \"a\" (func 0))
    (export \"$funcs\" (table 0)))
  (instance (;1;)
    (instantiate 2))
  (instance (;2;)
    (instantiate 0
      (import \"wasi_snapshot_preview1\" (instance 0))
      (import \"a\" (instance 1))))
  (instance (;3;)
    (instantiate 1
      (import \"wasi_snapshot_preview1\" (instance 0))
      (import \"$parent\" (instance 2))))
  (alias 3 \"a\" (func (;1;)))
  (alias 2 \"_start\" (func (;2;)))
  (alias 2 \"canonical_abi_realloc\" (func (;3;)))
  (alias 1 \"$funcs\" (table (;0;)))
  (alias 2 \"memory\" (memory (;0;)))
  (export \"_start\" (func 2))
  (export \"canonical_abi_realloc\" (func 3))
  (export \"memory\" (memory 0))
  (elem (;0;) (i32.const 0) funcref (ref.func 1)))"
        );

        Ok(())
    }
}
