
/// A dynamic adapter to dynamically link Wasm modules
///
///
/// Provider (hello.wasm)
///
/// ```
/// (module
///  ;; exports hello: () -> i32
/// )
/// ```
///
/// Consumer (hello-consumer.wasm)
///
/// ```
/// (module
///  ;; imports hello: (i32) -> ()
/// )
/// ```
///
/// Wit
/// hello: function() -> string
/// 
///
/// If the consumer module linked against the prover module,
/// the instantiation will fail, since the signatures don't
/// match.
///
/// The job of the dynamic adapter is create a top-level adapter module
/// that:
///
/// 1. Encodes the original consumer module
/// 2. Define an adapter functions for each function interface
/// 3. Instantiate the original module with the adapter functions
/// 4. Pass along functions that don't need to be adapted
///
///
/// (module
///   ;; N in the general case, 1 per dependency
///   (type (;0;) (instance
///     (export "hello" (func () (result i32)))
///     ;; export exports of realloc, free
///   ))
///
///   ;; N in the general case, 1 per dependency
///   (import "hello" (instance (;0;) (type 0)))
///
///   ;; Only 1
///   (module (;0;) ;; consumer
///     ;; the original consumer module
///   )
///
///   ;; instantiate 0 with an intermediate module that
///   ;; (i) has no imports
///   ;; (ii) exports a function that matches the consumer signature (i.e.(i32) -> ())
///   ;; (iii) performs an indirect call to a function in that table
///
///   ;; N in the general case, 1 per module dependency
///   ;; calling it proxy
///   (module (;1;) ;; intermediate
///     ;; defines function signatures that match the consumer
///     ;; exports these functions ^
///     ;; defines a a table of type extern ref func
///     ;; each function body will invoke the real function through an indirect_call
///     ;; the func ref table will be populated at the end
///   )
///
///   ;; instace 1 -> module 1 (intermediate)
///   ;; N in the general case, 1 per module dependency
///   (instance (;1;)
///     (instantiate 1))
///
///   ;; instance 2 -> module 0 (consumer)
///   ;; Only 1 instance of the consumer
///   (instance (;2;)
///     (instantiate 0
///       (import "..." (instance 1))
///     )
///   )
///
///   ;; This module glues everything together
///   ;; Import the consumer's memory
///   ;; Define an export functions that match the consumers import
///   ;; the adapter functions will be passed to the proxy module 
///   ;; through its table
///   ;; Must conditionally determine how the canonical abi is used
///   ;; depending on params/returns
///   ;; N in the general case, 1 per module dependency
///   (module (;2;) ;; provider-adapter
///     (type (;0;) (func () (result i32))) ;; original provider function (export type)
///     (type (;1;) (func (param i32))) ;; the adapter function (import type)
///
///     ;; I need to have access to provider's realloc index and consumers realloc index
///
///     (import "hello", "hello", (func (;0;) (type 0)))
///     (import "$consumer", "memory", (memory (;0;) 0)) ;; if module needs memory
///
///     (func (;0;) (type 0) (param i32) ;; (import type)
///       ;; call the original provider function
///       ;; get the return value
///       ;; emit copy instructions (we have access to both memories)
///       ;; the consumer and the provider adapter
///     )
///
///     (export "hello" (func 0)) ;; (exports import type)
///   )
///   
///   ;; N in the general case, 1 per module dependency
///   (instantiate (;3;)
///     (instantiate 2
///       (import "hello" (instance 0))
///       (import "$consumer" (instance 2))))
///
///   (alias 1 "$table" (table (;0;)))
///   (alias 3 "hello" (func 0))
///
///   (elem (;0;) (i32.const 0) funcref (ref.func 0))
/// )
///

use crate::{Module, linker};
use anyhow::Result;
use wasmparser::{FuncType, Type};
use std::collections::HashMap;
use crate::{module::Interface, adapter::call::CallAdapter};

pub const REALLOC_EXPORT_NAME: &str = "canonical_abi_realloc";
pub const FREE_EXPORT_NAME: &str = "canonical_abi_free";
pub const CANONICAL_ABI_MODULE_NAME: &str = "canonical_abi";
pub const CONSUMER_MODULE_NAME: &str = "$consumer";
pub const MEMORY_EXPORT_NAME: &str = "memory";

lazy_static::lazy_static! {
    pub static ref REALLOC_FUNC_TYPE: FuncType = {
        FuncType {
            params: Box::new([Type::I32, Type::I32, Type::I32, Type::I32]),
            returns: Box::new([Type::I32])
        }
    };
    pub static ref FREE_FUNC_TYPE: FuncType = {
        FuncType {
            params: Box::new([Type::I32, Type::I32, Type::I32]),
            returns: Box::new([])
        }
    };
}

pub struct DyAdapter<'a> {
    pub module: &'a Module<'a>,
}

#[derive(Hash, PartialEq, Eq, Debug)]
enum TypeEntry<'a> {
    Instance(String),
    FuncType(&'a FuncType),
    Module
}

#[derive(Hash, PartialEq, Eq, Debug)]
enum ModuleKind<'a> {
    Consumer(&'a str),
    ProviderAdapter(&'a str),
    Proxy(&'a str)
}

#[derive(Hash, PartialEq, Eq, Debug)]
enum InstanceKind<'a> {
    Consumer(&'a str),
    Proxy(&'a str),
    ProviderAdapter(&'a str),
    Provider(&'a str),
    Implicit(&'a str),
}

impl<'a> DyAdapter<'a> {
    pub fn new(module: &'a Module) -> Self {
        Self { module }
    }

    pub fn adapt(&self) -> Result<wasm_encoder::Module> {
        if !self.module.must_adapt {
            return Ok(self.module.encode());
        }

        let mut module = wasm_encoder::Module::new();
        let mut modules: HashMap<ModuleKind, u32> = HashMap::new();
        let mut instance_section = wasm_encoder::InstanceSection::new();
        let mut instances: HashMap<InstanceKind, u32> = HashMap::new();
        let mut types = HashMap::new();
        let mut module_section = wasm_encoder::ModuleSection::new();
        let mut alias_section = wasm_encoder::AliasSection::new();
        let mut export_section = wasm_encoder::ExportSection::new();
        let mut element_section = wasm_encoder::ElementSection::new();
        let mut table_info: HashMap<u32, (u32, Vec<u32>)> = HashMap::new();
        let interface_names: Vec<String> = self.module.interfaces.iter().map
            (|interface| interface.inner().name.clone()).collect();
        let mut imported_fn_count: u32 = 0;

        self.write_type_section(&mut module, &mut types);
        self.write_import_section(&mut module, &mut types, &mut instances, &interface_names, &mut imported_fn_count);

        self.write_module_section(&mut module_section, &mut modules);
        self.write_proxy_modules(&mut modules, &mut module_section);
        self.write_provider_adapter_modules(&mut modules, &mut module_section);

        self.write_proxy_instances(&mut modules, &mut instance_section, &mut instances);
        self.write_consumer_instance(&mut modules, &mut instance_section, &mut instances);
        self.write_provider_adapter_instances(&mut modules, &mut instance_section, &mut instances);
        self.write_alias_section(&mut instances, &mut alias_section, &mut export_section, &mut table_info, imported_fn_count);
        self.write_element_section(&mut element_section, &mut instances, &mut table_info);

        module.section(&module_section);
        module.section(&instance_section);
        module.section(&alias_section);
        module.section(&export_section);
        module.section(&element_section);

        let mut features = wasmparser::WasmFeatures::default();
        features.multi_memory = true;
        features.module_linking = true;

        let mut validator = wasmparser::Validator::new();
        validator.wasm_features(features);

        let module_clone = module.clone();

        let bytes = module_clone.finish();

        println!("{:?}", validator.validate_all(&bytes));

        Ok(module)
    }

    fn write_element_section(&self, elem_section: &mut wasm_encoder::ElementSection, instances: &mut HashMap<InstanceKind, u32>, tables: &mut HashMap<u32, (u32, Vec<u32>)>) {
        let proxies = instances.iter().filter_map(|(kind, idx)| {
            match kind {
                InstanceKind::Proxy(name) => Some((name, idx)),
                _ => None
            }
        });

        for (_, id) in proxies {
            let (table_id, func_ids) = tables.get(id).unwrap();
            let elements: Vec<wasm_encoder::Element> = func_ids.iter().map(|id| {
                wasm_encoder::Element::Func(*id)
            }).collect();

            elem_section.active(Some(*table_id), wasm_encoder::Instruction::I32Const(0), wasm_encoder::ValType::FuncRef, wasm_encoder::Elements::Expressions(&elements));
        }
    }

    fn write_alias_section(&self, instances: &mut HashMap<InstanceKind<'a>, u32>, alias_section: &mut wasm_encoder::AliasSection, export_section: &mut wasm_encoder::ExportSection, tables_hash: &mut HashMap<u32, (u32, Vec<u32>)>, imported_fn_count: u32) {

        let mut fns = imported_fn_count;
        let mut memories = 0;
        let mut tables = 0;
        let mut globals = 0;

        let adapters = instances.iter().filter_map(|(kind, idx)| {
            match kind {
                InstanceKind::ProviderAdapter(name) => Some((name, idx)),
                _ => None
            }
        });

        let proxies = instances.iter().filter_map(|(kind, idx)| {
            match kind {
                InstanceKind::Proxy(name) => Some((name, idx)),
                _ => None
            }
        });

        let mut proxied_fns: HashMap<&&str, Vec<u32>> = adapters.map(|(name, idx)| {
            let iface = self.module.interfaces.iter().find(|iface| iface.inner().name == *name).unwrap();
            let mut proxied_functions = vec![];
            for (f, _) in iface.iter() {
                alias_section.instance_export(*idx, wasm_encoder::ItemKind::Function, &f.name);
                proxied_functions.push(fns);
                fns += 1;
            }
            (name, proxied_functions)
        }).collect();

        for (kind, idx) in proxies {
            alias_section.instance_export(*idx, wasm_encoder::ItemKind::Table, "proxy_funcs");
            let result = proxied_fns.remove(kind).unwrap();
            tables_hash.insert(*idx, (tables, result));
            tables += 1;
        }

        let consumer_instance = instances.iter().find(|(kind, _)| {
            match kind {
                InstanceKind::Consumer(_) => true,
                _ => false
            }
        }).unwrap();

        for e in self.module.exports.iter() {
            let (kind, export) = match e.kind {
                wasmparser::ExternalKind::Function => {
                    let export_info = (wasm_encoder::ItemKind::Function, wasm_encoder::Export::Function(fns));
                    fns += 1;
                    export_info
                },

                wasmparser::ExternalKind::Table => {
                    let export_info = (wasm_encoder::ItemKind::Table, wasm_encoder::Export::Table(tables));
                    tables += 1;
                    export_info
                },
                wasmparser::ExternalKind::Memory => {
                    let export_info = (wasm_encoder::ItemKind::Memory, wasm_encoder::Export::Memory(memories));
                    memories += 1;
                    export_info
                },

                wasmparser::ExternalKind::Global => {
                    let export_info = (wasm_encoder::ItemKind::Global, wasm_encoder::Export::Global(globals));
                    globals += 1;
                    export_info
                },
                _ => unreachable!()
            };

            alias_section.instance_export(*consumer_instance.1, kind, e.field);
            export_section.export(e.field, export);
        }
    }

    fn write_proxy_instances(&self, modules: &mut HashMap<ModuleKind<'a>, u32>, instance_section: &mut wasm_encoder::InstanceSection, instances: &mut HashMap<InstanceKind<'a>, u32>) {
        let proxies = modules.iter().filter_map(|(kind, index)| match kind {
            ModuleKind::Proxy(name) => Some((name, index)),
            _ => None
        });

        for (name, index) in proxies {
            let instance_idx = instances.len() as u32;
            instance_section.instantiate(*index as u32, vec![]);
            instances.entry(InstanceKind::Proxy(name.clone()))
                .or_insert(instance_idx);
        }
    }

    fn write_consumer_instance(&self, modules: &mut HashMap<ModuleKind<'a>, u32>, instance_section: &mut wasm_encoder::InstanceSection, instances: &mut HashMap<InstanceKind<'a>, u32>) { 
        let consumer = modules.iter().find(|(kind, _)| match kind {
            ModuleKind::Consumer(_) => true,
            _ => false
        });

        let mut args = instances.iter().filter_map(|(kind, idx)| match kind {
            InstanceKind::Proxy(name) => Some((*name, wasm_encoder::Export::Instance(*idx))),
            InstanceKind::Implicit(name)=> Some((*name, wasm_encoder::Export::Instance(*idx))),
            _ => None
        }).collect::<Vec<_>>();

        if let Some((_, index)) = consumer {
            let instance_idx = instances.len() as u32;
            instance_section.instantiate(*index as u32, args);
            instances.entry(InstanceKind::Consumer(CONSUMER_MODULE_NAME))
                .or_insert(instance_idx);
        }
    }

    fn write_provider_adapter_instances(&self, modules: &mut HashMap<ModuleKind<'a>, u32>, instance_section: &mut wasm_encoder::InstanceSection, instances: &mut HashMap<InstanceKind<'a>, u32>) {
        let adapters = modules.iter().filter_map(|(kind, index)| match kind {
            ModuleKind::ProviderAdapter(name) => Some((name, index)),
            _ => None
        });

        let mut args = instances.iter().filter_map(|(kind, idx)| match kind {
            InstanceKind::Provider(name) => Some((*name, wasm_encoder::Export::Instance(*idx))),
            _ => None
        }).collect::<Vec<_>>();

        let consumer_instance = instances.iter().find(|(kind, _)| match kind {
            InstanceKind::Consumer(_) => true,
            _ => false
        });

        assert!(consumer_instance.is_some());
        args.push((CONSUMER_MODULE_NAME, wasm_encoder::Export::Instance(*consumer_instance.unwrap().1)));


        for (name, index) in adapters {
            let instance_idx = instances.len() as u32;
            instance_section.instantiate(*index as u32, args.clone());
            instances.entry(InstanceKind::ProviderAdapter(name.clone()))
                .or_insert(instance_idx);
        }
    }

    fn write_type_section(&self, module: &mut wasm_encoder::Module, types: &mut HashMap<TypeEntry<'a>, u32>) {
        let mut section = wasm_encoder::TypeSection::new();

        let import_types: Vec<&FuncType> = self.module.imports.iter().filter_map(|import| {
            if import.module == CANONICAL_ABI_MODULE_NAME {
                return None
            }
            Some(self.module.import_func_type(import).expect("expected import function"))
        }).collect();

        for ty in import_types {
            let idx = types.len() as u32;
            types.entry(TypeEntry::FuncType(ty)).or_insert_with(|| {
                section.function(
                    ty.params.iter().map(linker::to_val_type),
                    ty.returns.iter().map(linker::to_val_type),
                );
                idx
            });
        }

        for iface in self.module.interfaces.iter() {
            let mut entity_idxs: Vec<(&str, wasm_encoder::EntityType)> = Vec::new();
            for (func, info) in iface.iter() {
                let ty = &info.export_type;
                let next_idx = types.len() as u32;

                let fn_idx = types.entry(TypeEntry::FuncType(ty)).or_insert_with(|| {
                    section.function(
                        ty.params.iter().map(linker::to_val_type),
                        ty.returns.iter().map(linker::to_val_type),
                    );
                    next_idx
                });
                entity_idxs.push((&func.name, wasm_encoder::EntityType::Function(*fn_idx)));
            }

            if self.module.needs_memory_funcs {
                let params = REALLOC_FUNC_TYPE.params.iter().map(linker::to_val_type);
                let results = REALLOC_FUNC_TYPE.returns.iter().map(linker::to_val_type);
                let next_idx = types.len() as u32;
                let realloc_index = types.entry(TypeEntry::FuncType(&REALLOC_FUNC_TYPE)).or_insert_with(|| {
                    section.function(params, results);
                    next_idx
                });
                entity_idxs.push(
                    (REALLOC_EXPORT_NAME, wasm_encoder::EntityType::Function(*realloc_index))
                );

                let params = FREE_FUNC_TYPE.params.iter().map(linker::to_val_type);
                let results = FREE_FUNC_TYPE.returns.iter().map(linker::to_val_type);
                let next_idx = types.len() as u32;
                let free_index = types.entry(TypeEntry::FuncType(&FREE_FUNC_TYPE)).or_insert_with(|| {
                    section.function(params, results);
                    next_idx
                });

                entity_idxs.push(
                    (FREE_EXPORT_NAME, wasm_encoder::EntityType::Function(*free_index))
                );
            }

            if self.module.needs_memory {
                let mem_type = wasm_encoder::MemoryType {
                    limits: wasm_encoder::Limits {
                        min: 0,
                        max: None
                    },
                };
                entity_idxs.push((MEMORY_EXPORT_NAME, wasm_encoder::EntityType::Memory(mem_type)));
            }

            let idx = types.len() as u32;
            types.entry(TypeEntry::Instance(iface.inner().name.clone())).or_insert_with(|| {
                section.instance(entity_idxs);
                idx
            });
        }

        module.section(&section);
    }

    fn write_import_section(&self, module: &mut wasm_encoder::Module, types: &mut HashMap<TypeEntry<'a>, u32>, instances: &mut HashMap<InstanceKind<'a>, u32>, interfaces: &Vec<String>, imported_fn_count: &mut u32) {
        let mut section = wasm_encoder::ImportSection::new();

        let function_imports: Vec<(&str, Option<&str>, wasm_encoder::EntityType)> = self.module.imports.iter().filter_map(|import| {
            match import.ty {
                wasmparser::ImportSectionEntryType::Function(_) => {
                    if interfaces.contains(&import.module.to_string()) || (import.module.to_string() == CANONICAL_ABI_MODULE_NAME) { 
                        None
                    } else {
                        let ty = &TypeEntry::FuncType(self.module.import_func_type(import).expect("expected a function"));
                        Some((import.module, import.field, wasm_encoder::EntityType::Function(types[ty])))
                    }
                },
                _ => None
            }
        }).collect();

        for (module, field, entity) in function_imports {
            section.import(module, field, entity);
            let implicit_idx = instances.len() as u32;
            instances.entry(InstanceKind::Implicit(module)).or_insert(implicit_idx);
            *imported_fn_count += 1;
        }
       
        for iface in self.module.interfaces.iter() {
            let module_name = iface.inner().name.clone();
            // Assume for now that all interfaces have an instance type defined, which should
            // always be the case
            let instance_type_idx = types.get(&TypeEntry::Instance(module_name)).unwrap();
            section.import(&iface.inner().name, None, wasm_encoder::EntityType::Instance(*instance_type_idx));

            let instance_idx = instances.len() as u32;
            instances.entry(InstanceKind::Provider(&iface.inner().name)).or_insert(instance_idx);
        }

        module.section(&section);
    }

    fn write_module_section(&self, section: &mut wasm_encoder::ModuleSection, modules: &mut HashMap<ModuleKind, u32>) {
        let next_module_index = modules.len() as u32;
        modules.entry(ModuleKind::Consumer("consumer")).or_insert_with(|| {
            section.module(&self.module.encode());
            next_module_index
        });
    }

    fn create_proxy_module(&self, iface: &Interface) -> Option<wasm_encoder::Module> {
        let mut types_map = HashMap::new();
        let mut types = wasm_encoder::TypeSection::new();
        let mut functions = wasm_encoder::FunctionSection::new();
        let mut tables = wasm_encoder::TableSection::new();
        let mut exports = wasm_encoder::ExportSection::new();
        let mut code = wasm_encoder::CodeSection::new();

        let function_count = iface.inner().functions.len();
        for (fn_idx, (func, info)) in iface.iter().enumerate() {

            let import_type = &info.import_type;
            let next_idx = types_map.len() as u32;
            let fn_ty_idx = types_map.entry(import_type).or_insert_with(|| {
                types.function(
                    import_type.params.iter().map(linker::to_val_type),
                    import_type.returns.iter().map(linker::to_val_type),
                );

                next_idx
            });

            functions.function(*fn_ty_idx);

            // Define the proxy function and emit enough instructions to call it
            // through the functions table
            exports.export(func.name.as_str(), wasm_encoder::Export::Function(fn_idx as u32));

            let mut func = wasm_encoder::Function::new(std::iter::empty());

            for i in 0..import_type.params.len() {
                func.instruction(wasm_encoder::Instruction::LocalGet(i as u32));
            }

            // Put the original function index (provided through the table)
            // to indirectly call the function
            // When loading the tables, we'll need to ensure that 
            // provided functions are inserted in the interface definition order
            func.instruction(wasm_encoder::Instruction::I32Const(fn_idx as i32));
            func.instruction(wasm_encoder::Instruction::CallIndirect {
                ty: *fn_ty_idx,
                table: 0,
            });

            func.instruction(wasm_encoder::Instruction::End);
            code.function(&func);
        }

        tables.table(wasm_encoder::TableType {
            element_type: wasm_encoder::ValType::FuncRef,
            limits: wasm_encoder::Limits {
                min: function_count as u32,
                max: Some(function_count as u32),
            },
        });

        exports.export("proxy_funcs", wasm_encoder::Export::Table(0));

        let mut module = wasm_encoder::Module::new();
        module.section(&types);
        module.section(&functions);
        module.section(&tables);
        module.section(&exports);
        module.section(&code);

        Some(module)
    }   

    fn write_proxy_modules(&self, modules: &mut HashMap<ModuleKind<'a>, u32>, module_section: &mut wasm_encoder::ModuleSection) {
        for iface in self.module.interfaces.iter() {
            let next_module_index = modules.len() as u32;
            let module_kind_key = ModuleKind::Proxy(&iface.inner().name); 

            modules.entry(module_kind_key).or_insert_with(|| {
                let proxy_module = self.create_proxy_module(iface).unwrap();
                module_section.module(&proxy_module);
                next_module_index
            });
        }
    }

    fn create_provider_adapter_module(&self, iface: &'a Interface) -> Option<wasm_encoder::Module> {
        let mut types_map: HashMap<&'a FuncType, u32> = HashMap::new();
        let mut types = wasm_encoder::TypeSection::new();
        let mut functions = wasm_encoder::FunctionSection::new();
        let mut exports = wasm_encoder::ExportSection::new();
        let mut code = wasm_encoder::CodeSection::new();
        let mut imports = wasm_encoder::ImportSection::new();
        let mut next_type_idx = 0 as u32;
        let mut provider_free_index = None;
        let mut provider_realloc_index = None;
        let mut consumer_realloc_index = None;
        let mut fns = 0;
        let iface_name = &iface.inner().name;

        if self.module.needs_memory {
            imports.import(
                CONSUMER_MODULE_NAME,
                Some(MEMORY_EXPORT_NAME),
                wasm_encoder::EntityType::Memory(wasm_encoder::MemoryType {
                    limits: wasm_encoder::Limits {
                        min: 0,
                        max: None
                    },
                }),
            );

            imports.import(
                iface_name,
                Some(MEMORY_EXPORT_NAME),
                wasm_encoder::EntityType::Memory(wasm_encoder::MemoryType {
                    limits: wasm_encoder::Limits {
                        min: 0,
                        max: None
                    },
                }),
            );
        }


        if self.module.needs_memory_funcs {
            next_type_idx = types_map.len() as u32;
            types_map.entry(&REALLOC_FUNC_TYPE).or_insert_with(|| {
                types.function(
                    REALLOC_FUNC_TYPE.params.iter().map(linker::to_val_type),
                    REALLOC_FUNC_TYPE.returns.iter().map(linker::to_val_type),
                );
                next_type_idx
            });

            imports.import(
                iface_name,
                Some(REALLOC_EXPORT_NAME), 
                wasm_encoder::EntityType::Function(next_type_idx),
            );

            provider_realloc_index = Some(fns);
            fns += 1;

            next_type_idx = types_map.len() as u32;
            types_map.entry(&FREE_FUNC_TYPE).or_insert_with(|| {
                types.function(
                    FREE_FUNC_TYPE.params.iter().map(linker::to_val_type),
                    FREE_FUNC_TYPE.returns.iter().map(linker::to_val_type),
                );
                next_type_idx
            });

            imports.import(
                iface_name,
                Some(FREE_EXPORT_NAME),
                wasm_encoder::EntityType::Function(next_type_idx),
            );
            provider_free_index = Some(fns);
            fns += 1;

            imports.import(CONSUMER_MODULE_NAME, Some(REALLOC_EXPORT_NAME), wasm_encoder::EntityType::Function(provider_realloc_index.unwrap()));
            consumer_realloc_index = Some(fns);
            fns += 1;
        }

        for (_, (func, info)) in iface.iter().enumerate() {

            let export_type = &info.export_type;
            next_type_idx = types_map.len() as u32;
            types_map.entry(export_type).or_insert_with(|| {
                types.function(
                    export_type.params.iter().map(linker::to_val_type),
                    export_type.returns.iter().map(linker::to_val_type),
                );
                next_type_idx
            });

            let import_type = &info.import_type;
            next_type_idx = types_map.len() as u32;
            types_map.entry(import_type).or_insert_with(|| {
                types.function(
                    import_type.params.iter().map(linker::to_val_type),
                    import_type.returns.iter().map(linker::to_val_type),
                );
                next_type_idx
            });

            // import the original provider function
            imports.import(
                &iface.inner().name,
                Some(&func.name),
                wasm_encoder::EntityType::Function(*types_map.get(export_type).unwrap()),
            );
            let call_idx = fns;
            fns += 1;

            functions.function(*types_map.get(import_type).unwrap());
            exports.export(&func.name, wasm_encoder::Export::Function(fns as u32));
            //
            // TODO: hook resources
            let resource_fns = HashMap::new();

            let adapter = CallAdapter::new(
                &iface,
                &info.import_signature,
                func,
                call_idx,
                provider_realloc_index,
                provider_free_index,
                consumer_realloc_index, // consumer realloc index
                &resource_fns,
            );

            code.function(&adapter.adapt());

        }


        let mut module = wasm_encoder::Module::new();
        module.section(&types);
        module.section(&imports);
        module.section(&functions);
        module.section(&exports);
        module.section(&code);

        Some(module)
    }

    fn write_provider_adapter_modules(&self, modules: &mut HashMap<ModuleKind<'a>, u32>, module_section: &mut wasm_encoder::ModuleSection) {
        for iface in self.module.interfaces.iter() {
            let next_module_index = modules.len() as u32;
            let module_kind_key = ModuleKind::ProviderAdapter(&iface.inner().name); 

            modules.entry(module_kind_key).or_insert_with(|| {
                let adapter_module = self.create_provider_adapter_module(iface).unwrap();
                module_section.module(&adapter_module);
                next_module_index
            });
        }
    }
}
