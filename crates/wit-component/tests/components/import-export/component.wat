(component
  (type (;0;) (func (result string)))
  (type (;1;) 
    (instance
      (alias outer 1 0 (type (;0;)))
      (export "a" (type 0))
    )
  )
  (type (;2;) (tuple string u32 string))
  (type (;3;) (func (param "x" string) (result (type 2))))
  (type (;4;) (func))
  (module (;0;)
    (type (;0;) (func (param i32)))
    (type (;1;) (func (param i32 i32 i32 i32) (result i32)))
    (type (;2;) (func (param i32 i32 i32)))
    (type (;3;) (func (param i32 i32) (result i32)))
    (type (;4;) (func))
    (type (;5;) (func (result i32)))
    (import "foo" "a" (func (;0;) (type 0)))
    (func (;1;) (type 1) (param i32 i32 i32 i32) (result i32)
      unreachable
    )
    (func (;2;) (type 2) (param i32 i32 i32)
      unreachable
    )
    (func (;3;) (type 3) (param i32 i32) (result i32)
      unreachable
    )
    (func (;4;) (type 4)
      unreachable
    )
    (func (;5;) (type 5) (result i32)
      unreachable
    )
    (memory (;0;) 1)
    (export "memory" (memory 0))
    (export "canonical_abi_realloc" (func 1))
    (export "canonical_abi_free" (func 2))
    (export "a" (func 3))
    (export "bar#a" (func 4))
    (export "bar#b" (func 5))
  )
  (import "foo" (instance (;0;) (type 1)))
  (module (;1;)
    (type (;0;) (func (param i32)))
    (func (;0;) (type 0) (param i32)
      local.get 0
      i32.const 0
      call_indirect (type 0)
    )
    (table (;0;) 1 1 funcref)
    (export "0" (func 0))
    (export "$imports" (table 0))
  )
  (module (;2;)
    (type (;0;) (func (param i32)))
    (import "" "0" (func (;0;) (type 0)))
    (import "" "$imports" (table (;0;) 1 1 funcref))
    (elem (;0;) (i32.const 0) func 0)
  )
  (instance (;1;) (instantiate (module 1)))
  (alias export (instance 1) "0" (func (;0;)))
  (instance (;2;) core (export "a" (func 0)))
  (instance (;3;) (instantiate (module 0) (with "foo" (instance 2))))
  (alias export (instance 1) "$imports" (table (;0;)))
  (alias export (instance 0) "a" (func (;1;)))
  (func (;2;) (canon.lower utf8 (into (instance 3)) (func 1)))
  (instance (;4;) core (export "$imports" (table 0)) (export "0" (func 2)))
  (instance (;5;) (instantiate (module 2) (with "" (instance 4))))
  (alias export (instance 3) "a" (func (;3;)))
  (func (;4;) (canon.lift (type 3) utf8 (into (instance 3)) (func 3)))
  (alias export (instance 3) "bar#a" (func (;5;)))
  (alias export (instance 3) "bar#b" (func (;6;)))
  (func (;7;) (canon.lift (type 4) (func 5)))
  (func (;8;) (canon.lift (type 0) utf8 (into (instance 3)) (func 6)))
  (instance (;6;) (export "a" (func 7)) (export "b" (func 8)))
  (export "a" (func 4))
  (export "bar" (instance 6))
)