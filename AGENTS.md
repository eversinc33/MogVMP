# MogVMP Project Context

## Project overview

MogVMP is a work-in-progress devirtualizer/lifter for VMProtect 3.x through 3.5. The current approach is trace-based: given a VMProtected PE, an opcode trace, and an initial-state JSON file, the lifter uses Remill to lift VMProtect handlers to LLVM IR. LLVM optimization passes should then fold away the VM layer so the remaining LLVM IR represents the original virtualized code semantics, stripped of the virtualization layer.

The project target is correctness and principled devirtualization, not pattern hacks. If a proposed change relies on a workaround, cheat, hardcoded fix, test-specific assumption, or brittle special case, explicitly call that out to the user before implementing it and explain why it is needed.

## Build and test commands

The README is authoritative for build and usage instructions. Current common commands:

## Repository layout

- `src/main.cpp` - CLI entry point.
- `src/vmp_lifter.{h,cpp}` - core VMProtect lifting/devirtualization logic.
- `src/peload.{h,cpp}` - PE loading support.
- `src/passes/` - project-specific LLVM passes.
- `tests/lifter_golden_test.cpp` - golden behavior tests for lifted/devirtualized output.
- `tests/data/` - VMProtected sample data
- `remill/` - Remill submodule/dependency. Treat as third-party unless the task explicitly asks to modify it.
- `aux/tracer/` - PIN tracer used to find VMENTERs

## Development guidelines

- Prefer semantic fixes over output-shaping hacks.
- Do not hardcode expected IR strings, handler addresses, opcodes, traces, stack slots, or test fixture properties unless the user explicitly approves.
- If adding assumptions about VMProtect behavior, document them in code comments and/or the relevant design notes.
- Keep the trace-based pipeline working while making changes.
- Preserve or improve test coverage when changing lifter behavior.
- Use existing build/test commands before reporting completion when practical.
- Generated files in `build/`, temporary `.ll` outputs, and logs are usually not source changes.

## LLVM / Remill notes

- The intended simplification mechanism is LLVM optimization folding away lifted VM machinery.
- Avoid replacing that mechanism with ad-hoc recognizers unless discussed first.
- Favor IR that exposes semantics clearly to LLVM passes.
- Be careful when changing memory/register/stack modeling; small modeling shortcuts can produce convincing but incorrect devirtualized IR.

## Agent behavior preferences

- Be direct about uncertainty and tradeoffs.
- Before implementing a workaround, hardcoded fix, or brittle special case, stop and ask or clearly disclose it.
- Explain non-obvious reverse-engineering assumptions.
- When debugging, distinguish between:
  - incorrect trace/input state,
  - incorrect PE/handler decoding,
  - incorrect lifted semantics,
  - insufficient LLVM optimization/canonicalization,
  - test expectation problems.
- Prefer small, reviewable changes.
