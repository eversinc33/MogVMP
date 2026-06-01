"""
vmp_junk_remover.py
────────────────────
IDA Pro script: remove VMProtect junk instructions via backward liveness analysis.

Usage (in IDA's Python console):
    exec(open("/path/to/vmp_junk_remover.py").read())
    remove_junk(idc.get_screen_ea(), dry_run=True)   # inspect first
    remove_junk(idc.get_screen_ea(), dry_run=False)  # then patch

Requires: capstone (ships with IDA)
"""

import idc
import idaapi
import idautils
import ida_bytes
import ida_ua

import capstone
from capstone import x86_const as X86
from capstone import CS_OP_MEM, CS_OP_REG, CS_AC_WRITE, CS_AC_READ

# ─────────────────────────────────────────────────────────────────────────────
# Capstone context
# ─────────────────────────────────────────────────────────────────────────────
_cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
_cs.detail = True

EFLAGS_ID = X86.X86_REG_EFLAGS

# ─────────────────────────────────────────────────────────────────────────────
# Sub-register aliasing table
# Maps narrow register IDs → their 32-bit parent ID.
# If AL is written, EAX must be considered written (and vice-versa for liveness).
# ─────────────────────────────────────────────────────────────────────────────
def _build_alias_map():
    groups = [
        # 32-bit       16-bit          8-bit hi    8-bit lo
        (X86.X86_REG_EAX, X86.X86_REG_AX,  X86.X86_REG_AH, X86.X86_REG_AL),
        (X86.X86_REG_EBX, X86.X86_REG_BX,  X86.X86_REG_BH, X86.X86_REG_BL),
        (X86.X86_REG_ECX, X86.X86_REG_CX,  X86.X86_REG_CH, X86.X86_REG_CL),
        (X86.X86_REG_EDX, X86.X86_REG_DX,  X86.X86_REG_DH, X86.X86_REG_DL),
        (X86.X86_REG_ESI, X86.X86_REG_SI,  None,           None),
        (X86.X86_REG_EDI, X86.X86_REG_DI,  None,           None),
        (X86.X86_REG_ESP, X86.X86_REG_SP,  None,           None),
        (X86.X86_REG_EBP, X86.X86_REG_BP,  None,           None),
    ]
    alias = {}
    for members in groups:
        parent = members[0]
        for r in members:
            if r is not None:
                alias[r] = parent
    return alias

_ALIAS = _build_alias_map()

def normalize_reg(reg_id):
    """Map any sub-register to its 32-bit parent."""
    return _ALIAS.get(reg_id, reg_id)

def normalize_set(reg_set):
    return frozenset(normalize_reg(r) for r in reg_set)


# ─────────────────────────────────────────────────────────────────────────────
# Instructions that are NEVER junk regardless of liveness
# (have observable side effects beyond register/flag writes)
# ─────────────────────────────────────────────────────────────────────────────
NEVER_DEAD_MNEMS = frozenset({
    'pushf', 'popf', 'pushfd', 'popfd',   # VMP uses these for flag I/O
    'push', 'pop',                          # VM stack ops
    'call', 'ret', 'retn',
    'int', 'int3', 'syscall', 'sysenter',
    'out', 'in',
    'rdtsc', 'cpuid',
    'xchg',                                 # has implicit memory semantics in lock context
})


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Collect the handler's linear instruction stream
# ─────────────────────────────────────────────────────────────────────────────
def collect_linear_stream(start_ea, max_insns=1000):
    """
    Follow control flow from start_ea.
    - Transparently follows unconditional near jumps (VMP handler chaining)
    - Stops at: indirect jmp, jcc, ret, or max_insns reached
    Returns list of (ea, mnemonic) in execution order.
    """
    result = []
    ea = start_ea
    seen = set()

    while ea != idc.BADADDR and len(result) < max_insns:
        if ea in seen:
            print(f"  [!] Loop detected at {hex(ea)}, stopping")
            break
        seen.add(ea)

        flags = idc.get_full_flags(ea)
        if not ida_bytes.is_code(flags):
            break

        mnem = idc.print_insn_mnem(ea)
        if not mnem:
            break

        result.append(ea)

        if mnem == 'jmp':
            op_type = idc.get_operand_type(ea, 0)
            if op_type == idc.o_near:
                # Direct unconditional jump → follow transparently
                target = idc.get_operand_value(ea, 0)
                ea = target
                continue
            else:
                # Indirect jump (jmp eax, jmp [mem], jmp edi) = end of handler
                break

        # Conditional branch = end of handler body we care about
        if mnem.startswith('j') and mnem != 'jmp':
            break

        if mnem in ('ret', 'retn'):
            break

        ea = idc.next_head(ea)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Per-instruction semantics via capstone
# ─────────────────────────────────────────────────────────────────────────────
def get_insn_semantics(ea):
    """
    Returns (mnemonic, reads, writes, has_mem_write) for instruction at ea.
    reads/writes are frozensets of NORMALIZED (32-bit) capstone register IDs.
    EFLAGS is included as EFLAGS_ID.
    """
    size = idc.get_item_size(ea)
    if size <= 0:
        size = 15  # max x86 insn length, let capstone figure it out
    raw = ida_bytes.get_bytes(ea, size)
    if raw is None:
        return "", frozenset(), frozenset(), False

    for insn in _cs.disasm(raw, ea):
        mnem = insn.mnemonic

        try:
            rr, rw = insn.regs_access()
        except Exception:
            rr, rw = [], []

        reads  = normalize_set(rr)
        writes = normalize_set(rw)

        # Detect memory-destination operands (real side effect)
        has_mem_write = any(
            op.type == CS_OP_MEM and bool(op.access & CS_AC_WRITE)
            for op in insn.operands
        )

        return mnem, reads, writes, has_mem_write

    return "", frozenset(), frozenset(), False


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Backward liveness analysis
# ─────────────────────────────────────────────────────────────────────────────

# Registers that VMP's dispatcher always needs alive after any handler.
# Conservative: anything VMP uses as a VM register.
VMP_ALWAYS_LIVE = frozenset({
    X86.X86_REG_ESI,   # VM instruction pointer
    X86.X86_REG_EBP,   # VM stack pointer
    X86.X86_REG_EBX,   # rolling decode key
    X86.X86_REG_EDI,   # handler table base / return target
    X86.X86_REG_ESP,   # real stack (dispatch table lives here)
    EFLAGS_ID,         # VMP pushf/popf sequences read this
})


def liveness_analysis(addrs):
    """
    Backward liveness pass over a linear instruction list.
    Returns a set of addresses that are dead (safe to NOP).
    """
    # Pre-compute semantics for all instructions
    sem = {}
    for ea in addrs:
        sem[ea] = get_insn_semantics(ea)

    live = set(VMP_ALWAYS_LIVE)
    dead = set()

    for ea in reversed(addrs):
        mnem, reads, writes, has_mem_write = sem[ea]

        # Forced-live instructions
        if mnem.lower() in NEVER_DEAD_MNEMS:
            live = (live - writes) | reads
            continue

        # An instruction is dead junk if:
        #   1. No memory write side-effects
        #   2. All registers/flags it writes are currently dead (not in live)
        if not has_mem_write and writes and writes.isdisjoint(live):
            dead.add(ea)
            # Do NOT update liveness — this instruction is being removed
            continue

        # Update live set:  live = (live \ writes) ∪ reads
        live = (live - writes) | reads

    return dead


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Display + optionally patch
# ─────────────────────────────────────────────────────────────────────────────

def _reg_name(rid):
    """Get capstone register name, handle EFLAGS specially."""
    if rid == EFLAGS_ID:
        return "eflags"
    try:
        return _cs.reg_name(rid)
    except Exception:
        return str(rid)


def remove_junk(start_ea, dry_run=True, verbose=True):
    """
    Main entry point.

    start_ea  : Starting address (e.g. idc.get_screen_ea())
    dry_run   : If True, only print what would be removed. Set False to patch.
    verbose   : Print every instruction with live/dead annotation.
    """
    print(f"\n{'='*70}")
    print(f"VMP Junk Remover — starting at {hex(start_ea)}")
    print(f"{'='*70}")

    addrs = collect_linear_stream(start_ea)
    print(f"[*] Collected {len(addrs)} instructions across handler fragments\n")

    dead = liveness_analysis(addrs)

    if verbose:
        # Pre-compute for display
        sem = {ea: get_insn_semantics(ea) for ea in addrs}
        print(f"{'Address':>12}  {'Disasm':<40}  Status")
        print("-" * 75)
        for ea in addrs:
            mnem, reads, writes, hmw = sem[ea]
            disasm_text = idc.generate_disasm_line(ea, 0) or mnem
            status = "  <<< DEAD (JUNK)" if ea in dead else ""
            write_str = "{" + ",".join(_reg_name(r) for r in sorted(writes)) + "}"
            print(f"  {hex(ea):>10}  {disasm_text:<40}  writes={write_str}{status}")

    print(f"\n[*] {len(dead)}/{len(addrs)} instructions identified as junk")

    if dry_run:
        print("[!] DRY RUN — no patches applied.")
        print("[!] Call remove_junk(start_ea, dry_run=False) to apply patches.")
        return dead

    # Apply patches
    print("\n[*] Patching...")
    for ea in dead:
        size = idc.get_item_size(ea)
        disasm = idc.generate_disasm_line(ea, 0)
        ida_bytes.patch_bytes(ea, b'\x90' * size)
        print(f"  NOP'd {hex(ea)} ({size}B): {disasm}")

    # Reanalyze patched region
    idaapi.auto_wait()

    # Optionally refresh the view
    idaapi.refresh_idaview_anyway()

    print(f"\n[+] Done. {len(dead)} junk instructions replaced with NOPs.")
    return dead


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run on current cursor
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    # Dry run by default — inspect output before committing patches
    remove_junk(idc.get_screen_ea(), dry_run=True)
