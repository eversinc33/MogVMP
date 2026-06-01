#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "pin.H"

// Trace VMENTRY -> VM handlers -> VMEXIT -> native .text resume target.
//
// VMENTRY detection is intentionally trace-based and follows the current
// VMProtect layout assumption used by this project: the protected host code
// calls a VMENTRY stub in .text, and that stub eventually performs a direct
// jmp from .text into a .vmp* section.  PIN does not give us stripped function
// boundaries, so the VMENTRY "function start" is inferred from the most recent
// direct CALL target in .text, as requested.  If no such CALL target is known,
// we fall back to the JMP instruction address so the trace is not silently lost.

struct Section
{
    ADDRINT     start;
    ADDRINT     end;  // exclusive
    std::string imgName;
    std::string secName;
};

static std::vector<Section> g_sections;
static PIN_RWMUTEX          g_secLock;

static ADDRINT g_imageBase = 0;
static ADDRINT g_lastTextCallTarget = 0;
static bool    g_inVm = false;
static ADDRINT g_currentVmEnter = 0;
static ADDRINT g_currentVmExit = 0;

struct VmTraceEndpoints
{
    ADDRINT vmenter;
    ADDRINT vmexit;
};
static std::vector<VmTraceEndpoints> g_traces;

// Single JSON output path.
KNOB<std::string> KnobOutput(
    KNOB_MODE_WRITEONCE, "pintool", "o", "vmtrace.json", "Output JSON file containing all VM opcode traces"
);

static bool IsInNamedSection(ADDRINT addr, const char* name)
{
    PIN_RWMutexReadLock(&g_secLock);
    for (const auto& s : g_sections)
    {
        if (addr >= s.start && addr < s.end && s.secName == name)
        {
            PIN_RWMutexUnlock(&g_secLock);
            return true;
        }
    }
    PIN_RWMutexUnlock(&g_secLock);
    return false;
}

static bool IsInVmpSection(ADDRINT addr)
{
    PIN_RWMutexReadLock(&g_secLock);
    for (const auto& s : g_sections)
    {
        if (addr >= s.start && addr < s.end && s.secName.find(".vmp") != std::string::npos)
        {
            PIN_RWMutexUnlock(&g_secLock);
            return true;
        }
    }
    PIN_RWMutexUnlock(&g_secLock);
    return false;
}

static bool IsInTextSection(ADDRINT addr)
{
    return IsInNamedSection(addr, ".text");
}

static std::string HexAddr(ADDRINT addr)
{
    std::ostringstream oss;
    oss << "0x" << std::uppercase << std::hex << std::setw(static_cast<int>(sizeof(ADDRINT) * 2)) << std::setfill('0')
        << static_cast<UINT64>(addr);
    return oss.str();
}

static void AppendTraceTarget(ADDRINT target)
{
    // Endpoint-only tracer: handler/opcode targets are intentionally not logged.
}

VOID ImageLoad(IMG img, VOID*)
{
    std::string imgName = IMG_Name(img);
    size_t      slash = imgName.find_last_of("/\\");
    if (slash != std::string::npos)
        imgName = imgName.substr(slash + 1);

    if (IMG_IsMainExecutable(img))
        g_imageBase = IMG_LowAddress(img);

    PIN_RWMutexWriteLock(&g_secLock);
    for (SEC sec = IMG_SecHead(img); SEC_Valid(sec); sec = SEC_Next(sec))
    {
        if (!SEC_Mapped(sec) || SEC_Size(sec) == 0)
            continue;

        Section s;
        s.start = SEC_Address(sec);
        s.end = s.start + SEC_Size(sec);
        s.imgName = imgName;
        s.secName = SEC_Name(sec);
        g_sections.push_back(s);
    }
    PIN_RWMutexUnlock(&g_secLock);
}

// Track direct CALL targets in .text.  The next direct .text -> .vmp JMP uses
// this as the inferred start of the VMENTRY function.
VOID OnDirectCall(ADDRINT ip, ADDRINT target)
{
    if (IsInTextSection(ip) && IsInTextSection(target))
        g_lastTextCallTarget = target;

    // The VMENTRY stub calls the first opcode/handler after it entered .vmp.
    if (g_inVm && IsInVmpSection(ip) && IsInVmpSection(target))
        AppendTraceTarget(target);
}

VOID OnIndirectCall(ADDRINT ip, ADDRINT target)
{
    // Be permissive for the first handler call: record any call target that
    // stays in .vmp while the VM trace is active.
    if (g_inVm && IsInVmpSection(ip) && IsInVmpSection(target))
        AppendTraceTarget(target);
}

// Start a VM trace only on a direct JMP from .text into .vmp*.
VOID OnDirectJmp(ADDRINT ip, ADDRINT target)
{
    if (g_inVm || !IsInTextSection(ip) || !IsInVmpSection(target))
        return;

    ADDRINT vmentry = g_lastTextCallTarget ? g_lastTextCallTarget : ip;
    g_currentVmEnter = vmentry;
    g_currentVmExit = 0;
    g_inVm = true;
}

// Handler dispatch by indirect jmp reg.
VOID OnIndirectJmp(ADDRINT ip, ADDRINT target)
{
    if (!g_inVm)
        return;
    if (IsInVmpSection(ip) && IsInVmpSection(target))
        AppendTraceTarget(target);
}

// Handler dispatch by push reg; ret, and VMEXIT detection.  End the current VM
// trace only when a RET executed in .vmp* returns to .text.  The .text target is
// still part of the protected function's semantics: VMProtect can return from
// VMEXIT to a native resume/epilogue block (e.g. Project1 returns to 004071F4),
// so record that final concrete target instead of stopping at VMEXIT itself.
VOID OnRet(ADDRINT ip, ADDRINT target)
{
    if (!g_inVm || !IsInVmpSection(ip))
        return;

    if (IsInVmpSection(target))
    {
        AppendTraceTarget(target);
    }
    else if (IsInTextSection(target))
    {
        g_currentVmExit = target;
        g_traces.push_back({g_currentVmEnter, g_currentVmExit});
        g_inVm = false;
    }
}

VOID Instruction(INS ins, VOID*)
{
    if (INS_IsCall(ins))
    {
        if (INS_IsDirectBranchOrCall(ins))
        {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)OnDirectCall, IARG_INST_PTR, IARG_BRANCH_TARGET_ADDR, IARG_END);
        }
        else
        {
            INS_InsertCall(
                ins, IPOINT_BEFORE, (AFUNPTR)OnIndirectCall, IARG_INST_PTR, IARG_BRANCH_TARGET_ADDR, IARG_END
            );
        }
        return;
    }

    if (INS_Opcode(ins) == XED_ICLASS_JMP)
    {
        if (INS_IsDirectBranch(ins))
        {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)OnDirectJmp, IARG_INST_PTR, IARG_BRANCH_TARGET_ADDR, IARG_END);
        }
        else
        {
            INS_InsertCall(
                ins, IPOINT_BEFORE, (AFUNPTR)OnIndirectJmp, IARG_INST_PTR, IARG_BRANCH_TARGET_ADDR, IARG_END
            );
        }
        return;
    }

    if (INS_IsRet(ins))
    {
        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)OnRet, IARG_INST_PTR, IARG_BRANCH_TARGET_ADDR, IARG_END);
    }
}

VOID Fini(INT32, VOID*)
{
    std::ofstream out(KnobOutput.Value().c_str());
    out << "{\n  \"imagebase\": \"" << HexAddr(g_imageBase) << "\",\n  \"traces\": [\n";
    for (size_t i = 0; i < g_traces.size(); ++i)
    {
        out << "    { \"vmenter\": \"" << HexAddr(g_traces[i].vmenter) << "\", \"vmexit\": \""
            << HexAddr(g_traces[i].vmexit) << "\" }";
        if (i + 1 != g_traces.size())
            out << ",";
        out << "\n";
    }
    out << "  ]\n}\n";

    PIN_RWMutexFini(&g_secLock);
}

int main(int argc, char* argv[])
{
    PIN_InitSymbols();
    if (PIN_Init(argc, argv))
        return 1;

    PIN_RWMutexInit(&g_secLock);

    IMG_AddInstrumentFunction(ImageLoad, 0);
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);

    PIN_StartProgram();
    return 0;
}
