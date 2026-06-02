#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Memory model: a flat map from address -> byte
using Memory = std::map<uint64_t, uint8_t>;

// A PE section, stored by RVA so its absolute address tracks any later rebase
// (compute VA as info.image_base + rva).
struct PESection
{
    std::string name;
    uint32_t    rva = 0;
    uint32_t    virt_size = 0;
    uint32_t    characteristics = 0;
};

// A single imported symbol, keyed in PEInfo by its IAT slot RVA (the address a
// `call dword [slot]` reads through).
struct PEImport
{
    std::string dll;
    std::string symbol;       // empty when imported by ordinal
    uint16_t    ordinal = 0;
    bool        by_ordinal = false;
};

struct PEInfo
{
    uint64_t    image_base;
    std::string arch_name;  // "amd64" or "x86"
    std::string os_name;    // "windows"
    uint32_t    base_reloc_rva = 0;
    uint32_t    base_reloc_size = 0;

    std::vector<PESection>          sections;            // RVAs, image-base relative
    std::map<uint32_t, PEImport>    import_by_slot_rva;  // IAT slot RVA -> import
    // Original-thunk value -> import. The thunk value is the raw IMAGE_THUNK_DATA:
    // a name-RVA (high bit clear) for by-name imports, or 0x80000000|ordinal for
    // by-ordinal. VMProtect's import-protected external calls compute and push
    // this thunk value as the call target, so the VMEXIT EIP can be resolved here.
    std::map<uint32_t, PEImport>    import_by_orig_thunk;
};

namespace peload
{
bool LoadPE(const std::string &path, Memory &memory, PEInfo &info);
bool ApplyBaseRelocations(Memory &memory, const PEInfo &info, uint64_t new_base);
void RebaseMemory(Memory &memory, uint64_t old_base, uint64_t new_base);

// All take/return absolute VAs in the *current* info.image_base.
const PESection *SectionOf(const PEInfo &info, uint64_t addr);
bool             IsInVmpSection(const PEInfo &info, uint64_t addr);  // section name starts ".vmp"
const PEImport  *ImportAtSlot(const PEInfo &info, uint64_t slot_va);
// Resolve a raw original-thunk value (e.g. a VMEXIT EIP that the VM computed) to
// its import. Tries the value as-is and masked to the low 31 bits (name RVA).
const PEImport  *ImportByThunk(const PEInfo &info, uint32_t thunk_value);
}  // namespace peload