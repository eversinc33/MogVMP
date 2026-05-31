#pragma once
#include <cstdint>
#include <map>
#include <string>

// Memory model: a flat map from address -> byte
using Memory = std::map<uint64_t, uint8_t>;

struct PEInfo
{
    uint64_t    image_base;
    std::string arch_name;  // "amd64" or "x86"
    std::string os_name;    // "windows"
    uint32_t    base_reloc_rva = 0;
    uint32_t    base_reloc_size = 0;
};

namespace peload
{
bool LoadPE(const std::string &path, Memory &memory, PEInfo &info);
bool ApplyBaseRelocations(Memory &memory, const PEInfo &info, uint64_t new_base);
void RebaseMemory(Memory &memory, uint64_t old_base, uint64_t new_base);
}  // namespace peload