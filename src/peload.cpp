#include "peload.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

static bool ReadU16(const Memory &memory, uint64_t addr, uint16_t &out)
{
    auto b0 = memory.find(addr);
    auto b1 = memory.find(addr + 1);
    if (b0 == memory.end() || b1 == memory.end())
        return false;
    out = static_cast<uint16_t>(b0->second | (b1->second << 8));
    return true;
}

static bool ReadU32(const Memory &memory, uint64_t addr, uint32_t &out)
{
    out = 0;
    for (unsigned i = 0; i < 4; ++i)
    {
        auto it = memory.find(addr + i);
        if (it == memory.end())
            return false;
        out |= static_cast<uint32_t>(it->second) << (i * 8);
    }
    return true;
}

static void WriteU32(Memory &memory, uint64_t addr, uint32_t value)
{
    for (unsigned i = 0; i < 4; ++i) memory[addr + i] = static_cast<uint8_t>(value >> (i * 8));
}

namespace peload
{

bool LoadPE(const std::string &path, Memory &memory, PEInfo &info)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
    {
        std::cerr << "Cannot open: " << path << "\n";
        return false;
    }

    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    if (data.size() < 0x40)
    {
        std::cerr << "File too small to be a PE\n";
        return false;
    }

    // DOS header
    uint16_t mz_magic;
    std::memcpy(&mz_magic, data.data(), 2);
    if (mz_magic != 0x5A4D)
    {
        std::cerr << "Not a PE file (missing MZ magic)\n";
        return false;
    }

    uint32_t pe_offset;
    std::memcpy(&pe_offset, data.data() + 0x3c, 4);

    if (pe_offset + 24 > data.size())
    {
        std::cerr << "PE header out of bounds\n";
        return false;
    }

    // PE signature
    uint32_t sig;
    std::memcpy(&sig, data.data() + pe_offset, 4);
    if (sig != 0x00004550)
    {
        std::cerr << "Missing PE signature\n";
        return false;
    }

    // COFF header
    uint16_t machine, num_sections, opt_header_size;
    std::memcpy(&machine, data.data() + pe_offset + 4, 2);
    std::memcpy(&num_sections, data.data() + pe_offset + 6, 2);
    std::memcpy(&opt_header_size, data.data() + pe_offset + 20, 2);

    if (machine == 0x014c)
    {
        info.arch_name = "x86";
    }
    else
    {
        std::cerr << "Unsupported machine type: 0x" << std::hex << machine << "\n";
        return false;
    }
    info.os_name = "windows";

    // Optional header
    size_t opt_offset = pe_offset + 24;
    if (opt_offset + 2 > data.size())
    {
        std::cerr << "Optional header out of bounds\n";
        return false;
    }

    uint16_t opt_magic;
    std::memcpy(&opt_magic, data.data() + opt_offset, 2);

    if (opt_magic == 0x020b)
    {  // PE32+ (64-bit)
        std::memcpy(&info.image_base, data.data() + opt_offset + 24, 8);
    }
    else if (opt_magic == 0x010b)
    {  // PE32 (32-bit)
        uint32_t ib32;
        std::memcpy(&ib32, data.data() + opt_offset + 28, 4);
        info.image_base = ib32;
    }
    else
    {
        std::cerr << "Unknown optional header magic: 0x" << std::hex << opt_magic << "\n";
        return false;
    }

    // Base relocation data directory (index 5).  Used when a trace was
    // collected from an ASLR-rebased image.
    size_t data_dir_offset = 0;
    if (opt_magic == 0x010b)
        data_dir_offset = opt_offset + 96;
    else if (opt_magic == 0x020b)
        data_dir_offset = opt_offset + 112;
    if (data_dir_offset + 5 * 8 + 8 <= data.size())
    {
        std::memcpy(&info.base_reloc_rva, data.data() + data_dir_offset + 5 * 8, 4);
        std::memcpy(&info.base_reloc_size, data.data() + data_dir_offset + 5 * 8 + 4, 4);
    }

    // Section headers
    size_t sections_offset = opt_offset + opt_header_size;

    for (uint16_t i = 0; i < num_sections; i++)
    {
        size_t sh = sections_offset + i * 40;
        if (sh + 40 > data.size())
            break;

        uint32_t virt_size, virt_addr, raw_size, raw_offset;
        std::memcpy(&virt_size, data.data() + sh + 8, 4);
        std::memcpy(&virt_addr, data.data() + sh + 12, 4);
        std::memcpy(&raw_size, data.data() + sh + 16, 4);
        std::memcpy(&raw_offset, data.data() + sh + 20, 4);

        uint64_t va = info.image_base + virt_addr;
        size_t   copy_size = std::min({(size_t)raw_size, (size_t)virt_size, data.size() - (size_t)raw_offset});

        for (size_t j = 0; j < copy_size; j++)
        {
            memory[va + j] = data[raw_offset + j];
        }
    }

    return true;
}

bool ApplyBaseRelocations(Memory &memory, const PEInfo &info, uint64_t new_base)
{
    if (new_base == info.image_base)
        return true;
    if (info.base_reloc_rva == 0 || info.base_reloc_size == 0)
    {
        std::cerr << "Trace image base differs from PE image base, but PE has no base relocation table\n";
        return false;
    }

    const int64_t  delta = static_cast<int64_t>(new_base) - static_cast<int64_t>(info.image_base);
    uint64_t       cursor = info.image_base + info.base_reloc_rva;
    const uint64_t end = cursor + info.base_reloc_size;
    unsigned       patched = 0;

    while (cursor + 8 <= end)
    {
        uint32_t page_rva = 0;
        uint32_t block_size = 0;
        if (!ReadU32(memory, cursor, page_rva) || !ReadU32(memory, cursor + 4, block_size))
            return false;
        if (block_size < 8 || cursor + block_size > end)
            return false;

        const uint64_t entries = (block_size - 8) / 2;
        for (uint64_t i = 0; i < entries; ++i)
        {
            uint16_t entry = 0;
            if (!ReadU16(memory, cursor + 8 + i * 2, entry))
                return false;
            const uint16_t type = entry >> 12;
            const uint16_t off = entry & 0x0fff;
            if (type == 0)  // IMAGE_REL_BASED_ABSOLUTE
                continue;
            if (type != 3)  // IMAGE_REL_BASED_HIGHLOW for PE32
            {
                std::cerr << "Unsupported base relocation type: " << type << "\n";
                return false;
            }

            const uint64_t patch_addr = info.image_base + page_rva + off;
            uint32_t       value = 0;
            if (!ReadU32(memory, patch_addr, value))
                return false;
            WriteU32(memory, patch_addr, static_cast<uint32_t>(value + delta));
            ++patched;
        }
        cursor += block_size;
    }

    std::cout << "[*] Applied " << std::dec << patched << " base relocation(s)\n";
    return true;
}

void RebaseMemory(Memory &memory, uint64_t old_base, uint64_t new_base)
{
    if (old_base == new_base)
        return;

    Memory rebased;
    for (const auto &[addr, byte] : memory) rebased[(addr - old_base) + new_base] = byte;
    memory = std::move(rebased);
}

}  // namespace peload
