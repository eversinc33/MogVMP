#include <llvm/Support/raw_ostream.h>

#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "peload.h"
#include "vmp_lifter.h"

namespace
{

// Parse a CLI unsigned argument (decimal or 0x hex via base 0).
bool parse_unsigned_arg(const std::string &text, const char *name, unsigned &out)
{
    try
    {
        size_t idx   = 0;
        auto   value = std::stoul(text, &idx, 0);
        if (idx != text.size())
            throw std::invalid_argument("trailing characters");
        out = static_cast<unsigned>(value);
        return true;
    }
    catch (const std::exception &)
    {
        std::cerr << "Invalid " << name << ": " << text << "\n";
        return false;
    }
}

bool parse_u64_arg(const std::string &text, const char *name, uint64_t &out)
{
    try
    {
        size_t   idx   = 0;
        uint64_t value = std::stoull(text, &idx, 0);
        if (idx != text.size() || value == 0)
            throw std::invalid_argument("invalid address");
        out = value;
        return true;
    }
    catch (const std::exception &)
    {
        std::cerr << "Invalid " << name << ": " << text << "\n";
        return false;
    }
}

}  // namespace

int main(int argc, char *argv[])
{
    // Args
    bool                        save_intermediate = false;
    std::vector<uint64_t>       continue_vmentries;
    unsigned                    param_count = 0;
    bool                        have_param_count = false;
    VmpTrace                    trace;
    bool                        have_vmenter = false;
    std::vector<const char *>   positional;

    // Parse args
    for (int i = 1; i < argc; ++i)
    {
        std::string_view arg(argv[i]);
        if (arg == "--save-intermediate-steps")
        {
            save_intermediate = true;
        }
        else if (arg == "--continue")
        {
            // Comma-separated VMENTER addresses of the VMs reached after each VMEXIT,
            // in order. Enables following VMEXITs and resuming at the next listed VM.
            if (++i >= argc)
            {
                std::cerr << "Missing comma-separated address list after --continue\n";
                return 1;
            }
            std::string        token;
            std::istringstream ss(argv[i]);
            while (std::getline(ss, token, ','))
            {
                size_t b = token.find_first_not_of(" \t");
                size_t e = token.find_last_not_of(" \t");
                if (b == std::string::npos)
                    continue;
                token = token.substr(b, e - b + 1);

                uint64_t reentry = 0;
                if (!parse_u64_arg(token, "--continue vmentry", reentry))
                    return 1;
                continue_vmentries.push_back(reentry);
            }
        }
        else if (arg == "--args")
        {
            if (++i >= argc || !parse_unsigned_arg(argv[i], "--args", param_count))
                return 1;
            have_param_count = true;
        }
        else if (arg == "--vmenter")
        {
            if (++i >= argc || !parse_u64_arg(argv[i], "--vmenter", trace.vmenter))
                return 1;
            have_vmenter = true;
        }
        else if (arg == "--imagebase")
        {
            if (++i >= argc || !parse_u64_arg(argv[i], "--imagebase", trace.image_base))
                return 1;
        }
        else
        {
            positional.push_back(argv[i]);
        }
    }

    // Display usage on wrong args
    if (positional.size() != 2 || !have_vmenter)
    {
        std::cerr << "Usage: " << argv[0]
                  << " [--save-intermediate-steps] [--continue <vmentry,...>]"
                     " [--args <count>] --vmenter <0xADDR> [--imagebase <0xADDR>] <pe_file> <output.ll>\n"
                  << "  e.g. " << argv[0] << " --args 2 --vmenter 0x401000 vmp.exe lifted.ll\n"
                  << "  --args is optional; when omitted, argc is inferred from the stack pushes.\n"
                  << "  --continue lists the VMENTER addresses reached after each VMEXIT, in order.\n";
        return 1;
    }

    const std::string pe_path  = positional[0];
    const std::string out_path = positional[1];

    std::cout << "[*] Trace: vmenter=0x" << std::hex << trace.vmenter;
    if (trace.image_base != 0)
        std::cout << ", imagebase=0x" << trace.image_base;
    std::cout << std::dec << "\n";

    // Load PE
    Memory memory;
    PEInfo info{};
    if (!peload::LoadPE(pe_path, memory, info))
        return 1;
    std::cout << "[*] Image loaded: " << pe_path << "\n";
    if (trace.image_base != 0 && trace.image_base != info.image_base)
    {
        std::cout << "[*] Rebasing image to: 0x" << std::hex << trace.image_base << "\n";
        if (!peload::ApplyBaseRelocations(memory, info, trace.image_base))
            return 1;
        peload::RebaseMemory(memory, info.image_base, trace.image_base);
        info.image_base = trace.image_base;
    }
    if (have_param_count)
        std::cout << "    Symbolic parameters: " << std::dec << param_count << "\n";
    std::cout << "    PE image base : 0x" << std::hex << info.image_base << "\n";

    // Run lifter (nullopt -> infer argc from the VMENTER call's stack pushes)
    std::optional<unsigned> param_count_opt;
    if (have_param_count)
        param_count_opt = param_count;
    auto result = VmpLifter{}.run(memory, info, trace, param_count_opt, save_intermediate, continue_vmentries);
    if (!result)
        return 1;

    // Save output file
    std::error_code      ec;
    llvm::raw_fd_ostream out_file(out_path, ec);
    if (ec)
    {
        std::cerr << "Cannot open output file '" << out_path << "': " << ec.message() << "\n";
        return 1;
    }
    result.module->print(out_file, nullptr);
    std::cout << "[*] LLVM IR written to " << out_path << "\n";

    return 0;
}
