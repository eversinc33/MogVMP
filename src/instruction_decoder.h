#pragma once

#include <Zydis/Zydis.h>

#include <cstdint>
#include <optional>

#include "peload.h"

struct DecodedInstruction
{
    uint64_t                address = 0;
    uint8_t                 length = 0;
    ZydisMnemonic           mnemonic = ZYDIS_MNEMONIC_INVALID;
    ZydisDecodedInstruction instruction{};
    ZydisDecodedOperand     operands[ZYDIS_MAX_OPERAND_COUNT]{};
    std::optional<uint64_t> branch_target;
};

class InstructionDecoder
{
public:
    static std::optional<DecodedInstruction> decode(const Memory &memory, uint64_t addr);
    static bool                              is_ret(const DecodedInstruction &inst);
    static bool                              is_unconditional_jmp(const DecodedInstruction &inst);
    static bool                              is_direct_call(const DecodedInstruction &inst);
    static bool                              is_pop(const DecodedInstruction &inst);
    static std::optional<unsigned>           push_register_index(const DecodedInstruction &inst);

private:
    static const ZydisDecoder &x86_decoder();
};
