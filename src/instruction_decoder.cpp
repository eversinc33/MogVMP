#include "instruction_decoder.h"

#include <array>

static bool read_u8(const Memory &memory, uint64_t addr, uint8_t &out)
{
    auto it = memory.find(addr);
    if (it == memory.end())
        return false;
    out = it->second;
    return true;
}

const ZydisDecoder &InstructionDecoder::x86_decoder()
{
    static const ZydisDecoder decoder = []
    {
        ZydisDecoder d;
        ZydisDecoderInit(&d, ZYDIS_MACHINE_MODE_LEGACY_32, ZYDIS_STACK_WIDTH_32);
        return d;
    }();
    return decoder;
}

std::optional<DecodedInstruction> InstructionDecoder::decode(const Memory &memory, uint64_t addr)
{
    std::array<uint8_t, ZYDIS_MAX_INSTRUCTION_LENGTH> bytes{};
    bool                                              have_byte = false;
    for (size_t i = 0; i < bytes.size(); ++i)
    {
        if (!read_u8(memory, addr + i, bytes[i]))
            break;
        have_byte = true;
    }
    if (!have_byte)
        return std::nullopt;

    DecodedInstruction decoded;
    decoded.address = addr;
    if (!ZYAN_SUCCESS(
            ZydisDecoderDecodeFull(&x86_decoder(), bytes.data(), bytes.size(), &decoded.instruction, decoded.operands)
        ))
        return std::nullopt;

    decoded.length = decoded.instruction.length;
    decoded.mnemonic = decoded.instruction.mnemonic;
    for (unsigned i = 0; i < decoded.instruction.operand_count; ++i)
    {
        const auto &op = decoded.operands[i];
        if (op.type != ZYDIS_OPERAND_TYPE_IMMEDIATE || !op.imm.is_relative)
            continue;
        ZyanU64 target = 0;
        if (ZYAN_SUCCESS(ZydisCalcAbsoluteAddress(&decoded.instruction, &op, addr, &target)))
        {
            decoded.branch_target = static_cast<uint32_t>(target);
            break;
        }
    }
    return decoded;
}

bool InstructionDecoder::is_ret(const DecodedInstruction &inst)
{
    return inst.mnemonic == ZYDIS_MNEMONIC_RET;
}

bool InstructionDecoder::is_unconditional_jmp(const DecodedInstruction &inst)
{
    return inst.mnemonic == ZYDIS_MNEMONIC_JMP;
}

bool InstructionDecoder::is_direct_call(const DecodedInstruction &inst)
{
    return inst.mnemonic == ZYDIS_MNEMONIC_CALL && inst.branch_target.has_value();
}

bool InstructionDecoder::is_pop(const DecodedInstruction &inst)
{
    return inst.mnemonic == ZYDIS_MNEMONIC_POP || inst.mnemonic == ZYDIS_MNEMONIC_POPA ||
           inst.mnemonic == ZYDIS_MNEMONIC_POPAD || inst.mnemonic == ZYDIS_MNEMONIC_POPF ||
           inst.mnemonic == ZYDIS_MNEMONIC_POPFD;
}

std::optional<unsigned> InstructionDecoder::push_register_index(const DecodedInstruction &inst)
{
    if (inst.mnemonic != ZYDIS_MNEMONIC_PUSH || inst.instruction.operand_count_visible < 1 ||
        inst.operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER)
        return std::nullopt;

    switch (inst.operands[0].reg.value)
    {
        case ZYDIS_REGISTER_EAX: return 0;
        case ZYDIS_REGISTER_ECX: return 1;
        case ZYDIS_REGISTER_EDX: return 2;
        case ZYDIS_REGISTER_EBX: return 3;
        case ZYDIS_REGISTER_ESP: return 4;
        case ZYDIS_REGISTER_EBP: return 5;
        case ZYDIS_REGISTER_ESI: return 6;
        case ZYDIS_REGISTER_EDI: return 7;
        default: return std::nullopt;
    }
}
