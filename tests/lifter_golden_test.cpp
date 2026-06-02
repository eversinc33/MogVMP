#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

struct LifterRunResult
{
    std::string ir;
    std::string devirt;
    std::string log;
};

static std::string ReadFile(const fs::path &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("failed to open " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static std::string ShellQuote(const fs::path &path)
{
    std::string s = path.string();
    std::string q = "'";
    for (char c : s) q += (c == '\'') ? "'\\''" : std::string(1, c);
    q += "'";
    return q;
}

static std::string Trim(std::string s)
{
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

static std::string ToLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

static std::string ExtractDevirt(const std::string &ir)
{
    const std::string marker = "define i32 @devirt";
    const auto        start = ir.find(marker);
    if (start == std::string::npos)
        throw std::runtime_error("@devirt definition not found");

    const auto end = ir.find("\n}", start);
    if (end == std::string::npos)
        throw std::runtime_error("@devirt definition terminator not found");
    return ir.substr(start, end + 3 - start) + "\n";
}

static bool IsBasicBlockLabel(const std::string &line)
{
    // A basic-block label is "<name>:" optionally followed by a comment, e.g.
    // "entry:", "sub_4fdbad.exit:", "common.ret: ; preds = ...". The devirt entry
    // block is named after the terminal handler, so it is not always "entry:".
    auto colon = line.find(':');
    if (colon == std::string::npos || colon == 0)
        return false;
    return std::all_of(line.begin(), line.begin() + colon, [](unsigned char c) {
        return std::isalnum(c) || c == '_' || c == '.' || c == '-' || c == '$';
    });
}

static std::vector<std::string> DevirtInstructionLines(const std::string &devirt)
{
    std::istringstream       in(devirt);
    std::vector<std::string> lines;
    std::string              line;
    while (std::getline(in, line))
    {
        line = Trim(line);
        if (line.empty() || line == "}" || line.rfind("define ", 0) == 0 || IsBasicBlockLabel(line))
            continue;
        lines.push_back(line);
    }
    return lines;
}

static std::vector<std::string> RecoveredHandlerSequence(const std::string &log)
{
    constexpr std::string_view marker = "[*] Lifting handler 0x";
    std::istringstream         in(log);
    std::vector<std::string>   handlers;
    std::string                line;
    while (std::getline(in, line))
    {
        auto pos = line.find(marker);
        if (pos == std::string::npos)
            continue;
        handlers.push_back("0x" + ToLower(Trim(line.substr(pos + marker.size()))));
    }
    return handlers;
}

static fs::path TestTempDir()
{
    fs::path dir = fs::temp_directory_path() / ("lifter_golden_tests_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    return dir;
}

static LifterRunResult RunLifter(
    unsigned args, std::string_view exe_name, std::string_view vmenter, std::string_view stem,
    std::string_view continue_csv = ""
)
{
    fs::path temp_dir = TestTempDir();
    fs::path out_ll = temp_dir / (std::string(stem) + ".ll");
    fs::path log = temp_dir / (std::string(stem) + ".log");

    fs::path           lifter = LIFTER_BIN;
    fs::path           data_dir = TEST_DATA_DIR;
    std::ostringstream cmd;
    cmd << ShellQuote(lifter) << " --args " << args << " --vmenter " << vmenter;
    if (!continue_csv.empty())
        cmd << " --continue " << continue_csv;
    cmd << ' ' << ShellQuote(data_dir / exe_name) << ' ' << ShellQuote(out_ll) << " > " << ShellQuote(log) << " 2>&1";

    int rc = std::system(cmd.str().c_str());
    if (rc != 0)
        ADD_FAILURE() << "lifter failed, log:\n" << ReadFile(log);

    LifterRunResult r;
    r.log = ReadFile(log);
    if (rc == 0)
    {
        r.ir = ReadFile(out_ll);
        r.devirt = ExtractDevirt(r.ir);
    }
    return r;
}

static const LifterRunResult &RunVmp1TraceOnce()
{
    static const LifterRunResult result = RunLifter(2, "Project1.vmp.exe", "0x4040ED", "vmp1");
    return result;
}

static const LifterRunResult &RunVmp2TraceOnce()
{
    static const LifterRunResult result = RunLifter(3, "Project2.vmp.exe", "0x4040F2", "vmp2");
    return result;
}

static const LifterRunResult &RunVmp4TraceOnce()
{
    static const LifterRunResult result = RunLifter(3, "Project4.vmp.exe", "0x4040F2", "vmp4");
    return result;
}

static const LifterRunResult &RunBranch0Once()
{
    static const LifterRunResult result = RunLifter(1, "Branch0.vmp.exe", "0x403A5D", "branch0");
    return result;
}

static const LifterRunResult &RunDevirtMeChainOnce()
{
    // Multi-VM chain: the program exits each VM to call a native function, then
    // re-enters a new VM. --continue lists the next-VM VMENTER addresses in order.
    // VM1 -> GetTickCount -> VM2 -> sub_401000 (swprintf_s) -> VM3 -> MessageBoxW.
    static const LifterRunResult result =
        RunLifter(2, "devirtualizeme32_vmp_3.0.9_v1.exe", "0x0040C890", "devirtme", "0x4312d7,0x41F618");
    return result;
}

TEST(LifterGoldenTest, Vmp1TraceDevirtualizesToSymbolicAdd)
{
    const auto &run = RunVmp1TraceOnce();
    ASSERT_FALSE(run.devirt.empty()) << run.log;

    auto lines = DevirtInstructionLines(run.devirt);
    ASSERT_EQ(lines.size(), 2u) << run.devirt;
    EXPECT_NE(lines[0].find(" = add i32 %arg1, %arg0"), std::string::npos) << run.devirt;
    EXPECT_NE(lines[1].find("ret i32 "), std::string::npos) << run.devirt;
    EXPECT_NE(lines[1].find(lines[0].substr(0, lines[0].find(" = "))), std::string::npos) << run.devirt;
}

TEST(LifterGoldenTest, Vmp1TraceRecoversExpectedHandlerSequence)
{
    const auto &run = RunVmp1TraceOnce();
    ASSERT_FALSE(run.log.empty());

    const std::vector<std::string> expected = {
        "0x4040ed", "0x4a9a4e", "0x4b391d", "0x48ae12", "0x4e04a9", "0x488e93", "0x4d39e7", "0x4faa93",
        "0x4cca09", "0x4c4b9d", "0x4d62b5", "0x4a4b76", "0x4f845f", "0x4aa164", "0x4974ff", "0x4a9a4e",
        "0x4cabe3", "0x4f1b97", "0x4ac85d", "0x49715f", "0x4b391d", "0x4aeedb", "0x483eda", "0x497b65",
        "0x4bb142", "0x4dfbc0", "0x4b65ed", "0x4c072a", "0x4fdbad", "0x4cdfef",
    };

    EXPECT_EQ(RecoveredHandlerSequence(run.log), expected) << run.log;
}

TEST(LifterGoldenTest, Vmp2TraceDevirtualizesToXorAdd)
{
    const auto &run = RunVmp2TraceOnce();
    ASSERT_FALSE(run.devirt.empty()) << run.log;

    auto lines = DevirtInstructionLines(run.devirt);
    ASSERT_EQ(lines.size(), 4u) << run.devirt;
    EXPECT_NE(lines[0].find(" = xor i32 %arg1, %arg0"), std::string::npos) << run.devirt;
    EXPECT_NE(lines[1].find(" = add i32 %arg2, 3"), std::string::npos) << run.devirt;
    EXPECT_NE(lines[2].find(" = add i32 "), std::string::npos) << run.devirt;
    EXPECT_NE(lines[2].find(lines[0].substr(0, lines[0].find(" = "))), std::string::npos) << run.devirt;
    EXPECT_NE(lines[2].find(lines[1].substr(0, lines[1].find(" = "))), std::string::npos) << run.devirt;
    EXPECT_NE(lines[3].find("ret i32 "), std::string::npos) << run.devirt;
    EXPECT_NE(lines[3].find(lines[2].substr(0, lines[2].find(" = "))), std::string::npos) << run.devirt;
}

TEST(LifterGoldenTest, Vmp4TraceDevirtualizesToAddXor)
{
    const auto &run = RunVmp4TraceOnce();
    ASSERT_FALSE(run.devirt.empty()) << run.log;

    auto lines = DevirtInstructionLines(run.devirt);
    ASSERT_EQ(lines.size(), 3u) << run.devirt;
    EXPECT_NE(lines[0].find(" = add i32 %arg2, %arg1"), std::string::npos) << run.devirt;
    EXPECT_NE(lines[1].find(" = xor i32 "), std::string::npos) << run.devirt;
    EXPECT_NE(lines[1].find(lines[0].substr(0, lines[0].find(" = "))), std::string::npos) << run.devirt;
    EXPECT_NE(lines[1].find("%arg0"), std::string::npos) << run.devirt;
    EXPECT_NE(lines[2].find("ret i32 "), std::string::npos) << run.devirt;
    EXPECT_NE(lines[2].find(lines[1].substr(0, lines[1].find(" = "))), std::string::npos) << run.devirt;
}

TEST(LifterGoldenTest, Branch0DevirtualizesToSelectOnEqualsOne)
{
    // f(x) = x == 1 ? 0 : x  ->  select(arg0 == 1, 0, arg0). The VM branch must
    // survive devirtualization as a real argument-dependent select.
    const auto &run = RunBranch0Once();
    ASSERT_FALSE(run.devirt.empty()) << run.log;

    auto lines = DevirtInstructionLines(run.devirt);
    bool has_cmp_eq_one = false, has_select_zero = false, has_ret = false;
    for (const auto &line : lines)
    {
        if (line.find("icmp eq i32") != std::string::npos && line.find(", 1") != std::string::npos)
            has_cmp_eq_one = true;
        if (line.find("select i1") != std::string::npos && line.find("i32 0,") != std::string::npos)
            has_select_zero = true;
        if (line.find("ret i32 ") != std::string::npos)
            has_ret = true;
    }
    EXPECT_TRUE(has_cmp_eq_one) << run.devirt;   // x == 1
    EXPECT_TRUE(has_select_zero) << run.devirt;  // ? 0 : x
    EXPECT_TRUE(has_ret) << run.devirt;
    EXPECT_NE(run.devirt.find("%arg0"), std::string::npos) << run.devirt;
}

TEST(LifterGoldenTest, DevirtMeChainEmitsExternalCalls)
{
    // Multi-VM continuation: each VMEXIT calls a native function and re-enters a
    // new VM. The lifted @devirt must contain the three opaque calls in order, with
    // GetTickCount's result flowing through the MBA chain into sub_401000's args
    // (whose count is auto-recovered to 4, not the probed upper bound).
    const auto &run = RunDevirtMeChainOnce();
    ASSERT_FALSE(run.devirt.empty()) << run.log;

    auto lines = DevirtInstructionLines(run.devirt);
    EXPECT_EQ(lines.size(), 15u) << run.devirt;  // GetTickCount + 11 MBA ops + sub_401000 + MessageBoxW + ret

    // The three external calls, in program order.
    const auto tick = run.devirt.find("@GetTickCount(");
    const auto sprf = run.devirt.find("@sub_401000(");
    const auto mbox = run.devirt.find("@MessageBoxW(");
    ASSERT_NE(tick, std::string::npos) << run.devirt;
    ASSERT_NE(sprf, std::string::npos) << run.devirt;
    ASSERT_NE(mbox, std::string::npos) << run.devirt;
    EXPECT_LT(tick, sprf) << run.devirt;
    EXPECT_LT(sprf, mbox) << run.devirt;

    // sub_401000 (swprintf_s) keeps exactly 4 recovered args, the last being the
    // SSA-computed MBA value derived from GetTickCount (not a folded constant).
    const auto open = run.devirt.find('(', sprf);
    const auto close = run.devirt.find(')', open);
    ASSERT_NE(close, std::string::npos) << run.devirt;
    const std::string call_args = run.devirt.substr(open + 1, close - open - 1);
    EXPECT_EQ(std::count(call_args.begin(), call_args.end(), ',') + 1, 4) << call_args;
    const auto last_comma = call_args.rfind(',');
    EXPECT_NE(Trim(call_args.substr(last_comma + 1)).find("i32 %"), std::string::npos) << call_args;
}
