<p align="center">
  <img src="img/mog.png" width="400"> 

  <h1 align="center">MogVMP</h1>
  <p align="center">Devirtualizer for 32bit binaries protected by VMProtect 3.5.</p>
</p>

MogVMP lifts code to LLVM using [Remill](https://github.com/lifting-bits/remill) to recover the original semantics behind a virtualized function. Instead of modeling the VM specific handler semantics individually (like in my previous work on [byteshield](https://eversinc33.com/2026/05/07/llvm-devirtualizer)), the whole x86 assembly code of each handler is lifted. Junk code, as well as the virtualization layer,  is subsequently optimized away by LLVMs built-in optimization passes and a custom pass that does aliasing-aware constant propagation and store forwarding over memory allocas.

For recovery of the CFG, the main goal of this project was to have it work fully statically, without needing any opcode traces or merging CFGs from traced runs. Instead, opcode handlers are lifted and optimized incrementally from VMENTER on. After each lifted handler, the next handler materializes as a constant. If it doesn't, VMP is branching: in this case, the two possible targets are extracted and the lifting process is forked. This approach works well on CFGs without jumptables, with a caveat of being rather slow. 

That being said, **MogVMP is a PoC I created to learn Remill** - do not expect production quality, handling of edge-cases, clean code or even that it works reliably beyond the complexity of the test examples. Theres issues and work in progress. See the examples below for what can be lifted already.
 
## Examples

Simple Math function with a static CFG:

![](./img/lifted_math.png)

Branching example:

![](./img/lifted_branch.png)

## Usage

Run the binary on the target executable, witha VMENTER address and an output path:

```sh
./build/lifter --vmenter 0x004040ED tests/data/Project1.vmp.exe out.ll
```

- `--vmenter <0xADDR>` - address of the VMENTER to lift
- `--args <count>` - optional, number of args the function takes. infered from the caller if not supplied
- `--imagebase <0xADDR>` - optional address to rebase the PE to

For debugging, `--replay <handlers.txt>` skips discovery and replays a list of handler addresses (one per line):

### Finding VMENTERs

The PIN tool in `aux/tracer/` can be used to trace a VMProtected binary and capture virtualized function's VMENTERs.

Build the tool with the Visual Studio project in `C:\pin\source\tools\MyPinTool`, then run it via PIN against the protected binary:

```sh
pin.exe -t MyPinTool.dll -o trace.json -- target.vmp.exe [args]
```

## Build

```sh
git clone --recurse-submodules https://github.com/eversinc33/MogVMP && cd MogVMP
# If already checked out:
# git submodule update --init --recursive
cmake --preset default
cmake --build build
```

Run tests with:

```sh
ctest --test-dir build --output-on-failure
```

### Shoutouts

* [Ryan Weil](https://ryan-weil.github.io/), who happened to work on a similar project, for great discussions
* phage, for motivation, telling me VMP 3.5 was an easy target
* bakki, for giving the project its name:

<p align="center">
  <img src="img/bakki.png" width="400"> 
</p>


