- nand channel: path for moving data from dies into controller.  allows for parallelism.  probably 8 on my ssd
- nand die: silicon chip than can store data without power
- command queue: async messaging queue for the OS to tell the sdd what to read and be notified of completion
  - allows for a lot of parallelism
  - typicall one message/completion queue pair per cpu core
- typical block size is 4kb
- a command is a logical block address plus some number of blocks
- typically up to 128kb read per command, so 40 or so commands for a 5.3MB expert read
  - these can be parallelized across channels
- Flash Translation Layer: maps a command to physical pages across dies
- queue depth: commands being run in parallel
 - doesn't seem to matter with the expert sized reads.  1 vs 8 still maxes out at 3.5GB/s
- 


NAND perf
- each die has a cell array latency: microseconds to move a page into the io buffer
  - 50-100 micro seconds
- die interface moves from io buffer to the nvme controller
 - onfi 5.1 is 3,600 MT/s = 3,600MB/s
- a die is really a package of multiple dies that share the same io pins, so the onfi limit is the package limit
- nvme controller: really fast, not a bottleneck
- pcie bus ( 4 lanes)
 - gen 3: 3.5 GB/s
 - gen 4: 7 GB/s
 - gen 5: 14 GB/s
- TB4 external enclosure benchmarks max at around 3.5 GB/s, TB5 at 7-13 GB/s
- apple doesn't use pcie for internal components on a macbook
 - it uses fabric instead, which is optimized for unified memory architecture
 - it probably has at least the throughput as pcie gen 4 4x b/c of the 7 GB/s advertised ssd performance
 - pcie is used for thundeerbolt still
- data is split across packages.  research estimates the nvme controller interleave size is at the page level, 4-16KB
 - it's very likely a single expert will be well distributed across multiple packages.

