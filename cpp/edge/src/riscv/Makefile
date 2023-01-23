
CFLAGS := -Os -Wno-implicit
CFLAGS += -fno-builtin-printf -fno-common

SRC = riscv_predictor.c start.s
HDR = 
LDFLAGS := -std=gnu99 -static -lgcc -Tlinker.ld -march=rv32imf -mabi=ilp32 -Wa,-march=rv32imf -nostartfiles

main: $(SRC) $(HDR)
	riscv64-unknown-elf-gcc $(CFLAGS) $(SRC) $(LDFLAGS) -o main
	riscv64-unknown-elf-objcopy --gap-fill 0 -O verilog main main.hex
	riscv64-unknown-elf-objdump -D main > main.dump
	/usr/bin/python3 ./hex_dump.py main


clean:
	rm -f main main.hex main.dump

