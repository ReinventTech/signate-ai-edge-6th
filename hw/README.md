# RISCV + DPU のハードウェア作成に利用したファイル

## RISCV

- 160MHz駆動
- FPU有効

アドレス
- 命令メモリ (BRAM): 0x8200_0000 (8KB)
- データメモリ (BRAM): 0x8400_0000 (8KB)
- リセット (GPIO): 0x8001_0000
- 割り込み受信 (ARM): 0x8002_0000
- 割り込み通知 (VexRiscv): 0x8003_0000

## DPU

- B4096
- 290MHz/580MHz 駆動
- ALU8
- Leaky ReLU Disabled
- Softmax Disabled

## ファイル

- `bin/`: `/lib/firmware/xilinx` に配置するビルド済みのファイル
- `petalinux/`: reserved-memory の設定内容
- `vexriscv/`: VexRiscvのconfig
- `vivado/`: Vivadoで extensible vitis platform の作成に必要なスクリプトとファイル
- `vitis/`: Vitisでハードウェアビルドに必要な設定
- `system_wrapper.xsa`: Vitisで利用できる extensible vitis platform
