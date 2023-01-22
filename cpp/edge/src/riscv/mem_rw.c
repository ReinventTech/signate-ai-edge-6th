#define REG(address) *(volatile unsigned int*)(address)
#define REGF(address) *(volatile float*)(address)
#define DMEM_BASE  (0x10000000)
#define GPIO_BASE  (0x80030000)

int main(void)
{
    int i;

    REG(GPIO_BASE + 4) = 0; // 出力に設定
	REG(GPIO_BASE) = 0;

    for (i = 0; i < 100; i++) {
	    REGF(DMEM_BASE + i * 4) = (float)i / 100;
	}
	
	REGF(DMEM_BASE + 400) = 0;
	for (i = 0; i < 100; i++) {
	    REGF(DMEM_BASE + 400) += REGF(DMEM_BASE + i * 4);
	}
	
	// 結果が REGF(DMEM_BASE + 400) にあるはず
    
	REG(GPIO_BASE) = 0x01; // 終了通知
	while(1) {}

	return 0;
}

