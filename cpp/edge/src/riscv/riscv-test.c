

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <poll.h>

#define REG(address) *(volatile unsigned int*)(address)
#define REGF(address) *(volatile float*)(address)
#define GPIO_BASE (0x80010000)
#define IMEM_BASE (0x82000000)
#define DMEM_BASE (0x10000000)

unsigned int riscv_imm(unsigned int *IMEM);
unsigned int riscv_dmm(unsigned int *DMEM);

void setup_gpio_in();
void setup_gpio_out();
void wait_rising();

int main(void)
{
    unsigned int  i, inum, dnum;
    float a, b, c;
    unsigned int *iram, *dram, *gpio;
    int fd;
    struct pollfd pfd;
    char buf[1];
    unsigned int IMEM[4096];
    unsigned int DMEM[4096];

    printf("RISCV Test Program v0.14\n");

    setup_gpio_in();
    setup_gpio_out();

    if ((fd = open("/dev/mem", O_RDWR | O_SYNC)) < 0) {
        perror("open");
        return -1;
    }

    iram = (unsigned int*)mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IMEM_BASE);
    if (iram == MAP_FAILED) {
        perror("mmap iram");
        close(fd);
        return -1;
    }

    dram = (unsigned int*)mmap(NULL, 0x2000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, DMEM_BASE);
    if (dram == MAP_FAILED) {
        perror("mmap dram");
        close(fd);
        return -1;
    }

    gpio = (unsigned int*)mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, GPIO_BASE);
    if (gpio == MAP_FAILED) {
        perror("mmap gpio");
        close(fd);
        return -1;
    }

    pfd.events = POLLPRI;
    pfd.fd = open("/sys/class/gpio/gpio504/value", O_RDONLY);
    if (pfd.fd < 0) {
        perror("failed to open gpio504 value");
        exit(EXIT_FAILURE);
    }

    // 初回
    poll(&pfd, 1, -1);
    lseek(pfd.fd, 0, SEEK_SET);
    read(pfd.fd, buf, 1);

    // GPIO Direction
    REG(gpio + 1) = 0x00;
	  REG(gpio) = 0x02; // LED0

    // Write Program
    inum = riscv_imm(IMEM);
    for (i = 0; i < inum; i++) {
      REG(iram + i) = IMEM[i];
    }

    // Write Data
    dnum = riscv_dmm(DMEM);
    for (i = 0; i < inum; i++) {
      REG(dram + i) = DMEM[i];
    }

    // Run Program
    printf("Running...\n");
    REG(gpio) = 0x03; // LED1 + Reset off
    
    // Wait Program end
    poll(&pfd, 1, -1);
    lseek(pfd.fd, 0, SEEK_SET);
    read(pfd.fd, buf, 1);

  	REG(gpio) = 0x00; // Reset on

    // Check Result
    printf("result: %x, %f\n", REG(dram+100), REGF(dram+100));

    close(fd);
    close(pfd.fd);

    return 0;
}

/**
 * gpio495の設定をする
 */
void setup_gpio_out()
{
    int fd;

    // export gpio492
    fd = open("/sys/class/gpio/export", O_WRONLY);
    if (fd < 0) {
        perror("failed to open gpio export");
        exit(EXIT_FAILURE);
    }
    write(fd, "495", 4);
    close(fd);

    fd = open("/sys/class/gpio/gpio495/direction", O_WRONLY);
    if (fd < 0) {
        perror("failed to open gpio495 direction");
        exit(EXIT_FAILURE);
    }
    write(fd, "out", 4);
    close(fd);

    fd = open("/sys/class/gpio/gpio495/value", O_WRONLY);
    if (fd < 0) {
        perror("failed to open gpio495 value");
        exit(EXIT_FAILURE);
    }
    write(fd, "0", 2);
    close(fd);
}

/**
 * gpio500の設定をする
 */
void setup_gpio_in()
{
    int fd;

    fd = open("/sys/class/gpio/export", O_WRONLY);
    if (fd < 0) {
        perror("failed to open gpio export");
        exit(EXIT_FAILURE);
    }
    write(fd, "504", 4);
    close(fd);

    fd = open("/sys/class/gpio/gpio504/direction", O_WRONLY);
    if (fd < 0) {
        perror("failed to open gpio504 direction");
        exit(EXIT_FAILURE);
    }
    write(fd, "in", 3);
    close(fd);

    fd = open("/sys/class/gpio/gpio504/edge", O_WRONLY);
    if (fd < 0) {
        perror("failed to open gpio504 edge");
        exit(EXIT_FAILURE);
    }
    write(fd, "rising", 7);
    close(fd);
}