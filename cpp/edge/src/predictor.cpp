#include <vart/mm/host_flat_tensor_buffer.hpp>
#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>
#include <sys/mman.h>
#include <vector>
#include <memory>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <thread>
#include <mutex>
#include <utility>
#include <queue>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <poll.h>
#include <fstream>
#include <iostream>
#include "common.h"
typedef signed char i8;
typedef unsigned char u8;
typedef unsigned long long u64;

#define LIDAR_IMAGE_WIDTH 1024
#define LIDAR_IMAGE_HEIGHT 1024
#define LIDAR_IMAGE_DEPTH 24
#define N_BUFFERS 22
#define BUFFERS_AVAIL_ADDR_OFFSET 251658240 /* 8*30*1024*1024 */
#define BUFFER_UNIT_SIZE ((uintptr_t)(8*1024*1024))
#define FUNC_PREPROCESS 0
#define FUNC_REFINE 1
#define REG(address) *(volatile unsigned int*)(address)
#define REGF(address) *(volatile float*)(address)
#define GPIO_BASE (0x80010000)
#define IMEM_BASE (0x82000000)
#define DMEM_BASE (0x10000000)

char* base_addr = 0;
bool use_riscv = false;
bool visualize = false;
int last_preprocess_frame_idx = -1;
int last_dpu_frame_idx = -1;
int last_postprocess_frame_idx = -1;
int last_refine_frame_idx = -1;
uintptr_t LIDAR_IMAGE_BUFFER = 8*22*1024*1024;
uintptr_t RECORD_BUFFER = 8*26*1024*1024;
uintptr_t RISCV_ARGS_BUFFER = 8*29*1024*1024;
volatile bool* BUFFERS_AVAIL = 0;
volatile bool* RISCV_BUFFERS_AVAIL = 0;
volatile char* dram = 0;
volatile unsigned int* gpio = 0;
struct pollfd pfd;
std::mutex mutex_alloc, mutex_mfree;
std::mutex mutex_ralloc, mutex_rfree;
std::mutex mutex_riscv;
std::mutex mutex_lidar_image;
std::mutex mutex_records;

unsigned int riscv_imm(unsigned int *IMEM);
unsigned int riscv_dmm(unsigned int *DMEM);
void setup_gpio_in();
void setup_gpio_out();
void wait_rising();


class Bits {
public:
    u64* data;
    int size;

    inline Bits(bool* data, int size){
        this->data = (u64*)data;
        this->size = size;
    }

    inline bool get(int idx){
        int offset = idx / 64;
        int rem = idx % 64;
        return ((this->data[offset]>>rem) & 1) == 1;
    }

    inline void set(int idx, bool bit){
        int offset = idx / 64;
        int rem = idx % 64;
        this->data[offset] |= (1ull << rem);
    }
};


/**
 * Configure gpio495
 */
void setup_gpio_out(){
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
 * Configure gpio500
 */
void setup_gpio_in(){
    int fd;

    fd = open("/sys/class/gpio/export", O_WRONLY);
    if(fd < 0){
        perror("failed to open gpio export");
        exit(EXIT_FAILURE);
    }
    write(fd, "504", 4);
    close(fd);

    fd = open("/sys/class/gpio/gpio504/direction", O_WRONLY);
    if(fd < 0){
        perror("failed to open gpio504 direction");
        exit(EXIT_FAILURE);
    }
    write(fd, "in", 3);
    close(fd);

    fd = open("/sys/class/gpio/gpio504/edge", O_WRONLY);
    if(fd < 0){
        perror("failed to open gpio504 edge");
        exit(EXIT_FAILURE);
    }
    write(fd, "rising", 7);
    close(fd);
}

void* ralloc(){
    mutex_ralloc.lock();
    for(int i=0; i<N_BUFFERS; ++i){
        if(RISCV_BUFFERS_AVAIL[i]){
            RISCV_BUFFERS_AVAIL[i] = false;
            mutex_ralloc.unlock();
            return (void*)(dram + (uintptr_t)i*BUFFER_UNIT_SIZE);
        }
    }
    printf("ralloc failed\n");
    assert(false);
    return 0;
}

void rfree(void* ptr){
    mutex_rfree.lock();
    int idx = ((uintptr_t)ptr-(uintptr_t)dram) / (8*1024*1024);
    RISCV_BUFFERS_AVAIL[idx] = true;
    mutex_rfree.unlock();
}

void* alloc(){
    mutex_alloc.lock();
    for(int i=0; i<N_BUFFERS; ++i){
        if(BUFFERS_AVAIL[i]){
            BUFFERS_AVAIL[i] = false;
            mutex_alloc.unlock();
            return (void*)(base_addr + (uintptr_t)i*BUFFER_UNIT_SIZE);
        }
    }
    printf("alloc failed\n");
    assert(false);
    return 0;
}

void mfree(void* ptr){
    mutex_mfree.lock();
    int idx = ((uintptr_t)ptr-(uintptr_t)base_addr) / (8*1024*1024);
    BUFFERS_AVAIL[idx] = true;
    mutex_mfree.unlock();
}

void run_riscv(){
    // Run program
    REG(gpio) = 0x03; // LED1 + Reset off

    // Wait program end
    poll(&pfd, 1, -1);
    lseek(pfd.fd, 0, SEEK_SET);
    char buf[1];
    read(pfd.fd, buf, 1);

    REG(gpio) = 0x00; // Reset on
}

std::pair<i8*, i8*> riscv_preprocess(float* lidar_points, int n_points, float z_offset, int input_quant_scale, int frame_idx){
    mutex_riscv.lock();
    volatile float* riscv_lidar_points = (volatile float*)ralloc();
    u64* src = (u64*)lidar_points;
    u64* dst = (u64*)riscv_lidar_points;
    std::memcpy(dst, src, n_points*5/2*2*4);
    for(int i=n_points*5/2*2; i<n_points*5; ++i){
        riscv_lidar_points[i] = lidar_points[i];
    }
    volatile char* riscv_args = (volatile char*)(dram + RISCV_ARGS_BUFFER);
    volatile unsigned int* func = (volatile unsigned int*)riscv_args;
    *func = FUNC_PREPROCESS;
    volatile unsigned int* arg_lidar_points = (volatile unsigned int*)(riscv_args + 64);
    *arg_lidar_points = (long)riscv_lidar_points - (long)dram;
    volatile int* arg_n_points = (volatile int*)(riscv_args + 72);
    *arg_n_points = n_points;
    volatile float* arg_z_offset = (volatile float*)(riscv_args + 80);
    *arg_z_offset = z_offset;
    volatile int* arg_input_quant_scale = (volatile int*)(riscv_args + 88);
    *arg_input_quant_scale = input_quant_scale;

    volatile int* riscv_offsets = (volatile int*)ralloc();
    volatile unsigned int* arg_offsets = (volatile unsigned int*)(riscv_args + 96);
    *arg_offsets = (long)riscv_offsets - (long)dram;
    volatile i8* riscv_intensities = (volatile i8*)(riscv_offsets + 1024*1024);
    volatile unsigned int* arg_intensities = (volatile unsigned int*)(riscv_args + 104);
    *arg_intensities = (long)riscv_intensities - (long)dram;

    run_riscv();

    n_points = *arg_n_points;
    mutex_lidar_image.lock();
    i8* lidar_image = (i8*)(base_addr + LIDAR_IMAGE_BUFFER);
    i8* max_lidar_image = lidar_image + 1024*1024*24 + (frame_idx%4)*512*512;
    u64* tmp = (u64*)lidar_image;
    std::memset(tmp, 0, 1024*1024*24);
    tmp = (u64*)max_lidar_image;
    std::memset(tmp, 0, 512*512);
    for(int i=0; i<n_points; ++i){
        int offset = riscv_offsets[i];
        int max_offset = offset / LIDAR_IMAGE_DEPTH;
        int y = max_offset / 1024;
        int x = max_offset % 1024;
        max_offset = (y/2) * 512 + x/2;
        i8 intensity0 = lidar_image[offset];
        i8 intensity1 = riscv_intensities[i];
        lidar_image[offset] = (intensity0>intensity1? intensity0 : intensity1);
        max_lidar_image[max_offset] = (max_lidar_image[max_offset]<intensity1? intensity1 : max_lidar_image[max_offset]);
    }

    rfree((float*)riscv_lidar_points);
    rfree((int*)riscv_offsets);

    mutex_riscv.unlock();

    return {lidar_image, max_lidar_image};
}

std::pair<i8*, i8*> preprocess(float* lidar_points, int n_points, float z_offset, int input_quant_scale, int frame_idx){
    if(use_riscv){
        return riscv_preprocess(lidar_points, n_points, z_offset, input_quant_scale, frame_idx);
    }
    int* lidar_xs = (int*)alloc();
    int* lidar_ys = lidar_xs + 1024*1024/2;
    int* lidar_zs = lidar_ys + 1024*1024/2;
    i8* intensities = (i8*)(lidar_zs + 1024*1024/2);
    int offset = 0;
    int n_valid_points = 0;
    float scale = (float)(1 << input_quant_scale);
    for(int i=0; i<n_points; ++i){
        int x = (int)(lidar_points[offset]*10.0f+0.5f) + 512;
        int y = (int)(-lidar_points[offset+1]*10.0f+0.5f) + 512;
        int z = (int)((lidar_points[offset+2]+z_offset)*5.0f+0.5f);
        lidar_xs[i] = x;
        lidar_ys[i] = y;
        lidar_zs[i] = z;
        if(x>=0 && x<1024 && y>=0 && y<1024 && z>=0 && z<24){
            lidar_xs[n_valid_points] = x;
            lidar_ys[n_valid_points] = y;
            lidar_zs[n_valid_points] = z;
            float intensity = lidar_points[offset+3]*scale+0.5f;
            intensities[n_valid_points] = (intensity>127.0f? 127 :  (i8)intensity);
            if(intensities[n_valid_points]==0) intensities[n_valid_points] = 1;
            ++n_valid_points;
        }
        offset += 5;
    }
    i8* lidar_image = (i8*)(base_addr + LIDAR_IMAGE_BUFFER);
    u64* tmp = (u64*)lidar_image;
    std::memset(tmp, 0, 1024*1024*LIDAR_IMAGE_DEPTH);
    i8* max_lidar_image = lidar_image + 1024*1024*24 + (frame_idx%4)*512*512;
    tmp = (u64*)max_lidar_image;
    std::memset(tmp, 0, 1024*1024);
    for(int i=0; i<n_valid_points; ++i){
        int max_offset = lidar_ys[i] * 1024 + lidar_xs[i];
        int offset = max_offset*LIDAR_IMAGE_DEPTH + lidar_zs[i];
        max_offset /= 4;
        lidar_image[offset] = (lidar_image[offset]<intensities[i]? intensities[i] : lidar_image[offset]);
        max_lidar_image[max_offset] = (max_lidar_image[max_offset]<intensities[i]? intensities[i] : max_lidar_image[max_offset]);
    }
    mfree(lidar_xs);
    return {lidar_image, max_lidar_image};
}

const float sigmoid_table[256] = {1.2664165549094016e-14, 1.6261110446177924e-14, 2.08796791164589e-14, 2.6810038677817314e-14, 3.442477108469858e-14, 4.420228103640978e-14, 5.6756852326323996e-14, 7.287724095819161e-14, 9.357622968839299e-14, 1.2015425731770343e-13, 1.5428112031916497e-13, 1.9810087980485874e-13, 2.543665647376276e-13, 3.2661313427863805e-13, 4.193795658377786e-13, 5.384940217751136e-13, 6.914400106935423e-13, 8.878265478451776e-13, 1.1399918530430558e-12, 1.4637785141237662e-12, 1.8795288165355508e-12, 2.4133627718273897e-12, 3.0988191387122225e-12, 3.978962535821408e-12, 5.109089028037221e-12, 6.560200168110743e-12, 8.423463754397692e-12, 1.0815941557168708e-11, 1.3887943864771144e-11, 1.7832472907828393e-11, 2.289734845593124e-11, 2.940077739198032e-11, 3.7751345441365816e-11, 4.847368706035286e-11, 6.224144622520383e-11, 7.991959892315218e-11, 1.0261879630648827e-10, 1.3176514268359263e-10, 1.6918979223288784e-10, 2.1724399346070674e-10, 2.7894680920908113e-10, 3.581747929000289e-10, 4.599055376537186e-10, 5.905303995456778e-10, 7.582560422162385e-10, 9.736200303530205e-10, 1.2501528648238605e-09, 1.6052280526088547e-09, 2.0611536181902037e-09, 2.646573631904765e-09, 3.398267807946847e-09, 4.363462233903898e-09, 5.602796406145941e-09, 7.194132978569834e-09, 9.23744957664012e-09, 1.1861120010657661e-08, 1.522997951276035e-08, 1.955568070542584e-08, 2.5109990926928157e-08, 3.2241866333029355e-08, 4.1399375473943306e-08, 5.3157849718487075e-08, 6.825602910446286e-08, 8.764247451323235e-08, 1.12535162055095e-07, 1.4449800373124837e-07, 1.8553910183683314e-07, 2.38236909993343e-07, 3.059022269256247e-07, 3.927862002670442e-07, 5.043474082014517e-07, 6.475947982049267e-07, 8.315280276641321e-07, 1.067702870044147e-06, 1.3709572068578448e-06, 1.7603432133424856e-06, 2.2603242979035746e-06, 2.902311985211097e-06, 3.726639284186561e-06, 4.785094494890119e-06, 6.144174602214718e-06, 7.889262586245034e-06, 1.0129990980873921e-05, 1.3007128466476033e-05, 1.670142184809518e-05, 2.144494842091395e-05, 2.7535691114583473e-05, 3.5356250741744315e-05, 4.5397868702434395e-05, 5.829126566113865e-05, 7.484622751061123e-05, 9.610241549947396e-05, 0.00012339457598623172, 0.00015843621910252592, 0.00020342697805520653, 0.0002611903190957194, 0.0003353501304664781, 0.0004305570813246149, 0.0005527786369235996, 0.0007096703991005881, 0.0009110511944006454, 0.0011695102650555148, 0.0015011822567369917, 0.0019267346633274757, 0.0024726231566347743, 0.0031726828424851893, 0.004070137715896128, 0.005220125693558397, 0.0066928509242848554, 0.008577485413711984, 0.01098694263059318, 0.014063627043245475, 0.01798620996209156, 0.022977369910025615, 0.02931223075135632, 0.03732688734412946, 0.04742587317756678, 0.060086650174007626, 0.07585818002124355, 0.09534946489910949, 0.11920292202211755, 0.14804719803168948, 0.18242552380635635, 0.22270013882530884, 0.2689414213699951, 0.320821300824607, 0.3775406687981454, 0.43782349911420193, 0.5, 0.5621765008857981, 0.6224593312018546, 0.679178699175393, 0.7310585786300049, 0.7772998611746911, 0.8175744761936437, 0.8519528019683106, 0.8807970779778823, 0.9046505351008906, 0.9241418199787566, 0.9399133498259924, 0.9525741268224334, 0.9626731126558706, 0.9706877692486436, 0.9770226300899744, 0.9820137900379085, 0.9859363729567544, 0.9890130573694068, 0.991422514586288, 0.9933071490757153, 0.9947798743064417, 0.995929862284104, 0.9968273171575148, 0.9975273768433653, 0.9980732653366725, 0.998498817743263, 0.9988304897349445, 0.9990889488055994, 0.9992903296008995, 0.9994472213630764, 0.9995694429186754, 0.9996646498695336, 0.9997388096809043, 0.9997965730219448, 0.9998415637808975, 0.9998766054240137, 0.9999038975845005, 0.9999251537724895, 0.9999417087343389, 0.9999546021312976, 0.9999646437492582, 0.9999724643088853, 0.9999785550515792, 0.999983298578152, 0.9999869928715335, 0.9999898700090192, 0.9999921107374138, 0.9999938558253978, 0.9999952149055051, 0.9999962733607158, 0.9999970976880148, 0.999997739675702, 0.9999982396567868, 0.999998629042793, 0.9999989322971299, 0.9999991684719722, 0.9999993524052017, 0.9999994956525918, 0.9999996072137998, 0.999999694097773, 0.9999997617630899, 0.9999998144608981, 0.9999998555019962, 0.9999998874648379, 0.9999999123575255, 0.999999931743971, 0.9999999468421502, 0.9999999586006244, 0.9999999677581336, 0.999999974890009, 0.9999999804443193, 0.9999999847700205, 0.99999998813888, 0.9999999907625504, 0.9999999928058669, 0.9999999943972036, 0.9999999956365377, 0.9999999966017321, 0.9999999973534264, 0.9999999979388463, 0.999999998394772, 0.9999999987498471, 0.9999999990263799, 0.9999999992417439, 0.9999999994094697, 0.9999999995400946, 0.9999999996418252, 0.9999999997210531, 0.999999999782756, 0.9999999998308102, 0.999999999868235, 0.9999999998973812, 0.9999999999200804, 0.9999999999377585, 0.9999999999515263, 0.9999999999622486, 0.9999999999705993, 0.9999999999771028, 0.9999999999821676, 0.999999999986112, 0.999999999989184, 0.9999999999915765, 0.9999999999934397, 0.999999999994891, 0.999999999996021, 0.9999999999969011, 0.9999999999975866, 0.9999999999981204, 0.9999999999985363, 0.99999999999886, 0.9999999999991123, 0.9999999999993086, 0.9999999999994615, 0.9999999999995806, 0.9999999999996734, 0.9999999999997455, 0.9999999999998019, 0.9999999999998457, 0.9999999999998799, 0.9999999999999065, 0.9999999999999272, 0.9999999999999432, 0.9999999999999558, 0.9999999999999656, 0.9999999999999731, 0.9999999999999791, 0.9999999999999838};


void quaternion_to_matrix(float qt[4], float matrix[3][3]){
    float qt0_2 = qt[0] * qt[0];
    float qt1_2 = qt[1] * qt[1];
    float qt2_2 = qt[2] * qt[2];
    float qt3_2 = qt[3] * qt[3];
    float qt12 = qt[1] * qt[2];
    float qt13 = qt[1] * qt[3];
    float qt23 = qt[2] * qt[3];
    float qt01 = qt[0] * qt[1];
    float qt02 = qt[0] * qt[2];
    float qt03 = qt[0] * qt[3];
    float s = 2.0 / (qt0_2 + qt1_2 + qt2_2 + qt3_2);
    matrix[0][0] = 1.0f - s * (qt2_2 + qt3_2);
    matrix[0][1] = s * (qt12 - qt03);
    matrix[0][2] = s * (qt13 + qt02);
    matrix[1][0] = s * (qt12 + qt03);
    matrix[1][1] = 1.0f - s * (qt1_2 + qt3_2);
    matrix[1][2] = s * (qt23 - qt01);
    matrix[2][0] = s * (qt13 - qt02);
    matrix[2][1] = s * (qt23 + qt01);
    matrix[2][2] = 1.0f - s * (qt1_2 + qt2_2);
}

inline void rotate(float inp[3], float outp[3], float mx[3][3]){
    outp[0] = mx[0][0]*inp[0] + mx[0][1]*inp[1] + mx[0][2]*inp[2];
    outp[1] = mx[1][0]*inp[0] + mx[1][1]*inp[1] + mx[1][2]*inp[2];
    //outp[2] = mx[2][0]*inp[0] + mx[2][1]*inp[1] + mx[2][2]*inp[2];
}

inline void rotate_2d(float* inp, float* outp, float mx[2][2]){
    outp[0] = mx[0][0]*inp[0] + mx[0][1]*inp[1];
    outp[1] = mx[1][0]*inp[0] + mx[1][1]*inp[1];
}

Bits get_pedestrian_mask(u8* pedestrian_fy){
    bool* pedestrian_m = (bool*)alloc();
    Bits pedestrian_bits(pedestrian_m, 1024*1024);
    bool* buffer0 = pedestrian_m + 1024*1024 / 8;
    Bits bits0(buffer0, 1024*1024);
    bool* buffer1 = buffer0 + 1024*1024 / 8;
    Bits bits1(buffer1, 1024*1024);
    for(int i=0; i<1024*1024/64; ++i){
        u64 mask0 = 0, mask1 = 0;
        for(int j=0; j<64; ++j){
            mask0 |= ((u64)(pedestrian_fy[j]>126)) << j;
            mask1 |= ((u64)(pedestrian_fy[j]>109)) << j;
        }
        bits0.data[i] = pedestrian_bits.data[i] = mask0;
        bits1.data[i] = mask1;
        pedestrian_fy += 64;
    }
    for(int y=0; y<1023; ++y){
        for(int x=0; x<1024/64; ++x){
            pedestrian_bits.data[y*16+x] |= bits0.data[(y+1)*16+x];
            pedestrian_bits.data[(y+1)*16+x] |= bits0.data[y*16+x];
        }
    }
    for(int y=0; y<1024; ++y){
        for(int x=0; x<1024/64-1; ++x){
            pedestrian_bits.data[y*16+x+1] |= (pedestrian_bits.data[y*16+x+1]<<1) | (pedestrian_bits.data[y*16+x]>>63);
            pedestrian_bits.data[y*16+x] |= (pedestrian_bits.data[y*16+x]>>1) | ((pedestrian_bits.data[y*16+x+1]&1)<<63);
        }
        pedestrian_bits.data[y*16] |= (pedestrian_bits.data[y*16]<<1);
        pedestrian_bits.data[y*16+1024/64-1] |= (pedestrian_bits.data[y*16+1024/64-1]>>1);
    }
    for(int y=0; y<1024; ++y){
        for(int x=0; x<1024/64; ++x){
            pedestrian_bits.data[y*16+x] = (bits0.data[y*16+x] | ((~pedestrian_bits.data[y*16+x]) & bits1.data[y*16+x]));
        }
    }
    return pedestrian_bits;
}

Bits get_vehicle_mask(u8* vehicle_fy){
    bool* vehicle_m = (bool*)alloc();
    Bits vehicle_bits(vehicle_m, 1024*1024);
    bool* buffer0 = vehicle_m + 1024*1024 / 8;
    Bits bits0(buffer0, 1024*1024);
    bool* buffer1 = buffer0 + 1024*1024 / 8;
    Bits bits1(buffer1, 1024*1024);
    for(int i=0; i<1024*1024/64; ++i){
        u64 mask0 = 0, mask1 = 0;
        for(int j=0; j<64; ++j){
            mask0 |= ((u64)(vehicle_fy[j]>123)) << j;
            mask1 |= ((u64)(vehicle_fy[j]>118)) << j;
        }
        bits0.data[i] = vehicle_bits.data[i] = mask0;
        bits1.data[i] = mask1;
        vehicle_fy += 64;
    }
    for(int y=0; y<1021; ++y){
        for(int x=0; x<1024/64; ++x){
            vehicle_bits.data[y*16+x] |= bits0.data[(y+3)*16+x];
            vehicle_bits.data[(y+3)*16+x] |= bits0.data[y*16+x];
        }
    }
    for(int y=0; y<1024; ++y){
        for(int x=0; x<1024/64-3; ++x){
            vehicle_bits.data[y*16+x+1] |= (vehicle_bits.data[y*16+x+1]<<3) | (vehicle_bits.data[y*16+x]>>60);
            vehicle_bits.data[y*16+x] |= (vehicle_bits.data[y*16+x]>>3) | ((vehicle_bits.data[y*16+x+1]&7)<<61);
        }
        vehicle_bits.data[y*16] |= (vehicle_bits.data[y*16]<<3);
        vehicle_bits.data[y*16+1024/64-1] |= (vehicle_bits.data[y*16+1024/64-1]>>3);
    }
    for(int y=0; y<1024; ++y){
        for(int x=0; x<1024/64; ++x){
            vehicle_bits.data[y*16+x] = (bits0.data[y*16+x] | ((~vehicle_bits.data[y*16+x]) & bits1.data[y*16+x]));
        }
    }
    return vehicle_bits;
}

void cca(u8* p, Bits m, int* n_centroids, float* scores, int* areas, float* centroids){
    bool* checked = (bool*)alloc();
    u64* dst = (u64*)checked;
    std::memset(dst, 0, 1024*1024);
    *n_centroids = 0;
    int* coords = (int*)(checked + 1024*1024);
    for(int y=0; y<1024; ++y){
        int offset_y = y*1024;
        for(int x=0; x<1024; ++x){
            int offset = offset_y + x;
            if(checked[offset]) continue;
            if(!m.get(offset)){
                checked[offset] = true;
                continue;
            }
            u8 score = p[offset];
            int area = 1;
            int ys = y;
            int xs = x;
            coords[0] = x;
            coords[1] = y;
            checked[offset] = true;
            int idx = 0;
            int* head_coords = coords;
            int* tail_coords = coords + 2;
            while(idx<area){
                int tx = head_coords[0];
                int ty = head_coords[1];
                int d = ty*1024 + tx - 1;
                if(m.get(d) && !checked[d] && tx>0){
                    tail_coords[0] = tx - 1;
                    tail_coords[1] = ty;
                    ++area;
                    checked[d] = true;
                    ys += ty;
                    xs += tx - 1;
                    score = (score<p[d]? p[d] : score);
                    tail_coords += 2;
                }
                d += 2;
                if(m.get(d) && !checked[d] && tx<1023){
                    tail_coords[0] = tx + 1;
                    tail_coords[1] = ty;
                    ++area;
                    checked[d] = true;
                    ys += ty;
                    xs += tx + 1;
                    score = (score<p[d]? p[d] : score);
                    tail_coords += 2;
                }
                d -= 1025;
                if(m.get(d) && !checked[d] && ty>0){
                    tail_coords[0] = tx;
                    tail_coords[1] = ty - 1;
                    ++area;
                    checked[d] = true;
                    ys += ty - 1;
                    xs += tx;
                    score = (score<p[d]? p[d] : score);
                    tail_coords += 2;
                }
                d += 2048;
                if(m.get(d) && !checked[d] && ty<1023){
                    tail_coords[0] = tx;
                    tail_coords[1] = ty + 1;
                    ++area;
                    checked[d] = true;
                    ys += ty + 1;
                    xs += tx;
                    score = (score<p[d]? p[d] : score);
                    tail_coords += 2;
                }
                head_coords += 2;
                ++idx;
            }
            float cx = (float)xs / (float)area;
            float cy = (float)ys / (float)area;
            centroids[*n_centroids*2] = cx;
            centroids[*n_centroids*2+1] = cy;
            scores[*n_centroids] = sigmoid_table[score];
            areas[*n_centroids] = area;
            ++*n_centroids;
        }
    }
    mfree((void*)checked);
}

void postprocess(u8* quant_pedestrian_pred, u8* quant_vehicle_pred, float* pedestrian_centroid, float* pedestrian_confidence, int* n_pedestrians, float* vehicle_centroid, float* vehicle_confidence, int* n_vehicles, int frame_idx, u8* pred_records, float* ego_records){
    Bits pedestrian_m = get_pedestrian_mask(quant_pedestrian_pred);
    Bits vehicle_m = get_vehicle_mask(quant_vehicle_pred);

    int* pedestrian_areas = (int*)pedestrian_m.data + 1024*1024/2;
    cca(quant_pedestrian_pred, pedestrian_m, n_pedestrians, pedestrian_confidence, pedestrian_areas, pedestrian_centroid);

    int n_filtered_pedestrians = 0;
    for(int i=0; i<*n_pedestrians; ++i){
        if(pedestrian_areas[i]>78 && pedestrian_confidence[i]>0.37){
            pedestrian_areas[*n_pedestrians+n_filtered_pedestrians] = pedestrian_areas[i];
            pedestrian_confidence[*n_pedestrians+n_filtered_pedestrians] = pedestrian_confidence[i];
            pedestrian_centroid[(*n_pedestrians+n_filtered_pedestrians)*2] = pedestrian_centroid[i*2];
            pedestrian_centroid[(*n_pedestrians+n_filtered_pedestrians)*2+1] = pedestrian_centroid[i*2+1];
            ++n_filtered_pedestrians;
        }
    }
    *n_pedestrians += n_filtered_pedestrians;

    int* vehicle_areas = (int*)vehicle_m.data + 1024*1024/2;
    cca(quant_vehicle_pred, vehicle_m, n_vehicles, vehicle_confidence, vehicle_areas, vehicle_centroid);

    mfree((void*)pedestrian_m.data);
    mfree((void*)vehicle_m.data);
}

void scale_rotate_translate(float* centroids, int n_centroids, float ego_translation[3], float ego_rotation[4]){
    float mx[3][3] = {};
    quaternion_to_matrix(ego_rotation, mx);
    for(int i=0; i<n_centroids; ++i){
        float xyz[3] = { centroids[i*2] / 10.0f - 51.2f, -centroids[i*2+1] / 10.0f + 51.2f, 1.5};
        float rxyz[3] = {};
        rotate(xyz, rxyz, mx);
        rxyz[0] += ego_translation[0];
        rxyz[1] += ego_translation[1];
        centroids[i*2] = rxyz[0];
        centroids[i*2+1] = rxyz[1];
    }
}

void sort_predictions(float* preds, int n_preds){
    if(n_preds==0) return;
    float* sorted_preds = (float*)alloc();
    sorted_preds[0] = preds[0];
    sorted_preds[1] = preds[1];
    sorted_preds[2] = preds[2];
    sorted_preds[3] = preds[3];
    sorted_preds[4] = preds[4];
    for(int i=1; i<n_preds; ++i){
        for(int j=i-1; j>=0; --j){
            if(sorted_preds[j*5]<preds[i*5]){
                if(j==0){
                    for(int k=i; k>0; --k){
                        sorted_preds[k*5] = sorted_preds[(k-1)*5];
                        sorted_preds[k*5+1] = sorted_preds[(k-1)*5+1];
                        sorted_preds[k*5+2] = sorted_preds[(k-1)*5+2];
                        sorted_preds[k*5+3] = sorted_preds[(k-1)*5+3];
                        sorted_preds[k*5+4] = sorted_preds[(k-1)*5+4];
                    }
                    sorted_preds[0] = preds[i*5];
                    sorted_preds[1] = preds[i*5+1];
                    sorted_preds[2] = preds[i*5+2];
                    sorted_preds[3] = preds[i*5+3];
                    sorted_preds[4] = preds[i*5+4];
                    break;
                }
            }
            else{
                for(int k=i; k>j+1; --k){
                    sorted_preds[k*5] = sorted_preds[(k-1)*5];
                    sorted_preds[k*5+1] = sorted_preds[(k-1)*5+1];
                    sorted_preds[k*5+2] = sorted_preds[(k-1)*5+2];
                    sorted_preds[k*5+3] = sorted_preds[(k-1)*5+3];
                    sorted_preds[k*5+4] = sorted_preds[(k-1)*5+4];
                }
                sorted_preds[(j+1)*5] = preds[i*5];
                sorted_preds[(j+1)*5+1] = preds[i*5+1];
                sorted_preds[(j+1)*5+2] = preds[i*5+2];
                sorted_preds[(j+1)*5+3] = preds[i*5+3];
                sorted_preds[(j+1)*5+4] = preds[i*5+4];
                break;
            }
        }
    }
    for(int i=0; i<n_preds; ++i){
        preds[i*5] = sorted_preds[i*5];
        preds[i*5+1] = sorted_preds[i*5+1];
        preds[i*5+2] = sorted_preds[i*5+2];
        preds[i*5+3] = sorted_preds[i*5+3];
        preds[i*5+4] = sorted_preds[i*5+4];
    }
    mfree((void*)sorted_preds);
}

void riscv_refine_predictions(float* preds, i8* input_image, float* centroids, float* confidence, int* n_preds, float ego_translation[3], float ego_rotation[4], float max_dist, float fuzzy_dist, float fuzzy_rate, float refine_dist, int n_cutoff, bool is_pedestrian){
    mutex_riscv.lock();
    volatile float* riscv_preds = (volatile float*)ralloc();
    volatile float* riscv_centroids = riscv_preds + 1024*8;
    volatile float* riscv_confidence = riscv_centroids + 1024*8;

    for(int i=0; i<*n_preds*2; ++i){
        riscv_centroids[i] = centroids[i];
    }
    for(int i=0; i<*n_preds; ++i){
        riscv_confidence[i] = confidence[i];
    }

    volatile char* riscv_args = (volatile char*)(dram + RISCV_ARGS_BUFFER);
    volatile unsigned int* func = (volatile unsigned int*)riscv_args;
    *func = FUNC_REFINE;
    volatile unsigned int* arg_preds = (volatile unsigned int*)(riscv_args + 64);
    *arg_preds = (long)riscv_preds - (long)dram;
    volatile unsigned int* arg_centroids = (volatile unsigned int*)(riscv_args + 72);
    *arg_centroids = (long)riscv_centroids - (long)dram;
    volatile unsigned int* arg_confidence = (volatile unsigned int*)(riscv_args + 80);
    *arg_confidence = (long)riscv_confidence - (long)dram;
    volatile unsigned int* arg_n_preds = (volatile unsigned int*)(riscv_args + 88);
    *arg_n_preds = *n_preds;
    volatile float* arg_ego_translation = (volatile float*)(riscv_args + 96);
    arg_ego_translation[0] = ego_translation[0];
    arg_ego_translation[1] = ego_translation[1];
    arg_ego_translation[2] = ego_translation[2];
    volatile float* arg_ego_rotation = (volatile float*)(riscv_args + 112);
    arg_ego_rotation[0] = ego_rotation[0];
    arg_ego_rotation[1] = ego_rotation[1];
    arg_ego_rotation[2] = ego_rotation[2];
    arg_ego_rotation[3] = ego_rotation[3];
    volatile float* arg_max_dist = (volatile float*)(riscv_args + 128);
    *arg_max_dist = max_dist;
    volatile float* arg_fuzzy_dist = (volatile float*)(riscv_args + 136);
    *arg_fuzzy_dist = fuzzy_dist;
    volatile float* arg_fuzzy_rate = (volatile float*)(riscv_args + 144);
    *arg_fuzzy_rate = fuzzy_rate;
    volatile float* arg_refine_dist = (volatile float*)(riscv_args + 152);
    *arg_refine_dist = refine_dist;
    volatile int* arg_n_cutoff = (volatile int*)(riscv_args + 160);
    *arg_n_cutoff = n_cutoff;
    volatile bool* arg_is_pedestrian = (volatile bool*)(riscv_args + 168);
    *arg_is_pedestrian = is_pedestrian;

    run_riscv();

    *n_preds = *arg_n_preds;
    for(int i=0; i<*n_preds*3; ++i){
        preds[i] = riscv_preds[i];
    }

    rfree((float*)riscv_preds);
    mutex_riscv.unlock();
}

void refine_predictions(float* preds, i8* input_image, float* centroids, float* confidence, int* n_preds, float ego_translation[3], float ego_rotation[4], float max_dist, float fuzzy_dist, float fuzzy_rate, float refine_dist, int n_cutoff, bool is_pedestrian){
    if(use_riscv){
        riscv_refine_predictions(preds, input_image, centroids, confidence, n_preds, ego_translation, ego_rotation, max_dist, fuzzy_dist, fuzzy_rate, refine_dist, n_cutoff, is_pedestrian);
        return;
    }

    for(int i=0; i<*n_preds; ++i){
        int x = (int)(centroids[i*2]+0.5) / 2;
        int y = (int)(centroids[i*2+1]+0.5) / 2;
        int ox = x;
        int oy = y;
        bool has_points = false;
        int sx = x>4? (x-4) : 0;
        int sy = y>4? (y-4) : 0;
        int ex = x<512-4? (x+4) : (512-1);
        int ey = y<512-4? (y+4) : (512-1);
        for(int ry=sy; ry<ey; ++ry){
            int offset = ry * 512;
            for(int rx=sx; rx<ex; ++rx){
                if(input_image[offset+rx]>0){
                    has_points = true;
                    break;
                }
            }
            if(has_points) break;
        }
        if(!has_points){
            confidence[oy*(1024)+ox] *= 0.1;
        }
    }

    float mx[3][3] = {};
    quaternion_to_matrix(ego_rotation, mx);
    for(int i=0; i<*n_preds; ++i){
        float xyz[3] = { centroids[i*2] / 10.0f - 51.2f, -centroids[i*2+1] / 10.0f + 51.2f, 1.5};
        float rxyz[3] = {};
        rotate(xyz, rxyz, mx);
        rxyz[0] += ego_translation[0];
        rxyz[1] += ego_translation[1];
        centroids[i*2] = rxyz[0];
        centroids[i*2+1] = rxyz[1];
    }

    float* refined_preds = (float*)alloc();
    int n_refined_preds = 0;
    for(int i=0; i<*n_preds; ++i){
        float dx = centroids[i*2] - ego_translation[0];
        float dy = centroids[i*2+1] - ego_translation[1];
        float d = sqrt(dx*dx + dy*dy);
        if(d>max_dist) continue;
        refined_preds[n_refined_preds*5+1] = 1e10f;
        refined_preds[n_refined_preds*5+3] = centroids[i*2];
        refined_preds[n_refined_preds*5+4] = centroids[i*2+1];
        if(d>fuzzy_dist){
            refined_preds[n_refined_preds*5] = confidence[i] * fuzzy_rate;
            refined_preds[n_refined_preds*5+2] = confidence[i] * fuzzy_rate;
        }
        else{
            refined_preds[n_refined_preds*5] = confidence[i];
            refined_preds[n_refined_preds*5+2] = confidence[i];
        }
        ++n_refined_preds;
    }
    sort_predictions(refined_preds, n_refined_preds);
    n_refined_preds = (n_refined_preds>n_cutoff? n_cutoff : n_refined_preds);
    *n_preds = n_refined_preds;
    float m = 1.0 / refine_dist;
    for(int i=0; i<*n_preds && i<50; ++i){
        sort_predictions(refined_preds, *n_preds-i);
        preds[i*3] = refined_preds[3];
        preds[i*3+1] = refined_preds[4];
        preds[i*3+2] = refined_preds[0];
        refined_preds += 5;
        for(int n=0; n<*n_preds-i-1; ++n){
            float dx = refined_preds[n*5+3] - preds[i*3];
            float dy = refined_preds[n*5+4] - preds[i*3+1];
            float d = sqrt(dx*dx + dy*dy);
            refined_preds[n*5+1] = (refined_preds[n*5+1]>d? d : refined_preds[n*5+1]);
            if (is_pedestrian){
                if(refined_preds[n*5+1]==0.0f){
                    refined_preds[n*5] = refined_preds[n*5+2];
                }
                else{
                    float r = m * (refined_preds[n*5+1]>refine_dist? refine_dist: refined_preds[n*5+1]);
                    refined_preds[n*5] = refined_preds[n*5+2] * r;
                }
            }
            else{
                if(refined_preds[n*5+1]<0.4f){
                    refined_preds[n*5] = refined_preds[n*5+2] * 0.01f;
                }
                else{
                    float r = m * (refined_preds[n*5+1]>refine_dist? refine_dist: refined_preds[n*5+1]);
                    refined_preds[n*5] = refined_preds[n*5+2] * r;
                }
            }
        }
    }
    *n_preds = *n_preds>50? 50 : *n_preds;
    mfree((void*)refined_preds);
}

void composite_quaternions(float qt0[4], float qt1[4], float out_qt[4]){
    out_qt[0] = qt0[0]*qt1[0] - qt0[1]*qt1[1] - qt0[2]*qt1[2] - qt0[3]*qt1[3];
    out_qt[1] = qt0[0]*qt1[1] + qt0[1]*qt1[0] + qt0[2]*qt1[3] - qt0[3]*qt1[2];
    out_qt[2] = qt0[0]*qt1[2] - qt0[1]*qt1[3] + qt0[2]*qt1[0] + qt0[3]*qt1[1];
    out_qt[3] = qt0[0]*qt1[3] + qt0[1]*qt1[2] - qt0[2]*qt1[1] + qt0[3]*qt1[0];
}

void inverse_quaternion(float in_qt[4], float out_qt[4]){
    float d = in_qt[0]*in_qt[0] + in_qt[1]*in_qt[1] + in_qt[2]*in_qt[2] + in_qt[3]*in_qt[3];
    out_qt[0] = in_qt[0] / d;
    out_qt[1] = -in_qt[1] / d;
    out_qt[2] = -in_qt[2] / d;
    out_qt[3] = -in_qt[3] / d;
}

void merge_prev_preds(u8* pred, float ego_translation[3], float ego_rotation[4], int frame_idx, u8* pred_records, float* ego_records, int category){
    int offset0 = (frame_idx%2+1) * 1024*1024*2;
    int offset1 = ((frame_idx)%2) * 1024*1024*2;

    u8* pred0 = pred_records + offset0 + category*1024*1024;
    float* txyz0 = ego_records + (frame_idx%2+1) * 8;
    float* qt0 = ego_records + (frame_idx%2+1) * 8 + 3;

    u8* pred1 = pred_records + offset1 + category*1024*1024;
    float* txyz1 = ego_records + (frame_idx%2) * 8;
    float* qt1 = ego_records + (frame_idx%2) * 8 + 3;

    float dtx0 = (ego_translation[0] - txyz0[0]) * 10.0f;
    float dty0 = (ego_translation[1] - txyz0[1]) * 10.0f;
    float dtz0 = (ego_translation[2] - txyz0[2]) * 10.0f;

    float dtx1 = (ego_translation[0] - txyz1[0]) * 10.0f;
    float dty1 = (ego_translation[1] - txyz1[1]) * 10.0f;
    float dtz1 = (ego_translation[2] - txyz1[2]) * 10.0f;

    float iqt[4] = {};
    inverse_quaternion(ego_rotation, iqt);

    float dr0[4] = {};
    composite_quaternions((float*)qt0, iqt, dr0);
    float dr1[4] = {};
    composite_quaternions((float*)qt1, iqt, dr1);

    float _mx0[3][3] = {};
    float _mx1[3][3] = {};
    quaternion_to_matrix(dr0, _mx0);
    quaternion_to_matrix(dr1, _mx1);
    float mx0[2][2] = {{_mx0[0][0], _mx0[0][1]}, {_mx0[1][0], _mx0[1][1]}};
    float mx1[2][2] = {{_mx1[0][0], _mx1[0][1]}, {_mx1[1][0], _mx1[1][1]}};

    float sxyz0[2], sxyz1[2];
    for(int y=0; y<1024; ++y){
        float dxyz[3] = {-512.0f, (float)(y - 512), 0.0f};
        rotate_2d(dxyz, sxyz0, mx0);
        rotate_2d(dxyz, sxyz1, mx1);
        sxyz0[0] += dtx0;
        sxyz0[1] -= dty0;
        sxyz1[0] += dtx1;
        sxyz1[1] -= dty1;
        for(int x=0; x<1024; ++x){
            int sx0 = (int)sxyz0[0] + 512;
            int sy0 = (int)sxyz0[1] + 512;
            u8 v1 = (sx0>=0 && sx0<1024 && sy0>=0 && sy0<1024)? pred0[sy0*1024 + sx0] : 0;

            int sx1 = (int)sxyz1[0] + 512;
            int sy1 = (int)sxyz1[1] + 512;
            u8 v2 = (sx1>=0 && sx1<1024 && sy1>=0 && sy1<1024)? pred1[sy1*1024 + sx1] : 0;

            u8 v0 = pred[0];
            u8 a = v0>v2? v0 : v2;
            u8 b = a>v1? v1 : a;
            pred[0] = (b<v0? v0 : b);
            ++pred;
            sxyz0[0] += mx0[0][0];
            sxyz0[1] += mx0[1][0];
            sxyz1[0] += mx1[0][0];
            sxyz1[1] += mx1[1][0];
        }
    }
}

void run_dpu(vart::Runner* runner, i8* input_image, i8* pred){
    auto inputTensors = runner->get_input_tensors();
    auto outputTensors = runner->get_output_tensors();
    auto out_dims = outputTensors[0]->get_shape();
    auto in_dims = inputTensors[0]->get_shape();
    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
    std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
    std::vector<std::shared_ptr<xir::Tensor>> batchTensors;
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
                xir::Tensor::create(inputTensors[0]->get_name(), in_dims,
                    xir::DataType{xir::DataType::XINT, 8u})));
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
                input_image, batchTensors.back().get()));
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
                xir::Tensor::create(outputTensors[0]->get_name(), out_dims,
                    xir::DataType{xir::DataType::XINT, 8u})));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
                pred, batchTensors.back().get()));

    inputsPtr.clear();
    outputsPtr.clear();
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    mutex_lidar_image.unlock();
    runner->wait(job_id.first, -1);
}


void update_records(int frame_idx, u8* pedestrian_pred, u8* vehicle_pred, float ego_translation[3], float ego_rotation[4], u8* pred_records, float* ego_records){
    int pred_record_offset = (frame_idx%2) * 1024*1024*2;
    u64* src = (u64*)pedestrian_pred;
    u64* dst = (u64*)(pred_records+pred_record_offset);
    std::memcpy(dst, src, 1024*1024);
    pred_record_offset += 1024 * 1024;
    src = (u64*)vehicle_pred;
    dst = (u64*)(pred_records+pred_record_offset);
    std::memcpy(dst, src, 1024*1024);
    int ego_record_offset = (frame_idx%2) * 8;
    ego_records[ego_record_offset] = ego_translation[0];
    ego_records[ego_record_offset+1] = ego_translation[1];
    ego_records[ego_record_offset+2] = ego_translation[2];
    ego_records[ego_record_offset+3] = ego_rotation[0];
    ego_records[ego_record_offset+4] = ego_rotation[1];
    ego_records[ego_record_offset+5] = ego_rotation[2];
    ego_records[ego_record_offset+6] = ego_rotation[3];
}


i8* predict(float* lidar_points, int n_points, int input_quant_scale, int output_quant_scale, float ego_translation[3], float ego_rotation[4], float* pedestrian_preds, float* vehicle_preds, int* n_pedestrians, int* n_vehicles, vart::Runner* runner, int frame_idx, u8* pred_records, float* ego_records, std::string frame_id){
    std::chrono::system_clock::time_point t0 = std::chrono::system_clock::now();
    auto [input_image, max_input_image] = preprocess(lidar_points, n_points, 3.7, input_quant_scale, frame_idx);
    std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
    last_preprocess_frame_idx = frame_idx;


    while(last_dpu_frame_idx<frame_idx-1){
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    i8* quant_pred = (i8*)alloc();
    run_dpu(runner, (i8*)input_image, (i8*)quant_pred);
    std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
    last_dpu_frame_idx = frame_idx;

    while(last_postprocess_frame_idx<frame_idx-1){
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    u8* quant_pedestrian_pred = (u8*)alloc();
    u8* quant_vehicle_pred = quant_pedestrian_pred + 1024*1024;
    u8* src = (u8*)quant_pred;
    u8* dst_pedestrian = quant_pedestrian_pred;
    u8* dst_vehicle = quant_vehicle_pred;
    for(int y=0; y<1024; ++y){
        for(int x=0; x<1024; ++x){
            dst_pedestrian[0] = src[0] ^ 128;
            dst_vehicle[0] = src[1] ^ 128;
            src += 2;
            ++dst_pedestrian;
            ++dst_vehicle;
        }
    }
    mfree(quant_pred);

    if(frame_idx>=2){
        merge_prev_preds(quant_pedestrian_pred, ego_translation, ego_rotation, frame_idx, pred_records, ego_records, 0);
        merge_prev_preds(quant_vehicle_pred, ego_translation, ego_rotation, frame_idx, pred_records, ego_records, 1);
    }

    float* pedestrian_centroids = (float*)alloc();
    float* pedestrian_confidence = pedestrian_centroids + 1024*1024/2;
    float* vehicle_centroids = pedestrian_confidence + 1024*1024/2;
    float* vehicle_confidence = vehicle_centroids + 1024*1024/2;

    postprocess(quant_pedestrian_pred, quant_vehicle_pred, pedestrian_centroids, pedestrian_confidence, n_pedestrians, vehicle_centroids, vehicle_confidence, n_vehicles, frame_idx, pred_records, ego_records);

    update_records(frame_idx, quant_pedestrian_pred, quant_vehicle_pred, ego_translation, ego_rotation, pred_records, ego_records);
    mfree(quant_pedestrian_pred);

    std::chrono::system_clock::time_point t3 = std::chrono::system_clock::now();
    last_postprocess_frame_idx = frame_idx;


    while(last_refine_frame_idx<frame_idx-1){
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    refine_predictions(pedestrian_preds, max_input_image, pedestrian_centroids, pedestrian_confidence, n_pedestrians, ego_translation, ego_rotation, 40, 39, 0.95, 0.8, 55, true);
    refine_predictions(vehicle_preds, max_input_image, vehicle_centroids, vehicle_confidence, n_vehicles, ego_translation, ego_rotation, 50, 49.5, 0.9, 2.2, 60, false);
    mfree(pedestrian_centroids);
    std::chrono::system_clock::time_point t4 = std::chrono::system_clock::now();

    double d0 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0);
    double d1 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0);
    double d2 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.0);
    double d3 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() / 1000.0);
    double d_total = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t4 - t0).count() / 1000.0);
    std::cerr << "#" << frame_id << " Latency[ms] total:" << d_total << " preproc:" << d0 << " dpu:" << d1 << " cca:" << d2 << " refine:" << d3 << std::endl;
    last_refine_frame_idx = frame_idx;
    return max_input_image;
}

void predict_scene(char* output_path, char* xmodel_path){
    int frame_idx = 0;
    char* records = (char*)(base_addr + RECORD_BUFFER);
    u8* pred_records = (u8*)records;
    float* ego_records = (float*)(records + 1024*1024*2*2);

    auto graph = xir::Graph::deserialize(xmodel_path);
    auto subgraph = get_dpu_subgraph(graph.get());
    auto runner = vart::Runner::create_runner(subgraph[0], "run");
    auto outputTensors = runner->get_output_tensors();
    auto inputTensors = runner->get_input_tensors();
    int input_quant_scale = get_input_fix_point(inputTensors[0]);
    int output_quant_scale = get_output_fix_point(outputTensors[0]);
    bool first_frame = true;
    int last_write_frame_idx = -1;
    //printf("input scale: %d\n", input_quant_scale);
    //printf("output scale: %d\n", output_quant_scale);

    std::queue<std::thread*> prediction_threads;

    while(true){
        char frame_id[16];
        char lidar_path[256];
        float ego_translation[3];
        float ego_rotation[4];
        int status = scanf("%s %s %f %f %f %f %f %f %f", frame_id, lidar_path, &ego_translation[0], &ego_translation[1], &ego_translation[2], &ego_rotation[0], &ego_rotation[1], &ego_rotation[2], &ego_rotation[3]);
        if(status==EOF) break;

        auto pred_func = [&](
            int frame_idx, bool first_frame, std::string frame_id, std::string lidar_path,
            float ego_tx, float ego_ty, float ego_tz,
            float ego_rw, float ego_rx, float ego_ry, float ego_rz
        ){
            float* lidar_points = (float*)malloc(8*10*1024*1024);
            float* pedestrian_preds = (float*)malloc(8*10*1024*1024);
            float* vehicle_preds = (float*)malloc(8*10*1024*1024);
            FILE* ifp = fopen(lidar_path.c_str(), "r");
            int n_points = fread((float*)lidar_points, sizeof(float), 100000000, ifp) / 5;
            fclose(ifp);
            int n_pedestrians = 0;
            int n_vehicles = 0;
            float ego_trans[3] = {ego_tx, ego_ty, ego_tz};
            float ego_rot[4] = {ego_rw, ego_rx, ego_ry, ego_rz};
            u8* lidar_image = (u8*)predict(lidar_points, n_points, input_quant_scale, output_quant_scale, ego_trans, ego_rot, pedestrian_preds, vehicle_preds, &n_pedestrians, &n_vehicles, runner.get(), frame_idx, pred_records, ego_records, frame_id);
            if(visualize){
                u8* summary_image = (u8*)malloc(512*512*3);
                std::memset(summary_image+512*512, 128, 512*512*2);
                for(int y=0; y<512; ++y){
                    for(int x=0; x<512; ++x){
                        u8 p = lidar_image[512*y+x];
                        summary_image[y*512+x] = p*3;
                    }
                }
                float mx3[3][3] = {};
                float qt[4];
                inverse_quaternion(ego_rotation, qt);
                quaternion_to_matrix(qt, mx3);
                float mx[2][2] = {{mx3[0][0], mx3[0][1]}, {mx3[1][0], mx3[1][1]}};
                for(int i=0; i<n_vehicles; ++i){
                    float gxy[2] = {vehicle_preds[i*3], vehicle_preds[i*3+1]};
                    float gs = vehicle_preds[i*3+2];
                    float rxy[2];
                    gxy[0] -= ego_translation[0];
                    gxy[1] -= ego_translation[1];
                    rotate_2d(gxy, rxy, mx);
                    rxy[0] = (rxy[0] + 51.2f) * 10.0f;
                    rxy[1] = (-rxy[1] + 51.2f) * 10.0f;
                    int xy[2] = {
                        std::max(0, std::min(511, (int)rxy[0]/2)),
                        std::max(0, std::min(511, (int)rxy[1]/2)),
                    };
                    if(xy[0]>0 && xy[0]<511 && xy[1]>0 && xy[1]<511){
                        int s = std::min((int)(gs*gs*65535.0f)/256, 255);
                        for(int y = std::max(xy[1]-4, 0); y<=std::min(xy[1]+4, 511); ++y){
                            for(int x=std::max(xy[0]-4, 0); x<=std::min(xy[0]+4, 511); ++x){
                                summary_image[y*512+x] = 210*s / 256;
                            }
                        }
                        for(int y = std::max(xy[1]-4, 0); y<=std::min(xy[1]+4, 511); ++y){
                            for(int x=std::max(xy[0]-4, 0); x<=std::min(xy[0]+4, 511); ++x){
                                summary_image[512*512+y*512+x] = (16-128)*s/256 + 128;
                            }
                        }
                        for(int y = std::max(xy[1]-4, 0); y<=std::min(xy[1]+4, 511); ++y){
                            for(int x=std::max(xy[0]-4, 0); x<=std::min(xy[0]+4, 511); ++x){
                                summary_image[2*512*512+y*512+x] = (146-128)*s/256 + 128;
                            }
                        }
                    }
                }
                for(int i=0; i<n_pedestrians; ++i){
                    float gxy[2] = {pedestrian_preds[i*3], pedestrian_preds[i*3+1]};
                    float gs = pedestrian_preds[i*3+2];
                    float rxy[2];
                    gxy[0] -= ego_translation[0];
                    gxy[1] -= ego_translation[1];
                    rotate_2d(gxy, rxy, mx);
                    rxy[0] = (rxy[0] + 51.2f) * 10.0f;
                    rxy[1] = (-rxy[1] + 51.2f) * 10.0f;
                    int xy[2] = {
                        std::max(0, std::min(511, (int)rxy[0]/2)),
                        std::max(0, std::min(511, (int)rxy[1]/2)),
                    };
                    if(xy[0]>0 && xy[0]<511 && xy[1]>0 && xy[1]<511){
                        int s = std::min((int)(gs*512.0f), 255);
                        for(int y = std::max(xy[1]-2, 0); y<=std::min(xy[1]+2, 511); ++y){
                            for(int x=std::max(xy[0]-2, 0); x<=std::min(xy[0]+2, 511); ++x){
                                summary_image[y*512+x] = 106*s / 256;
                            }
                        }
                        for(int y = std::max(xy[1]-2, 0); y<=std::min(xy[1]+2, 511); ++y){
                            for(int x=std::max(xy[0]-2, 0); x<=std::min(xy[0]+2, 511); ++x){
                                summary_image[512*512+y*512+x] = (203-128)*s/256 + 128;
                            }
                        }
                        for(int y = std::max(xy[1]-2, 0); y<=std::min(xy[1]+2, 511); ++y){
                            for(int x=std::max(xy[0]-2, 0); x<=std::min(xy[0]+2, 511); ++x){
                                summary_image[2*512*512+y*512+x] = (63-128)*s/256 + 128;
                            }
                        }
                    }
                }
                for(int y = 256-4; y<=256+4; ++y){
                    for(int x=256-4; x<=256+4; ++x){
                        summary_image[y*512+x] = 114;
                    }
                }
                for(int y = 256-4; y<=256+4; ++y){
                    for(int x=256-4; x<=256+4; ++x){
                        summary_image[512*512+y*512+x] = 72;
                    }
                }
                for(int y = 256-4; y<=256+4; ++y){
                    for(int x=256-4; x<=256+4; ++x){
                        summary_image[2*512*512+y*512+x] = 216;
                    }
                }
                std::cout.write((const char*)summary_image, 512*512*3);
                free(summary_image);
            }
            while(last_write_frame_idx<frame_idx-1){
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            FILE* ofp = fopen(output_path, "a");
            if(!first_frame) fprintf(ofp, ", ");
            fprintf(ofp, "\"%s\": {", frame_id.c_str());
            if(n_pedestrians>0){
                fprintf(ofp, "\"pedestrian\": [");
                for(int i=0; i<n_pedestrians-1; ++i){
                    fprintf(ofp, "[%f, %f, %f], ", pedestrian_preds[i*3], pedestrian_preds[i*3+1], pedestrian_preds[i*3+2]);
                }
                fprintf(ofp, "[%f, %f, %f]", pedestrian_preds[(n_pedestrians-1)*3], pedestrian_preds[(n_pedestrians-1)*3+1], pedestrian_preds[(n_pedestrians-1)*3+2]);
                fprintf(ofp, "]");
            }
            if(n_vehicles>0){
                if(n_pedestrians>0){
                    fprintf(ofp, ", ");
                }
                fprintf(ofp, "\"vehicle\": [");
                for(int i=0; i<n_vehicles-1; ++i){
                    fprintf(ofp, "[%f, %f, %f], ", vehicle_preds[i*3], vehicle_preds[i*3+1], vehicle_preds[i*3+2]);
                }
                fprintf(ofp, "[%f, %f, %f]", vehicle_preds[(n_vehicles-1)*3], vehicle_preds[(n_vehicles-1)*3+1], vehicle_preds[(n_vehicles-1)*3+2]);
                fprintf(ofp, "]");
            }
            fprintf(ofp, "}\n");
            fclose(ofp);
            last_write_frame_idx = frame_idx;
            free(lidar_points);
            free(pedestrian_preds);
            free(vehicle_preds);
        };
        while(last_preprocess_frame_idx<frame_idx-1){
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        std::thread* pred_thread = new std::thread(
            pred_func, frame_idx, first_frame, std::string(frame_id), std::string(lidar_path),
            ego_translation[0], ego_translation[1], ego_translation[2],
            ego_rotation[0], ego_rotation[1], ego_rotation[2], ego_rotation[3]
        );
        prediction_threads.push(pred_thread);
        if(prediction_threads.size()>=2){
            auto thread = prediction_threads.front();
            prediction_threads.pop();
            thread->join();
            delete thread;
        }
        first_frame = false;
        ++frame_idx;
    }
    while(!prediction_threads.empty()){
        auto thread = prediction_threads.front();
        prediction_threads.pop();
        thread->join();
        delete thread;
    }
}

int main(int argc, char* argv[]){
    use_riscv = (strcmp(argv[3], "1") == 0);
    visualize = (strcmp(argv[4], "1") == 0);
    void* heap = 0;
    int ddr_fd = 0;
    if(use_riscv){
        char buf[1];
        unsigned int IMEM[4096];

        setup_gpio_in();
        setup_gpio_out();

        if((ddr_fd = open("/dev/mem", O_RDWR | O_SYNC)) < 0){
            perror("open");
            return -1;
        }

        volatile unsigned int* iram = (unsigned int*)mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_fd, IMEM_BASE);
        if (iram == MAP_FAILED){
            perror("mmap iram");
            close(ddr_fd);
            return -1;
        }

        dram = (char*)mmap(NULL, 0x10000000, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_fd, DMEM_BASE);
        if(dram == MAP_FAILED){
            perror("mmap dram");
            close(ddr_fd);
            return -1;
        }

        gpio = (unsigned int*)mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_fd, GPIO_BASE);
        if(gpio == MAP_FAILED) {
            perror("mmap gpio");
            close(ddr_fd);
            return -1;
        }

        pfd.events = POLLPRI;
        pfd.fd = open("/sys/class/gpio/gpio504/value", O_RDONLY);
        if(pfd.fd < 0){
            perror("failed to open gpio504 value");
            exit(EXIT_FAILURE);
        }

        // Initial setup
        poll(&pfd, 1, -1);
        lseek(pfd.fd, 0, SEEK_SET);
        read(pfd.fd, buf, 1);

        // GPIO direction
        REG(gpio + 1) = 0x00;
        //REG(gpio) = 0x02; // LED0

        // Write program
        unsigned int inum = riscv_imm(IMEM);
        for(int i=0; i<inum; ++i){
            REG(iram + i) = IMEM[i];
        }
        heap = malloc(256*1024*1024);
        base_addr = (char*)heap;
    }
    else{
        heap = malloc(256*1024*1024);
        base_addr = (char*)heap;
    }

    char* output_path = argv[1];
    char* xmodel_path = argv[2];
    BUFFERS_AVAIL = (volatile bool*)(base_addr + BUFFERS_AVAIL_ADDR_OFFSET);
    for(int i=0; i<N_BUFFERS; ++i){
        BUFFERS_AVAIL[i] = true;
    }
    if(use_riscv){
        RISCV_BUFFERS_AVAIL = (volatile bool*)(dram + BUFFERS_AVAIL_ADDR_OFFSET);
        for(int i=0; i<N_BUFFERS; ++i){
            RISCV_BUFFERS_AVAIL[i] = true;
        }
    }

    FILE* ofp = fopen(output_path, "w");
    fprintf(ofp, "{");
    fclose(ofp);
    predict_scene(output_path, xmodel_path);
    ofp = fopen(output_path, "a");
    fprintf(ofp, "}\n");
    fclose(ofp);

    if(use_riscv){
        close(ddr_fd);
        close(pfd.fd);
    }
    else{
        free(heap);
    }
}
