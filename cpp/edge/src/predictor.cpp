#include <cstdint>
#include <vart/mm/host_flat_tensor_buffer.hpp>
#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/util/data_type.hpp>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <poll.h>
#include "common.h"
typedef char BOOL;
typedef signed char int8_t;

//#define true 1
//#define false 0
#define LIDAR_IMAGE_WIDTH 1152
#define LIDAR_IMAGE_HEIGHT 1152
#define LIDAR_IMAGE_DEPTH 24
#define N_BUFFERS 18
#define BUFFERS_AVAIL_ADDR_OFFSET 251658240 /* 240*1024*1024 */
#define FUNC_PREPROCESS 0
#define FUNC_SUPPRESS 1
#define REG(address) *(volatile unsigned int*)(address)
#define REGF(address) *(volatile float*)(address)
#define GPIO_BASE (0x80010000)
#define IMEM_BASE (0x82000000)
#define DMEM_BASE (0x10000000)


char* base_addr = 0;
bool use_riscv = false;
uintptr_t BUFFERS[N_BUFFERS] = {
    0,
    10*1024*1024,
    20*1024*1024,
    30*1024*1024,
    40*1024*1024,
    50*1024*1024,
    60*1024*1024,
    70*1024*1024,
    80*1024*1024,
    90*1024*1024,
    100*1024*1024,
    110*1024*1024,
    120*1024*1024,
    130*1024*1024,
    140*1024*1024,
    150*1024*1024,
    160*1024*1024,
    170*1024*1024,
};
volatile bool* BUFFERS_AVAIL = 0;
volatile bool* RISCV_BUFFERS_AVAIL = 0;
uintptr_t LIDAR_IMAGE_BUFFER = 180*1024*1024;
uintptr_t RECORD_BUFFER = 220*1024*1024;
uintptr_t RISCV_ARGS_BUFFER = 239*1024*1024;
volatile unsigned int* iram = 0;
char* dram = 0;
volatile unsigned int* gpio = 0;
int ddr_fd = 0;
struct pollfd pfd;

unsigned int riscv_imm(unsigned int *IMEM);
unsigned int riscv_dmm(unsigned int *DMEM);
void setup_gpio_in();
void setup_gpio_out();
void wait_rising();


/**
 * gpio495の設定をする
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
 * gpio500の設定をする
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
    for(int i=0; i<N_BUFFERS; ++i){
        if(RISCV_BUFFERS_AVAIL[i]){
            RISCV_BUFFERS_AVAIL[i] = false;
            return (void*)(dram + BUFFERS[i]);
        }
    }
    return 0;
}

void rfree(void* ptr){
    int idx = ((uintptr_t)ptr-(uintptr_t)dram) / (10*1024*1024);
    RISCV_BUFFERS_AVAIL[idx] = true;
}

void* alloc(){
    for(int i=0; i<N_BUFFERS; ++i){
        if(BUFFERS_AVAIL[i]){
            BUFFERS_AVAIL[i] = false;
            return (void*)(base_addr + BUFFERS[i]);
        }
    }
    return 0;
}

void mfree(void* ptr){
    int idx = ((uintptr_t)ptr-(uintptr_t)base_addr) / (10*1024*1024);
    BUFFERS_AVAIL[idx] = true;
}

void run_riscv(){
    // Run Program
    REG(gpio) = 0x03; // LED1 + Reset off

    // Wait Program end
    poll(&pfd, 1, -1);
    lseek(pfd.fd, 0, SEEK_SET);
    char buf[1];
    read(pfd.fd, buf, 1);

    REG(gpio) = 0x00; // Reset on
}

int8_t* riscv_preprocess(float* lidar_points, int n_points, float z_offset, int input_quant_scale){
    std::chrono::system_clock::time_point t0 = std::chrono::system_clock::now();
    float* riscv_lidar_points = (float*)ralloc();
    for(int i=0; i<n_points*5; ++i){
        riscv_lidar_points[i] = lidar_points[i];
    }
    std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
    double d0 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0);
    printf("copy1 time[ms]: %lf\n", d0);
    char* riscv_args = 
        (char*)(dram + RISCV_ARGS_BUFFER);
    unsigned int* func = (unsigned int*)riscv_args;
    *func = FUNC_PREPROCESS;
    unsigned int* arg_lidar_points = (unsigned int*)(riscv_args + 64);
    *arg_lidar_points = (long)riscv_lidar_points - (long)dram;
    int* arg_n_points = (int*)(riscv_args + 72);
    *arg_n_points = n_points;
    float* arg_z_offset = (float*)(riscv_args + 80);
    *arg_z_offset = z_offset;
    int* arg_input_quant_scale = (int*)(riscv_args + 88);
    *arg_input_quant_scale = input_quant_scale;

    std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
    run_riscv();
    std::chrono::system_clock::time_point t3 = std::chrono::system_clock::now();
    double d1 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.0);
    printf("riscv time[ms]: %lf\n", d1);

    unsigned int* lidar_image_addr = (unsigned int*)(riscv_args + 96);
    int8_t* riscv_lidar_image = (int8_t*)(dram + *lidar_image_addr);
    printf("lidar image addr: %ld\n", (long)*lidar_image_addr);
    int8_t* lidar_image = (int8_t*)(base_addr + LIDAR_IMAGE_BUFFER);
    unsigned long long* dst = (unsigned long long*)lidar_image;
    unsigned long long* src = (unsigned long long*)riscv_lidar_image;
    std::chrono::system_clock::time_point t4 = std::chrono::system_clock::now();
    std::memcpy(dst, src, 1152*1152*24);
    //for(int i=0; i<1152*1152*24/8; ++i){
        //dst[i] = src[i];
        ////lidar_image[i] = riscv_lidar_image[i];
    //}
    std::chrono::system_clock::time_point t5 = std::chrono::system_clock::now();
    double d2 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count() / 1000.0);
    printf("copy2 time[ms]: %lf\n", d2);
    rfree(riscv_lidar_points);

    return lidar_image;
}

int8_t* preprocess(float* lidar_points, int n_points, float z_offset, int input_quant_scale){
    if(use_riscv){
        return riscv_preprocess(lidar_points, n_points, z_offset, input_quant_scale);
    }
    int* lidar_xs = (int*)alloc();
    int* lidar_ys = (int*)alloc();
    int* lidar_zs = (int*)alloc();
    int8_t* intensities = (int8_t*)alloc();
    int offset = 0;
    int n_valid_points = 0;
    float scale = (float)(1 << input_quant_scale);
    for(int i=0; i<n_points; ++i){
        int x = (int)(lidar_points[offset]*10.0f+0.5f) + 576;
        int y = (int)(-lidar_points[offset+1]*10.0f+0.5f) + 576;
        int z = (int)((lidar_points[offset+2]+z_offset)*5.0f+0.5f);
        lidar_xs[i] = x;
        lidar_ys[i] = y;
        lidar_zs[i] = z;
        if(x>=0 && x<1152 && y>=0 && y<1152 && z>=0 && z<24){
            lidar_xs[n_valid_points] = x;
            lidar_ys[n_valid_points] = y;
            lidar_zs[n_valid_points] = z;
            float intensity = lidar_points[offset+3]*scale+0.5f;
            intensities[n_valid_points] = (intensity>127.0f? 127 :  (int8_t)intensity);
            if(intensities[n_valid_points]==0) intensities[n_valid_points] = 1;
            ++n_valid_points;
        }
        offset += 5;
    }
    int8_t* lidar_image = (int8_t*)(base_addr + LIDAR_IMAGE_BUFFER);
    for(long long i=0; i<LIDAR_IMAGE_HEIGHT*LIDAR_IMAGE_WIDTH*LIDAR_IMAGE_DEPTH; ++i){
        lidar_image[i] = 0;
    }
    for(int i=0; i<n_valid_points; ++i){
        int offset = lidar_ys[i]*LIDAR_IMAGE_WIDTH*LIDAR_IMAGE_DEPTH + lidar_xs[i]*LIDAR_IMAGE_DEPTH + lidar_zs[i];
        lidar_image[offset] = (lidar_image[offset]<intensities[i]? intensities[i] : lidar_image[offset]);
    }
    mfree(lidar_xs);
    mfree(lidar_ys);
    mfree(lidar_zs);
    mfree(intensities);
    return lidar_image;
}

float* sigmoid(int* pred, int output_quant_scale){
    static const float sigmoid_table[256] = {1.2664165549094016e-14, 1.6261110446177924e-14, 2.08796791164589e-14, 2.6810038677817314e-14, 3.442477108469858e-14, 4.420228103640978e-14, 5.6756852326323996e-14, 7.287724095819161e-14, 9.357622968839299e-14, 1.2015425731770343e-13, 1.5428112031916497e-13, 1.9810087980485874e-13, 2.543665647376276e-13, 3.2661313427863805e-13, 4.193795658377786e-13, 5.384940217751136e-13, 6.914400106935423e-13, 8.878265478451776e-13, 1.1399918530430558e-12, 1.4637785141237662e-12, 1.8795288165355508e-12, 2.4133627718273897e-12, 3.0988191387122225e-12, 3.978962535821408e-12, 5.109089028037221e-12, 6.560200168110743e-12, 8.423463754397692e-12, 1.0815941557168708e-11, 1.3887943864771144e-11, 1.7832472907828393e-11, 2.289734845593124e-11, 2.940077739198032e-11, 3.7751345441365816e-11, 4.847368706035286e-11, 6.224144622520383e-11, 7.991959892315218e-11, 1.0261879630648827e-10, 1.3176514268359263e-10, 1.6918979223288784e-10, 2.1724399346070674e-10, 2.7894680920908113e-10, 3.581747929000289e-10, 4.599055376537186e-10, 5.905303995456778e-10, 7.582560422162385e-10, 9.736200303530205e-10, 1.2501528648238605e-09, 1.6052280526088547e-09, 2.0611536181902037e-09, 2.646573631904765e-09, 3.398267807946847e-09, 4.363462233903898e-09, 5.602796406145941e-09, 7.194132978569834e-09, 9.23744957664012e-09, 1.1861120010657661e-08, 1.522997951276035e-08, 1.955568070542584e-08, 2.5109990926928157e-08, 3.2241866333029355e-08, 4.1399375473943306e-08, 5.3157849718487075e-08, 6.825602910446286e-08, 8.764247451323235e-08, 1.12535162055095e-07, 1.4449800373124837e-07, 1.8553910183683314e-07, 2.38236909993343e-07, 3.059022269256247e-07, 3.927862002670442e-07, 5.043474082014517e-07, 6.475947982049267e-07, 8.315280276641321e-07, 1.067702870044147e-06, 1.3709572068578448e-06, 1.7603432133424856e-06, 2.2603242979035746e-06, 2.902311985211097e-06, 3.726639284186561e-06, 4.785094494890119e-06, 6.144174602214718e-06, 7.889262586245034e-06, 1.0129990980873921e-05, 1.3007128466476033e-05, 1.670142184809518e-05, 2.144494842091395e-05, 2.7535691114583473e-05, 3.5356250741744315e-05, 4.5397868702434395e-05, 5.829126566113865e-05, 7.484622751061123e-05, 9.610241549947396e-05, 0.00012339457598623172, 0.00015843621910252592, 0.00020342697805520653, 0.0002611903190957194, 0.0003353501304664781, 0.0004305570813246149, 0.0005527786369235996, 0.0007096703991005881, 0.0009110511944006454, 0.0011695102650555148, 0.0015011822567369917, 0.0019267346633274757, 0.0024726231566347743, 0.0031726828424851893, 0.004070137715896128, 0.005220125693558397, 0.0066928509242848554, 0.008577485413711984, 0.01098694263059318, 0.014063627043245475, 0.01798620996209156, 0.022977369910025615, 0.02931223075135632, 0.03732688734412946, 0.04742587317756678, 0.060086650174007626, 0.07585818002124355, 0.09534946489910949, 0.11920292202211755, 0.14804719803168948, 0.18242552380635635, 0.22270013882530884, 0.2689414213699951, 0.320821300824607, 0.3775406687981454, 0.43782349911420193, 0.5, 0.5621765008857981, 0.6224593312018546, 0.679178699175393, 0.7310585786300049, 0.7772998611746911, 0.8175744761936437, 0.8519528019683106, 0.8807970779778823, 0.9046505351008906, 0.9241418199787566, 0.9399133498259924, 0.9525741268224334, 0.9626731126558706, 0.9706877692486436, 0.9770226300899744, 0.9820137900379085, 0.9859363729567544, 0.9890130573694068, 0.991422514586288, 0.9933071490757153, 0.9947798743064417, 0.995929862284104, 0.9968273171575148, 0.9975273768433653, 0.9980732653366725, 0.998498817743263, 0.9988304897349445, 0.9990889488055994, 0.9992903296008995, 0.9994472213630764, 0.9995694429186754, 0.9996646498695336, 0.9997388096809043, 0.9997965730219448, 0.9998415637808975, 0.9998766054240137, 0.9999038975845005, 0.9999251537724895, 0.9999417087343389, 0.9999546021312976, 0.9999646437492582, 0.9999724643088853, 0.9999785550515792, 0.999983298578152, 0.9999869928715335, 0.9999898700090192, 0.9999921107374138, 0.9999938558253978, 0.9999952149055051, 0.9999962733607158, 0.9999970976880148, 0.999997739675702, 0.9999982396567868, 0.999998629042793, 0.9999989322971299, 0.9999991684719722, 0.9999993524052017, 0.9999994956525918, 0.9999996072137998, 0.999999694097773, 0.9999997617630899, 0.9999998144608981, 0.9999998555019962, 0.9999998874648379, 0.9999999123575255, 0.999999931743971, 0.9999999468421502, 0.9999999586006244, 0.9999999677581336, 0.999999974890009, 0.9999999804443193, 0.9999999847700205, 0.99999998813888, 0.9999999907625504, 0.9999999928058669, 0.9999999943972036, 0.9999999956365377, 0.9999999966017321, 0.9999999973534264, 0.9999999979388463, 0.999999998394772, 0.9999999987498471, 0.9999999990263799, 0.9999999992417439, 0.9999999994094697, 0.9999999995400946, 0.9999999996418252, 0.9999999997210531, 0.999999999782756, 0.9999999998308102, 0.999999999868235, 0.9999999998973812, 0.9999999999200804, 0.9999999999377585, 0.9999999999515263, 0.9999999999622486, 0.9999999999705993, 0.9999999999771028, 0.9999999999821676, 0.999999999986112, 0.999999999989184, 0.9999999999915765, 0.9999999999934397, 0.999999999994891, 0.999999999996021, 0.9999999999969011, 0.9999999999975866, 0.9999999999981204, 0.9999999999985363, 0.99999999999886, 0.9999999999991123, 0.9999999999993086, 0.9999999999994615, 0.9999999999995806, 0.9999999999996734, 0.9999999999997455, 0.9999999999998019, 0.9999999999998457, 0.9999999999998799, 0.9999999999999065, 0.9999999999999272, 0.9999999999999432, 0.9999999999999558, 0.9999999999999656, 0.9999999999999731, 0.9999999999999791, 0.9999999999999838};
    float* sigmoid_pred = (float*)alloc();
    //int scale = 7 - output_quant_scale;
    for(int i=0; i<1024*1024; ++i){
        sigmoid_pred[i] = sigmoid_table[pred[i]];
    }
    return sigmoid_pred;
}

void riscv_suppress_predictions_without_lidar_points(int8_t* input_image, float* centroid, float* confidence, int n_preds){
    std::chrono::system_clock::time_point t0 = std::chrono::system_clock::now();

    volatile float* riscv_centroid = (volatile float*)ralloc();
    volatile float* riscv_confidence = (volatile float*)ralloc();

    for(int i=0; i<n_preds*2; ++i){
        riscv_centroid[i] = centroid[i];
    }
    for(int i=0; i<n_preds; ++i){
        riscv_confidence[i] = confidence[i];
    }

    char* riscv_args = 
        (char*)(dram + RISCV_ARGS_BUFFER);
    unsigned int* func = (unsigned int*)riscv_args;
    *func = FUNC_SUPPRESS;
    unsigned int* arg_centroid = (unsigned int*)(riscv_args + 72);
    *arg_centroid = (long)riscv_centroid - (long)dram;
    unsigned int* arg_confidence = (unsigned int*)(riscv_args + 80);
    *arg_confidence = (long)riscv_confidence - (long)dram;
    int* arg_n_preds = (int*)(riscv_args + 88);
    *arg_n_preds = n_preds;

    run_riscv();

    for(int i=0; i<n_preds; ++i){
        confidence[i] = riscv_confidence[i];
    }

    rfree((float*)riscv_centroid);
    rfree((float*)riscv_confidence);
    std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
    double d0 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0);
    printf("riscv suppress time[ms]: %lf\n", d0);
}

void suppress_predictions_without_lidar_points(int8_t* input_image, float* centroid, float* confidence, int n_preds){
    if(use_riscv){
        riscv_suppress_predictions_without_lidar_points(
            input_image, centroid, confidence, n_preds
        );
        return;
    }
    for(int i=0; i<n_preds; ++i){
        int x = (int)(centroid[i*2]+0.5) + 64;
        int y = (int)(centroid[i*2+1]+0.5) + 64;
        BOOL has_points = false;
        int sx = x>7? (x-7) : 0;
        int sy = y>7? (y-7) : 0;
        int ex = x<LIDAR_IMAGE_WIDTH-7? (x+7) : (LIDAR_IMAGE_WIDTH-1);
        int ey = y<LIDAR_IMAGE_HEIGHT-7? (y+7) : (LIDAR_IMAGE_HEIGHT-1);
        for(int ry=sy; ry<ey; ++ry){
            int offset = ry * LIDAR_IMAGE_WIDTH;
            for(int rx=sx; rx<ex; ++rx){
                if(input_image[offset+rx]>0){
                    has_points = true;
                    break;
                }
            }
            if(has_points) break;
        }
        if(!has_points){
            confidence[y*(LIDAR_IMAGE_WIDTH)+x] *= 0.1;
        }
    }
}

BOOL* get_pedestrian_mask(float* pedestrian_fy){
    BOOL* buffer = (BOOL*)alloc();
    BOOL* pedestrian_m = (BOOL*)alloc();
    for(int i=0; i<LIDAR_IMAGE_WIDTH*LIDAR_IMAGE_HEIGHT; ++i){
        buffer[i] = pedestrian_m[i] = (pedestrian_fy[i]>0.28);
    }
    for(int y=0; y<1023; ++y){
        int offset0 = y * 1024;
        int offset1 = offset0 + 1024;
        for(int x=0; x<1024; ++x){
            pedestrian_m[offset0+x] |= buffer[offset1+x];
            pedestrian_m[offset1+x] |= buffer[offset0+x];
        }
    }
    for(int y=0; y<1024; ++y){
        int offset0 = y * 1024;
        int offset1 = offset0 + 1;
        for(int x=0; x<1023; ++x){
            pedestrian_m[offset0+x] |= buffer[offset1+x];
            pedestrian_m[offset1+x] |= buffer[offset0+x];
        }
    }
    for(int y=0; y<1024; ++y){
        int offset = y * 1024;
        for(int x=0; x<1024; ++x){
            pedestrian_m[offset+x] = buffer[offset+x] || ((!pedestrian_m[offset+x]) && pedestrian_fy[offset+x]>0.012);
        }
    }
    mfree((void*)buffer);
    return pedestrian_m;
}

BOOL* get_vehicle_mask(float* vehicle_fy){
    BOOL* buffer = (BOOL*)alloc();
    BOOL* vehicle_m = (BOOL*)alloc();
    for(int i=0; i<LIDAR_IMAGE_WIDTH*LIDAR_IMAGE_HEIGHT; ++i){
        buffer[i] = vehicle_m[i] = (vehicle_fy[i]>0.19);
    }
    for(int y=0; y<1021; ++y){
        int offset0 = y * 1024;
        int offset1 = offset0 + 3072;
        for(int x=0; x<1024; ++x){
            vehicle_m[offset0+x] |= buffer[offset1+x];
            vehicle_m[offset1+x] |= buffer[offset0+x];
        }
    }
    for(int y=0; y<1024; ++y){
        int offset0 = y * 1024;
        int offset1 = offset0 + 3;
        for(int x=0; x<1021; ++x){
            vehicle_m[offset0+x] |= buffer[offset1+x];
            vehicle_m[offset1+x] |= buffer[offset0+x];
        }
    }
    for(int y=0; y<1024; ++y){
        int offset = y * 1024;
        for(int x=0; x<1024; ++x){
            vehicle_m[offset+x] = buffer[offset+x] || ((!vehicle_m[offset+x]) && vehicle_fy[offset+x]>0.1);
        }
    }
    mfree((void*)buffer);
    return vehicle_m;
}

bool FALSE = false;

void cca(float* p, BOOL* m, int* n_centroids, float* scores, int* areas, float* centroids){
    std::chrono::system_clock::time_point t0 = std::chrono::system_clock::now();
    BOOL* checked = (BOOL*)alloc();
    unsigned long long* dst = (unsigned long long*)checked;
    std::memset(dst, 0, 1024*1024);
    *n_centroids = 0;
    int* coords = (int*)alloc();
    for(int y=0; y<1024; ++y){
        int offset_y = y*1024;
        for(int x=0; x<1024; ++x){
            int offset = offset_y + x;
            if(checked[offset]) continue;
            if(m[offset]==0){
                checked[offset] = true;
                continue;
            }
            float score = p[offset];
            int area = 1;
            long long ys = y;
            long long xs = x;
            coords[0] = x;
            coords[1] = y;
            checked[offset] = true;
            int idx = 0;
            while(idx<area){
                int tx = coords[idx*2+0];
                int ty = coords[idx*2+1];
                if(tx>0 && !checked[ty*1024+tx-1] && m[ty*1024+tx-1]==1){
                    coords[area*2+0] = tx - 1;
                    coords[area*2+1] = ty;
                    ++area;
                    checked[ty*1024+tx-1] = true;
                    ys += ty;
                    xs += tx - 1;
                    score = (score<p[ty*1024+tx-1]? p[ty*1024+tx-1] : score);
                }
                if(tx<1024-1 && !checked[ty*1024+tx+1] && m[ty*1024+tx+1]==1){
                    coords[area*2+0] = tx + 1;
                    coords[area*2+1] = ty;
                    ++area;
                    checked[ty*1024+tx+1] = true;
                    ys += ty;
                    xs += tx + 1;
                    score = (score<p[ty*1024+tx+1]? p[ty*1024+tx+1] : score);
                }
                if(ty>0 && !checked[(ty-1)*1024+tx] && m[(ty-1)*1024+tx]==1){
                    coords[area*2+0] = tx;
                    coords[area*2+1] = ty - 1;
                    ++area;
                    checked[(ty-1)*1024+tx] = true;
                    ys += ty - 1;
                    xs += tx;
                    score = (score<p[(ty-1)*1024+tx]? p[(ty-1)*1024+tx] : score);
                }
                if(ty<1024-1 && !checked[(ty+1)*1024+tx] && m[(ty+1)*1024+tx]==1){
                    coords[area*2+0] = tx;
                    coords[area*2+1] = ty + 1;
                    ++area;
                    checked[(ty+1)*1024+tx] = true;
                    ys += ty + 1;
                    xs += tx;
                    score = (score<p[(ty+1)*1024+tx]? p[(ty+1)*1024+tx] : score);
                }
                ++idx;
            }
            float cx = (float)xs / (float)area;
            float cy = (float)ys / (float)area;
            centroids[*n_centroids*2] = cx;
            centroids[*n_centroids*2+1] = cy;
            scores[*n_centroids] = score;
            areas[*n_centroids] = area;
            ++*n_centroids;
        }
    }
    mfree((void*)checked);
    mfree((void*)coords);
}

void postprocess(float* pedestrian_pred, float* vehicle_pred, float* pedestrian_centroid, float* pedestrian_confidence, int* n_pedestrians, float* vehicle_centroid, float* vehicle_confidence, int* n_vehicles, int frame_idx, float* records){
    BOOL* pedestrian_m = get_pedestrian_mask(pedestrian_pred);
    BOOL* vehicle_m = get_vehicle_mask(vehicle_pred);

    int* pedestrian_areas = (int*)alloc();
    cca(pedestrian_pred, pedestrian_m, n_pedestrians, pedestrian_confidence, pedestrian_areas, pedestrian_centroid);
    int n_filtered_pedestrians = 0;
    for(int i=0; i<*n_pedestrians; ++i){
        if(pedestrian_areas[i]>78 && pedestrian_confidence[i]>0.28){
            pedestrian_areas[*n_pedestrians+n_filtered_pedestrians] = pedestrian_areas[i];
            pedestrian_confidence[*n_pedestrians+n_filtered_pedestrians] = pedestrian_confidence[i];
            pedestrian_centroid[(*n_pedestrians+n_filtered_pedestrians)*2] = pedestrian_centroid[i*2];
            pedestrian_centroid[(*n_pedestrians+n_filtered_pedestrians)*2+1] = pedestrian_centroid[i*2+1];
            ++n_filtered_pedestrians;
        }
    }
    *n_pedestrians += n_filtered_pedestrians;

    int* vehicle_areas = (int*)alloc();
    cca(vehicle_pred, vehicle_m, n_vehicles, vehicle_confidence, vehicle_areas, vehicle_centroid);

    mfree((void*)pedestrian_m);
    mfree((void*)vehicle_m);
    mfree((void*)pedestrian_areas);
    mfree((void*)vehicle_areas);
}

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

void rotate(float inp[3], float outp[3], float mx[3][3]){
    outp[0] = mx[0][0]*inp[0] + mx[0][1]*inp[1] + mx[0][2]*inp[2];
    outp[1] = mx[1][0]*inp[0] + mx[1][1]*inp[1] + mx[1][2]*inp[2];
    //outp[2] = mx[2][0]*inp[0] + mx[2][1]*inp[1] + mx[2][2]*inp[2];
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

void refine_predictions(float* preds, int* n_preds, float ego_translation[3], float max_dist, float fuzzy_dist, float fuzzy_rate, float refine_dist, int n_cutoff, bool is_pedestrian){
    float* refined_preds = (float*)alloc();
    int n_refined_preds = 0;
    for(int i=0; i<*n_preds; ++i){
        float dx = preds[i*3] - ego_translation[0];
        float dy = preds[i*3+1] - ego_translation[1];
        float d = sqrt(dx*dx + dy*dy);
        if(d>max_dist) continue;
        refined_preds[n_refined_preds*5+1] = 1e10;
        refined_preds[n_refined_preds*5+3] = preds[i*3];
        refined_preds[n_refined_preds*5+4] = preds[i*3+1];
        if(d>fuzzy_dist){
            refined_preds[n_refined_preds*5] = preds[i*3+2] * fuzzy_rate;
            refined_preds[n_refined_preds*5+2] = preds[i*3+2] * fuzzy_rate;
        }
        else{
            refined_preds[n_refined_preds*5] = preds[i*3+2];
            refined_preds[n_refined_preds*5+2] = preds[i*3+2];
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
                if(refined_preds[n*5+1]==0.0){
                    refined_preds[n*5] = refined_preds[n*5+2];
                }
                else{
                    float r = m * (refined_preds[n*5+1]>refine_dist? refine_dist: refined_preds[n*5+1]);
                    refined_preds[n*5] = refined_preds[n*5+2] * r;
                }
            }
            else{
                if(refined_preds[n*5+1]<0.4){
                    refined_preds[n*5] = refined_preds[n*5+2] * 0.01;
                }
                else{
                    float r = m * (refined_preds[n*5+1]>refine_dist? refine_dist: refined_preds[n*5+1]);
                    refined_preds[n*5] = refined_preds[n*5+2] * r;
                }
            }
        }
    }
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

void merge_prev_preds(float* pred, float ego_translation[3], float ego_rotation[4], int frame_idx, float* records, int category){
    int offset0 = (frame_idx%2+1) * (1024*1024*2 + 8);
    int offset1 = ((frame_idx)%2) * (1024*1024*2 + 8);

    float* pred0 = records + offset0 + category*1024*1024;
    float* txyz0 = records + offset0 + 1024*1024*2;
    float* qt0 = records + offset0 + 1024*1024*2 + 3;

    float* pred1 = records + offset1 + category*1024*1024;
    float* txyz1 = records + offset1 + 1024*1024*2;
    float* qt1 = records + offset1 + 1024*1024*2 + 3;

    float dtx0 = ego_translation[0] - txyz0[0];
    float dty0 = ego_translation[1] - txyz0[1];
    float dtz0 = ego_translation[2] - txyz0[2];

    float dtx1 = ego_translation[0] - txyz1[0];
    float dty1 = ego_translation[1] - txyz1[1];
    float dtz1 = ego_translation[2] - txyz1[2];

    float iqt[4] = {};
    inverse_quaternion(ego_rotation, iqt);

    float dr0[4] = {};
    composite_quaternions((float*)qt0, iqt, dr0);
    float dr1[4] = {};
    composite_quaternions((float*)qt1, iqt, dr1);

    float* rpred0 = (float*)alloc();
    float mx[3][3] = {};
    quaternion_to_matrix(dr0, mx);
    for(int y=0; y<1024; ++y){
        for(int x=0; x<1024; ++x){
            float dxyz[3] = {(float)(x - 512), (float)(y - 512), 0.0};
            float sxyz[3] = {};
            rotate(dxyz, sxyz, mx);
            int sx = (int)(sxyz[0] + dtx0*10 + 0.5) + 512;
            int sy = (int)(sxyz[1] - dty0*10 + 0.5) + 512;
            if(sx>=0 && sx<1024 && sy>=0 && sy<1024){
                rpred0[y*1024 + x] = pred0[sy*1024 + sx];
            }
            else{
                rpred0[y*1024 + x] = 0;
            }
        }
    }

    float* rpred1 = (float*)alloc();
    quaternion_to_matrix(dr1, mx);
    for(int y=0; y<1024; ++y){
        for(int x=0; x<1024; ++x){
            float dxyz[3] = {(float)(x - 512), (float)(y - 512), 0.0};
            float sxyz[3] = {};
            rotate(dxyz, sxyz, mx);
            int sx = (int)(sxyz[0] + dtx1*10 + 0.5) + 512;
            int sy = (int)(sxyz[1] - dty1*10 + 0.5) + 512;
            if(sx>=0 && sx<1024 && sy>=0 && sy<1024){
                rpred1[y*1024 + x] = pred1[sy*1024 + sx];
            }
            else{
                rpred1[y*1024 + x] = 0;
            }
        }
    }

    for(int y=0; y<1024; ++y){
        for(int x=0; x<1024; ++x){
            float v0 = pred[y*1024 + x];
            float v1 = rpred0[y*1024 + x];
            float v2 = rpred1[y*1024 + x];
            float a = v0>v2? v0 : v2;
            float b = a>v1? v1 : a;
            pred[y*1024 + x] = (b<v0? v0 : b);
        }
    }

    mfree((void*)rpred0);
    mfree((void*)rpred1);
}

void run_dpu(vart::Runner* runner, int8_t* input_image, int8_t* pred){
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
    runner->wait(job_id.first, -1);
}


void predict(float* lidar_points, int n_points, int input_quant_scale, int output_quant_scale, float ego_translation[3], float ego_rotation[4], float* pedestrian_preds, float* vehicle_preds, int* n_pedestrians, int* n_vehicles, vart::Runner* runner, int frame_idx, float* records){
    std::chrono::system_clock::time_point t0 = std::chrono::system_clock::now();
    int8_t* input_image = preprocess(lidar_points, n_points, 3.7, input_quant_scale);
    std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();

    int8_t* quant_pred = (int8_t*)alloc();
    run_dpu(runner, (int8_t*)input_image, (int8_t*)quant_pred);
    std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();

    int* quant_pedestrian_pred = (int*)alloc();
    int* quant_vehicle_pred = (int*)alloc();
    for(int y=0; y<1024; ++y){
        for(int x=0; x<1024; ++x){
            int src = (y+64)*1152*2 + (x+64)*2;
            int dst = y*1024 + x;
            quant_pedestrian_pred[dst] = (int)quant_pred[src] + 128;
            quant_vehicle_pred[dst] = (int)quant_pred[src+1] + 128;
        }
    }
    mfree(quant_pred);

    float* pedestrian_pred = sigmoid(quant_pedestrian_pred, output_quant_scale);
    mfree(quant_pedestrian_pred);

    float* vehicle_pred = sigmoid(quant_vehicle_pred, output_quant_scale);
    mfree(quant_vehicle_pred);
    std::chrono::system_clock::time_point t3 = std::chrono::system_clock::now();

    float* pedestrian_centroids = (float*)alloc();
    float* pedestrian_confidence = (float*)alloc();
    float* vehicle_centroids = (float*)alloc();
    float* vehicle_confidence = (float*)alloc();

    if(frame_idx>=2){
        merge_prev_preds(pedestrian_pred, ego_translation, ego_rotation, frame_idx, records, 0);
        merge_prev_preds(vehicle_pred, ego_translation, ego_rotation, frame_idx, records, 1);
    }
    std::chrono::system_clock::time_point t4 = std::chrono::system_clock::now();

    postprocess(pedestrian_pred, vehicle_pred, pedestrian_centroids, pedestrian_confidence, n_pedestrians, vehicle_centroids, vehicle_confidence, n_vehicles, frame_idx, records);
    std::chrono::system_clock::time_point t5 = std::chrono::system_clock::now();
    //printf("pp n %d %d\n", *n_pedestrians, *n_vehicles);

    suppress_predictions_without_lidar_points(input_image, pedestrian_centroids, pedestrian_confidence, *n_pedestrians);
    std::chrono::system_clock::time_point t6 = std::chrono::system_clock::now();

    // update records
    int record_offset = (frame_idx%2) * (1024*1024*2+8);
    for(int i=0; i<1024*1024; ++i){
        records[i+record_offset] = pedestrian_pred[i];
    }
    record_offset += 1024 * 1024;
    for(int i=0; i<1024*1024; ++i){
        records[i+record_offset] = vehicle_pred[i];
    }
    record_offset += 1024 * 1024;
    records[record_offset] = ego_translation[0];
    records[record_offset+1] = ego_translation[1];
    records[record_offset+2] = ego_translation[2];
    records[record_offset+3] = ego_rotation[0];
    records[record_offset+4] = ego_rotation[1];
    records[record_offset+5] = ego_rotation[2];
    records[record_offset+6] = ego_rotation[3];

    mfree(pedestrian_pred);
    mfree(vehicle_pred);

    std::chrono::system_clock::time_point t7 = std::chrono::system_clock::now();

    scale_rotate_translate(pedestrian_centroids, *n_pedestrians, ego_translation, ego_rotation);
    scale_rotate_translate(vehicle_centroids, *n_vehicles, ego_translation, ego_rotation);

    for(int i=0; i<*n_pedestrians; ++i){
        pedestrian_preds[i*3] = pedestrian_centroids[i*2];
        pedestrian_preds[i*3+1] = pedestrian_centroids[i*2+1];
        pedestrian_preds[i*3+2] = pedestrian_confidence[i];
    }

    mfree(pedestrian_centroids);
    mfree(pedestrian_confidence);

    for(int i=0; i<*n_vehicles; ++i){
        vehicle_preds[i*3] = vehicle_centroids[i*2];
        vehicle_preds[i*3+1] = vehicle_centroids[i*2+1];
        vehicle_preds[i*3+2] = vehicle_confidence[i];
    }
    mfree(vehicle_centroids);
    mfree(vehicle_confidence);
    std::chrono::system_clock::time_point t8 = std::chrono::system_clock::now();

    refine_predictions(pedestrian_preds, n_pedestrians, ego_translation, 40, 39, 0.95, 0.8, 55, true);
    refine_predictions(vehicle_preds, n_vehicles, ego_translation, 50, 49.5, 0.9, 2.2, 60, false);
    *n_pedestrians = *n_pedestrians>50? 50 : *n_pedestrians;
    *n_vehicles = *n_vehicles>50? 50 : *n_vehicles;
    std::chrono::system_clock::time_point t9 = std::chrono::system_clock::now();

    double d0 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0);
    double d1 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0);
    double d2 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1000.0);
    double d3 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() / 1000.0);
    double d4 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count() / 1000.0);
    double d5 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count() / 1000.0);
    double d6 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t7 - t6).count() / 1000.0);
    double d7 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count() / 1000.0);
    double d8 = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t9 - t8).count() / 1000.0);
    double d_total = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t9 - t0).count() / 1000.0);
    printf("time[ms] total:%lf preproc:%lf dpu:%lf sigmoid:%lf affine:%lf postproc:%lf suppress:%lf record:%lf tx:%lf refine:%lf\n", d_total, d0, d1, d2, d3, d4, d5, d6, d7, d8);
}

void predict_scene(char* output_path, char* xmodel_path){
    printf("start scene\n");
    int frame_idx = 0;
    float* records = (float*)(base_addr + RECORD_BUFFER);
    float* ego_pose_records = (float*)alloc();
    float* pred_records = (float*)alloc();

    char frame_id[16];
    char lidar_path[256];
    float ego_translation[3];
    float ego_rotation[4];
    float* lidar_points = (float*)alloc();
    float* pedestrian_preds = (float*)alloc();
    float* vehicle_preds = (float*)alloc();
    auto graph = xir::Graph::deserialize(xmodel_path);
    auto subgraph = get_dpu_subgraph(graph.get());
    auto runner = vart::Runner::create_runner(subgraph[0], "run");
    auto outputTensors = runner->get_output_tensors();
    auto inputTensors = runner->get_input_tensors();
    int input_quant_scale = get_input_fix_point(inputTensors[0]);
    int output_quant_scale = get_output_fix_point(outputTensors[0]);
    bool first_frame = true;
    //printf("input scale: %d\n", input_quant_scale);
    //printf("output scale: %d\n", output_quant_scale);
    while(scanf("%s %s %f %f %f %f %f %f %f", frame_id, lidar_path, &ego_translation[0], &ego_translation[1], &ego_translation[2], &ego_rotation[0], &ego_rotation[1], &ego_rotation[2], &ego_rotation[3]) != EOF){
        printf("Frame ID: %s\n", frame_id);
        FILE* ifp = fopen(lidar_path, "r");
        int n_points = fread((float*)lidar_points, sizeof(float), 100000000, ifp) / 5;
        fclose(ifp);
        int n_pedestrians = 0;
        int n_vehicles = 0;
        predict(lidar_points, n_points, input_quant_scale, output_quant_scale, ego_translation, ego_rotation, pedestrian_preds, vehicle_preds, &n_pedestrians, &n_vehicles, runner.get(), frame_idx, records);
        FILE* ofp = fopen(output_path, "a");
        if(!first_frame) fprintf(ofp, ", ");
        fprintf(ofp, "\"%s\": {", frame_id);
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
        first_frame = false;
        ++frame_idx;
        if(frame_idx>=3){
            break;
        }
        //if(frame_idx>1) break;
    }
    mfree(ego_pose_records);
    mfree(pred_records);
    mfree(lidar_points);
    mfree(pedestrian_preds);
    mfree(vehicle_preds);
}

int main(int argc, char* argv[]){
    use_riscv = (strcmp(argv[3], "1") == 0);
    void* heap = 0;
    if(use_riscv){
        char buf[1];
        unsigned int IMEM[4096];

        setup_gpio_in();
        setup_gpio_out();

        //if((ddr_fd = open("/dev/mem", O_RDWR)) < 0){
        if((ddr_fd = open("/dev/mem", O_RDWR | O_SYNC)) < 0){
            perror("open");
            return -1;
        }

        iram = (unsigned int*)mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_fd, IMEM_BASE);
        if (iram == MAP_FAILED){
            perror("mmap iram");
            close(ddr_fd);
            return -1;
        }

        dram = (char*)mmap(NULL, 0x10000000, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_fd, DMEM_BASE);
        //base_addr = (char*)dram;
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

        // 初回
        poll(&pfd, 1, -1);
        lseek(pfd.fd, 0, SEEK_SET);
        read(pfd.fd, buf, 1);

        // GPIO Direction
        REG(gpio + 1) = 0x00;
        REG(gpio) = 0x02; // LED0

        // Write Program
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
