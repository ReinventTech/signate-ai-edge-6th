typedef signed char int8_t;
typedef char bool;

#define REG(address) *(volatile unsigned int*)(address)
#define REGF(address) *(volatile float*)(address)
#define REGC(address) *(volatile char*)(address)
#define REGC(address) *(volatile char*)(address)
#define DMEM_BASE  (0x10000000)
#define GPIO_BASE  (0x80030000)

#define LIDAR_IMAGE_WIDTH 1152
#define LIDAR_IMAGE_HEIGHT 1152
#define LIDAR_IMAGE_DEPTH 24
#define N_BUFFERS 18
#define BUFFERS_AVAIL_ADDR_OFFSET 251658240 /* 240*1024*1024 */

#define true 1
#define false 0

long base_addr = 0x10000000;
signed char ZERO = 0;

void* BUFFERS[N_BUFFERS] = {
    (void*)0,
    (void*)(10*1024*1024),
    (void*)(20*1024*1024),
    (void*)(30*1024*1024),
    (void*)(40*1024*1024),
    (void*)(50*1024*1024),
    (void*)(60*1024*1024),
    (void*)(70*1024*1024),
    (void*)(80*1024*1024),
    (void*)(90*1024*1024),
    (void*)(100*1024*1024),
    (void*)(110*1024*1024),
    (void*)(120*1024*1024),
    (void*)(130*1024*1024),
    (void*)(140*1024*1024),
    (void*)(150*1024*1024),
    (void*)(160*1024*1024),
    (void*)(170*1024*1024),
};
bool* BUFFERS_AVAIL = 0;
void* LIDAR_IMAGE_BUFFER = (void*)(180*1024*1024);
void* RECORD_BUFFER = (void*)(220*1024*1024);
void* RISCV_ARGS_BUFFER = (void*)(239*1024*1024);

void* alloc(){
    for(int i=0; i<N_BUFFERS; ++i){
        if(BUFFERS_AVAIL[i]){
            BUFFERS_AVAIL[i] = false;
            return (void*)((long)BUFFERS[i]+base_addr);
        }
    }
    return 0;
}

void mfree(void* ptr){
    int idx = ((long)ptr-base_addr) / (10*1024*1024);
    BUFFERS_AVAIL[idx] = true;
}

int8_t* preprocess(float* lidar_points, int n_points, float z_offset, int input_quant_scale){
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
    int8_t* lidar_image = (int8_t*)((long)LIDAR_IMAGE_BUFFER+base_addr);
    for(int i=0; i<LIDAR_IMAGE_HEIGHT*LIDAR_IMAGE_WIDTH*LIDAR_IMAGE_DEPTH; ++i){
        lidar_image[i] = ZERO;
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

int main(void)
{
    REG(GPIO_BASE + 4) = 0; // 出力に設定
	REG(GPIO_BASE) = 0;

    volatile void* args = (volatile void*)(DMEM_BASE + (long)RISCV_ARGS_BUFFER);

    char* func_name = (char*)args;

    if(func_name=="preprocess"){
        long* lidar_points_addr = (long*)((long)args + 64);
        float* lidar_points = (float*)(DMEM_BASE + *lidar_points_addr);
        int* n_points = (int*)((long)args + 72);
        float* z_offset = (float*)((long)args + 80);
        int* input_quant_scale = (int*)((long)args + 88);
        int8_t* lidar_image = preprocess(lidar_points, *n_points, *z_offset, *input_quant_scale);
        long* lidar_image_addr = (long*)((long)args + 96);
        *lidar_image_addr = (long)lidar_image;
    }

	REG(GPIO_BASE) = 0x01; // 終了通知
	while(1) {}

	return 0;
}
