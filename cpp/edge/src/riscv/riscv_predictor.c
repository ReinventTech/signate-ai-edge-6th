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

#define FUNC_PREPROCESS 0
#define FUNC_SUPPRESS 1

char* base_addr = (char*)0x10000000;
unsigned int BUFFERS[N_BUFFERS] = {
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
bool* BUFFERS_AVAIL = 0;
unsigned int LIDAR_IMAGE_BUFFER = 180*1024*1024;
unsigned int RECORD_BUFFER = 220*1024*1024;
unsigned int RISCV_ARGS_BUFFER = 239*1024*1024;

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
    int idx = ((unsigned int)ptr-(unsigned int)base_addr) / (10*1024*1024);
    BUFFERS_AVAIL[idx] = true;
}

/* Use this to initialize lidar_image, avoiding memset */
unsigned int ZERO = 0;

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
    int8_t* lidar_image = (int8_t*)(base_addr + LIDAR_IMAGE_BUFFER);
    unsigned int* tmp = (unsigned int*)lidar_image;
    for(int i=0; i<LIDAR_IMAGE_HEIGHT*LIDAR_IMAGE_WIDTH*LIDAR_IMAGE_DEPTH/4; ++i){
        tmp[i] = ZERO;
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

void suppress_predictions_without_lidar_points(int8_t* input_image, float* centroid, float* confidence, int n_preds){
    for(int i=0; i<n_preds; ++i){
        int x = (int)(centroid[i*2]+0.5f) + 64;
        int y = (int)(centroid[i*2+1]+0.5f) + 64;
        bool has_points = false;
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
        if(has_points==0){
            confidence[y*(LIDAR_IMAGE_WIDTH)+x] *= 0.1f;
        }
    }
}

int main(void)
{
    REG(GPIO_BASE + 4) = 0; // 出力に設定
	REG(GPIO_BASE) = 0;

    volatile char* args = (volatile char*)(RISCV_ARGS_BUFFER + DMEM_BASE);

    unsigned int* func = (unsigned int*)args;

    if(*func==FUNC_PREPROCESS){
        unsigned int* lidar_points_addr = (unsigned int*)(args + 64);
        float* lidar_points = (float*)(*lidar_points_addr + DMEM_BASE);
        int* n_points = (int*)(args + 72);
        float* z_offset = (float*)(args + 80);
        int* input_quant_scale = (int*)(args + 88);
        int8_t* lidar_image = preprocess(lidar_points, *n_points, *z_offset, *input_quant_scale);
        long* lidar_image_addr = (long*)(args + 96);
        *lidar_image_addr = (unsigned int)lidar_image - (unsigned int)base_addr;
    }
    else if(*func==FUNC_SUPPRESS){
        /*unsigned int* input_image_addr = (unsigned int*)(args + 64);*/
        /*int8_t* input_image = (int8_t*)(*input_image_addr + DMEM_BASE);*/
        int8_t* input_image = (int8_t*)(LIDAR_IMAGE_BUFFER + DMEM_BASE);
        unsigned int* centroid_addr = (unsigned int*)(args + 72);
        float* centroid = (float*)(*centroid_addr + DMEM_BASE);
        unsigned int* confidence_addr = (unsigned int*)(args + 80);
        float* confidence = (float*)(*confidence_addr + DMEM_BASE);
        int* n_preds = (int*)(args + 88);
        suppress_predictions_without_lidar_points(
            input_image, centroid, confidence, *n_preds
        );
    }

	REG(GPIO_BASE) = 0x01; // 終了通知
	while(1) {}

	return 0;
}
